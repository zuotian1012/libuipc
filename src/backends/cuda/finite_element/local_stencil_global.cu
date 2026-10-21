#include <finite_element/local_stencil_global.h>
#include <cuda_tool/cub.h>
#include <cuda_tool/eigen/atomic.h>
#include <kernel_cout.h>
#include <cmath>
#include <stdexcept>

namespace uipc::backend::cuda::fem
{
namespace
{
    namespace ct = cuda_tool;

    __global__ void global_stencil_mass_kernel(GlobalStencilContext     c,
                                               ct::CBufferView<Vector3> input,
                                               ct::BufferView<Vector3>  output,
                                               bool diagonal)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= c.masses.size())
            return;
        if(c.fixed(v))
            output(v).setZero();
        else if(diagonal)
            output(v).setConstant(c.masses(v));
        else
            output(v) = c.masses(v) * input(v);
    }

    __global__ void global_stencil_terms_kernel(ct::CBufferView<Vector4i> ids,
                                                ct::CBufferView<Matrix12x12> hessians,
                                                ct::CBufferView<IndexT>  fixed,
                                                ct::CBufferView<Vector3> input,
                                                ct::BufferView<Vector3>  output,
                                                bool diagonal)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if(e >= ids.size())
            return;
        const auto s = ids(e);
        Vector12   d = Vector12::Zero();
        for(int a = 0; a < 4; ++a)
        {
            UIPC_KERNEL_ASSERT(s(a) == -1 || (s(a) >= 0 && SizeT(s(a)) < fixed.size()),
                               "Global stencil primitive vertex out of range");
            if(!diagonal && s(a) >= 0 && !fixed(s(a)))
                d.segment<3>(3 * a) = input(s(a));
        }
        Vector12 value;
        if(diagonal)
            value = hessians(e).diagonal();
        else
            value = hessians(e) * d;
        for(int a = 0; a < 4; ++a)
            if(s(a) >= 0 && !fixed(s(a)))
            {
                const Vector3 block = value.segment<3>(3 * a);
                ct::eigen::atomic_add(output(s(a)), block);
            }
    }

    __global__ void global_stencil_init_kernel(GlobalStencilContext c,
                                               ct::CBufferView<Vector3> gradient,
                                               ct::BufferView<Vector3> diagonal,
                                               ct::BufferView<Vector3> direction,
                                               ct::BufferView<Vector3> r,
                                               ct::BufferView<Vector3> z,
                                               ct::BufferView<Vector3> p)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= c.masses.size())
            return;
        direction(v).setZero();
        for(int j = 0; j < 3; ++j)
        {
            const Float floor = c.masses(v) * 1e-8;
            UIPC_KERNEL_ASSERT(c.fixed(v) || (isfinite(floor) && floor > 0),
                               "Global stencil correction requires positive free mass");
            const Float value = diagonal(v)(j);
            UIPC_KERNEL_ASSERT(isfinite(value), "Nonfinite global Hessian diagonal");
            diagonal(v)(j) = c.fixed(v) ? 1 : 1 / fmax(fabs(value), floor);
        }
        r(v) = c.fixed(v) ? Vector3::Zero().eval() : (-gradient(v)).eval();
        z(v) = diagonal(v).cwiseProduct(r(v));
        p(v) = z(v);
    }

    __global__ void global_stencil_dot_kernel(ct::CBufferView<Vector3> a,
                                              ct::CBufferView<Vector3> b,
                                              ct::BufferView<Float>    values)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v < values.size())
            values(v) = a(v).dot(b(v));
    }

    __global__ void global_stencil_update_kernel(ct::BufferView<Vector3>  x,
                                                 ct::BufferView<Vector3>  r,
                                                 ct::BufferView<Vector3>  z,
                                                 ct::CBufferView<Vector3> p,
                                                 ct::CBufferView<Vector3> ap,
                                                 ct::CBufferView<Vector3> diagonal,
                                                 Float alpha)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= x.size())
            return;
        x(v) += alpha * p(v);
        r(v) -= alpha * ap(v);
        z(v) = diagonal(v).cwiseProduct(r(v));
    }

    __global__ void global_stencil_direction_kernel(ct::BufferView<Vector3>  p,
                                                    ct::CBufferView<Vector3> z,
                                                    Float beta)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v < p.size())
            p(v) = z(v) + beta * p(v);
    }

    __global__ void global_stencil_true_residual_kernel(ct::CBufferView<IndexT> fixed,
                                                        ct::CBufferView<Vector3> g,
                                                        ct::CBufferView<Vector3> ax,
                                                        ct::BufferView<Vector3> r)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v < r.size())
            r(v) = fixed(v) ? Vector3::Zero().eval() : (-g(v) - ax(v)).eval();
    }

    __global__ void global_stencil_fallback_kernel(ct::CBufferView<IndexT> fixed,
                                                   ct::CBufferView<Vector3> g,
                                                   ct::CBufferView<Vector3> diagonal,
                                                   ct::BufferView<Vector3> x)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v < x.size())
            x(v) = fixed(v) ? Vector3::Zero().eval() :
                              (-diagonal(v).cwiseProduct(g(v))).eval();
    }

    void validate(const GlobalStencilContext& c, SizeT input, SizeT output)
    {
        if(!input || input != c.masses.size() || output != input
           || c.fixed.size() != input || c.elements.size() != c.elastic_hessians.size()
           || c.contacts.size() != c.contact_hessians.size())
            throw std::runtime_error("Global stencil context size mismatch.");
    }

    void terms(const GlobalStencilContext& c,
               ct::CBufferView<Vector3>    input,
               ct::BufferView<Vector3>     output,
               bool                        diagonal)
    {
        if(c.elements.size())
            global_stencil_terms_kernel<<<(c.elements.size() + 63) / 64, 64>>>(
                c.elements, c.elastic_hessians, c.fixed, input, output, diagonal);
        if(c.contacts.size())
            global_stencil_terms_kernel<<<(c.contacts.size() + 63) / 64, 64>>>(
                c.contacts, c.contact_hessians, c.fixed, input, output, diagonal);
        CUDA_TOOL_CHECK(cudaGetLastError());
    }
}  // namespace

void GlobalStencilSolver::product(const GlobalStencilContext& c,
                                  ct::CBufferView<Vector3>    input,
                                  ct::BufferView<Vector3>     output)
{
    validate(c, input.size(), output.size());
    if(input.data() == output.data())
        throw std::runtime_error("Global stencil product requires distinct buffers.");
    global_stencil_mass_kernel<<<(input.size() + 127) / 128, 128>>>(c, input, output, false);
    terms(c, input, output, false);
}

Float GlobalStencilSolver::dot(ct::CBufferView<Vector3> a, ct::CBufferView<Vector3> b)
{
    m_values.resize_discard(a.size());
    m_scalar.resize_discard(1);
    global_stencil_dot_kernel<<<(a.size() + 127) / 128, 128>>>(a, b, m_values.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    ct::DeviceReduce().Sum(m_values.data(), m_scalar.data(), a.size());
    Float value;
    m_scalar.view().copy_to(&value);
    if(!std::isfinite(value))
        throw std::runtime_error("Nonfinite global stencil PCG reduction.");
    return value;
}

GlobalStencilResult GlobalStencilSolver::solve(const GlobalStencilContext& c,
                                               ct::CBufferView<Vector3>    g,
                                               ct::BufferView<Vector3>     x,
                                               IndexT max_iterations,
                                               Float  tolerance)
{
    validate(c, g.size(), x.size());
    if(g.data() == x.data())
        throw std::runtime_error("Global stencil solve requires distinct gradient and direction buffers.");
    if(max_iterations < 1 || !std::isfinite(tolerance) || tolerance <= 0 || tolerance >= 1)
        throw std::runtime_error("Invalid global stencil PCG controls.");
    const SizeT n = g.size();
    m_diagonal.resize_discard(n);
    m_r.resize_discard(n);
    m_z.resize_discard(n);
    m_p.resize_discard(n);
    m_ap.resize_discard(n);
    global_stencil_mass_kernel<<<(n + 127) / 128, 128>>>(c, g, m_diagonal.view(), true);
    terms(c, g, m_diagonal.view(), true);
    global_stencil_init_kernel<<<(n + 127) / 128, 128>>>(
        c, g, m_diagonal.view(), x, m_r.view(), m_z.view(), m_p.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    GlobalStencilResult result;
    const Float         initial_r2 = dot(m_r.cview(), m_r.cview());
    if(initial_r2 == 0)
        return result;
    Float rho = dot(m_r.cview(), m_z.cview());
    for(IndexT k = 0; k < max_iterations; ++k)
    {
        product(c, m_p.cview(), m_ap.view());
        const Float curvature = dot(m_p.cview(), m_ap.cview());
        if(curvature <= 0)
        {
            result.negative_curvature = true;
            break;
        }
        const Float alpha = rho / curvature;
        if(!std::isfinite(alpha) || alpha <= 0)
            throw std::runtime_error("Invalid global stencil PCG step.");
        global_stencil_update_kernel<<<(n + 127) / 128, 128>>>(
            x, m_r.view(), m_z.view(), m_p.cview(), m_ap.cview(), m_diagonal.cview(), alpha);
        CUDA_TOOL_CHECK(cudaGetLastError());
        ++result.iterations;
        const Float r2 = dot(m_r.cview(), m_r.cview());
        if(r2 <= tolerance * tolerance * initial_r2)
            break;
        const Float next_rho = dot(m_r.cview(), m_z.cview());
        global_stencil_direction_kernel<<<(n + 127) / 128, 128>>>(
            m_p.view(), m_z.cview(), next_rho / rho);
        CUDA_TOOL_CHECK(cudaGetLastError());
        rho = next_rho;
    }
    result.slope = dot(g, x);
    if(result.slope >= 0)
    {
        global_stencil_fallback_kernel<<<(n + 127) / 128, 128>>>(
            c.fixed, g, m_diagonal.cview(), x);
        CUDA_TOOL_CHECK(cudaGetLastError());
        result.fallback = true;
        result.slope    = dot(g, x);
    }
    if(result.slope >= 0)
        throw std::runtime_error("Global stencil correction is not a descent direction.");
    // Report the true residual of the submitted direction, not recursive CG r.
    product(c, x, m_ap.view());
    global_stencil_true_residual_kernel<<<(n + 127) / 128, 128>>>(
        c.fixed, g, m_ap.cview(), m_r.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    result.relative_residual = std::sqrt(dot(m_r.cview(), m_r.cview()) / initial_r2);
    return result;
}
}  // namespace uipc::backend::cuda::fem
