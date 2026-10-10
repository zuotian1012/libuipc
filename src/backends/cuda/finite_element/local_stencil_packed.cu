#include <finite_element/local_stencil_packed.h>
#include <finite_element/local_stencil_csfd.h>
#include <finite_element/local_stencil.h>
#include <cuda_tool/cub.h>
#include <cuda_tool/eigen/atomic.h>
#include <kernel_cout.h>
#include <limits>
#include <stdexcept>

namespace uipc::backend::cuda::fem
{
namespace
{
    namespace ct = cuda_tool;
    template <typename T>
    void overwrite_resize(ct::DeviceBuffer<T>& buffer, SizeT size)
    {
        if(size > buffer.capacity())
            buffer.reserve_discard(size + size / 10);
        buffer.resize_discard(size);
    }
    __global__ void packed_count_kernel(ct::CBufferView<Vector4i> stencils,
                                        ct::BufferView<IndexT>    counts)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= stencils.size())
            return;
        for(int a = 0; a < 4; ++a)
            if(stencils(k)(a) >= 0)
                atomicAdd(&counts(stencils(k)(a)), 1);
    }
    __global__ void packed_slots_kernel(ct::CBufferView<Vector4i> stencils,
                                        ct::BufferView<IndexT>    counts,
                                        ct::BufferView<IndexT>    refs,
                                        ct::BufferView<Vector4i>  slots,
                                        IndexT                    stride)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= stencils.size())
            return;
        Vector4i output = Vector4i::Constant(-1);
        for(int a = 0; a < 4; ++a)
        {
            const int v = stencils(k)(a);
            if(v < 0)
                continue;
            const int lane = atomicAdd(&counts(v), 1);
            UIPC_KERNEL_ASSERT(lane < stride, "Packed stencil slot overflow");
            output(a)       = v * stride + lane;
            refs(output(a)) = k;
        }
        slots(k) = output;
    }
    // One energy primitive enumerates each incident solve stencil exactly once.
    // The same topology pass counts and then emits its pairs into scanned storage.
    template <bool Fill>
    __global__ void packed_pairs_kernel(PackedStencilContext    c,
                                        ct::CBufferView<IndexT> counts,
                                        ct::CBufferView<IndexT> refs,
                                        IndexT                  stride,
                                        ct::BufferView<IndexT>  pair_counts,
                                        ct::CBufferView<IndexT> offsets,
                                        ct::BufferView<PackedStencilPair> pairs)
    {
        const int e  = blockIdx.x * blockDim.x + threadIdx.x;
        const int ne = c.elements.size();
        if(e >= ne + c.contacts.size())
            return;
        if(e < ne && c.dt == 0)
        {
            if constexpr(!Fill)
                pair_counts(e) = 0;
            return;
        }
        const Vector4i ids = e < ne ? c.elements(e) : c.contacts(e - ne);
        int            n   = 0;
        for(int a = 0; a < 4; ++a)
        {
            const int v = ids(a);
            if(v < 0)
                continue;
            for(int lane = 0; lane < counts(v); ++lane)
            {
                const int  k       = refs(v * stride + lane);
                const auto target  = c.stencils(k);
                bool       visited = false;
                for(int b = 0; b < a; ++b)
                    if(ids(b) >= 0)
                        for(int j = 0; j < 4; ++j)
                            visited |= ids(b) == target(j);
                if(visited)
                    continue;
                if constexpr(Fill)
                {
                    PackedStencilPair pair;
                    pair.target = k;
                    pair.term   = e;
                    for(int i = 0; i < 4; ++i)
                        if(ids(i) >= 0)
                            for(int j = 0; j < 4; ++j)
                                if(ids(i) == target(j))
                                    pair.local(i) = j;
                    pairs(offsets(e) + n) = pair;
                }
                ++n;
            }
        }
        if constexpr(!Fill)
            pair_counts(e) = n;
    }
    __global__ void packed_cg_init_kernel(PackedStencilContext c,
                                          ct::BufferView<PackedStencilCG> states)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= states.size())
            return;
        PackedStencilCG s;
        s.g.setZero();
        if(!c.colors.size() || c.colors(k) == c.color)
            for(int a = 0; a < 4; ++a)
            {
                const int v = c.stencils(k)(a);
                if(v >= 0 && !c.fixed(v))
                    s.g.segment<3>(3 * a) =
                        c.elastic_gradients(v) + c.contact_gradients(v);
            }
        s.r         = -s.g;
        s.p         = s.r;
        s.r2        = s.r.squaredNorm();
        s.tolerance = 1e-12 + 1e-10 * sqrt(s.r2);
        s.active    = sqrt(s.r2) > s.tolerance;
        UIPC_KERNEL_ASSERT(stencil_finite(s.r2), "Nonfinite packed stencil gradient");
        states(k) = s;
    }
    __global__ void packed_distribute_kernel(PackedStencilContext      c,
                                             ct::CBufferView<Vector4i> slots,
                                             ct::BufferView<PackedStencilCG> states,
                                             ct::BufferView<Vector3>  p,
                                             ct::BufferView<Vector12> products,
                                             bool                     final)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= states.size())
            return;
        auto& s = states(k);
        if(final && s.g.squaredNorm() > 0 && s.g.dot(s.x) >= 0)
        {
            s.x        = -s.g;
            s.fallback = 1;
        }
        const Vector12 d            = final    ? s.x :
                                      s.active ? s.p :
                                                 Vector12::Zero().eval();
        Vector12       mass_product = Vector12::Zero();
        for(int a = 0; a < 4; ++a)
        {
            const int v = c.stencils(k)(a);
            if(v < 0)
                continue;
            const Vector3 q =
                c.fixed(v) ? Vector3::Zero().eval() : d.segment<3>(3 * a).eval();
            p(slots(k)(a))                 = q;
            mass_product.segment<3>(3 * a) = c.masses(v) * q;
        }
        products(k) = mass_product;
    }
    __global__ void packed_product_kernel(PackedStencilContext c,
                                          ct::CBufferView<PackedStencilPair> pairs,
                                          ct::CBufferView<Vector4i> slots,
                                          ct::CBufferView<Vector3>  p,
                                          ct::BufferView<Vector12>  products)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= pairs.size())
            return;
        const auto pair = pairs(i);
        Vector12   d    = Vector12::Zero();
        for(int a = 0; a < 4; ++a)
            if(pair.local(a) >= 0)
                d.segment<3>(3 * a) = p(slots(pair.target)(pair.local(a)));
        if(d.squaredNorm() == 0)
            return;
        Vector12 hd;
        if(pair.term < c.elements.size())
        {
            const auto term = c.terms(pair.term);
            if(term.type == 10)
            {
                const auto v = c.elements(pair.term);
                Matrix3x3  ds;
                for(int a = 0; a < 3; ++a)
                    ds.col(a) = c.xs(v(a + 1)) - c.xs(v(0));
                hd = snk_stencil_product_csfd(ds * term.inverse,
                                              term.inverse,
                                              term.mu,
                                              term.lambda,
                                              term.weight * c.dt * c.dt,
                                              d);
            }
            else
                hd = c.elastic_hessians(pair.term) * d;
        }
        else
            hd = c.contact_hessians(pair.term - c.elements.size()) * d;
        for(int a = 0; a < 4; ++a)
        {
            const int b = pair.local(a);
            if(b >= 0 && !c.fixed(c.stencils(pair.target)(b)))
                for(int j = 0; j < 3; ++j)
                    atomicAdd(&products(pair.target)(3 * b + j), hd(3 * a + j));
        }
    }
    __global__ void packed_cg_advance_kernel(ct::BufferView<PackedStencilCG> states,
                                             ct::CBufferView<Vector12> products)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= states.size())
            return;
        auto& s = states(k);
        if(!s.active)
            return;
        const Vector12 hp        = products(k);
        const Float    curvature = s.p.dot(hp);
        UIPC_KERNEL_ASSERT(stencil_finite(curvature), "Nonfinite packed stencil curvature");
        if(curvature <= 0)
        {
            s.negative = 1;
            s.active   = 0;
            return;
        }
        const Float alpha = s.r2 / curvature;
        s.x += alpha * s.p;
        s.r -= alpha * hp;
        const Float next = s.r.squaredNorm();
        UIPC_KERNEL_ASSERT(stencil_finite(next), "Nonfinite packed stencil residual");
        s.p  = s.r + (next / s.r2) * s.p;
        s.r2 = next;
        ++s.iterations;
        s.active = sqrt(next) > s.tolerance;
    }
    __global__ void packed_cg_finish_kernel(ct::CBufferView<PackedStencilCG> states,
                                            ct::CBufferView<Vector12> products,
                                            ct::BufferView<Vector12> directions,
                                            ct::BufferView<Float>    slopes,
                                            ct::BufferView<Vector4> diagnostics)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= states.size())
            return;
        const auto  s        = states(k);
        const Float residual = (s.g + products(k)).norm();
        UIPC_KERNEL_ASSERT(stencil_finite(residual) && stencil_finite(s.x.squaredNorm()),
                           "Invalid packed stencil direction");
        directions(k) = s.x;
        slopes(k)     = s.g.dot(s.x);
        if(diagnostics.size())
            diagnostics(k) =
                Vector4(s.iterations, s.negative, s.fallback, residual / (s.g.norm() + 1e-30));
    }
}  // namespace

void LocalStencilPackedSolver::build(const PackedStencilContext& c)
{
    const int n = c.xs.size(), k = c.stencils.size(),
              e = c.elements.size() + c.contacts.size();
    overwrite_resize(m_counts, n);
    overwrite_resize(m_scalar, 1);
    overwrite_resize(m_slots, k);
    CUDA_TOOL_CHECK(cudaMemset(m_counts.data(), 0, n * sizeof(IndexT)));
    packed_count_kernel<<<(k + 127) / 128, 128>>>(c.stencils, m_counts.view());
    cuda_tool::DeviceReduce().Max(m_counts.data(), m_scalar.data(), n);
    m_scalar.view().copy_to(&m_stride);
    const SizeT slots = static_cast<SizeT>(n) * m_stride;
    if(slots > static_cast<SizeT>(std::numeric_limits<IndexT>::max()))
        throw std::runtime_error("Packed stencil direction layout exceeds int32 capacity");
    overwrite_resize(m_refs, slots);
    overwrite_resize(m_p, slots);
    CUDA_TOOL_CHECK(cudaMemset(m_counts.data(), 0, n * sizeof(IndexT)));
    packed_slots_kernel<<<(k + 127) / 128, 128>>>(
        c.stencils, m_counts.view(), m_refs.view(), m_slots.view(), m_stride);
    overwrite_resize(m_pair_counts, e + 1);
    overwrite_resize(m_pair_offsets, e + 1);
    CUDA_TOOL_CHECK(cudaMemset(m_pair_counts.data() + e, 0, sizeof(IndexT)));
    packed_pairs_kernel<false><<<(e + 127) / 128, 128>>>(
        c, m_counts.cview(), m_refs.cview(), m_stride, m_pair_counts.view(), {}, {});
    cuda_tool::DeviceScan().ExclusiveSum(m_pair_counts.data(), m_pair_offsets.data(), e + 1);
    IndexT pairs;
    m_pair_offsets.cview().subview(e, 1).copy_to(&pairs);
    if(pairs < 0)
        throw std::runtime_error("Packed stencil energy incidence overflow");
    overwrite_resize(m_pairs, pairs);
    packed_pairs_kernel<true><<<(e + 127) / 128, 128>>>(c,
                                                        m_counts.cview(),
                                                        m_refs.cview(),
                                                        m_stride,
                                                        {},
                                                        m_pair_offsets.cview(),
                                                        m_pairs.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
}
void LocalStencilPackedSolver::product(const PackedStencilContext& c, bool final)
{
    const int k = c.stencils.size();
    packed_distribute_kernel<<<(k + 127) / 128, 128>>>(
        c, m_slots.cview(), m_cg.view(), m_p.view(), m_products.view(), final);
    if(m_pairs.size())
        packed_product_kernel<<<(m_pairs.size() + 127) / 128, 128>>>(
            c, m_pairs.cview(), m_slots.cview(), m_p.cview(), m_products.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
}
void LocalStencilPackedSolver::solve(const PackedStencilContext&     c,
                                     cuda_tool::BufferView<Vector12> directions,
                                     cuda_tool::BufferView<Float>    slopes,
                                     cuda_tool::BufferView<Vector4> diagnostics)
{
    const int k = c.stencils.size();
    if(!k)
        return;
    build(c);
    overwrite_resize(m_cg, k);
    overwrite_resize(m_products, k);
    packed_cg_init_kernel<<<(k + 127) / 128, 128>>>(c, m_cg.view());
    for(int iteration = 0; iteration < 12; ++iteration)
    {
        product(c, false);
        packed_cg_advance_kernel<<<(k + 127) / 128, 128>>>(m_cg.view(),
                                                           m_products.cview());
    }
    product(c, true);
    packed_cg_finish_kernel<<<(k + 127) / 128, 128>>>(
        m_cg.cview(), m_products.cview(), directions, slopes, diagnostics);
    CUDA_TOOL_CHECK(cudaGetLastError());
}
}  // namespace uipc::backend::cuda::fem
