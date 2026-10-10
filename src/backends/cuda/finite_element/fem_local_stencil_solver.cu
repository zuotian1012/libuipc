#include <finite_element/fem_local_stencil_solver.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/local_stencil.h>
#include <finite_element/local_stencil_elastic.h>
#include <finite_element/local_stencil_operator.h>
#include <sim_engine.h>
#include <finite_element/fem_dytopo_effect_receiver.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <collision_detection/global_trajectory_filter.h>
#include <cuda_tool/eigen/atomic.h>
#include <contact_system/global_contact_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <kernel_cout.h>
#include <cuda_tool/cub.h>
#include <uipc/common/timer.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <chrono>

namespace uipc::backend
{
template <>
class SimSystemCreator<cuda::FEMLocalStencilSolver>
{
  public:
    static U<cuda::FEMLocalStencilSolver> create(SimEngine& engine)
    {
        auto scene = dynamic_cast<cuda::SimEngine&>(engine).world().scene();
        const auto& config = scene.config();
        if(config.find<std::string>("newton/solver")->view()[0] != "local_stencil")
            return nullptr;
        if(config.find<IndexT>("diff_sim/enable")->view()[0]
           || config.find<std::string>("contact/constitution")->view()[0] != "ipc"
           || config.find<std::string>("integrator/type")->view()[0] != "bdf1")
            throw SimEngineException("local_stencil requires diff_sim disabled, ipc pipeline and bdf1.");
        if(config.find<IndexT>("contact/enable")->view()[0]
           && config.find<IndexT>("contact/friction/enable")->view()[0])
            throw SimEngineException("local_stencil contact currently requires friction disabled.");
        if(config.find<Float>("newton/velocity_tol_relative")->view()[0] != 0)
            throw SimEngineException("local_stencil uses the absolute newton/velocity_tol; set velocity_tol_relative to zero.");
        const auto uids = scene.constitution_tabular().uids();
        if(uids.empty()
           || std::any_of(uids.begin(),
                          uids.end(),
                          [](U64 uid) {
                              return uid != 10 && uid != 11 && uid != 819 && uid != 17;
                          }))
            throw SimEngineException("local_stencil supports SNK1 tetrahedra, BW/NH cloth, DiscreteShellBending and fixed vertices.");
        return uipc::make_unique<cuda::FEMLocalStencilSolver>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
namespace
{
    namespace ct = cuda_tool;

    __global__ void stencil_element_derivatives_kernel(ct::CBufferView<Vector4i> elements,
                                                       ct::CBufferView<Vector3> xs,
                                                       ct::CBufferView<fem::StencilElasticTerm> terms,
                                                       ct::BufferView<Vector12> gradients,
                                                       ct::BufferView<Matrix12x12> hessians,
                                                       Float dt,
                                                       bool  inertia_only,
                                                       bool assemble_tet_hessian)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= elements.size())
            return;
        if(inertia_only)
        {
            gradients(t).setZero();
            hessians(t).setZero();
            return;
        }
        Vector3 p[4];
        for(int a = 0; a < 4; ++a)
            if(elements(t)(a) >= 0)
                p[a] = xs(elements(t)(a));
        fem::stencil_elastic_derivatives(terms(t), p, dt, gradients(t), hessians(t), assemble_tet_hessian);
    }

    __global__ void stencil_primary_owners_kernel(ct::CBufferView<IndexT> offsets,
                                                  ct::CBufferView<IndexT> incident,
                                                  ct::CBufferView<IndexT> colors,
                                                  ct::BufferView<IndexT> owners,
                                                  IndexT color_offset,
                                                  IndexT color,
                                                  bool   hybrid)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= owners.size())
            return;
        int owner = -1;
        if(hybrid)
            for(int k = offsets(v); k < offsets(v + 1); ++k)
                if(colors(color_offset + incident(k)) == color)
                {
                    UIPC_KERNEL_ASSERT(owner < 0, "Conflicting stencils in independent color");
                    owner = incident(k);
                }
        owners(v) = owner;
    }

    __global__ void stencil_vertex_derivatives_kernel(
        ct::CBufferView<Vector4i>    elements,
        ct::CBufferView<Vector3>     xs,
        ct::CBufferView<Vector3>     predicted,
        ct::CBufferView<Float>       masses,
        ct::CBufferView<IndexT>      offsets,
        ct::CBufferView<IndexT>      incident,
        ct::CBufferView<Vector12>    gradients,
        ct::CBufferView<Matrix12x12> hessians,
        ct::BufferView<Vector3>      vertex_gradients,
        ct::BufferView<Matrix3x3>    diagonal,
        bool                         assemble_diagonal)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= xs.size())
            return;
        Vector3   g = masses(v) * (xs(v) - predicted(v));
        Matrix3x3 H = masses(v) * Matrix3x3::Identity();
        for(int k = offsets(v); k < offsets(v + 1); ++k)
        {
            const int e = incident(k);
            for(int a = 0; a < 4; ++a)
                if(elements(e)(a) == v)
                {
                    g += gradients(e).segment<3>(3 * a);
                    if(assemble_diagonal)
                        H += hessians(e).block<3, 3>(3 * a, 3 * a);
                }
        }
        vertex_gradients(v) = g;
        diagonal(v)         = H;
    }

    __global__ void stencil_local_solve_kernel(ct::CBufferView<Vector4i> elements,
                                               ct::CBufferView<Vector4i> contacts,
                                               ct::CBufferView<IndexT> fixed,
                                               ct::CBufferView<IndexT> vertex_offsets,
                                               ct::CBufferView<IndexT> vertex_elements,
                                               ct::CBufferView<Vector3> vertex_gradients,
                                               ct::CBufferView<Matrix3x3> diagonal,
                                               ct::CBufferView<Matrix12x12> hessians,
                                               fem::LocalStencilContactView contact,
                                               ct::BufferView<Vector12> directions,
                                               ct::BufferView<Float> slopes,
                                               ct::BufferView<Vector4> diagnostics,
                                               ct::CBufferView<IndexT> active_colors,
                                               IndexT active_color,
                                               bool   csfd,
                                               ct::CBufferView<fem::StencilElasticTerm> terms,
                                               ct::CBufferView<Vector3> xs,
                                               ct::CBufferView<Float>   masses,
                                               Float                    dt,
                                               ct::CBufferView<Vector4i> update_stencils)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= directions.size())
            return;
        if(active_colors.size() && active_colors(t) != active_color)
        {
            directions(t).setZero();
            slopes(t) = 0;
            if(diagnostics.size())
                diagnostics(t).setZero();
            return;
        }
        const Vector4i stencil = update_stencils.size() ? update_stencils(t) :
                                 t < elements.size()    ? elements(t) :
                                                       contacts(t - elements.size());
        if(csfd)
        {
            Vector12 gradient = Vector12::Zero();
            for(int a = 0; a < 4; ++a)
                if(stencil(a) >= 0 && !fixed(stencil(a)))
                    gradient.segment<3>(3 * a) = vertex_gradients(stencil(a))
                                                 + contact.gradients(stencil(a));
            const fem::LocalStencilCSFDOperator product{
                stencil, elements, terms, xs, masses, fixed, vertex_offsets, vertex_elements, hessians, contact, dt};
            const auto result = fem::local_stencil_direction(gradient, product);
            UIPC_KERNEL_ASSERT(result.valid, "Invalid CSFD stencil direction %d", t);
            directions(t) = result.displacement;
            slopes(t)     = gradient.dot(result.displacement);
            if(diagnostics.size())
                diagnostics(t) =
                    Vector4(result.iterations,
                            result.nonpositive_curvature,
                            result.gradient_fallback,
                            result.residual_norm / (gradient.norm() + 1e-30));
            return;
        }
        fem::LocalStencilSystem system;
        // Shared diagonal blocks and gradients are assembled once per vertex.
        // Off-diagonal blocks are still the exact restriction of ALL incident
        // elastic terms; no local energy, pin, or neighbor coupling is dropped.
        for(int i = 0; i < 4; ++i)
        {
            const int v = stencil(i);
            if(v < 0)
                continue;
            system.gradient.segment<3>(3 * i)        = vertex_gradients(v);
            system.hessian.block<3, 3>(3 * i, 3 * i) = diagonal(v);
            for(int entry = vertex_offsets(v); entry < vertex_offsets(v + 1); ++entry)
            {
                const int      e   = vertex_elements(entry);
                const Vector4i ids = elements(e);
                int            a   = 0;
                while(a < 4 && ids(a) != v)
                    ++a;
                for(int j = i + 1; j < 4; ++j)
                    if(stencil(j) >= 0)
                        for(int b = 0; b < 4; ++b)
                            if(ids(b) == stencil(j))
                            {
                                system.hessian.block<3, 3>(3 * i, 3 * j) +=
                                    hessians(e).block<3, 3>(3 * a, 3 * b);
                                system.hessian.block<3, 3>(3 * j, 3 * i) +=
                                    hessians(e).block<3, 3>(3 * b, 3 * a);
                            }
            }
        }
        contact.add(system, stencil);
        for(int i = 0; i < 4; ++i)
            if(stencil(i) < 0 || fixed(stencil(i)))
                system.fix_vertex(i);
        // Section 5.1: solve the same complete local problem for every color.
        // Priority is a submission rule, never an extra Dirichlet constraint.
        const auto result = fem::local_stencil_direction(system);
        UIPC_KERNEL_ASSERT(result.valid, "Invalid local stencil direction %d", t);
        directions(t) = result.displacement;
        slopes(t)     = system.gradient.dot(result.displacement);
        if(diagnostics.size())
            diagnostics(t) =
                Vector4(result.iterations,
                        result.nonpositive_curvature,
                        result.gradient_fallback,
                        result.residual_norm / (system.gradient.norm() + 1e-30));
    }

    __global__ void stencil_scatter_contact_kernel(ct::CBufferView<Vector4i> contacts,
                                                   ct::CBufferView<Vector12> directions,
                                                   ct::BufferView<Vector3> dx,
                                                   ct::CBufferView<IndexT> primary_owner,
                                                   IndexT elastic_count,
                                                   Float  weight)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= contacts.size())
            return;
        for(int i = 0; i < 4; ++i)
            if(contacts(t)(i) >= 0 && primary_owner(contacts(t)(i)) < 0)
            {
                const Vector3 d =
                    weight * directions(t + elastic_count).segment<3>(3 * i);
                ct::eigen::atomic_add(dx(contacts(t)(i)), d);
            }
    }

    __global__ void stencil_gs_contact_owners_kernel(ct::CBufferView<Vector4i> contacts,
                                                     ct::CBufferView<IndexT> colors,
                                                     ct::BufferView<IndexT> owners,
                                                     IndexT elastic_count,
                                                     IndexT color)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= contacts.size() || colors(elastic_count + t) != color)
            return;
        for(int a = 0; a < 4; ++a)
            if(contacts(t)(a) >= 0)
            {
                const int v = contacts(t)(a);
                UIPC_KERNEL_ASSERT(owners(v) < 0, "Conflicting GS contact owner");
                owners(v) = elastic_count + t;
            }
    }

    __global__ void stencil_gather_kernel(ct::CBufferView<Vector4i> elements,
                                          ct::CBufferView<Vector4i> contacts,
                                          ct::CBufferView<Vector3>  xs,
                                          ct::CBufferView<Float>    masses,
                                          ct::CBufferView<IndexT>   fixed,
                                          ct::CBufferView<IndexT>   offsets,
                                          ct::CBufferView<IndexT> incident_elements,
                                          ct::CBufferView<Vector12> directions,
                                          ct::BufferView<Vector3> displacements,
                                          ct::BufferView<Float>   residuals,
                                          ct::BufferView<Float> gradient_norms,
                                          ct::CBufferView<Vector3> contact_gradients,
                                          ct::CBufferView<IndexT> primary_owner,
                                          ct::BufferView<Vector3> total_gradients,
                                          bool  hybrid,
                                          Float weight,
                                          Float dt)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= xs.size())
            return;
        Vector3 dx       = Vector3::Zero();
        Vector3 gradient = total_gradients(v) + contact_gradients(v);
        for(int entry = offsets(v); entry < offsets(v + 1); ++entry)
        {
            const int t = incident_elements(entry);
            for(int i = 0; i < 4; ++i)
                if(elements(t)(i) == v)
                {
                    dx += directions(t).segment<3>(3 * i);
                }
        }
        if(hybrid && primary_owner(v) >= 0)
        {
            const int      t = primary_owner(v);
            const Vector4i ids =
                t < elements.size() ? elements(t) : contacts(t - elements.size());
            for(int a = 0; a < 4; ++a)
                if(ids(a) == v)
                    dx = directions(t).segment<3>(3 * a);
        }
        displacements(v) = fixed(v) ? Vector3::Zero().eval() : (weight * dx).eval();
        total_gradients(v) = fixed(v) ? Vector3::Zero().eval() : gradient;
        residuals(v)       = fixed(v) ? 0 : gradient.norm() / (masses(v) * dt);
        gradient_norms(v)  = fixed(v) ? 0 : gradient.squaredNorm();
        UIPC_KERNEL_ASSERT(fem::stencil_finite(residuals(v)),
                           "Nonfinite stencil residual at vertex %d",
                           v);
    }

    __global__ void stencil_hybrid_average_kernel(ct::BufferView<Vector3> dx,
                                                  ct::CBufferView<Vector3> gradients,
                                                  ct::CBufferView<IndexT> owners,
                                                  ct::CBufferView<IndexT> overlap,
                                                  ct::BufferView<Float> slopes)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= dx.size())
            return;
        if(owners(v) < 0)
            dx(v) /= overlap(v);
        slopes(v) = gradients(v).dot(dx(v));
        // Owned vertices retain the primary stencil's filtered displacement
        // exactly. A rejected local direction is a zero Jacobi contribution,
        // but still participates in the shared-vertex multiplicity.
    }

    __global__ void stencil_determinants_kernel(ct::CBufferView<Vector4i> ids,
                                                ct::CBufferView<Vector3>  xs,
                                                ct::CBufferView<fem::StencilElasticTerm> terms,
                                                ct::BufferView<Float> determinants)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= ids.size())
            return;
        const auto s     = ids(t);
        const auto term  = terms(t);
        Float      value = 1;
        if(term.type == 10)
            value = fem::F(xs(s(0)), xs(s(1)), xs(s(2)), xs(s(3)), term.inverse).determinant();
        else if(term.type != 17)
            value = fem::stencil_triangle_measure(xs(s(0)), xs(s(1)), xs(s(2))) / term.rest_area2
                    - 1e-12;
        determinants(t) = fem::stencil_finite(value) ? value : -1;
    }

    __global__ void stencil_inertial_proposal_kernel(ct::CBufferView<Vector3> xs,
                                                     ct::CBufferView<Vector3> predicted,
                                                     ct::CBufferView<Vector3> gradient,
                                                     ct::CBufferView<IndexT> fixed,
                                                     ct::BufferView<Vector3> proposal,
                                                     ct::BufferView<Float> slopes)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= xs.size())
            return;
        proposal(v) = fixed(v) ? Vector3::Zero().eval() : (predicted(v) - xs(v)).eval();
        slopes(v) = gradient(v).dot(proposal(v));
    }

    __global__ void stencil_combined_cap_kernel(ct::CBufferView<Vector4i> elements,
                                                ct::CBufferView<fem::StencilElasticTerm> terms,
                                                ct::CBufferView<Vector3> xs,
                                                ct::CBufferView<Vector3> dx,
                                                ct::BufferView<Float>    caps)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if(e >= elements.size())
            return;
        Vector3 start[4], end[4];
        for(int a = 0; a < 4; ++a)
            if(elements(e)(a) >= 0)
            {
                const int v = elements(e)(a);
                start[a]    = xs(v);
                end[a]      = xs(v) + dx(v);
            }
        caps(e) = fem::stencil_elastic_cap(terms(e), start, end);
    }

    __global__ void stencil_scale_proposal_kernel(ct::BufferView<Vector3> dx, Float alpha)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v < dx.size())
            dx(v) *= alpha;
    }

    __global__ void sos_active_overlap_kernel(ct::CBufferView<IndexT> offsets,
                                              ct::CBufferView<IndexT> entries,
                                              ct::CBufferView<Vector12> contact_gradients,
                                              ct::CBufferView<IndexT> elastic_offsets,
                                              ct::BufferView<IndexT> overlap,
                                              bool                   warm)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= overlap.size())
            return;
        IndexT degree = warm ? 0 : elastic_offsets(v + 1) - elastic_offsets(v);
        for(int k = offsets(v); k < offsets(v + 1); ++k)
            if(contact_gradients(entries(k)).squaredNorm() > 0)
                ++degree;
        overlap(v) = degree;
    }
    __global__ void sos_update_stencils_kernel(ct::CBufferView<IndexT> overlap,
                                               ct::CBufferView<Vector4i> elements,
                                               ct::CBufferView<Vector4i> contacts,
                                               ct::CBufferView<Vector12> contact_gradients,
                                               ct::BufferView<Vector4i> stencils,
                                               IndexT prefix,
                                               bool   warm)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= stencils.size())
            return;
        if(i < prefix)
            stencils(i) = !warm           ? elements(i) :
                          overlap(i) == 0 ? Vector4i(i, -1, -1, -1) :
                                            Vector4i::Constant(-1).eval();
        else
            stencils(i) = contact_gradients(i - prefix).squaredNorm() > 0 ?
                              contacts(i - prefix) :
                              Vector4i::Constant(-1).eval();
    }
    __global__ void sos_warm_gather_kernel(ct::CBufferView<Vector4i> contacts,
                                           ct::CBufferView<IndexT>   offsets,
                                           ct::CBufferView<IndexT>   entries,
                                           ct::CBufferView<Vector12> directions,
                                           ct::CBufferView<Float>    masses,
                                           ct::CBufferView<IndexT>   fixed,
                                           ct::CBufferView<Vector3> contact_gradient,
                                           ct::BufferView<Vector3> gradient,
                                           ct::BufferView<Vector3> dx,
                                           ct::BufferView<Float>   residuals,
                                           ct::BufferView<Float>   norms,
                                           ct::BufferView<Float>   slopes,
                                           Float                   dt)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= dx.size())
            return;
        Vector3 d      = Vector3::Zero();
        int     degree = 0;
        for(int k = offsets(v); k < offsets(v + 1); ++k)
            if(contacts(entries(k))(0) >= 0)
                ++degree;
        if(degree == 0)
            d = directions(v).head<3>();
        else
        {
            for(int k = offsets(v); k < offsets(v + 1); ++k)
            {
                const int e = entries(k);
                for(int a = 0; a < 4; ++a)
                    if(contacts(e)(a) == v)
                        d += directions(dx.size() + e).segment<3>(3 * a);
            }
            d /= degree;
        }
        Vector3 g = gradient(v) + contact_gradient(v);
        if(fixed(v))
        {
            g.setZero();
            d.setZero();
        }
        gradient(v)  = g;
        dx(v)        = d;
        residuals(v) = g.norm() / (masses(v) * dt);
        norms(v)     = g.squaredNorm();
        slopes(v)    = g.dot(d);
    }

    __global__ void sos_predictor_kernel(ct::CBufferView<Vector3> xs,
                                         ct::CBufferView<Vector3> predicted,
                                         ct::CBufferView<Vector3> velocity,
                                         ct::BufferView<Vector3> previous_velocity,
                                         ct::CBufferView<IndexT> fixed,
                                         ct::BufferView<Vector3> dx,
                                         Float                   dt,
                                         Float                   previous_dt)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= xs.size())
            return;
        const Vector3 external = (predicted(v) - xs(v) - dt * velocity(v)) / (dt * dt);
        const Vector3 acceleration =
            previous_dt > 0 ?
                ((velocity(v) - previous_velocity(v)) / previous_dt).eval() :
                external;
        dx(v) =
            fixed(v) ?
                Vector3::Zero().eval() :
                (dt * velocity(v) + .25 * dt * dt * (acceleration + external)).eval();
        previous_velocity(v) = velocity(v);
    }

    __global__ void stencil_priority_proposal_kernel(ct::CBufferView<Vector4i> elements,
                                                     ct::CBufferView<IndexT> owners,
                                                     ct::CBufferView<IndexT> fixed,
                                                     ct::CBufferView<Vector12> directions,
                                                     ct::BufferView<Vector3> dx)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= dx.size())
            return;
        dx(v).setZero();
        const int e = owners(v);
        if(e < 0 || fixed(v))
            return;
        for(int a = 0; a < 4; ++a)
            if(elements(e)(a) == v)
                dx(v) = directions(e).segment<3>(3 * a);
    }

    __global__ void stencil_add_contact_gradient_kernel(ct::BufferView<Vector3> gradients,
                                                        ct::CBufferView<Vector3> contacts,
                                                        ct::CBufferView<IndexT> fixed)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v < gradients.size())
        {
            gradients(v) += contacts(v);
            if(fixed(v))
                gradients(v).setZero();
        }
    }

    __global__ void stencil_jacobi_submission_kernel(ct::CBufferView<Vector4i> elements,
                                                     ct::CBufferView<Vector4i> contacts,
                                                     ct::CBufferView<Vector4i> update_stencils,
                                                     ct::CBufferView<IndexT> owners,
                                                     ct::CBufferView<Vector3> gradients,
                                                     ct::CBufferView<Vector12> original,
                                                     ct::BufferView<Vector12> directions,
                                                     ct::BufferView<Float> slopes)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if(e >= directions.size())
            return;
        const auto ids   = update_stencils.size() ? update_stencils(e) :
                           e < elements.size()    ? elements(e) :
                                                    contacts(e - elements.size());
        Vector12   d     = original(e);
        Float      slope = 0;
        for(int a = 0; a < 4; ++a)
        {
            const int v = ids(a);
            if(v < 0 || owners(v) >= 0)
                d.segment<3>(3 * a).setZero();
            else
                slope += gradients(v).dot(d.segment<3>(3 * a));
        }
        directions(e) = d;
        slopes(e)     = slope;
    }

    __global__ void stencil_jacobi_gather_kernel(ct::CBufferView<Vector4i> elements,
                                                 ct::CBufferView<Vector4i> contacts,
                                                 ct::CBufferView<IndexT> offsets,
                                                 ct::CBufferView<IndexT> incident,
                                                 ct::CBufferView<IndexT> contact_offsets,
                                                 ct::CBufferView<IndexT> contact_incident,
                                                 ct::CBufferView<IndexT> owners,
                                                 ct::CBufferView<IndexT> overlap,
                                                 ct::CBufferView<IndexT> fixed,
                                                 ct::CBufferView<Vector12> directions,
                                                 ct::BufferView<Vector3> dx)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= dx.size())
            return;
        Vector3 sum = Vector3::Zero();
        if(owners(v) < 0 && !fixed(v))
        {
            for(int k = offsets(v); k < offsets(v + 1); ++k)
                for(int a = 0; a < 4; ++a)
                    if(elements(incident(k))(a) == v)
                        sum += directions(incident(k)).segment<3>(3 * a);
            for(int k = contact_offsets(v); k < contact_offsets(v + 1); ++k)
                for(int a = 0; a < 4; ++a)
                    if(contacts(contact_incident(k))(a) == v)
                        sum += directions(elements.size() + contact_incident(k)).segment<3>(3 * a);
            sum /= overlap(v);
        }
        dx(v) = sum;
    }

    __global__ void stencil_batch_proposal_kernel(ct::CBufferView<Vector3> start,
                                                  ct::CBufferView<Vector3> xs,
                                                  ct::CBufferView<Vector3> gradient,
                                                  ct::BufferView<Vector3> dx,
                                                  ct::BufferView<Float> slopes)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= xs.size())
            return;
        dx(v)     = xs(v) - start(v);
        slopes(v) = gradient(v).dot(dx(v));
    }
}  // namespace

REGISTER_SIM_SYSTEM(FEMLocalStencilSolver);

void FEMLocalStencilSolver::do_build()
{
    m_fem             = &require<FiniteElementMethod>();
    m_vertex_reporter = &require<FiniteElementVertexReporter>();
    if(world().scene().config().find<IndexT>("contact/enable")->view()[0])
    {
        m_contact_receiver  = &require<FEMDyTopoEffectReceiver>();
        m_trajectory_filter = &require<GlobalTrajectoryFilter>();
        m_contact_manager   = &require<GlobalContactManager>();
        m_vertices          = &require<GlobalVertexManager>();
        m_surface           = &require<GlobalSimplicialSurfaceManager>();
        m_planes            = &require<HalfPlane>();
        m_plane_reporter    = &require<HalfPlaneVertexReporter>();
    }
    on_init_scene([this] { init(); });
    on_rebuild_scene(
        [this]
        {
            if(!world().scene().pending_geometries().empty()
               || !world().scene().pending_destroy_ids().empty())
                throw SimEngineException("local_stencil currently requires static topology; rebuild the World after geometry changes.");
        });
}

void FEMLocalStencilSolver::init()
{
    const auto& config = world().scene().config();
    const auto& sweep =
        config.find<std::string>("newton/local_stencil/sweep")->view()[0];
    m_hybrid = sweep == "hybrid";
    m_jacobi = sweep == "jacobi";
    m_gs     = sweep == "gs";
    m_local_filter =
        sweep != "additive"
        && config.find<std::string>("newton/local_stencil/globalization")->view()[0] == "local";
    m_sequential_hybrid =
        config.find<std::string>("newton/local_stencil/hybrid_submission")->view()[0]
        == "sequential";
    if(m_sequential_hybrid && (!m_hybrid || !m_local_filter))
        throw SimEngineException("Sequential hybrid submission requires hybrid/local globalization.");
    m_inertial_guess =
        config.find<std::string>("newton/local_stencil/initial_guess")->view()[0] == "inertial";
    m_color_limit = config.find<IndexT>("newton/local_stencil/color_limit")->view()[0];
    m_gradient_stopping =
        config.find<std::string>("newton/local_stencil/stopping")->view()[0] == "gradient_l2";
    m_gradient_tolerance =
        config.find<Float>("newton/local_stencil/gradient_tol")->view()[0];
    m_primary_coverage =
        config.find<std::string>("newton/local_stencil/color_coverage")->view()[0] == "primary";
    m_inner_sweeps =
        config.find<IndexT>("newton/local_stencil/inner_sweeps")->view()[0];
    m_restart_schedule =
        config.find<IndexT>("newton/local_stencil/restart_schedule_on_recover")->view()[0] != 0;
    m_local_merge_acceptance =
        config.find<std::string>("newton/local_stencil/merge_acceptance")->view()[0] == "local";
    if(m_local_merge_acceptance && !m_local_filter)
        throw SimEngineException("Local merge acceptance requires local globalization.");
    m_convert_ipc_stiffness =
        config.find<std::string>("newton/local_stencil/barrier_stiffness")->view()[0] == "ipc";
    m_expand_barriers_only =
        config.find<std::string>("newton/local_stencil/barrier_activation")->view()[0]
        == "expanded";
    m_custom_barriers =
        config.find<std::string>("newton/local_stencil/barrier_model")->view()[0] == "sos";
    if(m_custom_barriers && !m_local_filter)
        throw SimEngineException("SoS customized barriers require local globalization.");
    m_warm_enabled =
        config.find<IndexT>("newton/local_stencil/warm_start")->view()[0] != 0;
    const auto warm_inversion =
        config.find<std::string>("newton/local_stencil/warm_start_inversion")->view()[0];
    if(warm_inversion != "iterative" && warm_inversion != "after" && warm_inversion != "after_local")
        throw SimEngineException("warm_start_inversion must be iterative, after or after_local.");
    m_warm_post_inversion   = warm_inversion != "iterative";
    m_warm_local_projection = warm_inversion == "after_local";
    m_warm_relaxation =
        config.find<Float>("newton/local_stencil/warm_start_relaxation")->view()[0];
    if(!std::isfinite(m_warm_relaxation) || m_warm_relaxation <= 0 || m_warm_relaxation > 1)
        throw SimEngineException("warm_start_relaxation must be in (0,1].");
    if(m_warm_enabled && !m_custom_barriers)
        throw SimEngineException("Inertia-barrier warm start requires barrier_model=sos.");
    if(m_warm_enabled)
        m_inertial_guess = false;
    m_packed_product =
        config.find<std::string>("newton/local_stencil/hessian_product")->view()[0] == "packed";
    if(m_packed_product && !m_local_filter)
        throw SimEngineException("Packed stencil products require local globalization.");
    m_csfd_product =
        config.find<std::string>("newton/local_stencil/hessian_product")->view()[0] == "csfd";
    if(m_inner_sweeps < 1 || m_inner_sweeps > 256
       || ((!m_hybrid || !m_local_filter) && m_inner_sweeps != 1))
        throw SimEngineException("local_stencil inner_sweeps must be in [1,256]; batching requires hybrid/local globalization.");
    if(!std::isfinite(m_gradient_tolerance) || m_gradient_tolerance <= 0)
        throw SimEngineException("local_stencil gradient_tol must be finite and positive.");
    if(m_color_limit < 1 || m_color_limit > 64)
        throw SimEngineException("local_stencil color_limit must be in [1,64].");
    m_tolerance =
        world().scene().config().find<Float>("newton/velocity_tol")->view()[0];
    const auto vertex_count = m_fem->xs().size();
    if(vertex_count == 0 || m_fem->codim_0ds().size() || m_fem->codim_1ds().size())
        throw SimEngineException("local_stencil requires tetrahedra and/or triangle cloth, with positive vertex mass.");
    init_elastic();
    const auto count = m_elastic_stencils.size();
    if(count == 0)
        throw SimEngineException("local_stencil requires nonempty elastic stencils.");
    vector<Vector4i> elements(count);
    m_elastic_stencils.view().copy_to(elements.data());
    vector<Float> masses(vertex_count);
    m_fem->masses().copy_to(masses.data());
    vector<vector<IndexT>> incident(vertex_count);
    for(SizeT t = 0; t < count; ++t)
        for(int i = 0; i < 4; ++i)
        {
            const auto v = elements[t](i);
            if(v == -1)
                continue;
            if(v < 0 || v >= vertex_count)
                throw SimEngineException("local_stencil: elastic stencil vertex index out of bounds.");
            incident[v].push_back(static_cast<IndexT>(t));
        }
    vector<IndexT> vertex_offsets{0}, vertex_elements;
    SizeT          max_degree = 1;
    for(SizeT v = 0; v < vertex_count; ++v)
    {
        if(incident[v].empty() || !std::isfinite(masses[v]) || masses[v] <= 0)
            throw SimEngineException("local_stencil requires every vertex to have positive mass and incident elastic elements.");
        max_degree = std::max(max_degree, incident[v].size());
        vertex_elements.insert(
            vertex_elements.end(), incident[v].begin(), incident[v].end());
        vertex_offsets.push_back(static_cast<IndexT>(vertex_elements.size()));
    }
    // One uniform weight preserves descent of the summed local directions.
    // Per-vertex averaging would in general change the sign of g.dot(dx).
    m_weight = 1.0 / static_cast<Float>(max_degree);
    init_coloring(elements, incident);
    m_vertex_offsets.copy_from(vertex_offsets);
    m_vertex_elements.copy_from(vertex_elements);
    m_element_gradients.resize_discard(count);
    m_element_hessians.resize_discard(count);
    m_directions.resize_discard(count);
    m_residuals.resize_discard(vertex_count);
    m_gradient_norms.resize_discard(vertex_count);
    m_inertial_proposal.resize_discard(vertex_count);
    m_outer_start.resize_discard(vertex_count);
    m_outer_gradients.resize_discard(vertex_count);
    m_first_proposal.resize_discard(vertex_count);
    m_determinants.resize_discard(count);
    m_scalar.resize_discard(1);
    m_contact_gradients.resize_discard(vertex_count);
    m_contact_counts.resize_discard(vertex_count + 1);
    m_contact_offsets.resize_discard(vertex_count + 1);
    m_contact_cursors.resize_discard(vertex_count);
    m_overlap.resize_discard(vertex_count);
    m_max_overlap.resize_discard(1);
    m_primary_owner.resize_discard(vertex_count);
    m_total_gradients.resize_discard(vertex_count);
    m_vertex_hessians.resize_discard(vertex_count);
    m_global_slopes.resize_discard(vertex_count);
    m_barrier_counts.resize_discard(vertex_count + 1);
    m_barrier_offsets.resize_discard(vertex_count + 1);
    m_barrier_cursors.resize_discard(vertex_count);
    m_local_statistics.resize_discard(5);
    logger::info("Local stencil solver: {} elastic stencils, {} vertices, max overlap {}",
                 count,
                 vertex_count,
                 max_degree);
}

bool FEMLocalStencilSolver::begin_warm_start()
{
    if(m_warm_frame == engine().frame())
        return false;
    m_warm_frame        = engine().frame();
    m_warm_active       = true;
    m_warm_iterations   = 0;
    m_warm_finishing    = false;
    m_warm_needs_rebase = false;
    if(m_warm_post_inversion)
    {
        m_warm_global_origin.resize_discard(m_vertices->positions().size());
        m_warm_global_origin.view().copy_from(m_vertices->positions());
    }
    prepare_custom_contacts(0);  // Capture physical-step origin before prediction.
    const int vertices = static_cast<int>(m_fem->xs().size());
    m_previous_velocity.resize_preserve(vertices);
    sos_predictor_kernel<<<(vertices + 127) / 128, 128>>>(m_fem->xs(),
                                                          m_fem->x_tildes(),
                                                          m_fem->vs(),
                                                          m_previous_velocity.view(),
                                                          m_fem->is_fixed(),
                                                          m_fem->m_impl.dxs.view(),
                                                          world().scene().dt(),
                                                          m_previous_dt);
    CUDA_TOOL_CHECK(cudaGetLastError());
    m_previous_dt = world().scene().dt();
    m_residual = m_gradient_norm = std::numeric_limits<Float>::infinity();
    cap_combined_elastic_step();
    logger::info("SoS acceleration predictor: frame={}; swept pairs and global CCD follow",
                 engine().frame());
    return true;
}

void FEMLocalStencilSolver::solve()
{
    m_inner_index = 0;
    if(m_warm_finishing)
    {
        m_warm_finishing = false;
        if(m_history_changed)
        {
            // Global CCD can discover new pairs along the rebased path. The
            // engine then rejects that proposal at the physical frame origin.
            // Its old warm residual does not certify the expanded objective.
            // Keep the new history and solve Eq.14 again before handing off.
            m_warm_needs_rebase = false;
            logger::info("SoS post-warm objective expanded; continuing inertia-barrier solve");
        }
        else
        {
            m_warm_active = false;
            logger::info("SoS post-warm geometric filtering finished; enabling elasticity");
        }
    }
    if(m_warm_enabled && begin_warm_start())
        return;
    if(m_warm_active)
    {
        solve_sweep(true);
        ++m_warm_iterations;
        logger::info("SoS warm start: iteration={}, velocity_residual={}, gradient_l2={}",
                     m_warm_iterations,
                     m_residual,
                     m_gradient_norm);
        if(!residual_converged())
            return;
        if(m_warm_post_inversion && m_warm_needs_rebase)
        {
            const int vertices = static_cast<int>(m_fem->xs().size());
            // Solve Eq.14 first, then bound the complete guess from the last
            // physical state. Do not ask an inversion-constrained iterate to
            // satisfy the unconstrained inertia/barrier stationarity condition.
            stencil_batch_proposal_kernel<<<(vertices + 127) / 128, 128>>>(
                m_step_origin.cview(),
                m_fem->xs(),
                m_total_gradients.cview(),
                m_fem->m_impl.dxs.view(),
                m_global_slopes.view());
            CUDA_TOOL_CHECK(cudaGetLastError());
            m_fem->m_impl.xs.view().copy_from(m_step_origin.cview());
            m_warm_finishing = true;
            m_residual = m_gradient_norm = std::numeric_limits<Float>::infinity();
            if(m_warm_local_projection)
                project_warm_guess_locally();
            cap_combined_elastic_step();
            logger::info("SoS warm start converged in {} passes; submitting post-warm feasible path",
                         m_warm_iterations);
            return;
        }
        m_warm_active = false;
        logger::info("SoS warm start complete: {} Jacobi passes; enabling elasticity",
                     m_warm_iterations);
    }
    if(m_inner_sweeps == 1)
    {
        solve_sweep(true);
        return;
    }
    const int   vertices = static_cast<int>(m_fem->xs().size());
    const SizeT bytes    = vertices * sizeof(Vector3);
    CUDA_TOOL_CHECK(cudaMemcpy(m_outer_start.data(), m_fem->m_impl.xs.data(), bytes, cudaMemcpyDeviceToDevice));
    Float outer_residual = 0, outer_gradient_norm = 0;
    int   accepted = 0;
    try
    {
        for(; m_inner_index < m_inner_sweeps; ++m_inner_index)
        {
            solve_sweep(m_inner_index == 0);
            if(m_inner_index == 0)
            {
                outer_residual      = m_residual;
                outer_gradient_norm = m_gradient_norm;
                CUDA_TOOL_CHECK(cudaMemcpy(m_outer_gradients.data(),
                                           m_total_gradients.data(),
                                           bytes,
                                           cudaMemcpyDeviceToDevice));
                CUDA_TOOL_CHECK(cudaMemcpy(m_first_proposal.data(),
                                           m_fem->m_impl.dxs.data(),
                                           bytes,
                                           cudaMemcpyDeviceToDevice));
            }
            if(converged())
                break;
            if(accept_inner_step())
                ++accepted;
        }
        stencil_batch_proposal_kernel<<<(vertices + 127) / 128, 128>>>(
            m_outer_start.cview(),
            m_fem->xs(),
            m_outer_gradients.cview(),
            m_fem->m_impl.dxs.view(),
            m_global_slopes.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
        cuda_tool::DeviceReduce().Sum(m_global_slopes.data(), m_scalar.data(), vertices);
        Float slope;
        m_scalar.view().copy_to(&slope);
        if(!std::isfinite(slope))
            throw SimEngineException("Nonfinite accumulated stencil proposal.");
        if(slope >= 0 && !m_local_merge_acceptance)
            CUDA_TOOL_CHECK(cudaMemcpy(
                m_fem->m_impl.dxs.data(), m_first_proposal.data(), bytes, cudaMemcpyDeviceToDevice));
    }
    catch(...)
    {
        CUDA_TOOL_CHECK(cudaMemcpy(m_fem->m_impl.xs.data(), m_outer_start.data(), bytes, cudaMemcpyDeviceToDevice));
        CUDA_TOOL_CHECK(cudaMemset(m_fem->m_impl.dxs.data(), 0, bytes));
        throw;
    }
    // Only the outer start has a refreshed global contact residual. The engine
    // must certify the accumulated proposal before any inner convergence counts.
    CUDA_TOOL_CHECK(cudaMemcpy(m_fem->m_impl.xs.data(), m_outer_start.data(), bytes, cudaMemcpyDeviceToDevice));
    m_residual      = outer_residual;
    m_gradient_norm = outer_gradient_norm;
    cap_combined_elastic_step();
    logger::info("Local stencil batch: outer={}, color_passes={}, accepted_inner={}, velocity_residual={}, gradient_l2={}",
                 engine().newton_iter(),
                 std::min(m_inner_index + 1, m_inner_sweeps),
                 accepted,
                 m_residual,
                 m_gradient_norm);
}

void FEMLocalStencilSolver::solve_sweep(bool first)
{
    const bool  hybrid          = m_hybrid && !m_warm_active;
    const bool  gs              = m_gs && !m_warm_active;
    const bool  jacobi          = m_jacobi || m_warm_active;
    const char* audit_directory = std::getenv("UIPC_STENCIL_AUDIT");
    const bool  audit =
        audit_directory && *audit_directory && first
        && (engine().newton_iter() == 0 || (engine().newton_iter() + 1) % 128 == 0);
    const char* profile_setting = std::getenv("UIPC_STENCIL_PROFILE");
    const bool  profile = profile_setting && std::string(profile_setting) == "1"
                         && engine().newton_iter() == 0 && first;
    auto       started = std::chrono::steady_clock::now();
    const auto mark    = [&](const char* name)
    {
        if(profile)
        {
            CUDA_TOOL_CHECK(cudaDeviceSynchronize());
            const auto now = std::chrono::steady_clock::now();
            logger::info(
                "Stencil phase {}: {} ms",
                name,
                std::chrono::duration<double, std::milli>(now - started).count());
            started = now;
        }
    };
    Timer       timer{"Local Stencil Solve"};
    const Float dt       = world().scene().dt();
    const int   count    = static_cast<int>(m_elastic_stencils.cview().size());
    const int   vertices = static_cast<int>(m_fem->xs().size());
    stencil_element_derivatives_kernel<<<(count + 63) / 64, 64>>>(
        m_elastic_stencils.cview(),
        m_fem->xs(),
        m_elastic_terms.cview(),
        m_element_gradients.view(),
        m_element_hessians.view(),
        dt,
        m_warm_active,
        !m_packed_product || (audit_directory && *audit_directory));
    CUDA_TOOL_CHECK(cudaGetLastError());
    if(first)
    {
        prepare_contacts();
        if(m_local_filter)
            prepare_barriers();
        if(gs)
            prepare_gs_coloring();
    }
    if(m_inner_sweeps > 1 || m_custom_barriers || m_packed_product)
        refresh_frozen_contacts();
    mark("derivatives/contact");
    const Vector2i pass =
        gs ? Vector2i(0, static_cast<IndexT>(m_sweep_index++ % m_gs_color_count)) :
        hybrid ? m_color_passes[m_sweep_index++ % m_color_passes.size()] :
                 Vector2i::Zero().eval();
    const int solve_prefix = m_warm_active ? vertices : count;
    const int stencil_count = solve_prefix + static_cast<int>(m_contact_stencils.size());
    if(m_custom_barriers)
    {
        m_update_stencils.resize_discard(stencil_count);
        sos_active_overlap_kernel<<<(vertices + 127) / 128, 128>>>(
            m_barrier_offsets.cview(),
            m_barrier_entries.cview(),
            m_frozen_contact_gradients.cview(),
            m_vertex_offsets.cview(),
            m_overlap.view(),
            m_warm_active);
        sos_update_stencils_kernel<<<(stencil_count + 127) / 128, 128>>>(
            m_overlap.cview(),
            m_elastic_stencils.cview(),
            m_contact_stencils.cview(),
            m_frozen_contact_gradients.cview(),
            m_update_stencils.view(),
            solve_prefix,
            m_warm_active);
        CUDA_TOOL_CHECK(cudaGetLastError());
    }
    const auto update_contacts =
        m_custom_barriers ?
            m_update_stencils.cview().subview(solve_prefix, m_contact_stencils.size()) :
            m_contact_stencils.cview();
    if(stencil_count > m_directions.capacity())
        m_directions.reserve_discard(stencil_count + stencil_count / 10);
    m_directions.resize_discard(stencil_count);
    m_direction_slopes.resize_discard(stencil_count);
    m_cg_diagnostics.resize_discard(audit ? stencil_count : 0);
    stencil_vertex_derivatives_kernel<<<(vertices + 127) / 128, 128>>>(
        m_elastic_stencils.cview(),
        m_fem->xs(),
        m_fem->x_tildes(),
        m_fem->masses(),
        m_vertex_offsets.cview(),
        m_vertex_elements.cview(),
        m_element_gradients.cview(),
        m_element_hessians.cview(),
        m_total_gradients.view(),
        m_vertex_hessians.view(),
        !m_packed_product);
    CUDA_TOOL_CHECK(cudaGetLastError());
    if(m_packed_product)
    {
        fem::PackedStencilContext context;
        context.elements = m_elastic_stencils.cview();
        context.contacts = m_contact_stencils.cview();
        if(m_custom_barriers)
            context.stencils = m_update_stencils.cview();
        else
        {
            m_packed_supports.resize_discard(stencil_count);
            m_packed_supports.view().subview(0, count).copy_from(
                m_elastic_stencils.cview());
            if(m_contact_stencils.size())
                m_packed_supports.view()
                    .subview(count, m_contact_stencils.size())
                    .copy_from(m_contact_stencils.cview());
            context.stencils = m_packed_supports.cview();
        }
        context.terms             = m_elastic_terms.cview();
        context.xs                = m_fem->xs();
        context.elastic_gradients = m_total_gradients.cview();
        context.contact_gradients = m_contact_gradients.cview();
        context.masses            = m_fem->masses();
        context.fixed             = m_fem->is_fixed();
        context.colors = gs ? m_gs_colors.cview() : ct::CBufferView<IndexT>{};
        context.color  = pass(1);
        context.elastic_hessians = m_element_hessians.cview();
        context.contact_hessians = m_frozen_contact_hessians.cview();
        context.dt               = m_warm_active ? 0 : dt;
        m_packed_solver.solve(context,
                              m_directions.view(),
                              m_direction_slopes.view(),
                              m_cg_diagnostics.view());
    }
    else
    {
        stencil_local_solve_kernel<<<(stencil_count + 63) / 64, 64>>>(
            m_elastic_stencils.cview(),
            m_contact_stencils.cview(),
            m_fem->is_fixed(),
            m_vertex_offsets.cview(),
            m_vertex_elements.cview(),
            m_total_gradients.cview(),
            m_vertex_hessians.cview(),
            m_element_hessians.cview(),
            contact_view(),
            m_directions.view(),
            m_direction_slopes.view(),
            m_cg_diagnostics.view(),
            gs ? m_gs_colors.cview() : ct::CBufferView<IndexT>{},
            pass(1),
            m_csfd_product || m_warm_active,
            m_elastic_terms.cview(),
            m_fem->xs(),
            m_fem->masses(),
            m_warm_active ? 0 : dt,
            m_custom_barriers ? m_update_stencils.cview() : ct::CBufferView<Vector4i>{});
        CUDA_TOOL_CHECK(cudaGetLastError());
    }
    mark("directions");
    if(hybrid && m_sequential_hybrid)
    {
        m_unfiltered_directions.resize_discard(m_directions.size());
        m_unfiltered_directions.view().copy_from(m_directions.cview());
    }
    if(m_local_filter)
        filter_local_steps();
    mark("filtering");
    if(m_warm_active)
    {
        sos_warm_gather_kernel<<<(vertices + 127) / 128, 128>>>(
            update_contacts,
            m_barrier_offsets.cview(),
            m_barrier_entries.cview(),
            m_directions.cview(),
            m_fem->masses(),
            m_fem->is_fixed(),
            m_contact_gradients.cview(),
            m_total_gradients.view(),
            m_fem->m_impl.dxs.view(),
            m_residuals.view(),
            m_gradient_norms.view(),
            m_global_slopes.view(),
            dt);
        CUDA_TOOL_CHECK(cudaGetLastError());
    }
    else
    {
        // All directions and their local CCD/Armijo filters use the same frozen x.
        // Only now select priority owners for the displacement submission.
        stencil_primary_owners_kernel<<<(vertices + 127) / 128, 128>>>(
            m_vertex_offsets.cview(),
            m_vertex_elements.cview(),
            gs ? m_gs_colors.cview() : m_colors.cview(),
            m_primary_owner.view(),
            pass(0) * count,
            pass(1),
            hybrid || gs);
        CUDA_TOOL_CHECK(cudaGetLastError());
        if(hybrid)
            logger::info("Local stencil hybrid sweep: scheme={}, color={}", pass(0), pass(1));
        const int contact_count = static_cast<int>(m_contact_stencils.size());
        if(gs && contact_count)
            stencil_gs_contact_owners_kernel<<<(contact_count + 127) / 128, 128>>>(
                update_contacts, m_gs_colors.cview(), m_primary_owner.view(), count, pass(1));
        CUDA_TOOL_CHECK(cudaGetLastError());
        if(gs)
            logger::info("Local stencil GS sweep: color={}, colors={}", pass(1), m_gs_color_count);
        stencil_gather_kernel<<<(vertices + 127) / 128, 128>>>(
            m_elastic_stencils.cview(),
            update_contacts,
            m_fem->xs(),
            m_fem->masses(),
            m_fem->is_fixed(),
            m_vertex_offsets.cview(),
            m_vertex_elements.cview(),
            m_directions.cview(),
            m_fem->m_impl.dxs.view(),
            m_residuals.view(),
            m_gradient_norms.view(),
            m_contact_gradients.cview(),
            m_primary_owner.cview(),
            m_total_gradients.view(),
            hybrid || gs,
            hybrid || jacobi || gs ? 1.0 : m_weight,
            dt);
        CUDA_TOOL_CHECK(cudaGetLastError());
        if(contact_count && !gs)
            stencil_scatter_contact_kernel<<<(contact_count + 127) / 128, 128>>>(
                update_contacts,
                m_directions.cview(),
                m_fem->m_impl.dxs.view(),
                m_primary_owner.cview(),
                count,
                hybrid || jacobi ? 1.0 : m_weight);
        CUDA_TOOL_CHECK(cudaGetLastError());
    }
    if(hybrid || jacobi || gs)
    {
        if(!m_warm_active)
            stencil_hybrid_average_kernel<<<(vertices + 127) / 128, 128>>>(
                m_fem->m_impl.dxs.view(),
                m_total_gradients.cview(),
                m_primary_owner.cview(),
                m_overlap.cview(),
                m_global_slopes.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
        // Local descent/CCD certificates do not transfer through averaging or
        // priority selection. The engine still filters this combined proposal.
        cuda_tool::DeviceReduce().Sum(m_global_slopes.data(), m_scalar.data(), vertices);
        Float slope;
        m_scalar.view().copy_to(&slope);
        if(!std::isfinite(slope))
            throw SimEngineException("Nonfinite combined local-stencil slope.");
        logger::info("Local stencil combined directional slope: {}", slope);
        if(slope >= 0 && !m_local_merge_acceptance)
        {
            // A scalar backtrack cannot repair an uphill direction. Reject the
            // entire proposal and try the next color at the unchanged iterate;
            // do not alter Jacobi weights or report this as convergence.
            CUDA_TOOL_CHECK(cudaMemset(m_fem->m_impl.dxs.data(), 0, vertices * sizeof(Vector3)));
            logger::info("Local stencil rejected non-descent merged proposal");
        }
    }
    cuda_tool::DeviceReduce().Max(m_residuals.data(), m_scalar.data(), vertices);
    m_scalar.view().copy_to(&m_residual);
    logger::info("Local stencil velocity residual: {} m/s", m_residual);
    cuda_tool::DeviceReduce().Sum(m_gradient_norms.data(), m_scalar.data(), vertices);
    Float gradient_squared_norm;
    m_scalar.view().copy_to(&gradient_squared_norm);
    m_gradient_norm = std::sqrt(gradient_squared_norm);
    if(!std::isfinite(m_gradient_norm))
        throw SimEngineException("Nonfinite local-stencil gradient L2.");
    logger::info("Local stencil gradient L2: {}", m_gradient_norm);
    if(m_inertial_guess && engine().newton_iter() == 0 && first && !converged())
    {
        stencil_inertial_proposal_kernel<<<(vertices + 127) / 128, 128>>>(
            m_fem->xs(),
            m_fem->x_tildes(),
            m_total_gradients.cview(),
            m_fem->is_fixed(),
            m_inertial_proposal.view(),
            m_global_slopes.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
        cuda_tool::DeviceReduce().Sum(m_global_slopes.data(), m_scalar.data(), vertices);
        Float slope;
        m_scalar.view().copy_to(&slope);
        if(std::isfinite(slope) && slope < 0)
        {
            CUDA_TOOL_CHECK(cudaMemcpy(m_fem->m_impl.dxs.data(),
                                       m_inertial_proposal.data(),
                                       vertices * sizeof(Vector3),
                                       cudaMemcpyDeviceToDevice));
            logger::info("Local stencil inertial initial proposal (global CCD/energy acceptance required): {}",
                         slope);
        }
    }
    if(hybrid && m_sequential_hybrid && !converged())
        submit_sequential_hybrid();
    // The residual certifies the current iterate, not a subsequent proposal.
    // The engine checks convergence before stepping; keep that certified state.
    if(converged())
        CUDA_TOOL_CHECK(cudaMemset(m_fem->m_impl.dxs.data(), 0, vertices * sizeof(Vector3)));
    else
        cap_combined_elastic_step();
    if(audit_directory && *audit_directory && first)
    {
        const auto iteration = engine().newton_iter();
        if(iteration == 0 || (iteration + 1) % 128 == 0 || converged())
        {
            const auto folder = std::filesystem::path(audit_directory)
                                / std::to_string(engine().frame());
            std::filesystem::create_directories(folder);
            const auto prefix =
                folder
                / (std::to_string(iteration)
                   + (m_inner_sweeps > 1 ? "-inner-" + std::to_string(m_inner_index) : ""));
            auto dump = [&](const char* name, auto view)
            {
                using T =
                    std::remove_const_t<std::remove_reference_t<decltype(*view.data())>>;
                vector<T> host(view.size());
                view.copy_to(host.data());
                std::ofstream out(prefix.string() + name, std::ios::binary);
                out.exceptions(std::ios::failbit | std::ios::badbit);
                out.write(reinterpret_cast<const char*>(host.data()),
                          host.size() * sizeof(T));
            };
            dump("-x.bin", m_fem->xs());
            dump("-predicted.bin", m_fem->x_tildes());
            dump("-mass.bin", m_fem->masses());
            dump("-gradient.bin", m_total_gradients.cview());
            dump("-dx.bin", m_fem->m_impl.dxs.cview());
            dump("-directions.bin", m_directions.cview());
            dump("-element-gradient.bin", m_element_gradients.cview());
            dump("-cg.bin", m_cg_diagnostics.cview());
            dump("-owners.bin", m_primary_owner.cview());
            dump("-contacts.bin", m_contact_stencils.cview());
            dump("-contact-hessians.bin", m_frozen_contact_hessians.cview());
            dump("-hessians.bin", m_element_hessians.cview());
            if(iteration == 0)
            {
                dump("-elements.bin", m_elastic_stencils.cview());
            }
        }
    }
}

void FEMLocalStencilSolver::submit_sequential_hybrid()
{
    const int n     = static_cast<int>(m_fem->xs().size());
    const int count = static_cast<int>(m_elastic_stencils.size());
    m_hybrid_start.resize_discard(n);
    m_hybrid_gradient.resize_discard(n);
    m_hybrid_start.view().copy_from(m_fem->xs());
    m_hybrid_gradient.view().copy_from(m_total_gradients.cview());
    const auto refresh_derivatives = [&]
    {
        stencil_element_derivatives_kernel<<<(count + 63) / 64, 64>>>(
            m_elastic_stencils.cview(),
            m_fem->xs(),
            m_elastic_terms.cview(),
            m_element_gradients.view(),
            m_element_hessians.view(),
            world().scene().dt(),
            false,
            !m_packed_product || std::getenv("UIPC_STENCIL_AUDIT"));
        refresh_frozen_contacts();
        stencil_vertex_derivatives_kernel<<<(n + 127) / 128, 128>>>(
            m_elastic_stencils.cview(),
            m_fem->xs(),
            m_fem->x_tildes(),
            m_fem->masses(),
            m_vertex_offsets.cview(),
            m_vertex_elements.cview(),
            m_element_gradients.cview(),
            m_element_hessians.cview(),
            m_total_gradients.view(),
            m_vertex_hessians.view(),
            !m_packed_product);
        stencil_add_contact_gradient_kernel<<<(n + 127) / 128, 128>>>(
            m_total_gradients.view(), m_contact_gradients.cview(), m_fem->is_fixed());
        CUDA_TOOL_CHECK(cudaGetLastError());
    };
    try
    {
        // First submit only the independently colored stencils. All local
        // Newton directions were computed at the original frozen state.
        stencil_priority_proposal_kernel<<<(n + 127) / 128, 128>>>(
            m_elastic_stencils.cview(),
            m_primary_owner.cview(),
            m_fem->is_fixed(),
            m_directions.cview(),
            m_fem->m_impl.dxs.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
        cap_combined_elastic_step();
        accept_inner_step();

        // The Jacobi trial is the actual unowned part of each original
        // direction, evaluated at the committed GS state. Its slope, energy
        // and continuous bounds must all refer to that same trial.
        refresh_derivatives();
        const int stencils = static_cast<int>(m_directions.size());
        stencil_jacobi_submission_kernel<<<(stencils + 127) / 128, 128>>>(
            m_elastic_stencils.cview(),
            m_contact_stencils.cview(),
            m_custom_barriers ? m_update_stencils.cview() : ct::CBufferView<Vector4i>{},
            m_primary_owner.cview(),
            m_total_gradients.cview(),
            m_unfiltered_directions.cview(),
            m_directions.view(),
            m_direction_slopes.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
        filter_local_steps();
        stencil_jacobi_gather_kernel<<<(n + 127) / 128, 128>>>(
            m_elastic_stencils.cview(),
            m_contact_stencils.cview(),
            m_vertex_offsets.cview(),
            m_vertex_elements.cview(),
            m_barrier_offsets.cview(),
            m_barrier_entries.cview(),
            m_primary_owner.cview(),
            m_overlap.cview(),
            m_fem->is_fixed(),
            m_directions.cview(),
            m_fem->m_impl.dxs.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
        cap_combined_elastic_step();
        accept_inner_step();
        stencil_batch_proposal_kernel<<<(n + 127) / 128, 128>>>(
            m_hybrid_start.cview(),
            m_fem->xs(),
            m_hybrid_gradient.cview(),
            m_fem->m_impl.dxs.view(),
            m_global_slopes.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
    }
    catch(...)
    {
        m_fem->m_impl.xs.view().copy_from(m_hybrid_start.cview());
        m_total_gradients.view().copy_from(m_hybrid_gradient.cview());
        throw;
    }
    // External acceptance must still certify the complete path and discover
    // new pairs. Inner batches also need their energy cache at this origin.
    m_fem->m_impl.xs.view().copy_from(m_hybrid_start.cview());
    m_total_gradients.view().copy_from(m_hybrid_gradient.cview());
    filter_local_steps(true);
    // Optional audit blocks must describe the restored origin, not the
    // intermediate GS state used for the second search.
    if(const char* audit = std::getenv("UIPC_STENCIL_AUDIT"); audit && *audit)
        refresh_derivatives();
}

void FEMLocalStencilSolver::cap_combined_elastic_step()
{

    const int count    = static_cast<int>(m_elastic_stencils.size());
    const int vertices = static_cast<int>(m_fem->xs().size());
    // Independent local certificates do not certify a priority/Jacobi
    // mixture or an inertial predictor. Bound the actual combined path,
    // including interior tetrahedra that surface CCD does not monitor.
    stencil_combined_cap_kernel<<<(count + 127) / 128, 128>>>(m_elastic_stencils.cview(),
                                                              m_elastic_terms.cview(),
                                                              m_fem->xs(),
                                                              m_fem->m_impl.dxs.cview(),
                                                              m_determinants.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    cuda_tool::DeviceReduce().Min(m_determinants.data(), m_scalar.data(), count);
    Float cap;
    m_scalar.view().copy_to(&cap);
    if(!std::isfinite(cap) || cap < 0 || cap > 1)
        throw SimEngineException("Invalid combined local-stencil inversion bound.");
    if(m_warm_active && m_warm_post_inversion && !m_warm_finishing)
    {
        // An already feasible warm path needs no second projection from origin.
        // Still measure the full path (including interior tets), without capping
        // the Eq.14 scratch solve. If any proposal is unsafe, project at handoff.
        m_warm_needs_rebase |= cap < 1;
        return;
    }
    if(cap < 1)
    {
        stencil_scale_proposal_kernel<<<(vertices + 127) / 128, 128>>>(
            m_fem->m_impl.dxs.view(), cap);
        CUDA_TOOL_CHECK(cudaGetLastError());
        logger::info("Local stencil combined inversion cap: {}", cap);
    }
}

bool FEMLocalStencilSolver::converged() const
{
    return !m_warm_active && residual_converged();
}

bool FEMLocalStencilSolver::residual_converged() const
{
    return !m_history_changed
           && (m_gradient_stopping ? m_gradient_norm <= m_gradient_tolerance :
                                     m_residual <= m_tolerance);
}

bool FEMLocalStencilSolver::do_dump(DumpInfo& info)
{
    std::ofstream out(fmt::format(
        "{}sweep.{}.json", info.dump_path(UIPC_RELATIVE_SOURCE_FILE), info.frame()));
    out << Json{{"sweep", m_sweep_index},
                {"passes", m_color_passes.size()},
                {"previous_dt", m_previous_dt}};
    return out.good()
           && (!m_warm_enabled
               || m_velocity_dump.dump(fmt::format("{}velocity.{}",
                                                   info.dump_path(UIPC_RELATIVE_SOURCE_FILE),
                                                   info.frame()),
                                       m_previous_velocity));
}

bool FEMLocalStencilSolver::do_try_recover(RecoverInfo& info)
{
    if(m_restart_schedule && !m_warm_enabled)
    {
        m_recovered_sweep = 0;
        logger::info("Local stencil checkpoint fork: physical state recovered with color schedule restarted");
        return true;
    }
    std::ifstream in(fmt::format(
        "{}sweep.{}.json", info.dump_path(UIPC_RELATIVE_SOURCE_FILE), info.frame()));
    const Json    value = Json::parse(in, nullptr, false);
    if(value.is_discarded() || !value.contains("sweep") || !value.contains("passes")
       || !value["sweep"].is_number_unsigned() || !value["passes"].is_number_unsigned()
       || value["passes"].get<SizeT>() != m_color_passes.size())
        return false;
    m_recovered_sweep = m_restart_schedule ? 0 : value["sweep"].get<SizeT>();
    m_recovered_dt    = value.value("previous_dt", Float{0});
    if(m_warm_enabled)
    {
        if(!value.contains("previous_dt")
           || !m_velocity_dump.load(fmt::format(
               "{}velocity.{}", info.dump_path(UIPC_RELATIVE_SOURCE_FILE), info.frame())))
            return false;
        if(m_recovered_dt > 0
           && m_velocity_dump.view<Vector3>().size() != m_fem->xs().size())
            return false;
    }
    return true;
}

void FEMLocalStencilSolver::do_apply_recover(RecoverInfo&)
{
    m_sweep_index = m_recovered_sweep;
    m_previous_dt = m_recovered_dt;
    if(m_warm_enabled)
        m_velocity_dump.apply_to(m_previous_velocity);
    m_warm_frame = m_history_frame = ~SizeT{0};
    m_warm_active                  = false;
    m_warm_finishing               = false;
    m_history.reset();
}

bool FEMLocalStencilSolver::valid_configuration()
{
    if(m_warm_active && m_warm_post_inversion && !m_warm_finishing)
        return true;  // Scratch Eq.14 iterate; never a completed physical frame.
    const int count = static_cast<int>(m_elastic_stencils.cview().size());
    stencil_determinants_kernel<<<(count + 127) / 128, 128>>>(m_elastic_stencils.cview(),
                                                              m_fem->xs(),
                                                              m_elastic_terms.cview(),
                                                              m_determinants.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    cuda_tool::DeviceReduce().Min(m_determinants.data(), m_scalar.data(), count);
    Float minimum;
    m_scalar.view().copy_to(&minimum);
    return std::isfinite(minimum) && minimum > 0;
}
}  // namespace uipc::backend::cuda
