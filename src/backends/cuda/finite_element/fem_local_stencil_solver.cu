#include <finite_element/fem_local_stencil_solver.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/local_stencil.h>
#include <finite_element/local_stencil_elastic.h>
#include <sim_engine.h>
#include <finite_element/fem_dytopo_effect_receiver.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <collision_detection/global_trajectory_filter.h>
#include <cuda_tool/eigen/atomic.h>
#include <contact_system/global_contact_manager.h>
#include <global_geometry/global_vertex_manager.h>
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
                                                       Float dt)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= elements.size())
            return;
        Vector3 p[4];
        for(int a = 0; a < 4; ++a)
            if(elements(t)(a) >= 0)
                p[a] = xs(elements(t)(a));
        fem::stencil_elastic_derivatives(terms(t), p, dt, gradients(t), hessians(t));
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
        ct::BufferView<Matrix3x3>    diagonal)
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
                                               ct::BufferView<Float> slopes)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= directions.size())
            return;
        const Vector4i stencil =
            t < elements.size() ? elements(t) : contacts(t - elements.size());
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

    __global__ void stencil_gather_kernel(ct::CBufferView<Vector4i> elements,
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
            const int t = primary_owner(v);
            for(int a = 0; a < 4; ++a)
                if(elements(t)(a) == v)
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
    m_hybrid = config.find<std::string>("newton/local_stencil/sweep")->view()[0] == "hybrid";
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
    if(m_inner_sweeps < 1 || m_inner_sweeps > 256 || (!m_hybrid && m_inner_sweeps != 1))
        throw SimEngineException("local_stencil inner_sweeps must be in [1,256]; additive requires one.");
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

void FEMLocalStencilSolver::solve()
{
    m_inner_index = 0;
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
        if(slope >= 0)
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
        dt);
    CUDA_TOOL_CHECK(cudaGetLastError());
    if(first)
    {
        prepare_contacts();
        if(m_hybrid)
            prepare_barriers();
    }
    if(m_inner_sweeps > 1)
        refresh_frozen_contacts();
    mark("derivatives/contact");
    const int stencil_count = count + static_cast<int>(m_contact_stencils.size());
    if(stencil_count > m_directions.capacity())
        m_directions.reserve_discard(stencil_count + stencil_count / 10);
    m_directions.resize_discard(stencil_count);
    m_direction_slopes.resize_discard(stencil_count);
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
        m_vertex_hessians.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
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
        m_direction_slopes.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    mark("directions");
    if(m_hybrid)
        filter_local_steps();
    mark("filtering");
    // All directions and their local CCD/Armijo filters use the same frozen x.
    // Only now select priority owners for the displacement submission.
    const Vector2i pass =
        m_hybrid ? m_color_passes[m_sweep_index++ % m_color_passes.size()] :
                   Vector2i::Zero().eval();
    stencil_primary_owners_kernel<<<(vertices + 127) / 128, 128>>>(
        m_vertex_offsets.cview(),
        m_vertex_elements.cview(),
        m_colors.cview(),
        m_primary_owner.view(),
        pass(0) * count,
        pass(1),
        m_hybrid);
    CUDA_TOOL_CHECK(cudaGetLastError());
    if(m_hybrid)
        logger::info("Local stencil hybrid sweep: scheme={}, color={}", pass(0), pass(1));
    stencil_gather_kernel<<<(vertices + 127) / 128, 128>>>(
        m_elastic_stencils.cview(),
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
        m_hybrid,
        m_hybrid ? 1.0 : m_weight,
        dt);
    CUDA_TOOL_CHECK(cudaGetLastError());
    const int contact_count = static_cast<int>(m_contact_stencils.size());
    if(contact_count)
        stencil_scatter_contact_kernel<<<(contact_count + 127) / 128, 128>>>(
            m_contact_stencils.cview(),
            m_directions.cview(),
            m_fem->m_impl.dxs.view(),
            m_primary_owner.cview(),
            count,
            m_hybrid ? 1.0 : m_weight);
    CUDA_TOOL_CHECK(cudaGetLastError());
    if(m_hybrid)
    {
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
        if(slope >= 0)
        {
            // A scalar backtrack cannot repair an uphill direction. Reject the
            // entire proposal and try the next color at the unchanged iterate;
            // do not alter Jacobi weights or report this as convergence.
            CUDA_TOOL_CHECK(cudaMemset(m_fem->m_impl.dxs.data(), 0, vertices * sizeof(Vector3)));
            logger::info("Local stencil rejected non-descent hybrid proposal");
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
    // The residual certifies the current iterate, not a subsequent proposal.
    // The engine checks convergence before stepping; keep that certified state.
    if(converged())
        CUDA_TOOL_CHECK(cudaMemset(m_fem->m_impl.dxs.data(), 0, vertices * sizeof(Vector3)));
    else
        cap_combined_elastic_step();
    if(const char* directory = std::getenv("UIPC_STENCIL_AUDIT"); directory && *directory)
    {
        const auto iteration = engine().newton_iter();
        if(iteration == 0 || (iteration + 1) % 128 == 0 || converged())
        {
            const auto folder =
                std::filesystem::path(directory) / std::to_string(engine().frame());
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
            if(iteration == 0)
            {
                dump("-elements.bin", m_elastic_stencils.cview());
                dump("-hessians.bin", m_element_hessians.cview());
            }
        }
    }
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
    return m_gradient_stopping ? m_gradient_norm <= m_gradient_tolerance :
                                 m_residual <= m_tolerance;
}

bool FEMLocalStencilSolver::do_dump(DumpInfo& info)
{
    std::ofstream out(fmt::format(
        "{}sweep.{}.json", info.dump_path(UIPC_RELATIVE_SOURCE_FILE), info.frame()));
    out << Json{{"sweep", m_sweep_index}, {"passes", m_color_passes.size()}};
    return out.good();
}

bool FEMLocalStencilSolver::do_try_recover(RecoverInfo& info)
{
    if(m_restart_schedule)
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
    m_recovered_sweep = value["sweep"].get<SizeT>();
    return true;
}

void FEMLocalStencilSolver::do_apply_recover(RecoverInfo&)
{
    m_sweep_index = m_recovered_sweep;
}

bool FEMLocalStencilSolver::valid_configuration()
{
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
