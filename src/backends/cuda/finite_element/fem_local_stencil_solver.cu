#include <finite_element/fem_local_stencil_solver.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/local_stencil.h>
#include <finite_element/local_stencil_snk.h>
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
           || std::any_of(
               uids.begin(), uids.end(), [](U64 uid) { return uid != 10; }))
            throw SimEngineException("local_stencil currently supports only StableNeoHookean tetrahedra and fixed vertices.");
        return uipc::make_unique<cuda::FEMLocalStencilSolver>(engine);
    }
};
}  // namespace uipc::backend

namespace uipc::backend::cuda
{
namespace
{
    namespace ct = cuda_tool;

    __global__ void stencil_element_derivatives_kernel(ct::CBufferView<Vector4i> tets,
                                                       ct::CBufferView<Vector3> xs,
                                                       ct::CBufferView<Matrix3x3> inverses,
                                                       ct::CBufferView<Float> volumes,
                                                       ct::CBufferView<Float> mu,
                                                       ct::CBufferView<Float> lambda,
                                                       ct::BufferView<Vector12> gradients,
                                                       ct::BufferView<Matrix12x12> hessians,
                                                       Float dt)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= tets.size())
            return;
        const auto      tet = tets(t);
        const Matrix3x3 F =
            fem::F(xs(tet(0)), xs(tet(1)), xs(tet(2)), xs(tet(3)), inverses(t));
        fem::snk_stencil_element(
            F, inverses(t), mu(t), lambda(t), volumes(t) * dt * dt, gradients(t), hessians(t));
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

    __global__ void stencil_local_solve_kernel(ct::CBufferView<Vector4i> tets,
                                               ct::CBufferView<Vector4i> contacts,
                                               ct::CBufferView<Vector3> xs,
                                               ct::CBufferView<Vector3> predicted,
                                               ct::CBufferView<Float>  masses,
                                               ct::CBufferView<IndexT> fixed,
                                               ct::CBufferView<IndexT> offsets,
                                               ct::CBufferView<IndexT> neighbors,
                                               ct::CBufferView<IndexT> vertex_offsets,
                                               ct::CBufferView<IndexT> vertex_tets,
                                               ct::CBufferView<Vector12> gradients,
                                               ct::CBufferView<Matrix12x12> hessians,
                                               fem::LocalStencilContactView contact,
                                               ct::BufferView<Vector12> directions,
                                               ct::BufferView<Float> slopes,
                                               ct::CBufferView<IndexT> primary_owner,
                                               bool hybrid)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= directions.size())
            return;
        const Vector4i stencil = t < tets.size() ? tets(t) : contacts(t - tets.size());
        fem::LocalStencilSystem system;
        for(int i = 0; i < 4; ++i)
            if(stencil(i) >= 0)
                system.add_inertia(i,
                                   masses(stencil(i)),
                                   xs(stencil(i)) - predicted(stencil(i)));
        if(t < tets.size())
        {
            for(int entry = offsets(t); entry < offsets(t + 1); ++entry)
            {
                const int e = neighbors(entry);
                system.add_element(stencil, tets(e), gradients(e), hessians(e));
            }
        }
        else
        {
            // The contact stencil can cross bodies. Include the union of all
            // incident elastic energies once, even when vertices share a tet.
            for(int a = 0; a < 4; ++a)
            {
                const int v = stencil(a);
                if(v < 0)
                    continue;
                for(int k = vertex_offsets(v); k < vertex_offsets(v + 1); ++k)
                {
                    const int e    = vertex_tets(k);
                    bool      seen = false;
                    for(int b = 0; b < a; ++b)
                        for(int j = 0; j < 4; ++j)
                            seen = seen || stencil(b) == tets(e)(j);
                    if(!seen)
                        system.add_element(stencil, tets(e), gradients(e), hessians(e));
                }
            }
        }
        contact.add(system, stencil);
        for(int i = 0; i < 4; ++i)
            if(stencil(i) < 0 || fixed(stencil(i))
               || (hybrid && primary_owner(stencil(i)) >= 0
                   && primary_owner(stencil(i)) != t))
                system.fix_vertex(i);
        const auto result = fem::local_stencil_direction(system);
        UIPC_KERNEL_ASSERT(result.valid, "Invalid local stencil direction %d", t);
        directions(t) = result.displacement;
        slopes(t)     = system.gradient.dot(result.displacement);
    }

    __global__ void stencil_scatter_contact_kernel(ct::CBufferView<Vector4i> contacts,
                                                   ct::CBufferView<Vector12> directions,
                                                   ct::BufferView<Vector3> dx,
                                                   ct::CBufferView<IndexT> primary_owner,
                                                   IndexT tet_count,
                                                   Float  weight)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= contacts.size())
            return;
        for(int i = 0; i < 4; ++i)
            if(contacts(t)(i) >= 0 && primary_owner(contacts(t)(i)) < 0)
            {
                const Vector3 d = weight * directions(t + tet_count).segment<3>(3 * i);
                ct::eigen::atomic_add(dx(contacts(t)(i)), d);
            }
    }

    __global__ void stencil_gather_kernel(ct::CBufferView<Vector4i> tets,
                                          ct::CBufferView<Vector3>  xs,
                                          ct::CBufferView<Vector3>  predicted,
                                          ct::CBufferView<Float>    masses,
                                          ct::CBufferView<IndexT>   fixed,
                                          ct::CBufferView<IndexT>   offsets,
                                          ct::CBufferView<IndexT> incident_tets,
                                          ct::CBufferView<Vector12> gradients,
                                          ct::CBufferView<Vector12> directions,
                                          ct::BufferView<Vector3> displacements,
                                          ct::BufferView<Float>   residuals,
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
        Vector3 dx = Vector3::Zero();
        Vector3 gradient = masses(v) * (xs(v) - predicted(v)) + contact_gradients(v);
        for(int entry = offsets(v); entry < offsets(v + 1); ++entry)
        {
            const int t = incident_tets(entry);
            for(int i = 0; i < 4; ++i)
                if(tets(t)(i) == v)
                {
                    dx += directions(t).segment<3>(3 * i);
                    gradient += gradients(t).segment<3>(3 * i);
                }
        }
        if(hybrid && primary_owner(v) >= 0)
        {
            const int t = primary_owner(v);
            for(int a = 0; a < 4; ++a)
                if(tets(t)(a) == v)
                    dx = directions(t).segment<3>(3 * a);
        }
        displacements(v) = fixed(v) ? Vector3::Zero().eval() : (weight * dx).eval();
        total_gradients(v) = fixed(v) ? Vector3::Zero().eval() : gradient;
        residuals(v)       = fixed(v) ? 0 : gradient.norm() / (masses(v) * dt);
        UIPC_KERNEL_ASSERT(fem::stencil_finite(residuals(v)),
                           "Nonfinite stencil residual at vertex %d",
                           v);
    }

    __global__ void stencil_hybrid_average_kernel(ct::BufferView<Vector3> dx,
                                                  ct::CBufferView<Vector3> gradients,
                                                  ct::CBufferView<IndexT> owners,
                                                  ct::CBufferView<IndexT> overlap,
                                                  ct::BufferView<Float> slopes,
                                                  bool  fallback,
                                                  Float uniform_weight)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= dx.size())
            return;
        if(owners(v) < 0)
            dx(v) *= fallback ? overlap(v) * uniform_weight : 1.0 / overlap(v);
        slopes(v) = gradients(v).dot(dx(v));
    }

    __global__ void stencil_determinants_kernel(ct::CBufferView<Vector4i> tets,
                                                ct::CBufferView<Vector3>  xs,
                                                ct::CBufferView<Matrix3x3> inverses,
                                                ct::BufferView<Float> determinants)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= tets.size())
            return;
        const auto  tet = tets(t);
        const Float det =
            fem::F(xs(tet(0)), xs(tet(1)), xs(tet(2)), xs(tet(3)), inverses(t)).determinant();
        determinants(t) = fem::stencil_finite(det) ? det : -1;
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
    m_color_limit = config.find<IndexT>("newton/local_stencil/color_limit")->view()[0];
    if(m_color_limit < 1 || m_color_limit > 64)
        throw SimEngineException("local_stencil color_limit must be in [1,64].");
    m_tolerance =
        world().scene().config().find<Float>("newton/velocity_tol")->view()[0];
    const auto count        = m_fem->tets().size();
    const auto vertex_count = m_fem->xs().size();
    if(count == 0 || vertex_count == 0 || m_fem->codim_0ds().size()
       || m_fem->codim_1ds().size() || m_fem->codim_2ds().size())
        throw SimEngineException("local_stencil requires a nonempty tetrahedral FEM scene.");
    vector<Float> mu(count), lambda(count);
    m_fem->for_each(
        world().scene().geometries(),
        [&](const FiniteElementMethod::ForEachInfo& info, geometry::SimplicialComplex& sc)
        {
            if(info.geo_info().dim_uid.uid != 10 || sc.dim() != 3)
                throw SimEngineException("local_stencil requires StableNeoHookean tetrahedra.");
            const auto mus     = sc.tetrahedra().find<Float>("mu")->view();
            const auto lambdas = sc.tetrahedra().find<Float>("lambda")->view();
            const auto offset  = info.geo_info().primitive_offset;
            for(SizeT i = 0; i < mus.size(); ++i)
            {
                if(!std::isfinite(mus[i]) || !std::isfinite(lambdas[i])
                   || mus[i] <= 0 || lambdas[i] <= 0)
                    throw SimEngineException("local_stencil requires positive finite material parameters.");
                mu[offset + i]     = mus[i];
                lambda[offset + i] = lambdas[i];
            }
        });
    m_mu.copy_from(mu);
    m_lambda.copy_from(lambda);

    vector<Vector4i> tets(count);
    m_fem->tets().copy_to(tets.data());
    vector<Float> masses(vertex_count);
    m_fem->masses().copy_to(masses.data());
    vector<vector<IndexT>> incident(vertex_count);
    for(SizeT t = 0; t < count; ++t)
        for(int i = 0; i < 4; ++i)
        {
            const auto v = tets[t](i);
            if(v < 0 || v >= vertex_count)
                throw SimEngineException("local_stencil: tetrahedron vertex index out of bounds.");
            incident[v].push_back(static_cast<IndexT>(t));
        }
    vector<IndexT> vertex_offsets{0}, vertex_tets;
    SizeT          max_degree = 1;
    for(SizeT v = 0; v < vertex_count; ++v)
    {
        if(incident[v].empty() || !std::isfinite(masses[v]) || masses[v] <= 0)
            throw SimEngineException("local_stencil requires every vertex to have positive mass and incident tetrahedra.");
        max_degree = std::max(max_degree, incident[v].size());
        vertex_tets.insert(vertex_tets.end(), incident[v].begin(), incident[v].end());
        vertex_offsets.push_back(static_cast<IndexT>(vertex_tets.size()));
    }
    // One uniform weight preserves descent of the summed local directions.
    // Per-vertex averaging would in general change the sign of g.dot(dx).
    m_weight = 1.0 / static_cast<Float>(max_degree);
    vector<IndexT> stencil_offsets{0}, stencil_neighbors;
    for(const auto& tet : tets)
    {
        vector<IndexT> neighbors;
        for(int i = 0; i < 4; ++i)
            neighbors.insert(neighbors.end(),
                             incident[tet(i)].begin(),
                             incident[tet(i)].end());
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                        neighbors.end());
        stencil_neighbors.insert(
            stencil_neighbors.end(), neighbors.begin(), neighbors.end());
        stencil_offsets.push_back(static_cast<IndexT>(stencil_neighbors.size()));
    }
    init_coloring(tets, incident);
    m_vertex_offsets.copy_from(vertex_offsets);
    m_vertex_tets.copy_from(vertex_tets);
    m_stencil_offsets.copy_from(stencil_offsets);
    m_stencil_neighbors.copy_from(stencil_neighbors);
    m_element_gradients.resize_discard(count);
    m_element_hessians.resize_discard(count);
    m_directions.resize_discard(count);
    m_residuals.resize_discard(vertex_count);
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
    m_global_slopes.resize_discard(vertex_count);
    m_barrier_counts.resize_discard(vertex_count + 1);
    m_barrier_offsets.resize_discard(vertex_count + 1);
    m_barrier_cursors.resize_discard(vertex_count);
    m_local_statistics.resize_discard(5);
    logger::info("Local stencil solver: {} tetrahedra, {} vertices, max overlap {}",
                 count,
                 vertex_count,
                 max_degree);
}

void FEMLocalStencilSolver::solve()
{
    Timer       timer{"Local Stencil Solve"};
    const Float dt       = world().scene().dt();
    const int   count    = static_cast<int>(m_fem->tets().size());
    const int   vertices = static_cast<int>(m_fem->xs().size());
    stencil_element_derivatives_kernel<<<(count + 63) / 64, 64>>>(
        m_fem->tets(),
        m_fem->xs(),
        m_fem->Dm3x3_invs(),
        m_fem->rest_volumes(),
        m_mu.cview(),
        m_lambda.cview(),
        m_element_gradients.view(),
        m_element_hessians.view(),
        dt);
    CUDA_TOOL_CHECK(cudaGetLastError());
    prepare_contacts();
    const int stencil_count = count + static_cast<int>(m_contact_stencils.size());
    if(stencil_count > m_directions.capacity())
        m_directions.reserve_discard(stencil_count + stencil_count / 10);
    m_directions.resize_discard(stencil_count);
    m_direction_slopes.resize_discard(stencil_count);
    const Vector2i pass =
        m_hybrid ? m_color_passes[m_sweep_index++ % m_color_passes.size()] :
                   Vector2i::Zero().eval();
    stencil_primary_owners_kernel<<<(vertices + 127) / 128, 128>>>(
        m_vertex_offsets.cview(),
        m_vertex_tets.cview(),
        m_colors.cview(),
        m_primary_owner.view(),
        pass(0) * count,
        pass(1),
        m_hybrid);
    CUDA_TOOL_CHECK(cudaGetLastError());
    if(m_hybrid)
        logger::info("Local stencil hybrid sweep: scheme={}, color={}", pass(0), pass(1));
    stencil_local_solve_kernel<<<(stencil_count + 63) / 64, 64>>>(
        m_fem->tets(),
        m_contact_stencils.cview(),
        m_fem->xs(),
        m_fem->x_tildes(),
        m_fem->masses(),
        m_fem->is_fixed(),
        m_stencil_offsets.cview(),
        m_stencil_neighbors.cview(),
        m_vertex_offsets.cview(),
        m_vertex_tets.cview(),
        m_element_gradients.cview(),
        m_element_hessians.cview(),
        contact_view(),
        m_directions.view(),
        m_direction_slopes.view(),
        m_primary_owner.cview(),
        m_hybrid);
    CUDA_TOOL_CHECK(cudaGetLastError());
    if(m_hybrid)
        filter_local_steps();
    stencil_gather_kernel<<<(vertices + 127) / 128, 128>>>(m_fem->tets(),
                                                           m_fem->xs(),
                                                           m_fem->x_tildes(),
                                                           m_fem->masses(),
                                                           m_fem->is_fixed(),
                                                           m_vertex_offsets.cview(),
                                                           m_vertex_tets.cview(),
                                                           m_element_gradients.cview(),
                                                           m_directions.cview(),
                                                           m_fem->m_impl.dxs.view(),
                                                           m_residuals.view(),
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
            m_global_slopes.view(),
            false,
            m_weight);
        CUDA_TOOL_CHECK(cudaGetLastError());
        cuda_tool::DeviceReduce().Sum(m_global_slopes.data(), m_scalar.data(), vertices);
        Float slope;
        m_scalar.view().copy_to(&slope);
        // Per-vertex Jacobi averaging is not always a descent direction. Keep
        // committed primary values, but use a uniform secondary weight if needed.
        if(slope > 0)
        {
            stencil_hybrid_average_kernel<<<(vertices + 127) / 128, 128>>>(
                m_fem->m_impl.dxs.view(),
                m_total_gradients.cview(),
                m_primary_owner.cview(),
                m_overlap.cview(),
                m_global_slopes.view(),
                true,
                m_weight);
            CUDA_TOOL_CHECK(cudaGetLastError());
            logger::info("Local stencil hybrid: uniform secondary descent safeguard");
        }
    }
    cuda_tool::DeviceReduce().Max(m_residuals.data(), m_scalar.data(), vertices);
    m_scalar.view().copy_to(&m_residual);
    logger::info("Local stencil velocity residual: {} m/s", m_residual);
}

bool FEMLocalStencilSolver::converged() const
{
    return m_residual <= m_tolerance;
}

bool FEMLocalStencilSolver::valid_configuration()
{
    const int count = static_cast<int>(m_fem->tets().size());
    stencil_determinants_kernel<<<(count + 127) / 128, 128>>>(
        m_fem->tets(), m_fem->xs(), m_fem->Dm3x3_invs(), m_determinants.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    cuda_tool::DeviceReduce().Min(m_determinants.data(), m_scalar.data(), count);
    Float minimum;
    m_scalar.view().copy_to(&minimum);
    return std::isfinite(minimum) && minimum > 0;
}
}  // namespace uipc::backend::cuda
