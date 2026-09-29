#include <finite_element/fem_local_stencil_solver.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/local_stencil.h>
#include <finite_element/local_stencil_snk.h>
#include <sim_engine.h>
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
        if(config.find<IndexT>("contact/enable")->view()[0]
           || config.find<IndexT>("diff_sim/enable")->view()[0]
           || config.find<std::string>("contact/constitution")->view()[0] != "ipc"
           || config.find<std::string>("integrator/type")->view()[0] != "bdf1")
            throw SimEngineException("local_stencil requires contact disabled, diff_sim disabled, ipc pipeline and bdf1.");
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

    __global__ void stencil_local_solve_kernel(ct::CBufferView<Vector4i> tets,
                                               ct::CBufferView<Vector3>  xs,
                                               ct::CBufferView<Vector3> predicted,
                                               ct::CBufferView<Float>  masses,
                                               ct::CBufferView<IndexT> fixed,
                                               ct::CBufferView<IndexT> offsets,
                                               ct::CBufferView<IndexT> neighbors,
                                               ct::CBufferView<Vector12> gradients,
                                               ct::CBufferView<Matrix12x12> hessians,
                                               ct::BufferView<Vector12> directions)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= tets.size())
            return;
        const auto              tet = tets(t);
        fem::LocalStencilSystem system;
        for(int i = 0; i < 4; ++i)
            system.add_inertia(i, masses(tet(i)), xs(tet(i)) - predicted(tet(i)));
        for(int entry = offsets(t); entry < offsets(t + 1); ++entry)
        {
            const int neighbor = neighbors(entry);
            system.add_element(tet, tets(neighbor), gradients(neighbor), hessians(neighbor));
        }
        for(int i = 0; i < 4; ++i)
            if(fixed(tet(i)))
                system.fix_vertex(i);
        const auto result = fem::local_stencil_direction(system);
        UIPC_KERNEL_ASSERT(result.valid, "Invalid local stencil direction at tetrahedron %d", t);
        directions(t) = result.displacement;
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
                                          Float                   weight,
                                          Float                   dt)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= xs.size())
            return;
        Vector3 dx       = Vector3::Zero();
        Vector3 gradient = masses(v) * (xs(v) - predicted(v));
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
        displacements(v) = fixed(v) ? Vector3::Zero().eval() : (weight * dx).eval();
        residuals(v) = fixed(v) ? 0 : gradient.norm() / (masses(v) * dt);
        UIPC_KERNEL_ASSERT(fem::stencil_finite(residuals(v)),
                           "Nonfinite stencil residual at vertex %d",
                           v);
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
    m_fem = &require<FiniteElementMethod>();
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
    stencil_local_solve_kernel<<<(count + 63) / 64, 64>>>(m_fem->tets(),
                                                          m_fem->xs(),
                                                          m_fem->x_tildes(),
                                                          m_fem->masses(),
                                                          m_fem->is_fixed(),
                                                          m_stencil_offsets.cview(),
                                                          m_stencil_neighbors.cview(),
                                                          m_element_gradients.cview(),
                                                          m_element_hessians.cview(),
                                                          m_directions.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
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
                                                           m_weight,
                                                           dt);
    CUDA_TOOL_CHECK(cudaGetLastError());
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
