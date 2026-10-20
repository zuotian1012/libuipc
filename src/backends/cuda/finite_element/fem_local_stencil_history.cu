#include <finite_element/fem_local_stencil_solver.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <finite_element/sos_contact.h>
#include <global_geometry/global_simplicial_surface_manager.h>
#include <global_geometry/global_vertex_manager.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <contact_system/global_contact_manager.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <utils/primitive_d_hat.h>
#include <utils/codim_thickness.h>
#include <utils/distance/ccd.h>
#include <cuda_tool/cub.h>
#include <sim_engine.h>

namespace uipc::backend::cuda
{
namespace
{
    namespace ct = cuda_tool;
    __global__ void sos_candidates_kernel(ct::CBufferView<Vector2i> pts,
                                          ct::CBufferView<Vector2i> ees,
                                          ct::CBufferView<IndexT> surface_vertices,
                                          ct::CBufferView<Vector2i> edges,
                                          ct::CBufferView<Vector3i> triangles,
                                          ct::CBufferView<Vector3> plane_positions,
                                          ct::CBufferView<Vector3> plane_normals,
                                          ct::CBufferView<Vector3> origin,
                                          ct::CBufferView<Vector3> rest,
                                          ct::CBufferView<Vector3> xs,
                                          ct::CBufferView<Vector3> dx,
                                          ct::CBufferView<Float>   thicknesses,
                                          ct::CBufferView<Float>   d_hats,
                                          ct::CBufferView<IndexT>  ids,
                                          ct::CBufferView<IndexT>  subscene_ids,
                                          ct::CBuffer2DView<IndexT> contact_mask,
                                          ct::CBuffer2DView<IndexT> subscene_mask,
                                          ct::CDense2D<ContactCoeff> table,
                                          ct::BufferView<fem::SoSBarrierRecord> result,
                                          IndexT vertex_offset,
                                          IndexT plane_offset,
                                          Float  alpha,
                                          Float  dt,
                                          bool   convert_stiffness,
                                          bool   expand_only)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= result.size())
            return;
        using namespace sym::codim_ipc_simplex_contact;
        fem::SoSBarrierRecord record;
        Vector4i              global = Vector4i::Constant(-1);
        StencilBarrierTerm    term;
        Float                 dh = 0, stiffness = 0;
        int                   index = k;
        if(index < pts.size())
        {
            const auto pair  = pts(index);
            global(0)        = surface_vertices(pair(0));
            global.tail<3>() = triangles(pair(1));
            record.key       = fem::sos_pair_key(0, global(0), pair(1));
            term.type        = 0;
        }
        else if((index -= pts.size()) < ees.size())
        {
            const auto pair  = ees(index);
            global.head<2>() = edges(pair(0));
            global.tail<2>() = edges(pair(1));
            record.key       = fem::sos_pair_key(1, pair(0), pair(1));
            term.type        = 1;
        }
        else
        {
            index -= ees.size();
            const int v  = surface_vertices(index / plane_positions.size());
            const int h  = index % plane_positions.size();
            const int hv = h + plane_offset;
            if(!contact_mask(ids(v), ids(hv))
               || !subscene_mask(subscene_ids(v), subscene_ids(hv)))
            {
                result(k) = record;
                return;
            }
            global(0)           = v;
            record.key          = fem::sos_pair_key(2, v, h);
            term.type           = 4;
            term.plane_position = plane_positions(h);
            term.plane_normal   = plane_normals(h);
            term.thickness      = thicknesses(v);
            dh                  = d_hats(v);
            stiffness           = table(ids(v), ids(hv)).kappa;
        }
        Vector3  p[4], end[4], motion[4], reference[4];
        Vector4  thick = Vector4::Zero(), hats = Vector4::Zero();
        Vector4i cids = Vector4i::Zero();
        for(int a = 0; a < 4; ++a)
            if(global(a) >= 0)
            {
                const int v = global(a) - vertex_offset;
                UIPC_KERNEL_ASSERT(v >= 0 && v < xs.size(), "SoS contact requires FEM vertices");
                record.vertices(a) = v;
                p[a]               = xs(v);
                motion[a]          = alpha * dx(v);
                end[a]             = p[a] + motion[a];
                reference[a]       = origin(v);
                thick(a)           = thicknesses(global(a));
                hats(a)            = d_hats(global(a));
                cids(a)            = ids(global(a));
            }
        if(term.type == 0)
        {
            term.thickness = PT_thickness(thick(0), thick(1), thick(2), thick(3));
            dh        = PT_d_hat(hats(0), hats(1), hats(2), hats(3));
            stiffness = PT_kappa(table, cids);
        }
        else if(term.type == 1)
        {
            const auto ids = record.vertices;
            distance::edge_edge_mollifier_threshold(rest(ids(0)),
                                                    rest(ids(1)),
                                                    rest(ids(2)),
                                                    rest(ids(3)),
                                                    Float{1e-3},
                                                    record.mollifier_eps);
            term.thickness = EE_thickness(thick(0), thick(1), thick(2), thick(3));
            dh        = EE_d_hat(hats(0), hats(1), hats(2), hats(3));
            stiffness = EE_kappa(table, cids);
        }
        const Float range = term.thickness + dh;
        bool        keep  = fem::sos_distance_squared(term, p) < range * range
                    || fem::sos_distance_squared(term, end) < range * range;
        if(!keep && alpha > 0 && term.type < 2)
        {
            Float toi = 1;
            if(term.type == 0)
                keep = distance::point_triangle_ccd(
                    p[0], p[1], p[2], p[3], motion[0], motion[1], motion[2], motion[3], .2, term.thickness, 100, toi);
            else
                keep = distance::edge_edge_ccd(
                    p[0], p[1], p[2], p[3], motion[0], motion[1], motion[2], motion[3], .2, term.thickness, 100, toi);
        }
        if(!keep)
        {
            record.key = ~U64{0};
            result(k)  = record;
            return;
        }
        const Float clearance =
            std::sqrt(fem::sos_distance_squared(term, reference)) - term.thickness;
        UIPC_KERNEL_ASSERT(clearance > 0 && fem::stencil_finite(clearance),
                           "Infeasible SoS frame-origin clearance");
        if(convert_stiffness)
            stiffness = fem::sos_distance_stiffness(stiffness, dh, term.thickness);
        record.parameters = fem::SoSBarrierParameters::from_reference(
            clearance, dh, stiffness * dt * dt, expand_only);
        record.thickness      = term.thickness;
        record.plane_position = term.plane_position;
        record.plane_normal   = term.plane_normal;
        result(k)             = record;
    }
    __global__ void sos_history_terms_kernel(ct::CBufferView<fem::SoSBarrierRecord> records,
                                             ct::BufferView<Vector4i> stencils,
                                             ct::BufferView<StencilBarrierTerm> terms,
                                             ct::BufferView<IndexT> counts)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= records.size())
            return;
        const auto         record = records(i);
        StencilBarrierTerm term;
        term.customized     = true;
        term.type           = (record.key >> 62) == 2 ? 4 : (record.key >> 62);
        term.kappa_dt2      = record.parameters.kappa;
        term.d_hat          = record.parameters.activation_distance;
        term.thickness      = record.thickness;
        term.mollifier_eps  = record.mollifier_eps;
        term.plane_position = record.plane_position;
        term.plane_normal   = record.plane_normal;
        terms(i)            = term;
        stencils(i)         = record.vertices;
        for(int a = 0; a < 4; ++a)
            if(record.vertices(a) >= 0)
                atomicAdd(&counts(record.vertices(a)), 1);
    }
    __global__ void sos_history_rows_kernel(ct::CBufferView<Vector4i> stencils,
                                            ct::CBufferView<IndexT>   offsets,
                                            ct::BufferView<IndexT>    cursors,
                                            ct::BufferView<IndexT>    entries)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= stencils.size())
            return;
        for(int a = 0; a < 4; ++a)
            if(stencils(i)(a) >= 0)
            {
                const int v = stencils(i)(a);
                entries(offsets(v) + atomicAdd(&cursors(v), 1)) = i;
            }
    }
    __global__ void sos_overlap_kernel(ct::CBufferView<IndexT> counts,
                                       ct::CBufferView<IndexT> elastic_offsets,
                                       ct::BufferView<IndexT>  overlap)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v < overlap.size())
            overlap(v) = counts(v) + elastic_offsets(v + 1) - elastic_offsets(v);
    }
}  // namespace

void FEMLocalStencilSolver::prepare_custom_contacts(Float alpha)
{
    const SizeT frame = engine().frame();
    if(m_history_frame != frame)
    {
        m_history_frame = frame;
        m_history.reset();
        m_step_origin.resize_discard(m_fem->xs().size());
        m_step_origin.view().copy_from(m_fem->xs());
    }
    const SizeT previous = m_history.records().size();
    if(m_trajectory_filter)
    {
        const auto simplex = m_trajectory_filter->find<SimplexTrajectoryFilter>();
        const auto pts =
            simplex ? simplex->candidate_PTs() : ct::CBufferView<Vector2i>{};
        const auto ees =
            simplex ? simplex->candidate_EEs() : ct::CBufferView<Vector2i>{};
        const SizeT count =
            pts.size() + ees.size()
            + m_surface->surf_vertices().size() * m_planes->positions().size();
        m_history_candidates.resize_discard(count);
        if(count)
        {
            sos_candidates_kernel<<<(count + 127) / 128, 128>>>(
                pts,
                ees,
                m_surface->surf_vertices(),
                m_surface->surf_edges(),
                m_surface->surf_triangles(),
                m_planes->positions(),
                m_planes->normals(),
                m_step_origin.cview(),
                m_fem->x_bars(),
                m_fem->xs(),
                m_fem->m_impl.dxs.cview(),
                m_vertices->thicknesses(),
                m_vertices->d_hats(),
                m_vertices->contact_element_ids(),
                m_vertices->subscene_element_ids(),
                m_contact_manager->contact_mask_tabular(),
                m_contact_manager->subscene_mask_tabular(),
                m_contact_manager->contact_tabular().viewer(),
                m_history_candidates.view(),
                m_vertex_reporter->vertex_offset(),
                m_plane_reporter->vertex_offset(),
                alpha,
                world().scene().dt(),
                m_convert_ipc_stiffness,
                m_expand_barriers_only);
            CUDA_TOOL_CHECK(cudaGetLastError());
            m_history.append(m_history_candidates.cview());
        }
    }
    m_history_changed |= previous != m_history.records().size();
    build_custom_barriers();
}

void FEMLocalStencilSolver::build_custom_barriers()
{
    const int count    = static_cast<int>(m_history.records().size());
    const int vertices = static_cast<int>(m_fem->xs().size());
    m_contact_stencils.resize_discard(count);
    m_barrier_terms.resize_discard(count);
    CUDA_TOOL_CHECK(cudaMemset(m_barrier_counts.data(), 0, (vertices + 1) * sizeof(IndexT)));
    CUDA_TOOL_CHECK(cudaMemset(m_barrier_cursors.data(), 0, vertices * sizeof(IndexT)));
    if(count)
        sos_history_terms_kernel<<<(count + 127) / 128, 128>>>(
            m_history.records(),
            m_contact_stencils.view(),
            m_barrier_terms.view(),
            m_barrier_counts.view());
    ct::DeviceScan().ExclusiveSum(m_barrier_counts.data(), m_barrier_offsets.data(), vertices + 1);
    m_barrier_entries.resize_discard(4 * count);
    if(count)
        sos_history_rows_kernel<<<(count + 127) / 128, 128>>>(
            m_contact_stencils.cview(),
            m_barrier_offsets.cview(),
            m_barrier_cursors.view(),
            m_barrier_entries.view());
    sos_overlap_kernel<<<(vertices + 127) / 128, 128>>>(
        m_barrier_counts.cview(), m_vertex_offsets.cview(), m_overlap.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
}

bool FEMLocalStencilSolver::prepare_global_objective(Float alpha)
{
    const SizeT previous = m_history.records().size();
    if(m_custom_barriers)
        prepare_custom_contacts(alpha);
    // A predicted warm-start path deliberately precedes its barrier objective.
    // Newton directions, however, must be recomputed when new barriers have
    // nonzero force at the current iterate (their activation distance is the
    // frame-origin clearance, not the narrow-phase discovery distance).
    // A budgeted projected initializer, like the acceleration predictor, is
    // not a Newton direction certified against the old objective. Evaluate it
    // against the expanded energy and full geometric checks, then rebuild all
    // derivatives for the mandatory elasticity solve. No warm residual is
    // promoted to a physical-frame convergence certificate.
    // Guarded handoff likewise evaluates the proposal on the expanded full
    // objective; it is not an old-objective Newton direction to invalidate.
    const bool initial_guess =
        m_warm_active
        && (m_warm_iterations == 0
            || (m_warm_finishing && (warm_budget_exhausted() || m_full_objective_handoff)));
    return previous != m_history.records().size() && !initial_guess;
}
}  // namespace uipc::backend::cuda
