#include <finite_element/fem_local_stencil_solver.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <global_geometry/global_vertex_manager.h>
#include <contact_system/global_contact_manager.h>
#include <implicit_geometry/half_plane.h>
#include <implicit_geometry/half_plane_vertex_reporter.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <utils/primitive_d_hat.h>
#include <utils/codim_thickness.h>
#include <cuda_tool/cub.h>

namespace uipc::backend::cuda
{
namespace
{
    namespace ct = cuda_tool;
    __global__ void stencil_barrier_terms_kernel(ct::CBufferView<Vector4i> stencils,
                                                 ct::CBufferView<Vector2i> phs,
                                                 ct::BufferView<StencilBarrierTerm> terms,
                                                 ct::BufferView<IndexT> counts,
                                                 ct::CDense2D<ContactCoeff> table,
                                                 ct::CBufferView<IndexT> ids,
                                                 ct::CBufferView<Float> thicknesses,
                                                 ct::CBufferView<Float> d_hats,
                                                 ct::CBufferView<Vector3> plane_positions,
                                                 ct::CBufferView<Vector3> plane_normals,
                                                 IndexT vertex_offset,
                                                 IndexT plane_offset,
                                                 IndexT npt,
                                                 IndexT nee,
                                                 IndexT npe,
                                                 IndexT npp,
                                                 Float  dt)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= stencils.size())
            return;
        using namespace sym::codim_ipc_simplex_contact;
        const Vector4i s     = stencils(k);
        Vector4i       cids  = Vector4i::Zero();
        Vector4        thick = Vector4::Zero(), dh = Vector4::Zero();
        for(int a = 0; a < 4; ++a)
            if(s(a) >= 0)
            {
                const int v = s(a) + vertex_offset;
                cids(a)     = ids(v);
                thick(a)    = thicknesses(v);
                dh(a)       = d_hats(v);
                atomicAdd(&counts(s(a)), 1);
            }
        StencilBarrierTerm term;
        if(k < npt)
        {
            term.type      = 0;
            term.kappa_dt2 = PT_kappa(table, cids) * dt * dt;
            term.thickness = PT_thickness(thick(0), thick(1), thick(2), thick(3));
            term.d_hat = PT_d_hat(dh(0), dh(1), dh(2), dh(3));
        }
        else if(k < npt + nee)
        {
            term.type      = 1;
            term.kappa_dt2 = EE_kappa(table, cids) * dt * dt;
            term.thickness = EE_thickness(thick(0), thick(1), thick(2), thick(3));
            term.d_hat = EE_d_hat(dh(0), dh(1), dh(2), dh(3));
        }
        else if(k < npt + nee + npe)
        {
            term.type      = 2;
            term.kappa_dt2 = PE_kappa(table, cids.head<3>()) * dt * dt;
            term.thickness = PE_thickness(thick(0), thick(1), thick(2));
            term.d_hat     = PE_d_hat(dh(0), dh(1), dh(2));
        }
        else if(k < npt + nee + npe + npp)
        {
            term.type      = 3;
            term.kappa_dt2 = PP_kappa(table, cids.head<2>()) * dt * dt;
            term.thickness = PP_thickness(thick(0), thick(1));
            term.d_hat     = PP_d_hat(dh(0), dh(1));
        }
        else
        {
            term.type   = 4;
            const int h = phs(k - npt - nee - npe - npp)(1);
            term.kappa_dt2 = table(cids(0), ids(h + plane_offset)).kappa * dt * dt;
            term.thickness      = thick(0);
            term.d_hat          = dh(0);
            term.plane_position = plane_positions(h);
            term.plane_normal   = plane_normals(h);
        }
        terms(k) = term;
    }
    __global__ void stencil_barrier_rows_kernel(ct::CBufferView<Vector4i> stencils,
                                                ct::CBufferView<IndexT> offsets,
                                                ct::BufferView<IndexT>  cursors,
                                                ct::BufferView<IndexT>  entries)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= stencils.size())
            return;
        for(int a = 0; a < 4; ++a)
            if(stencils(k)(a) >= 0)
            {
                const int v = stencils(k)(a);
                entries(offsets(v) + atomicAdd(&cursors(v), 1)) = k;
            }
    }
}  // namespace

void FEMLocalStencilSolver::prepare_barriers()
{
    if(m_custom_barriers)
        return;
    const int n = static_cast<int>(m_contact_stencils.size());
    CUDA_TOOL_CHECK(
        cudaMemset(m_barrier_counts.data(), 0, m_barrier_counts.size() * sizeof(IndexT)));
    CUDA_TOOL_CHECK(cudaMemset(
        m_barrier_cursors.data(), 0, m_barrier_cursors.size() * sizeof(IndexT)));
    if(n > m_barrier_terms.capacity())
        m_barrier_terms.reserve_discard(n + n / 10);
    m_barrier_terms.resize_discard(n);
    if(n == 0)
    {
        CUDA_TOOL_CHECK(cudaMemset(
            m_barrier_offsets.data(), 0, m_barrier_offsets.size() * sizeof(IndexT)));
        m_barrier_entries.resize_discard(0);
        return;
    }
    auto simplex = m_trajectory_filter->find<SimplexTrajectoryFilter>();
    auto plane   = m_trajectory_filter->find<VertexHalfPlaneTrajectoryFilter>();
    const int npt = simplex ? static_cast<int>(simplex->PTs().size()) : 0;
    const int nee = simplex ? static_cast<int>(simplex->EEs().size()) : 0;
    const int npe = simplex ? static_cast<int>(simplex->PEs().size()) : 0;
    const int npp = simplex ? static_cast<int>(simplex->PPs().size()) : 0;
    auto      phs = plane ? plane->PHs() : ct::CBufferView<Vector2i>{};
    stencil_barrier_terms_kernel<<<(n + 127) / 128, 128>>>(
        m_contact_stencils.cview(),
        phs,
        m_barrier_terms.view(),
        m_barrier_counts.view(),
        m_contact_manager->contact_tabular().viewer(),
        m_vertices->contact_element_ids(),
        m_vertices->thicknesses(),
        m_vertices->d_hats(),
        m_planes->positions(),
        m_planes->normals(),
        m_vertex_reporter->vertex_offset(),
        m_plane_reporter->vertex_offset(),
        npt,
        nee,
        npe,
        npp,
        world().scene().dt());
    CUDA_TOOL_CHECK(cudaGetLastError());
    ct::DeviceScan().ExclusiveSum(m_barrier_counts.data(),
                                  m_barrier_offsets.data(),
                                  static_cast<int>(m_barrier_counts.size()));
    const int entries =
        4 * (npt + nee) + 3 * npe + 2 * npp + static_cast<int>(phs.size());
    if(entries > m_barrier_entries.capacity())
        m_barrier_entries.reserve_discard(entries + entries / 10);
    m_barrier_entries.resize_discard(entries);
    stencil_barrier_rows_kernel<<<(n + 127) / 128, 128>>>(m_contact_stencils.cview(),
                                                          m_barrier_offsets.cview(),
                                                          m_barrier_cursors.view(),
                                                          m_barrier_entries.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
}
}  // namespace uipc::backend::cuda
