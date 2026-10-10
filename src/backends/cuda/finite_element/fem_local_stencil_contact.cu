#include <finite_element/fem_local_stencil_solver.h>
#include <finite_element/fem_dytopo_effect_receiver.h>
#include <finite_element/finite_element_vertex_reporter.h>
#include <collision_detection/global_trajectory_filter.h>
#include <collision_detection/simplex_trajectory_filter.h>
#include <collision_detection/vertex_half_plane_trajectory_filter.h>
#include <cuda_tool/cub.h>
#include <cuda_tool/eigen/atomic.h>
#include <kernel_cout.h>

namespace uipc::backend::cuda
{
namespace
{
    namespace ct = cuda_tool;

    __global__ void stencil_contact_reset_kernel(ct::BufferView<Vector3> gradients,
                                                 ct::BufferView<IndexT> counts,
                                                 ct::BufferView<IndexT> cursors,
                                                 ct::BufferView<IndexT> overlap,
                                                 ct::CBufferView<IndexT> vertex_offsets)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= gradients.size())
            return;
        gradients(v).setZero();
        counts(v)  = 0;
        cursors(v) = 0;
        overlap(v) = vertex_offsets(v + 1) - vertex_offsets(v);
        if(v == 0)
            counts(gradients.size()) = 0;
    }

    __global__ void stencil_contact_gradient_kernel(ct::CDoubletVectorView<Float, 3> source,
                                                    ct::BufferView<Vector3> gradients,
                                                    IndexT vertex_offset)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= source.doublet_count())
            return;
        auto [i, g] = source(k);
        const int v = i - vertex_offset;
        UIPC_KERNEL_ASSERT(v >= 0 && v < gradients.size(), "Contact vertex outside FEM range");
        ct::eigen::atomic_add(gradients(v), g);
    }

    __global__ void stencil_contact_count_kernel(ct::CTripletMatrixView<Float, 3> hessians,
                                                 ct::BufferView<IndexT> counts,
                                                 IndexT vertex_offset)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= hessians.triplet_count())
            return;
        auto [i, j, H] = hessians(k);
        i -= vertex_offset;
        j -= vertex_offset;
        UIPC_KERNEL_ASSERT(i >= 0 && j >= 0 && i < counts.size() - 1
                               && j < counts.size() - 1,
                           "Contact Hessian outside FEM range");
        atomicAdd(&counts(i), 1);
        if(i != j)
            atomicAdd(&counts(j), 1);
    }

    __global__ void stencil_contact_rows_kernel(ct::CTripletMatrixView<Float, 3> hessians,
                                                ct::CBufferView<IndexT> offsets,
                                                ct::BufferView<IndexT>  cursors,
                                                ct::BufferView<IndexT>  entries,
                                                IndexT vertex_offset)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= hessians.triplet_count())
            return;
        auto [i, j, H] = hessians(k);
        i -= vertex_offset;
        j -= vertex_offset;
        entries(offsets(i) + atomicAdd(&cursors(i), 1)) = k;
        if(i != j)
            entries(offsets(j) + atomicAdd(&cursors(j), 1)) = -k - 1;
    }

    __global__ void stencil_contact_topology_kernel(ct::CBufferView<Vector4i> pts,
                                                    ct::CBufferView<Vector4i> ees,
                                                    ct::CBufferView<Vector3i> pes,
                                                    ct::CBufferView<Vector2i> pps,
                                                    ct::CBufferView<Vector2i> phs,
                                                    ct::BufferView<Vector4i> stencils,
                                                    ct::BufferView<IndexT> overlap,
                                                    IndexT vertex_offset)
    {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if(k >= stencils.size())
            return;
        int      t        = k;
        Vector4i vertices = Vector4i::Constant(-1);
        int      count    = 4;
        if(t < pts.size())
            vertices = pts(t);
        else if((t -= pts.size()) < ees.size())
            vertices = ees(t);
        else if((t -= ees.size()) < pes.size())
        {
            vertices.head<3>() = pes(t);
            count              = 3;
        }
        else if((t -= pes.size()) < pps.size())
        {
            vertices.head<2>() = pps(t);
            count              = 2;
        }
        else
        {
            t -= pps.size();
            vertices(0) = phs(t)(0);  // The plane has no dynamic coordinates.
            count       = 1;
        }
        for(int a = 0; a < count; ++a)
        {
            vertices(a) -= vertex_offset;
            UIPC_KERNEL_ASSERT(vertices(a) >= 0 && vertices(a) < overlap.size(),
                               "local_stencil contact requires FEM vertices");
            atomicAdd(&overlap(vertices(a)), 1);
        }
        stencils(k) = vertices;
    }
}  // namespace

void FEMLocalStencilSolver::prepare_contacts()
{
    if(m_custom_barriers)
    {
        prepare_custom_contacts(0);
        m_history_changed = false;  // Derivatives will use the newly built list.
        return;
    }
    const int vertices = static_cast<int>(m_contact_gradients.size());
    stencil_contact_reset_kernel<<<(vertices + 127) / 128, 128>>>(
        m_contact_gradients.view(),
        m_contact_counts.view(),
        m_contact_cursors.view(),
        m_overlap.view(),
        m_vertex_offsets.cview());
    CUDA_TOOL_CHECK(cudaGetLastError());
    if(!m_contact_receiver)
    {
        m_contact_stencils.resize_discard(0);
        CUDA_TOOL_CHECK(
            cudaMemset(m_contact_offsets.data(), 0, (vertices + 1) * sizeof(IndexT)));
        return;
    }

    const int  offset    = m_vertex_reporter->vertex_offset();
    const auto gradients = m_contact_receiver->gradients();
    const auto hessians  = m_contact_receiver->hessians();
    const int  ng        = static_cast<int>(gradients.doublet_count());
    const int  nh        = static_cast<int>(hessians.triplet_count());
    if(ng)
        stencil_contact_gradient_kernel<<<(ng + 127) / 128, 128>>>(
            gradients, m_contact_gradients.view(), offset);
    if(nh)
        stencil_contact_count_kernel<<<(nh + 127) / 128, 128>>>(
            hessians, m_contact_counts.view(), offset);
    CUDA_TOOL_CHECK(cudaGetLastError());
    ct::DeviceScan().ExclusiveSum(m_contact_counts.data(), m_contact_offsets.data(), vertices + 1);
    // Every upper-triangular block contributes at most two row entries.
    const SizeT capacity = 2 * static_cast<SizeT>(nh);
    if(capacity > m_contact_entries.capacity())
        m_contact_entries.reserve_discard(capacity + capacity / 10);
    m_contact_entries.resize_discard(capacity);
    if(nh)
        stencil_contact_rows_kernel<<<(nh + 127) / 128, 128>>>(hessians,
                                                               m_contact_offsets.cview(),
                                                               m_contact_cursors.view(),
                                                               m_contact_entries.view(),
                                                               offset);
    CUDA_TOOL_CHECK(cudaGetLastError());

    auto simplex = m_trajectory_filter->find<SimplexTrajectoryFilter>();
    auto plane   = m_trajectory_filter->find<VertexHalfPlaneTrajectoryFilter>();
    const auto pts = simplex ? simplex->PTs() : ct::CBufferView<Vector4i>{};
    const auto ees = simplex ? simplex->EEs() : ct::CBufferView<Vector4i>{};
    const auto pes = simplex ? simplex->PEs() : ct::CBufferView<Vector3i>{};
    const auto pps = simplex ? simplex->PPs() : ct::CBufferView<Vector2i>{};
    const auto phs = plane ? plane->PHs() : ct::CBufferView<Vector2i>{};
    const SizeT count = pts.size() + ees.size() + pes.size() + pps.size() + phs.size();
    if(count > m_contact_stencils.capacity())
        m_contact_stencils.reserve_discard(count + count / 10);
    m_contact_stencils.resize_discard(count);
    if(count)
        stencil_contact_topology_kernel<<<(count + 127) / 128, 128>>>(
            pts, ees, pes, pps, phs, m_contact_stencils.view(), m_overlap.view(), offset);
    CUDA_TOOL_CHECK(cudaGetLastError());
    ct::DeviceReduce().Max(m_overlap.data(), m_max_overlap.data(), vertices);
    IndexT maximum;
    m_max_overlap.view().copy_to(&maximum);
    m_weight = 1.0 / maximum;
}

fem::LocalStencilContactView FEMLocalStencilSolver::contact_view() const
{
    if(m_inner_sweeps > 1 || m_custom_barriers || m_packed_product)
        return {m_contact_gradients.cview(),
                m_barrier_offsets.cview(),
                m_barrier_entries.cview(),
                {},
                0,
                m_contact_stencils.cview(),
                m_frozen_contact_hessians.cview(),
                true};
    return {m_contact_gradients.cview(),
            m_contact_offsets.cview(),
            m_contact_entries.cview(),
            m_contact_receiver ? m_contact_receiver->hessians() :
                                 ct::CTripletMatrixView<Float, 3>{},
            m_vertex_reporter->vertex_offset()};
}
}  // namespace uipc::backend::cuda
