#include <finite_element/fem_local_stencil_solver.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/local_stencil_elastic.h>
#include <finite_element/local_stencil_inversion.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <utils/distance/ccd.h>
#include <utils/distance/distance_flagged.h>
#include <utils/codim_thickness.h>

namespace uipc::backend::cuda
{
namespace
{
    namespace ct = cuda_tool;
    __device__ Vector3 local_motion(IndexT v, const Vector4i& stencil, const Vector12& direction)
    {
        for(int a = 0; a < 4; ++a)
            if(stencil(a) == v)
                return direction.segment<3>(3 * a);
        return Vector3::Zero();
    }
    __device__ bool visited_before(const Vector4i& term, const Vector4i& stencil, int a)
    {
        for(int b = 0; b < a; ++b)
            if(stencil(b) >= 0)
                for(int j = 0; j < 4; ++j)
                    if(term(j) == stencil(b))
                        return true;
        return false;
    }

    struct LocalObjective
    {
        ct::CBufferView<Vector4i>                elements, contacts;
        ct::CBufferView<Vector3>                 xs, rest, predicted;
        ct::CBufferView<Float>                   masses;
        ct::CBufferView<fem::StencilElasticTerm> terms;
        ct::CBufferView<IndexT> vertex_offsets, vertex_elements, barrier_offsets, barrier_entries;
        ct::CBufferView<StencilBarrierTerm> barriers;
        Float                               dt;

        __device__ void element_positions(
            int e, const Vector4i& s, const Vector12& d, Float alpha, Vector3* p) const
        {
            for(int a = 0; a < 4; ++a)
                if(elements(e)(a) >= 0)
                    p[a] = xs(elements(e)(a))
                           + alpha * local_motion(elements(e)(a), s, d);
        }

        __device__ Float barrier_energy(int e, const Vector4i& s, const Vector12& d, Float alpha) const
        {
            using namespace sym::codim_ipc_simplex_contact;
            const auto term = barriers(e);
            const auto ids  = contacts(e);
            Vector3    p[4];
            for(int a = 0; a < 4; ++a)
                if(ids(a) >= 0)
                    p[a] = xs(ids(a)) + alpha * local_motion(ids(a), s, d);
            Float D = 0;
            if(term.type == 0)
                distance::point_triangle_distance2(
                    distance::point_triangle_distance_flag(p[0], p[1], p[2], p[3]),
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    D);
            if(term.type == 1)
                distance::edge_edge_distance2(
                    distance::edge_edge_distance_flag(p[0], p[1], p[2], p[3]),
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    D);
            if(term.type == 2)
                distance::point_edge_distance2(
                    distance::point_edge_distance_flag(p[0], p[1], p[2]), p[0], p[1], p[2], D);
            if(term.type == 3)
                distance::point_point_distance2(p[0], p[1], D);
            if(term.type == 4)
            {
                const Float gap = (p[0] - term.plane_position).dot(term.plane_normal);
                if(gap <= term.thickness)
                    return std::numeric_limits<Float>::infinity();
                D = gap * gap;
            }
            const auto range = D_range(term.thickness, term.d_hat);
            if(D <= range(0))
                return std::numeric_limits<Float>::infinity();
            if(D >= range(1))
                return 0;
            const Float k = term.kappa_dt2, h = term.d_hat, thickness = term.thickness;
            if(term.type == 0)
                return PT_barrier_energy(
                    distance::point_triangle_distance_flag(p[0], p[1], p[2], p[3]),
                    k,
                    h,
                    thickness,
                    p[0],
                    p[1],
                    p[2],
                    p[3]);
            if(term.type == 1)
            {
                const auto flag =
                    distance::edge_edge_distance_flag(p[0], p[1], p[2], p[3]);
                return mollified_EE_barrier_energy(flag,
                                                   k,
                                                   h,
                                                   thickness,
                                                   rest(ids(0)),
                                                   rest(ids(1)),
                                                   rest(ids(2)),
                                                   rest(ids(3)),
                                                   p[0],
                                                   p[1],
                                                   p[2],
                                                   p[3]);
            }
            if(term.type == 2)
                return PE_barrier_energy(distance::point_edge_distance_flag(p[0], p[1], p[2]),
                                         k,
                                         h,
                                         thickness,
                                         p[0],
                                         p[1],
                                         p[2]);
            if(term.type == 3)
                return PP_barrier_energy(
                    distance::point_point_distance_flag(p[0], p[1]), k, h, thickness, p[0], p[1]);
            return sym::ipc_vertex_half_contact::PH_barrier_energy(
                k, h, thickness, p[0], term.plane_position, term.plane_normal);
        }

        __device__ Float energy(const Vector4i& s, const Vector12& d, Float alpha) const
        {
            Float energy = 0;
            for(int a = 0; a < 4; ++a)
            {
                const int v = s(a);
                if(v < 0)
                    continue;
                const Vector3 x = xs(v) + alpha * d.segment<3>(3 * a);
                energy += 0.5 * masses(v) * (x - predicted(v)).squaredNorm();
                for(int k = vertex_offsets(v); k < vertex_offsets(v + 1); ++k)
                {
                    const int e = vertex_elements(k);
                    if(visited_before(elements(e), s, a))
                        continue;
                    Vector3 p[4];
                    element_positions(e, s, d, alpha, p);
                    energy += dt * dt * fem::stencil_elastic_energy(terms(e), p);
                }
                for(int k = barrier_offsets(v); k < barrier_offsets(v + 1); ++k)
                {
                    const int e = barrier_entries(k);
                    if(!visited_before(contacts(e), s, a))
                        energy += barrier_energy(e, s, d, alpha);
                }
            }
            return energy;
        }

        __device__ Float contact_cap(int e, const Vector4i& s, const Vector12& d) const
        {
            const auto term = barriers(e);
            const auto ids  = contacts(e);
            Vector3    p[4], dx[4];
            for(int a = 0; a < 4; ++a)
                if(ids(a) >= 0)
                {
                    p[a]  = xs(ids(a));
                    dx[a] = local_motion(ids(a), s, d);
                }
            Float           toi = 1;
            bool            hit = false;
            constexpr Float eta = 0.2;
            if(term.type == 0)
                hit = distance::point_triangle_ccd(
                    p[0], p[1], p[2], p[3], dx[0], dx[1], dx[2], dx[3], eta, term.thickness, 100, toi);
            if(term.type == 1)
                hit = distance::edge_edge_ccd(
                    p[0], p[1], p[2], p[3], dx[0], dx[1], dx[2], dx[3], eta, term.thickness, 100, toi);
            if(term.type == 2)
                hit = distance::point_edge_ccd(
                    p[0], p[1], p[2], dx[0], dx[1], dx[2], eta, term.thickness, 100, toi);
            if(term.type == 3)
                hit = distance::point_point_ccd(
                    p[0], p[1], dx[0], dx[1], eta, term.thickness, 100, toi);
            if(term.type == 4)
            {
                const Float gap =
                    (p[0] - term.plane_position).dot(term.plane_normal) - term.thickness;
                const Float speed = dx[0].dot(term.plane_normal);
                if(gap <= 0)
                    return 0;
                return speed < 0 ? min(Float{1}, 0.8 * gap / -speed) : Float{1};
            }
            return hit ? max(Float{0}, min(Float{1}, toi)) : Float{1};
        }

        __device__ Float cap(const Vector4i& s, const Vector12& d, int& ccd, int& inversion) const
        {
            Float contact_alpha = 1, inversion_alpha = 1;
            for(int a = 0; a < 4; ++a)
            {
                const int v = s(a);
                if(v < 0)
                    continue;
                for(int k = barrier_offsets(v); k < barrier_offsets(v + 1); ++k)
                {
                    const int e = barrier_entries(k);
                    if(!visited_before(contacts(e), s, a))
                        contact_alpha = min(contact_alpha, contact_cap(e, s, d));
                }
                for(int k = vertex_offsets(v); k < vertex_offsets(v + 1); ++k)
                {
                    const int e = vertex_elements(k);
                    if(visited_before(elements(e), s, a))
                        continue;
                    Vector3 p[4], end[4];
                    element_positions(e, s, d, 0, p);
                    element_positions(e, s, d, 1, end);
                    inversion_alpha =
                        min(inversion_alpha, fem::stencil_elastic_cap(terms(e), p, end));
                }
            }
            ccd       = contact_alpha < 1;
            inversion = inversion_alpha < 1;
            return min(contact_alpha, inversion_alpha);
        }
    };

    __global__ void stencil_local_linesearch_kernel(LocalObjective objective,
                                                    ct::BufferView<Vector12> directions,
                                                    ct::CBufferView<Float> slopes,
                                                    ct::BufferView<IndexT> statistics)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= directions.size())
            return;
        Vector12 d = directions(t);
        if(d.squaredNorm() == 0)
            return;
        const Vector4i s   = t < objective.elements.size() ?
                                 objective.elements(t) :
                                 objective.contacts(t - objective.elements.size());
        int            ccd = 0, inversion = 0;
        Float          alpha = objective.cap(s, d, ccd, inversion);
        const Float    E0    = objective.energy(s, d, 0);
        UIPC_KERNEL_ASSERT(fem::stencil_finite(E0), "Nonfinite local stencil initial energy");
        int  backtracks = 0;
        bool accepted   = false;
        for(; alpha > 0 && backtracks < 32; ++backtracks)
        {
            const Float E = objective.energy(s, d, alpha);
            if(fem::stencil_finite(E) && E <= E0 + 1e-4 * alpha * slopes(t))
            {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        directions(t) = accepted ? (alpha * d).eval() : Vector12::Zero().eval();
        if(ccd)
            atomicAdd(&statistics(0), 1);
        if(inversion)
            atomicAdd(&statistics(1), 1);
        atomicAdd(&statistics(2), backtracks);
        atomicAdd(&statistics(accepted ? 3 : 4), 1);
    }
}  // namespace

void FEMLocalStencilSolver::filter_local_steps()
{
    prepare_barriers();
    CUDA_TOOL_CHECK(cudaMemset(m_local_statistics.data(), 0, 5 * sizeof(IndexT)));
    LocalObjective objective{m_elastic_stencils.cview(),
                             m_contact_stencils.cview(),
                             m_fem->xs(),
                             m_fem->x_bars(),
                             m_fem->x_tildes(),
                             m_fem->masses(),
                             m_elastic_terms.cview(),
                             m_vertex_offsets.cview(),
                             m_vertex_elements.cview(),
                             m_barrier_offsets.cview(),
                             m_barrier_entries.cview(),
                             m_barrier_terms.cview(),
                             world().scene().dt()};
    const int      n = static_cast<int>(m_directions.size());
    stencil_local_linesearch_kernel<<<(n + 63) / 64, 64>>>(objective,
                                                           m_directions.view(),
                                                           m_direction_slopes.cview(),
                                                           m_local_statistics.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    IndexT statistics[5];
    m_local_statistics.view().copy_to(statistics);
    logger::info("Local stencil filtering: ccd_caps={}, inversion_caps={}, backtracks={}, accepted={}, rejected={}",
                 statistics[0],
                 statistics[1],
                 statistics[2],
                 statistics[3],
                 statistics[4]);
}
}  // namespace uipc::backend::cuda
