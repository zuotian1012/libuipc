#include <finite_element/fem_local_stencil_solver.h>
#include <sim_engine.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/local_stencil_elastic.h>
#include <finite_element/local_stencil_inversion.h>
#include <finite_element/sos_contact.h>
#include <contact_system/contact_models/codim_ipc_simplex_normal_contact_function.h>
#include <contact_system/contact_models/ipc_vertex_half_plane_contact_function.h>
#include <utils/distance/ccd.h>
#include <utils/distance/distance_flagged.h>
#include <utils/codim_thickness.h>
#include <cuda_tool/cub.h>
#include <uipc/common/timer.h>
#include <cstdlib>

namespace uipc::backend::cuda
{
namespace
{
    namespace ct = cuda_tool;
    bool energy_audit_enabled()
    {
        static const bool enabled = []
        {
            const char* value = std::getenv("UIPC_STENCIL_ENERGY_AUDIT");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        return enabled;
    }
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
        ct::CBufferView<Matrix3x3>          deformation;
        ct::CBufferView<Float> elastic_energy, barrier_energy_at_start;

        bool                      inertia_only = false;
        ct::CBufferView<Vector4i> update_stencils;
        Float                     direction_fraction = 1;
        bool                      defer_inversion    = false;
        ct::CBufferView<IndexT>   submission_overlap;
        ct::CBufferView<Vector3>  search_gradient, search_contact_gradient;

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
            if(term.customized)
                return fem::sos_contact_energy(term, p, D);
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

        struct Evaluation
        {
            Float change = 0, alpha = 1;
            int   ccd = 0, inversion = 0;
        };

        // Evaluate the exact incident energy increment and, on the first
        // trial, its continuous feasibility cap in one adjacency traversal.
        __device__ Evaluation evaluate(const Vector4i& s, const Vector12& d, Float alpha, bool bound) const
        {
            Evaluation result;
            Float      contact_alpha = 1, inversion_alpha = 1;
            for(int a = 0; a < 4; ++a)
            {
                const int v = s(a);
                if(v < 0)
                    continue;
                const Vector3 move = alpha * d.segment<3>(3 * a);
                result.change +=
                    masses(v)
                    * ((xs(v) - predicted(v)).dot(move) + 0.5 * move.squaredNorm());
                for(int k = vertex_offsets(v);
                    !defer_inversion && k < vertex_offsets(v + 1);
                    ++k)
                {
                    const int  e   = vertex_elements(k);
                    const auto ids = elements(e);
                    if(visited_before(ids, s, a))
                        continue;
                    const auto term = terms(e);
                    Vector3    original[4], motion[4];
                    for(int j = 0; j < 4; ++j)
                        if(ids(j) >= 0)
                        {
                            if(term.type != 10)
                                original[j] = xs(ids(j));
                            motion[j] = local_motion(ids(j), s, d);
                        }
                    Float change;
                    if(term.type == 10)
                    {
                        const Matrix3x3 F = deformation(e);
                        const Matrix3x3 dF =
                            fem::F(motion[0], motion[1], motion[2], motion[3], term.inverse);
                        if(bound)
                            inversion_alpha =
                                min(inversion_alpha, fem::stencil_inversion_cap(F, dF));
                        const Matrix3x3 increment = alpha * dF;
                        const Matrix3x3 trial     = F + increment;
                        change                    = trial.determinant() > 0 ?
                                                        term.weight
                                         * fem::snk_stencil_energy_change(
                                             F, increment, term.mu, term.lambda) :
                                                        std::numeric_limits<Float>::infinity();
                    }
                    else
                    {
                        Vector3 trial[4], end[4];
                        for(int j = 0; j < 4; ++j)
                            if(ids(j) >= 0)
                            {
                                trial[j] = original[j] + alpha * motion[j];
                                if(bound)
                                    end[j] = original[j] + motion[j];
                            }
                        if(bound)
                            inversion_alpha =
                                min(inversion_alpha,
                                    fem::stencil_elastic_cap(term, original, end));
                        change = fem::stencil_elastic_energy(term, trial)
                                 - elastic_energy(e);
                    }
                    if(!inertia_only)
                        result.change += dt * dt * change;
                }
                for(int k = barrier_offsets(v); k < barrier_offsets(v + 1); ++k)
                {
                    const int e = barrier_entries(k);
                    if(visited_before(contacts(e), s, a))
                        continue;
                    if(bound)
                        contact_alpha = min(contact_alpha, contact_cap(e, s, d));
                    result.change +=
                        barrier_energy(e, s, d, alpha) - barrier_energy_at_start(e);
                }
            }
            result.ccd       = contact_alpha < 1;
            result.inversion = inversion_alpha < 1;
            result.alpha     = min(contact_alpha, inversion_alpha);
            return result;
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
    };

    // Rebuilt at each frozen iterate, shared by all incident local trials and
    // the merged inner search. Directions and acceptance inequalities are unchanged.
    __global__ void stencil_filter_state_kernel(LocalObjective objective,
                                                ct::BufferView<Matrix3x3> deformation,
                                                ct::BufferView<Float> elastic_energy,
                                                ct::BufferView<Float> barrier_energy)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < objective.elements.size())
        {
            const auto ids  = objective.elements(i);
            const auto term = objective.terms(i);
            Vector3    p[4];
            for(int a = 0; a < 4; ++a)
                if(ids(a) >= 0)
                    p[a] = objective.xs(ids(a));
            Matrix3x3 F      = Matrix3x3::Zero();
            Float     energy = 0;
            if(term.type == 10)
                F = fem::F(p[0], p[1], p[2], p[3], term.inverse);
            else
                energy = fem::stencil_elastic_energy(term, p);
            deformation(i)    = F;
            elastic_energy(i) = energy;
        }
        else if(i < objective.elements.size() + objective.contacts.size())
        {
            const int e = i - objective.elements.size();
            barrier_energy(e) =
                objective.barrier_energy(e, objective.contacts(e), Vector12::Zero(), 0);
        }
    }

    __global__ void stencil_frozen_derivatives_kernel(ct::CBufferView<Vector4i> ids,
                                                      ct::CBufferView<StencilBarrierTerm> terms,
                                                      ct::CBufferView<Vector3> xs,
                                                      ct::CBufferView<Vector3> rest,
                                                      ct::BufferView<Vector12> gradients,
                                                      ct::BufferView<Matrix12x12> hessians)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if(e >= ids.size())
            return;
        using namespace sym::codim_ipc_simplex_contact;
        const auto s    = ids(e);
        const auto term = terms(e);
        Vector3    p[4];
        for(int a = 0; a < 4; ++a)
            if(s(a) >= 0)
                p[a] = xs(s(a));
        Float D = 0;
        if(term.type == 0)
            distance::point_triangle_distance2(
                distance::point_triangle_distance_flag(p[0], p[1], p[2], p[3]),
                p[0],
                p[1],
                p[2],
                p[3],
                D);
        else if(term.type == 1)
            distance::edge_edge_distance2(
                distance::edge_edge_distance_flag(p[0], p[1], p[2], p[3]), p[0], p[1], p[2], p[3], D);
        else if(term.type == 2)
            distance::point_edge_distance2(
                distance::point_edge_distance_flag(p[0], p[1], p[2]), p[0], p[1], p[2], D);
        else if(term.type == 3)
            distance::point_point_distance2(p[0], p[1], D);
        else
        {
            const Float gap = (p[0] - term.plane_position).dot(term.plane_normal);
            UIPC_KERNEL_ASSERT(gap > term.thickness, "Infeasible frozen plane barrier %d", e);
            D = gap * gap;
        }
        Vector12    g = Vector12::Zero();
        Matrix12x12 H = Matrix12x12::Zero();
        if(term.customized)
        {
            fem::sos_contact_derivatives(term, p, g, H);
            gradients(e) = g;
            hessians(e)  = H;
            return;
        }
        const auto range = D_range(term.thickness, term.d_hat);
        UIPC_KERNEL_ASSERT(fem::stencil_finite(D) && D > range(0),
                           "Infeasible frozen barrier %d",
                           e);
        if(D < range(1))
        {
            const Float k = term.kappa_dt2, h = term.d_hat, t = term.thickness;
            if(term.type == 0)
                PT_barrier_gradient_hessian(
                    g,
                    H,
                    distance::point_triangle_distance_flag(p[0], p[1], p[2], p[3]),
                    k,
                    h,
                    t,
                    p[0],
                    p[1],
                    p[2],
                    p[3]);
            else if(term.type == 1)
                mollified_EE_barrier_gradient_hessian(
                    g,
                    H,
                    distance::edge_edge_distance_flag(p[0], p[1], p[2], p[3]),
                    k,
                    h,
                    t,
                    rest(s(0)),
                    rest(s(1)),
                    rest(s(2)),
                    rest(s(3)),
                    p[0],
                    p[1],
                    p[2],
                    p[3]);
            else if(term.type == 2)
            {
                Vector9   g9;
                Matrix9x9 H9;
                PE_barrier_gradient_hessian(
                    g9,
                    H9,
                    distance::point_edge_distance_flag(p[0], p[1], p[2]),
                    k,
                    h,
                    t,
                    p[0],
                    p[1],
                    p[2]);
                g.head<9>()             = g9;
                H.topLeftCorner<9, 9>() = H9;
            }
            else if(term.type == 3)
            {
                Vector6   g6;
                Matrix6x6 H6;
                PP_barrier_gradient_hessian(
                    g6, H6, distance::point_point_distance_flag(p[0], p[1]), k, h, t, p[0], p[1]);
                g.head<6>()             = g6;
                H.topLeftCorner<6, 6>() = H6;
            }
            else
            {
                Vector3   g3;
                Matrix3x3 H3;
                sym::ipc_vertex_half_contact::PH_barrier_gradient_hessian(
                    g3, H3, k, h, t, p[0], term.plane_position, term.plane_normal);
                g.head<3>()             = g3;
                H.topLeftCorner<3, 3>() = H3;
            }
        }
        gradients(e) = g;
        hessians(e)  = H;
    }

    __global__ void stencil_frozen_gather_kernel(ct::CBufferView<Vector4i> ids,
                                                 ct::CBufferView<Vector12> gradients,
                                                 ct::CBufferView<IndexT> offsets,
                                                 ct::CBufferView<IndexT> entries,
                                                 ct::BufferView<Vector3> result)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= result.size())
            return;
        Vector3 g = Vector3::Zero();
        for(int k = offsets(v); k < offsets(v + 1); ++k)
        {
            const int e = entries(k);
            for(int a = 0; a < 4; ++a)
                if(ids(e)(a) == v)
                    g += gradients(e).segment<3>(3 * a);
        }
        result(v) = g;
    }

    __global__ void stencil_inner_contact_cap_kernel(LocalObjective objective,
                                                     ct::CBufferView<Vector3> dx,
                                                     ct::BufferView<Float> caps)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if(e >= objective.contacts.size())
            return;
        const auto ids = objective.contacts(e);
        Vector12   d   = Vector12::Zero();
        for(int a = 0; a < 4; ++a)
            if(ids(a) >= 0)
                d.segment<3>(3 * a) = dx(ids(a));
        caps(e) = objective.contact_cap(e, ids, d);
    }

    // Clip the actual combined warm path, not isolated coordinate directions.
    // Re-evaluate after per-vertex scaling: unequal caps at an element's vertices
    // do not inherit the element's original scalar certificate.
    __global__ void stencil_warm_path_caps_kernel(LocalObjective objective,
                                                  ct::CBufferView<Vector3> dx,
                                                  ct::BufferView<Float>    caps)
    {
        const int e = blockIdx.x * blockDim.x + threadIdx.x;
        if(e >= caps.size())
            return;
        if(e < objective.elements.size())
        {
            const auto ids = objective.elements(e);
            Vector3    p[4], q[4];
            for(int a = 0; a < 4; ++a)
                if(ids(a) >= 0)
                {
                    p[a] = objective.xs(ids(a));
                    q[a] = p[a] + dx(ids(a));
                }
            caps(e) = fem::stencil_elastic_cap(objective.terms(e), p, q);
        }
        else
        {
            const int  c   = e - objective.elements.size();
            const auto ids = objective.contacts(c);
            Vector12   d   = Vector12::Zero();
            for(int a = 0; a < 4; ++a)
                if(ids(a) >= 0)
                    d.segment<3>(3 * a) = dx(ids(a));
            caps(e) = objective.contact_cap(c, ids, d);
        }
    }
    __global__ void stencil_warm_local_cap_kernel(LocalObjective objective,
                                                  ct::CBufferView<Float>  caps,
                                                  ct::BufferView<Vector3> dx)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v >= dx.size())
            return;
        Float cap = 1;
        for(int k = objective.vertex_offsets(v); k < objective.vertex_offsets(v + 1); ++k)
            cap = min(cap, caps(objective.vertex_elements(k)));
        for(int k = objective.barrier_offsets(v); k < objective.barrier_offsets(v + 1); ++k)
            cap = min(cap, caps(objective.elements.size() + objective.barrier_entries(k)));
        dx(v) *= cap;
    }

    __global__ void stencil_inner_energy_kernel(LocalObjective objective,
                                                ct::CBufferView<Vector3> dx,
                                                Float                    alpha,
                                                ct::BufferView<Float> changes)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= changes.size())
            return;
        int e = i;
        if(e < objective.xs.size())
        {
            const Vector3 d = alpha * dx(e);
            changes(i)      = objective.masses(e)
                         * ((objective.xs(e) - objective.predicted(e)).dot(d)
                            + .5 * d.squaredNorm());
        }
        else if((e -= objective.xs.size()) < objective.elements.size())
        {
            const auto ids  = objective.elements(e);
            const auto term = objective.terms(e);
            Vector3    p[4], q[4], d[4];
            for(int a = 0; a < 4; ++a)
                if(ids(a) >= 0)
                {
                    p[a] = objective.xs(ids(a));
                    d[a] = alpha * dx(ids(a));
                    q[a] = p[a] + d[a];
                }
            Float change;
            if(term.type == 10)
            {
                const Matrix3x3 F = objective.deformation(e);
                const Matrix3x3 dF = fem::F(d[0], d[1], d[2], d[3], term.inverse);
                change = (F + dF).determinant() > 0 ?
                             term.weight
                                 * fem::snk_stencil_energy_change(F, dF, term.mu, term.lambda) :
                             std::numeric_limits<Float>::infinity();
            }
            else
                change = fem::stencil_elastic_energy(term, q)
                         - objective.elastic_energy(e);
            changes(i) = objective.inertia_only ? 0 : objective.dt * objective.dt * change;
        }
        else
        {
            e -= objective.elements.size();
            const auto ids = objective.contacts(e);
            Vector12   d   = Vector12::Zero();
            for(int a = 0; a < 4; ++a)
                if(ids(a) >= 0)
                    d.segment<3>(3 * a) = dx(ids(a));
            changes(i) = objective.barrier_energy(e, ids, d, alpha)
                         - objective.barrier_energy_at_start(e);
        }
    }

    __global__ void stencil_objective_energy_kernel(LocalObjective objective,
                                                    ct::BufferView<Float> values)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= values.size())
            return;
        int e = i;
        if(e < objective.xs.size())
            values(i) = .5 * objective.masses(e)
                        * (objective.xs(e) - objective.predicted(e)).squaredNorm();
        else if((e -= objective.xs.size()) < objective.elements.size())
        {
            Vector3    p[4];
            const auto ids = objective.elements(e);
            for(int a = 0; a < 4; ++a)
                if(ids(a) >= 0)
                    p[a] = objective.xs(ids(a));
            values(i) = objective.inertia_only ?
                            0 :
                            objective.dt * objective.dt
                                * fem::stencil_elastic_energy(objective.terms(e), p);
        }
        else
        {
            e -= objective.elements.size();
            values(i) =
                objective.barrier_energy(e, objective.contacts(e), Vector12::Zero(), 0);
        }
    }

    __global__ void stencil_inner_slope_kernel(ct::CBufferView<Vector3> g,
                                               ct::CBufferView<Vector3> dx,
                                               ct::BufferView<Float>    slopes)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v < g.size())
            slopes(v) = g(v).dot(dx(v));
    }

    __global__ void stencil_inner_commit_kernel(ct::BufferView<Vector3>  xs,
                                                ct::CBufferView<Vector3> dx,
                                                Float                    alpha)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;
        if(v < xs.size())
            xs(v) += alpha * dx(v);
    }

    __global__ void stencil_local_linesearch_kernel(LocalObjective objective,
                                                    ct::BufferView<Vector12> directions,
                                                    ct::CBufferView<Float> slopes,
                                                    ct::BufferView<IndexT> statistics)
    {
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if(t >= directions.size())
            return;
        const Vector12 original = directions(t);
        Vector12       d        = original;
        if(d.squaredNorm() == 0)
            return;
        const Vector4i s     = objective.update_stencils.size() ?
                                   objective.update_stencils(t) :
                               t < objective.elements.size() ?
                                   objective.elements(t) :
                                   objective.contacts(t - objective.elements.size());
        Float          slope = slopes(t);
        if(objective.submission_overlap.size())
        {
            // Jacobi submits each component divided by that vertex's overlap.
            // Unequal degrees can turn a raw descent direction uphill. Check
            // the actual contribution, including CCD and Armijo, before summing.
            // Keep directions in raw units; the gather performs this division.
            slope = 0;
            for(int a = 0; a < 4; ++a)
                if(const int v = s(a); v >= 0)
                {
                    d.segment<3>(3 * a) /= max(objective.submission_overlap(v), 1);
                    Vector3 g = objective.search_gradient(v);
                    if(objective.search_contact_gradient.size())
                        g += objective.search_contact_gradient(v);
                    slope += g.dot(d.segment<3>(3 * a));
                }
        }
        if(!fem::stencil_finite(slope) || slope >= 0)
        {
            directions(t).setZero();
            atomicAdd(&statistics(4), 1);
            return;
        }
        auto      evaluation = objective.evaluate(s, d, 1, true);
        const int ccd = evaluation.ccd, inversion = evaluation.inversion;
        Float     alpha = min(evaluation.alpha, objective.direction_fraction);
        if(alpha > 0 && alpha < 1)
            evaluation = objective.evaluate(s, d, alpha, false);
        int  backtracks = 0;
        bool accepted   = false;
        for(; alpha > 0 && backtracks < 32; ++backtracks)
        {
            if(fem::stencil_finite(evaluation.change)
               && evaluation.change <= 1e-4 * alpha * slope)
            {
                accepted = true;
                break;
            }
            alpha *= 0.5;
            if(backtracks + 1 < 32)
                evaluation = objective.evaluate(s, d, alpha, false);
        }
        directions(t) = accepted ? (alpha * original).eval() : Vector12::Zero().eval();
        if(ccd)
            atomicAdd(&statistics(0), 1);
        if(inversion)
            atomicAdd(&statistics(1), 1);
        atomicAdd(&statistics(2), backtracks);
        atomicAdd(&statistics(accepted ? 3 : 4), 1);
    }
}  // namespace

void FEMLocalStencilSolver::project_warm_guess_locally()
{
    LocalObjective objective;
    objective.elements        = m_elastic_stencils.cview();
    objective.contacts        = m_contact_stencils.cview();
    objective.terms           = m_elastic_terms.cview();
    objective.barriers        = m_barrier_terms.cview();
    objective.xs              = m_fem->xs();
    objective.vertex_offsets  = m_vertex_offsets.cview();
    objective.vertex_elements = m_vertex_elements.cview();
    objective.barrier_offsets = m_barrier_offsets.cview();
    objective.barrier_entries = m_barrier_entries.cview();
    const int count    = objective.elements.size() + objective.contacts.size();
    const int vertices = objective.xs.size();
    m_inner_values.resize_discard(count);
    Float first_cap = 1;
    for(int pass = 0; pass < 32; ++pass)
    {
        stencil_warm_path_caps_kernel<<<(count + 127) / 128, 128>>>(
            objective, m_fem->m_impl.dxs.cview(), m_inner_values.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
        ct::DeviceReduce().Min(m_inner_values.data(), m_scalar.data(), count);
        Float cap;
        m_scalar.view().copy_to(&cap);
        if(!std::isfinite(cap) || cap < 0 || cap > 1)
            throw SimEngineException("Invalid local warm projection bound.");
        if(pass == 0)
            first_cap = cap;
        if(cap == 1)
        {
            logger::info("SoS local warm projection: {} passes, initial minimum cap={}",
                         pass,
                         first_cap);
            return;
        }
        stencil_warm_local_cap_kernel<<<(vertices + 127) / 128, 128>>>(
            objective, m_inner_values.cview(), m_fem->m_impl.dxs.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
    }
    // The caller always certifies the resulting full path with combined
    // inversion and regular CCD, including pairs outside the retained history.
    logger::info("SoS local warm projection reached 32 passes; final global certification follows");
}

void FEMLocalStencilSolver::refresh_frozen_contacts()
{
    const int n        = static_cast<int>(m_contact_stencils.size());
    const int vertices = static_cast<int>(m_fem->xs().size());
    m_frozen_contact_gradients.resize_discard(n);
    m_frozen_contact_hessians.resize_discard(n);
    if(n)
        stencil_frozen_derivatives_kernel<<<(n + 63) / 64, 64>>>(
            m_contact_stencils.cview(),
            m_barrier_terms.cview(),
            m_fem->xs(),
            m_fem->x_bars(),
            m_frozen_contact_gradients.view(),
            m_frozen_contact_hessians.view());
    stencil_frozen_gather_kernel<<<(vertices + 127) / 128, 128>>>(
        m_contact_stencils.cview(),
        m_frozen_contact_gradients.cview(),
        m_barrier_offsets.cview(),
        m_barrier_entries.cview(),
        m_contact_gradients.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
}

bool FEMLocalStencilSolver::accept_inner_step(const char* stage)
{
    Timer     timer{"Stencil Merged Inner Acceptance"};
    const int vertices = static_cast<int>(m_fem->xs().size());
    stencil_inner_slope_kernel<<<(vertices + 127) / 128, 128>>>(
        m_total_gradients.cview(), m_fem->m_impl.dxs.cview(), m_global_slopes.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    ct::DeviceReduce().Sum(m_global_slopes.data(), m_scalar.data(), vertices);
    Float slope;
    m_scalar.view().copy_to(&slope);
    if(!std::isfinite(slope))
        throw SimEngineException("Nonfinite frozen-objective slope.");
    if(slope >= 0 && !m_local_merge_acceptance)
    {
        if(energy_audit_enabled())
            logger::info("SoS energy audit rejected slope: frame={}, outer={}, inner={}, stage={}, slope={}",
                         engine().frame(),
                         engine().newton_iter(),
                         m_inner_index,
                         stage,
                         slope);
        return false;
    }
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
                             world().scene().dt(),
                             m_filter_deformation.cview(),
                             m_filter_elastic_energy.cview(),
                             m_filter_barrier_energy.cview(),
                             m_warm_active};
    Float          alpha    = 1;
    const int      contacts = static_cast<int>(m_contact_stencils.size());
    if(contacts)
    {
        m_inner_values.resize_discard(contacts);
        stencil_inner_contact_cap_kernel<<<(contacts + 127) / 128, 128>>>(
            objective, m_fem->m_impl.dxs.cview(), m_inner_values.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
        ct::DeviceReduce().Min(m_inner_values.data(), m_scalar.data(), contacts);
        m_scalar.view().copy_to(&alpha);
        if(!std::isfinite(alpha) || alpha < 0 || alpha > 1)
            throw SimEngineException("Invalid frozen-contact combined CCD bound.");
    }
    const int count = vertices + static_cast<int>(m_elastic_stencils.size()) + contacts;
    m_inner_values.resize_discard(count);
    for(int trial = 0; trial < 32 && alpha > 0; ++trial, alpha *= .5)
    {
        stencil_inner_energy_kernel<<<(count + 127) / 128, 128>>>(
            objective, m_fem->m_impl.dxs.cview(), alpha, m_inner_values.view());
        CUDA_TOOL_CHECK(cudaGetLastError());
        ct::DeviceReduce().Sum(m_inner_values.data(), m_scalar.data(), count);
        Float change;
        m_scalar.view().copy_to(&change);
        const bool accepted =
            std::isfinite(change)
            && (m_local_merge_acceptance || change <= 1e-4 * alpha * slope);
        if(energy_audit_enabled())
            logger::info("SoS energy audit: frame={}, outer={}, inner={}, warm={}, stage={}, slope={}, alpha={}, change={}, accepted={}",
                         engine().frame(),
                         engine().newton_iter(),
                         m_inner_index,
                         m_warm_active,
                         stage,
                         slope,
                         alpha,
                         change,
                         accepted);
        if(accepted)
        {
            stencil_inner_commit_kernel<<<(vertices + 127) / 128, 128>>>(
                m_fem->m_impl.xs.view(), m_fem->m_impl.dxs.cview(), alpha);
            CUDA_TOOL_CHECK(cudaGetLastError());
            return true;
        }
    }
    logger::info("Local stencil rejected merged inner step; retaining frozen iterate");
    return false;
}

void FEMLocalStencilSolver::filter_local_steps(bool state_only, bool jacobi_submission)
{
    Timer     timer{"Stencil Local Filtering"};
    const int elastic_count = static_cast<int>(m_elastic_stencils.size());
    const int contact_count = static_cast<int>(m_contact_stencils.size());
    m_filter_deformation.resize_discard(elastic_count);
    m_filter_elastic_energy.resize_discard(elastic_count);
    m_filter_barrier_energy.resize_discard(contact_count);
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
                             world().scene().dt(),
                             m_filter_deformation.cview(),
                             m_filter_elastic_energy.cview(),
                             m_filter_barrier_energy.cview(),
                             m_warm_active,
                             m_custom_barriers ? m_update_stencils.cview() :
                                                 ct::CBufferView<Vector4i>{},
                             m_warm_active ? m_warm_relaxation : 1,
                             m_warm_active && m_warm_post_inversion && !m_warm_finishing};
    if(m_warm_active || m_jacobi || jacobi_submission)
    {
        objective.submission_overlap = m_overlap.cview();
        objective.search_gradient    = m_total_gradients.cview();
        // The sequential second sweep has already assembled the full gradient
        // at the committed GS state. Other paths keep contact forces separate.
        if(!jacobi_submission)
            objective.search_contact_gradient = m_contact_gradients.cview();
    }
    stencil_filter_state_kernel<<<(elastic_count + contact_count + 127) / 128, 128>>>(
        objective,
        m_filter_deformation.view(),
        m_filter_elastic_energy.view(),
        m_filter_barrier_energy.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    if(state_only)
        return;
    const int n = static_cast<int>(m_directions.size());
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
Float FEMLocalStencilSolver::objective_energy()
{
    // Guarded handoff compares origin and candidate using the same full
    // objective and current history, while warm convergence stays disabled.
    const bool     warm_objective = m_warm_active && !guarded_warm_handoff();
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
                             world().scene().dt(),
                             {},
                             {},
                             {},
                             warm_objective};
    const SizeT    count =
        m_fem->xs().size() + m_elastic_stencils.size() + m_contact_stencils.size();
    m_objective_values.resize_discard(count);
    stencil_objective_energy_kernel<<<(count + 127) / 128, 128>>>(
        objective, m_objective_values.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    ct::DeviceReduce().Sum(m_objective_values.data(), m_scalar.data(), count);
    Float energy;
    m_scalar.view().copy_to(&energy);
    if(energy_audit_enabled())
        logger::info("SoS objective audit: frame={}, outer={}, warm={}, energy={}",
                     engine().frame(),
                     engine().newton_iter(),
                     warm_objective,
                     energy);
    return energy;
}
}  // namespace uipc::backend::cuda
