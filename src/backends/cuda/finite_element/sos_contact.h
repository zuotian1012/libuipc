#pragma once
#include <finite_element/local_stencil_barrier.h>
#include <finite_element/sos_barrier.h>
#include <utils/distance/distance_flagged.h>
#include <utils/distance/edge_edge_mollifier.h>

namespace uipc::backend::cuda::fem
{
// Real closest-feature branches, full primitive coordinates. At a feature
// boundary use the same branch for energy, gradient and Hessian.
UIPC_GENERIC inline Float sos_distance_squared(const StencilBarrierTerm& term,
                                               const Vector3*            p)
{
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
    else
    {
        const Float gap = (p[0] - term.plane_position).dot(term.plane_normal);
        D               = gap > 0 ? gap * gap : 0;
    }
    return D;
}
UIPC_GENERIC inline Float sos_contact_energy(const StencilBarrierTerm& term,
                                             const Vector3*            p,
                                             Float                     D)
{
    const Float energy =
        sos_barrier(std::sqrt(D) - term.thickness, {term.d_hat, term.kappa_dt2}).energy;
    // A zero mollifier must never hide an infeasible configuration.
    if(!stencil_finite(energy))
        return energy;
    Float m = 1;
    if(term.type == 1 && term.mollifier_eps > 0)
        distance::edge_edge_mollifier(p[0], p[1], p[2], p[3], term.mollifier_eps, m);
    return m * energy;
}
UIPC_GENERIC inline void sos_contact_derivatives(const StencilBarrierTerm& term,
                                                 const Vector3*            p,
                                                 Vector12&                 g,
                                                 Matrix12x12&              H)
{
    g.setZero();
    H.setZero();
    const Float                D = sos_distance_squared(term, p);
    const SoSBarrierParameters parameters{term.d_hat, term.kappa_dt2};
    if(std::sqrt(D) - term.thickness >= term.d_hat)
        return;
    Vector12    gD = Vector12::Zero();
    Matrix12x12 HD = Matrix12x12::Zero();
    if(term.type == 0)
    {
        const auto flag =
            distance::point_triangle_distance_flag(p[0], p[1], p[2], p[3]);
        distance::point_triangle_distance2_gradient(flag, p[0], p[1], p[2], p[3], gD);
        distance::point_triangle_distance2_hessian(flag, p[0], p[1], p[2], p[3], HD);
    }
    else if(term.type == 1)
    {
        const auto flag = distance::edge_edge_distance_flag(p[0], p[1], p[2], p[3]);
        distance::edge_edge_distance2_gradient(flag, p[0], p[1], p[2], p[3], gD);
        distance::edge_edge_distance2_hessian(flag, p[0], p[1], p[2], p[3], HD);
    }
    else
    {
        const Float gap = (p[0] - term.plane_position).dot(term.plane_normal);
        const auto  b   = sos_barrier(gap - term.thickness, parameters);
        g.head<3>()     = b.first * term.plane_normal;
        H.topLeftCorner<3, 3>() =
            b.second * term.plane_normal * term.plane_normal.transpose();
        return;
    }
    sos_barrier_derivatives<12>(D, term.thickness, parameters, gD, HD, g, H);
    if(term.type == 1 && term.mollifier_eps > 0
       && distance::need_mollify(p[0], p[1], p[2], p[3], term.mollifier_eps))
    {
        Float       m;
        Vector12    gm;
        Matrix12x12 Hm;
        distance::edge_edge_mollifier(p[0], p[1], p[2], p[3], term.mollifier_eps, m);
        distance::edge_edge_mollifier_gradient(p[0], p[1], p[2], p[3], term.mollifier_eps, gm);
        distance::edge_edge_mollifier_hessian(p[0], p[1], p[2], p[3], term.mollifier_eps, Hm);
        const Float b = sos_barrier(std::sqrt(D) - term.thickness, parameters).energy;
        H = (m * H + b * Hm + gm * g.transpose() + g * gm.transpose()).eval();
        g = (m * g + b * gm).eval();
    }
}
}  // namespace uipc::backend::cuda::fem
