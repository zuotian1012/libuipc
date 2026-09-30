#pragma once
#include <finite_element/local_stencil_elastic_data.h>
#include <finite_element/local_stencil_snk.h>
#include <finite_element/local_stencil_inversion.h>
#include <finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.h>
#include <finite_element/constitutions/neo_hookean_shell_2d_function.h>
#include <finite_element/constitutions/discrete_shell_bending_function.h>

namespace uipc::backend::cuda::fem
{
UIPC_GENERIC inline Float stencil_triangle_measure(const Vector3& a,
                                                   const Vector3& b,
                                                   const Vector3& c)
{
    return (b - a).cross(c - a).norm();
}

UIPC_GENERIC inline Float stencil_shell_energy(const StencilElasticTerm& t, const Vector3* p)
{
    if(t.type == 17)
        return t.weight
               * sym::discrete_shell_bending::E(
                   p[0], p[1], p[2], p[3], t.length, t.height, t.angle, t.mu);
    if(stencil_triangle_measure(p[0], p[1], p[2]) <= 1e-12 * t.rest_area2)
        return std::numeric_limits<Float>::infinity();
    if(t.type == 819)
    {
        namespace BW = sym::strainlimiting_baraff_witkin_shell_2d;
        const Matrix<Float, 3, 2> F = BW::Ds3x2(p[0], p[1], p[2]) * t.shell_inverse;
        return t.weight * BW::E(F, Vector2(1, 0), Vector2(0, 1), t.lambda, t.mu, t.strain_rate);
    }
    Vector9 X;
    for(int a = 0; a < 3; ++a)
        X.segment<3>(3 * a) = p[a];
    Float E;
    sym::neo_hookean_shell_2d::E(E, t.lambda, t.mu, X, t.shell_inverse);
    return t.weight * E;
}

UIPC_GENERIC inline Float stencil_elastic_energy(const StencilElasticTerm& t, const Vector3* p)
{
    if(t.type == 10)
    {
        const Matrix3x3 F = fem::F(p[0], p[1], p[2], p[3], t.inverse);
        if(F.determinant() <= 0)
            return std::numeric_limits<Float>::infinity();
        Float E;
        snk1::E(E, t.mu, t.lambda, F);
        return t.weight * E;
    }
    return stencil_shell_energy(t, p);
}

// Exact, unprojected Hessian of the existing BW energy. Its standard solver
// helper deliberately uses a PSD approximation; Newton-CG here handles raw H.
UIPC_GENERIC inline Matrix6x6 stencil_bw_hessian(const Matrix<Float, 3, 2>& F,
                                                 Float stretch,
                                                 Float shear,
                                                 Float rate)
{
    Matrix6x6 H = Matrix6x6::Zero();
    for(int a = 0; a < 2; ++a)
    {
        const Vector3 f = F.col(a);
        const Float   r = f.norm(), d = r - 1;
        const Float   first         = 2 * d + (r > 1 ? 3 * rate * d * d : 0);
        const Float   second        = 2 + (r > 1 ? 6 * rate * d : 0);
        const Vector3 u             = f / r;
        H.block<3, 3>(3 * a, 3 * a) = stretch
                                      * ((first / r) * Matrix3x3::Identity()
                                         + (second - first / r) * (u * u.transpose()));
    }
    const Vector3 u = F.col(0), v = F.col(1);
    const Float   s = u.dot(v);
    H.block<3, 3>(0, 0) += 2 * shear * (v * v.transpose());
    H.block<3, 3>(3, 3) += 2 * shear * (u * u.transpose());
    H.block<3, 3>(0, 3) = 2 * shear * (v * u.transpose() + s * Matrix3x3::Identity());
    H.block<3, 3>(3, 0) = H.block<3, 3>(0, 3).transpose();
    return H;
}

UIPC_GENERIC inline void stencil_shell_derivatives(const StencilElasticTerm& t,
                                                   const Vector3*            p,
                                                   Float                     dt,
                                                   Vector12&                 g,
                                                   Matrix12x12&              H)
{
    const Float scale = t.weight * dt * dt;
    g.setZero();
    H.setZero();
    if(t.type == 17)
    {
        namespace DSB = sym::discrete_shell_bending;
        DSB::dEdx(g, p[0], p[1], p[2], p[3], t.length, t.height, t.angle, t.mu);
        DSB::ddEddx(H, p[0], p[1], p[2], p[3], t.length, t.height, t.angle, t.mu);
    }
    else if(t.type == 819)
    {
        namespace BW = sym::strainlimiting_baraff_witkin_shell_2d;
        const Matrix<Float, 3, 2> F = BW::Ds3x2(p[0], p[1], p[2]) * t.shell_inverse;
        const auto          J = BW::dFdX(t.shell_inverse);
        Matrix<Float, 3, 2> P;
        BW::dEdF(P, F, Vector2(1, 0), Vector2(0, 1), t.lambda, t.mu, t.strain_rate);
        g.head<9>() = J.transpose() * BW::flatten(P);
        H.topLeftCorner<9, 9>() =
            J.transpose() * stencil_bw_hessian(F, t.lambda, t.mu, t.strain_rate) * J;
    }
    else
    {
        Vector9   X, G;
        Matrix9x9 B;
        for(int a = 0; a < 3; ++a)
            X.segment<3>(3 * a) = p[a];
        sym::neo_hookean_shell_2d::dEdX(G, t.lambda, t.mu, X, t.shell_inverse);
        sym::neo_hookean_shell_2d::ddEddX(B, t.lambda, t.mu, X, t.shell_inverse);
        g.head<9>()             = G;
        H.topLeftCorner<9, 9>() = B;
    }
    g *= scale;
    H *= scale;
}

UIPC_GENERIC inline void stencil_elastic_derivatives(const StencilElasticTerm& t,
                                                     const Vector3* p,
                                                     Float          dt,
                                                     Vector12&      g,
                                                     Matrix12x12&   H)
{
    const Float scale = t.weight * dt * dt;
    if(t.type == 10)
    {
        snk_stencil_element(
            fem::F(p[0], p[1], p[2], p[3], t.inverse), t.inverse, t.mu, t.lambda, scale, g, H);
        return;
    }
    stencil_shell_derivatives(t, p, dt, g, H);
}

// Freeze the CURRENT normal over a trial. Positive projected area prevents
// collapse anywhere along the quadratic triangle trajectory, while permitting
// arbitrary accumulated rotations over successive accepted steps.
UIPC_GENERIC inline Float stencil_triangle_cap(const Vector3* p, const Vector3* q, int a, int b, int c)
{
    Matrix3x3 F, dF;
    F.col(0)  = p[b] - p[a];
    F.col(1)  = p[c] - p[a];
    F.col(2)  = F.col(0).cross(F.col(1)).normalized();
    dF.col(0) = q[b] - q[a] - F.col(0);
    dF.col(1) = q[c] - q[a] - F.col(1);
    dF.col(2).setZero();
    return stencil_inversion_cap(F, dF);
}

UIPC_GENERIC inline Float stencil_elastic_cap(const StencilElasticTerm& t,
                                              const Vector3*            p,
                                              const Vector3*            q)
{
    if(t.type == 10)
    {
        const Matrix3x3 F  = fem::F(p[0], p[1], p[2], p[3], t.inverse);
        const Matrix3x3 dF = fem::F(q[0], q[1], q[2], q[3], t.inverse) - F;
        return stencil_inversion_cap(F, dF);
    }
    Float cap = stencil_triangle_cap(p, q, 0, 1, 2);
    if(t.type == 17)
        cap = std::min(cap, stencil_triangle_cap(p, q, 3, 2, 1));
    return cap;
}
}  // namespace uipc::backend::cuda::fem
