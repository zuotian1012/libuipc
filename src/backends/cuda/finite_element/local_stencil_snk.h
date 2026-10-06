#pragma once
#include <finite_element/fem_utils.h>
#include <finite_element/constitutions/stable_neo_hookean_3d_function.h>

namespace uipc::backend::cuda::fem
{
// Exact polynomial energy increment. Subtracting two SNK1 energies loses tiny
// descent steps against its nonzero rest-energy offset (mu^2 / (2 lambda)).
UIPC_GENERIC inline Float snk_stencil_energy_change(const Matrix3x3& F,
                                                    const Matrix3x3& dF,
                                                    Float            mu,
                                                    Float            lambda)
{
    const Vector3 a = F.col(0), b = F.col(1), c = F.col(2);
    const Vector3 da = dF.col(0), db = dF.col(1), dc = dF.col(2);
    const Float   higher = da.dot(db.cross(c)) + da.dot(b.cross(dc))
                         + a.dot(db.cross(dc)) + da.dot(db.cross(dc));
    const Float dJ = da.dot(b.cross(c)) + a.dot(db.cross(c)) + a.dot(b.cross(dc)) + higher;
    Matrix3x3 stress;
    snk1::dEdF(stress, mu, lambda, F);
    return (stress.array() * dF.array()).sum() + 0.5 * mu * dF.squaredNorm()
           + (lambda * (F.determinant() - 1) - mu) * higher + 0.5 * lambda * dJ * dJ;
}

// Unprojected directional stress derivative for the EXISTING SNK1 energy.
// Keep this experimental path separate from the production SPD assembly.
UIPC_GENERIC inline Matrix3x3 snk_stress_direction(const Matrix3x3& F,
                                                   const Matrix3x3& dF,
                                                   Float            mu,
                                                   Float            lambda)
{
    Matrix3x3 cofactor, dcofactor;
    for(int i = 0; i < 3; ++i)
    {
        const int j     = (i + 1) % 3;
        const int k     = (i + 2) % 3;
        cofactor.col(i) = F.col(j).cross(F.col(k));
        dcofactor.col(i) = dF.col(j).cross(F.col(k)) + F.col(j).cross(dF.col(k));
    }
    const Float dJ = (cofactor.array() * dF.array()).sum();
    return mu * dF + lambda * dJ * cofactor + (lambda * (F.determinant() - 1) - mu) * dcofactor;
}

// volume_dt2 = rest volume * dt^2, matching libuipc's incremental potential.
// Positive material parameters and a valid rest mesh are caller preconditions.
UIPC_GENERIC inline void snk_stencil_element(const Matrix3x3& F,
                                             const Matrix3x3& Dm_inverse,
                                             Float            mu,
                                             Float            lambda,
                                             Float            volume_dt2,
                                             Vector12&        gradient,
                                             Matrix12x12&     hessian)
{
    const auto shapes = tetrahedron_shape_gradients(Dm_inverse);
    Matrix3x3  stress;
    snk1::dEdF(stress, mu, lambda, F);
    for(int i = 0; i < 4; ++i)
        gradient.segment<3>(3 * i) = volume_dt2 * stress * shapes.col(i);

    for(int j = 0; j < 12; ++j)
    {
        Matrix3x3 dF       = Matrix3x3::Zero();
        dF.row(j % 3)      = shapes.col(j / 3).transpose();
        const Matrix3x3 dP = snk_stress_direction(F, dF, mu, lambda);
        for(int i = 0; i < 4; ++i)
            hessian.block<3, 1>(3 * i, j) = volume_dt2 * dP * shapes.col(i);
    }
}
}  // namespace uipc::backend::cuda::fem
