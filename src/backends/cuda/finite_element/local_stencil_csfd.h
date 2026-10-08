#pragma once
#include <finite_element/local_stencil_snk.h>

namespace uipc::backend::cuda::fem
{
// Complex arithmetic used for directional stress derivatives. Eigen's complex
// dot/cross/norm operations conjugate their inputs and are unsuitable for the
// analytic continuation required by CSFD; these scalar operations never do.
struct StencilComplex
{
    Float                       real = 0, imag = 0;
    UIPC_GENERIC StencilComplex operator+(StencilComplex b) const
    {
        return {real + b.real, imag + b.imag};
    }
    UIPC_GENERIC StencilComplex operator-(StencilComplex b) const
    {
        return {real - b.real, imag - b.imag};
    }
    UIPC_GENERIC StencilComplex operator*(StencilComplex b) const
    {
        return {real * b.real - imag * b.imag, real * b.imag + imag * b.real};
    }
    UIPC_GENERIC StencilComplex operator*(Float b) const
    {
        return {real * b, imag * b};
    }
};

// Imag(P(F+i*h*dF))/h, without subtracting nearly equal real stresses.
// The step is dimensionless because F and dF are dimensionless. Caller freezes
// the real iterate during a CG solve. This evaluates the existing SNK1 material;
// it must not silently replace it by a different neo-Hookean potential.
UIPC_GENERIC inline Matrix3x3 snk_stress_csfd(const Matrix3x3& F, const Matrix3x3& dF, Float mu, Float lambda)
{
    constexpr Float h = 1e-20;
    StencilComplex  z[3][3], cof[3][3];
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            z[r][c] = {F(r, c), h * dF(r, c)};
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
        {
            const int a = (r + 1) % 3, b = (r + 2) % 3;
            const int j = (c + 1) % 3, k = (c + 2) % 3;
            cof[r][c] = z[a][j] * z[b][k] - z[a][k] * z[b][j];
        }
    StencilComplex J;
    for(int r = 0; r < 3; ++r)
        J = J + z[r][0] * cof[r][0];
    const auto pressure = (J - StencilComplex{1, 0}) * lambda - StencilComplex{mu, 0};
    Matrix3x3 result;
    for(int r = 0; r < 3; ++r)
        for(int c = 0; c < 3; ++c)
            result(r, c) = (z[r][c] * mu + cof[r][c] * pressure).imag / h;
    return result;
}

UIPC_GENERIC inline Vector12 snk_stencil_product_csfd(const Matrix3x3& F,
                                                      const Matrix3x3& inverse,
                                                      Float            mu,
                                                      Float            lambda,
                                                      Float volume_dt2,
                                                      const Vector12& direction)
{
    const auto shapes = tetrahedron_shape_gradients(inverse);
    Matrix3x3  dF     = Matrix3x3::Zero();
    for(int a = 0; a < 4; ++a)
        dF += direction.segment<3>(3 * a) * shapes.col(a).transpose();
    const Matrix3x3 dP = snk_stress_csfd(F, dF, mu, lambda);
    Vector12        result;
    for(int a = 0; a < 4; ++a)
        result.segment<3>(3 * a) = volume_dt2 * dP * shapes.col(a);
    return result;
}
}  // namespace uipc::backend::cuda::fem
