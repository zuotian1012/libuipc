#pragma once
#include <finite_element/local_stencil.h>

namespace uipc::backend::cuda::fem
{
// A positive determinant alone is unreliable near coplanarity. Bound the
// cancellation in the six triple products before certifying a physical state.
// This scale-relative numerical guard is not a material compression limit or
// a convergence test. Ambiguous signs must fail even when det(F) rounds positive.
UIPC_GENERIC inline Float stencil_determinant_margin(const Matrix3x3& F)
{
    const Vector3 a = F.col(0), b = F.col(1), c = F.col(2);
    const Float   permanent = abs(a(0)) * (abs(b(1) * c(2)) + abs(b(2) * c(1)))
                            + abs(a(1)) * (abs(b(0) * c(2)) + abs(b(2) * c(0)))
                            + abs(a(2)) * (abs(b(0) * c(1)) + abs(b(1) * c(0)));
    const Float determinant = F.determinant();
    if(!stencil_finite(determinant) || !stencil_finite(permanent))
        return -1;
    return determinant - 64 * std::numeric_limits<Float>::epsilon() * permanent;
}
}  // namespace uipc::backend::cuda::fem
