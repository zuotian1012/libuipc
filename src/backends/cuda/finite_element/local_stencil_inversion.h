#pragma once
#include <finite_element/local_stencil.h>

namespace uipc::backend::cuda::fem
{
// Earliest positive zero of det(F + t dF), including a tangential zero that
// endpoint-sign tests miss. Derivative roots partition the cubic into monotone
// intervals; bisection brackets the first unsafe point, then keeps 80% of it.
UIPC_GENERIC inline Float stencil_inversion_cap(const Matrix3x3& F, const Matrix3x3& dF)
{
    const Vector3 a = F.col(0), b = F.col(1), c = F.col(2);
    const Vector3 da = dF.col(0), db = dF.col(1), dc = dF.col(2);
    const Float   c0 = a.dot(b.cross(c));
    const Float c1 = da.dot(b.cross(c)) + a.dot(db.cross(c)) + a.dot(b.cross(dc));
    const Float c2 = da.dot(db.cross(c)) + da.dot(b.cross(dc)) + a.dot(db.cross(dc));
    const Float c3 = da.dot(db.cross(dc));
    if(!stencil_finite(c0) || !stencil_finite(c1) || !stencil_finite(c2)
       || !stencil_finite(c3) || c0 <= 0)
        return 0;
    const Float scale = abs(c1) + abs(c2) + abs(c3);
    if(scale == 0)
        return 1;
    const Float tol       = 64 * std::numeric_limits<Float>::epsilon() * scale;
    Float       points[4] = {0, 1, 1, 1};
    int         count     = 2;
    const Float A = 3 * c3, B = 2 * c2, C = c1;
    if(abs(A) <= tol)
    {
        if(abs(B) > tol)
        {
            const Float r = -C / B;
            if(r > 0 && r < 1)
                points[count++] = r;
        }
    }
    else
    {
        const Float disc = B * B - 4 * A * C;
        if(disc >= 0)
        {
            const Float q  = -0.5 * (B + (B >= 0 ? sqrt(disc) : -sqrt(disc)));
            const Float r0 = q / A;
            const Float r1 = q == 0 ? -B / (2 * A) : C / q;
            if(r0 > 0 && r0 < 1)
                points[count++] = r0;
            if(r1 > 0 && r1 < 1 && count < 4)
                points[count++] = r1;
        }
    }
    for(int i = 1; i < count; ++i)
        for(int j = i; j > 0 && points[j] < points[j - 1]; --j)
        {
            const Float tmp = points[j];
            points[j]       = points[j - 1];
            points[j - 1]   = tmp;
        }
    for(int i = 1; i < count; ++i)
    {
        const Float t     = points[i];
        const Float value = ((c3 * t + c2) * t + c1) * t + c0;
        if(value <= c0 * 1e-12)
        {
            Float lo = points[i - 1], hi = t;
            for(int k = 0; k < 48; ++k)
            {
                const Float mid = (lo + hi) * 0.5;
                if(((c3 * mid + c2) * mid + c1) * mid + c0 > c0 * 1e-12)
                    lo = mid;
                else
                    hi = mid;
            }
            return 0.8 * lo;
        }
    }
    return 1;
}
}  // namespace uipc::backend::cuda::fem
