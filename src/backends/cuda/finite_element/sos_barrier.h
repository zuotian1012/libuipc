#pragma once
#include <finite_element/local_stencil.h>

namespace uipc::backend::cuda::fem
{
// Equation 11 uses a distance, not libuipc's squared-distance IPC parameter.
// The caller supplies positive geometric clearance (distance minus thickness).
// kappa includes the incremental-potential time scale exactly once.
// The native IPC dispatcher uses a squared-distance log^2 barrier, whereas
// Eq.11 uses a distance log barrier. Their kappas have different dimensions.
// Match normal force at clearance c=h/2, before per-pair compensation. This
// explicit reference calibration does not equate the two potentials elsewhere.
UIPC_GENERIC inline Float sos_distance_stiffness(Float ipc_kappa, Float h, Float thickness)
{
    const Float c = .5 * h, q = c * c + 2 * thickness * c,
                range = h * h + 2 * thickness * h;
    const Float delta = q - range, logarithm = log(q / range);
    const Float native_derivative =
        ipc_kappa * (2 * delta * logarithm * logarithm + 2 * delta * delta * logarithm / q)
        * 2 * (c + thickness);
    const Float distance_derivative = -(c - h) * (2 * log(c / h) + (c - h) / c);
    return native_derivative / distance_derivative;
}

struct SoSBarrierParameters
{
    Float activation_distance = 0;
    Float kappa               = 0;

    UIPC_GENERIC static SoSBarrierParameters from_reference(Float reference_clearance,
                                                            Float global_distance,
                                                            Float global_kappa)
    {
        assert(reference_clearance > 0 && global_distance > 0 && global_kappa >= 0);
        return {reference_clearance, global_kappa * global_distance / reference_clearance};
    }
};

struct SoSBarrierValue
{
    Float energy = 0, first = 0, second = 0;
};

UIPC_GENERIC inline SoSBarrierValue sos_barrier(Float clearance, SoSBarrierParameters parameters)
{
    const Float h = parameters.activation_distance;
    if(clearance <= 0 || !stencil_finite(clearance))
        return {std::numeric_limits<Float>::infinity(),
                std::numeric_limits<Float>::quiet_NaN(),
                std::numeric_limits<Float>::quiet_NaN()};
    if(clearance >= h)
        return {};
    const Float delta = clearance - h;
    // log1p avoids cancellation for newly activated pairs near d == d_hat.
    // log(d/h) remains necessary when d/h is tiny and delta/h rounds to -1.
    const Float logarithm = clearance > .5 * h ? log1p(delta / h) : log(clearance / h);
    const Float ratio = delta / clearance;
    return {-parameters.kappa * delta * delta * logarithm,
            -parameters.kappa * delta * (2 * logarithm + ratio),
            -parameters.kappa * (2 * logarithm + 4 * ratio - ratio * ratio)};
}

// Complete chain rule through q=||distance||^2 and d=sqrt(q)-thickness.
// Both rank-one curvature and distance curvature are required. A frozen real
// closest-feature branch supplies grad(q) and H(q); no PSD projection is used.
template <int N>
UIPC_GENERIC inline void sos_barrier_derivatives(Float squared_distance,
                                                 Float thickness,
                                                 SoSBarrierParameters parameters,
                                                 const Vector<Float, N>& distance_gradient,
                                                 const Matrix<Float, N, N>& distance_hessian,
                                                 Vector<Float, N>&    gradient,
                                                 Matrix<Float, N, N>& hessian)
{
    const Float distance = sqrt(squared_distance);
    const auto  b        = sos_barrier(distance - thickness, parameters);
    const Float first    = b.first / (2 * distance);
    const Float second   = b.second / (4 * squared_distance)
                         - b.first / (4 * squared_distance * distance);
    gradient = first * distance_gradient;
    hessian  = first * distance_hessian
              + second * distance_gradient * distance_gradient.transpose();
}
}  // namespace uipc::backend::cuda::fem
