#pragma once
#include <type_define.h>
#include <limits>
#include <cassert>

namespace uipc::backend::cuda::fem
{
// Stage-one SoS building block. The caller supplies every energy term incident
// on these four vertices, exactly once. This is a restriction of the full
// objective, NOT the Hessian of just the central tetrahedron.
struct LocalStencilSystem
{
    Vector12    gradient = Vector12::Zero();
    Matrix12x12 hessian  = Matrix12x12::Zero();

    UIPC_GENERIC void add_inertia(int local_vertex, Float mass, const Vector3& offset)
    {
        assert(local_vertex >= 0 && local_vertex < 4);
        assert(mass >= 0 && mass <= std::numeric_limits<Float>::max());
        gradient.segment<3>(3 * local_vertex) += mass * offset;
        hessian.block<3, 3>(3 * local_vertex, 3 * local_vertex).diagonal().array() += mass;
    }

    UIPC_GENERIC void add_element(const Vector4i&    stencil,
                                  const Vector4i&    element,
                                  const Vector12&    element_gradient,
                                  const Matrix12x12& element_hessian)
    {
        int map[4] = {-1, -1, -1, -1};
        for(int i = 0; i < 4; ++i)
            if(stencil(i) >= 0)
                for(int a = 0; a < 4; ++a)
                    if(stencil(i) == element(a))
                        map[i] = a;
        // Most incident elements share only one vertex. Find the matching
        // blocks once instead of repeating vertex comparisons for every block.
        for(int i = 0; i < 4; ++i)
        {
            if(map[i] < 0)
                continue;
            gradient.segment<3>(3 * i) += element_gradient.segment<3>(3 * map[i]);
            for(int j = 0; j < 4; ++j)
                if(map[j] >= 0)
                    hessian.block<3, 3>(3 * i, 3 * j) +=
                        element_hessian.block<3, 3>(3 * map[i], 3 * map[j]);
        }
    }

    // Call only AFTER all incident contributions have been accumulated.
    UIPC_GENERIC void fix_vertex(int local_vertex)
    {
        assert(local_vertex >= 0 && local_vertex < 4);
        gradient.segment<3>(3 * local_vertex).setZero();
        hessian.middleRows<3>(3 * local_vertex).setZero();
        hessian.middleCols<3>(3 * local_vertex).setZero();
        hessian.block<3, 3>(3 * local_vertex, 3 * local_vertex).setIdentity();
    }
};

UIPC_GENERIC inline bool stencil_finite(Float value)
{
    const Float limit = std::numeric_limits<Float>::max();
    return value == value && value <= limit && value >= -limit;
}

struct LocalStencilDirection
{
    Vector12 displacement          = Vector12::Zero();
    Float    residual_norm         = 0;
    int      iterations            = 0;
    bool     valid                 = false;
    bool     converged             = false;
    bool     nonpositive_curvature = false;
    bool     gradient_fallback     = false;
};

// Projection-free Newton-CG, starting from zero. Uses the early-termination
// alternative explicitly described in SoS section 4.1, rather than choosing
// an undocumented sigma for Algorithm 1's negative-curvature replacement.
// A first-iteration curvature failure falls back to steepest descent.
// Twelve steps are a budget, not an exact-solve guarantee in floating point.
template <typename HessianProduct>
UIPC_GENERIC inline LocalStencilDirection local_stencil_direction(
    const Vector12&       gradient,
    const HessianProduct& product,
    Float                 relative_tolerance = 1e-10,
    Float                 absolute_tolerance = 1e-12)
{
    LocalStencilDirection result;
    if(!stencil_finite(relative_tolerance) || relative_tolerance < 0
       || !stencil_finite(absolute_tolerance) || absolute_tolerance < 0)
        return result;
    for(int i = 0; i < 12; ++i)
    {
        if(!stencil_finite(gradient(i)))
            return result;
    }

    Vector12 residual  = -gradient;
    Vector12 direction = residual;
    Float    r2        = residual.squaredNorm();
    if(!stencil_finite(r2))
        return result;
    const Float tolerance = absolute_tolerance + relative_tolerance * sqrt(r2);
    if(!stencil_finite(tolerance))
        return result;
    if(sqrt(r2) <= tolerance)
    {
        result.residual_norm = sqrt(r2);
        result.valid         = true;
        result.converged     = true;
        return result;
    }

    for(int iteration = 0; iteration < 12 && sqrt(r2) > tolerance; ++iteration)
    {
        const Vector12 Hd        = product(direction);
        const Float    curvature = direction.dot(Hd);
        if(!stencil_finite(curvature))
            return result;
        if(curvature <= 0)
        {
            result.nonpositive_curvature = true;
            break;
        }
        const Float alpha = r2 / curvature;
        if(!stencil_finite(alpha))
            return result;
        result.displacement += alpha * direction;
        residual -= alpha * Hd;
        const Float next_r2 = residual.squaredNorm();
        if(!stencil_finite(next_r2))
            return result;
        direction = residual + (next_r2 / r2) * direction;
        r2        = next_r2;
        ++result.iterations;
    }

    // A descent check also protects a truncated solve from roundoff breakdown.
    if(gradient.squaredNorm() > 0 && gradient.dot(result.displacement) >= 0)
    {
        result.displacement      = -gradient;
        result.gradient_fallback = true;
    }
    for(int i = 0; i < 12; ++i)
        if(!stencil_finite(result.displacement(i)))
            return result;
    result.residual_norm = (gradient + product(result.displacement)).norm();
    result.valid         = stencil_finite(result.residual_norm);
    result.converged     = result.valid && result.residual_norm <= tolerance;
    return result;
}

struct LocalStencilMatrixProduct
{
    const Matrix12x12&    matrix;
    UIPC_GENERIC Vector12 operator()(const Vector12& direction) const
    {
        return matrix * direction;
    }
};

// Preserve the assembled operator as a control and an independent HVP oracle.
UIPC_GENERIC inline LocalStencilDirection local_stencil_direction(
    const LocalStencilSystem& system, Float relative_tolerance = 1e-10, Float absolute_tolerance = 1e-12)
{
    for(int i = 0; i < 12; ++i)
        for(int j = 0; j < 12; ++j)
            if(!stencil_finite(system.hessian(i, j)))
                return {};
    return local_stencil_direction(system.gradient,
                                   LocalStencilMatrixProduct{system.hessian},
                                   relative_tolerance,
                                   absolute_tolerance);
}

struct LocalStencilStep
{
    LocalStencilDirection direction;
    Vector12              displacement  = Vector12::Zero();
    Float                 alpha         = 0;
    Float                 energy_before = 0;
    Float                 energy_after  = 0;
    int                   trials        = 0;
    bool                  accepted      = false;
};

// EnergyAtOffset evaluates the SAME incident objective with all other DOFs
// frozen. It must return +infinity for an inadmissible trial. This Armijo search
// does not implement CCD or a continuous inversion check.
template <typename EnergyAtOffset>
UIPC_GENERIC LocalStencilStep local_stencil_step(const LocalStencilSystem& system,
                                                 const EnergyAtOffset& energy)
{
    LocalStencilStep result;
    result.direction = local_stencil_direction(system);
    if(!result.direction.valid)
        return result;
    result.energy_before = energy(Vector12::Zero());
    result.energy_after  = result.energy_before;
    const Float slope    = system.gradient.dot(result.direction.displacement);
    if(!stencil_finite(result.energy_before) || !stencil_finite(slope) || slope >= 0)
        return result;

    Float alpha = 1;
    for(int trial = 0; trial < 32; ++trial, alpha *= 0.5)
    {
        ++result.trials;
        const Vector12 displacement = alpha * result.direction.displacement;
        const Float    trial_energy = energy(displacement);
        if(stencil_finite(trial_energy)
           && trial_energy <= result.energy_before + 1e-4 * alpha * slope)
        {
            result.displacement = displacement;
            result.alpha        = alpha;
            result.energy_after = trial_energy;
            result.accepted     = true;
            break;
        }
    }
    return result;
}
}  // namespace uipc::backend::cuda::fem
