#include <finite_element/local_stencil.h>
#include <finite_element/local_stencil_snk.h>
#include <finite_element/local_stencil_csfd.h>
#include <finite_element/local_stencil_packed.h>
#include <finite_element/sos_barrier_history.h>
#include <finite_element/sos_contact.h>
#include <contact_system/contact_models/codim_ipc_contact_function.h>
#include <finite_element/local_stencil_inversion.h>
#include <finite_element/local_stencil_geometry.h>
#include <cuda_tool/cuda_tool.h>
#include <app/app.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>

namespace local_stencil_test
{
using namespace uipc;
using namespace uipc::backend::cuda;
namespace ct = uipc::backend::cuda_tool;

// A small, deliberately coupled non-contact FEM problem. Both tetrahedra
// contribute to either stencil, including the shared-face off-diagonal blocks.
struct Patch
{
    Vector3   x[5];
    Vector3   predicted[5];
    Float     mass[5];
    int       fixed[5] = {1, 0, 0, 0, 0};
    Vector4i  tets[2];
    Matrix3x3 inverse[2];
    Float     volume[2];
    Float     dt     = 0.08;
    Float     mu     = 80;
    Float     lambda = 120;
};

Patch make_patch()
{
    Patch   patch;
    Vector3 rest[5] = {Vector3{0, 0, 0},
                       Vector3{1, 0, 0},
                       Vector3{0, 1, 0},
                       Vector3{0, 0, 1},
                       Vector3{1, 1, 1}};
    patch.tets[0]   = Vector4i{0, 1, 2, 3};
    patch.tets[1]   = Vector4i{1, 4, 2, 3};
    for(int i = 0; i < 5; ++i)
    {
        patch.x[i]         = rest[i];
        patch.predicted[i] = rest[i] + Vector3{0.025 * i, -0.09, 0.02 * i};
        patch.mass[i]      = 0.25 + 0.05 * i;
    }
    patch.predicted[0] = rest[0];
    patch.x[2] += Vector3{0.13, -0.04, 0.07};
    patch.x[4] += Vector3{0.08, 0.11, -0.06};
    for(int t = 0; t < 2; ++t)
    {
        Matrix3x3 Dm;
        for(int j = 0; j < 3; ++j)
            Dm.col(j) = rest[patch.tets[t](j + 1)] - rest[patch.tets[t](0)];
        patch.inverse[t] = Dm.inverse();
        patch.volume[t]  = Dm.determinant() / 6;
    }
    return patch;
}

__host__ __device__ Matrix3x3 deformation(const Patch& patch, int t, const Vector3* x)
{
    Matrix3x3 Ds;
    for(int j = 0; j < 3; ++j)
        Ds.col(j) = x[patch.tets[t](j + 1)] - x[patch.tets[t](0)];
    return Ds * patch.inverse[t];
}

__host__ __device__ Float energy(const Patch& patch, const Vector3* x)
{
    Float value = 0;
    for(int i = 0; i < 5; ++i)
        value += 0.5 * patch.mass[i] * (x[i] - patch.predicted[i]).squaredNorm();
    for(int t = 0; t < 2; ++t)
    {
        const Matrix3x3 F = deformation(patch, t, x);
        // Endpoint guard only; this testbed makes no continuous inversion claim.
        if(F.determinant() <= 0)
            return std::numeric_limits<Float>::infinity();
        Float elastic;
        snk1::E(elastic, patch.mu, patch.lambda, F);
        value += patch.dt * patch.dt * patch.volume[t] * elastic;
    }
    return value;
}

struct PatchEnergy
{
    const Patch*              patch;
    int                       stencil;
    __host__ __device__ Float operator()(const Vector12& displacement) const
    {
        Vector3 trial[5];
        for(int i = 0; i < 5; ++i)
            trial[i] = patch->x[i];
        for(int i = 0; i < 4; ++i)
            trial[patch->tets[stencil](i)] += displacement.segment<3>(3 * i);
        return energy(*patch, trial);
    }
};

__host__ __device__ fem::LocalStencilSystem assemble(const Patch& patch,
                                                     int          stencil,
                                                     bool constrain = true)
{
    fem::LocalStencilSystem system;
    const Vector4i          vertices = patch.tets[stencil];
    for(int i = 0; i < 4; ++i)
    {
        const int v = vertices(i);
        system.add_inertia(i, patch.mass[v], patch.x[v] - patch.predicted[v]);
    }
    for(int t = 0; t < 2; ++t)
    {
        Vector12    g;
        Matrix12x12 H;
        fem::snk_stencil_element(deformation(patch, t, patch.x),
                                 patch.inverse[t],
                                 patch.mu,
                                 patch.lambda,
                                 patch.dt * patch.dt * patch.volume[t],
                                 g,
                                 H);
        system.add_element(vertices, patch.tets[t], g, H);
    }
    if(constrain)
        for(int i = 0; i < 4; ++i)
            if(patch.fixed[vertices(i)])
                system.fix_vertex(i);
    return system;
}

using Vector15 = Vector<Float, 15>;
using Matrix15 = Matrix<Float, 15, 15>;

void assemble_global(const Patch& patch, Vector15& g, Matrix15& H)
{
    g.setZero();
    H.setZero();
    for(int i = 0; i < 5; ++i)
    {
        g.segment<3>(3 * i) = patch.mass[i] * (patch.x[i] - patch.predicted[i]);
        H.block<3, 3>(3 * i, 3 * i).diagonal().setConstant(patch.mass[i]);
    }
    for(int t = 0; t < 2; ++t)
    {
        Vector12    local_g;
        Matrix12x12 local_H;
        fem::snk_stencil_element(deformation(patch, t, patch.x),
                                 patch.inverse[t],
                                 patch.mu,
                                 patch.lambda,
                                 patch.dt * patch.dt * patch.volume[t],
                                 local_g,
                                 local_H);
        for(int i = 0; i < 4; ++i)
        {
            const int a = patch.tets[t](i);
            g.segment<3>(3 * a) += local_g.segment<3>(3 * i);
            for(int j = 0; j < 4; ++j)
                H.block<3, 3>(3 * a, 3 * patch.tets[t](j)) +=
                    local_H.block<3, 3>(3 * i, 3 * j);
        }
    }
    for(int i = 0; i < 5; ++i)
        if(patch.fixed[i])
        {
            g.segment<3>(3 * i).setZero();
            H.middleRows<3>(3 * i).setZero();
            H.middleCols<3>(3 * i).setZero();
            H.block<3, 3>(3 * i, 3 * i).setIdentity();
        }
}

__global__ void stencil_step_kernel(ct::CBufferView<Patch> patches,
                                    ct::BufferView<fem::LocalStencilStep> results)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= patches.size() || i >= results.size())
        return;
    const Patch& patch = patches(i);
    results(i) = fem::local_stencil_step(assemble(patch, 0), PatchEnergy{&patch, 0});
}

TEST_CASE("Local stencil includes neighboring FEM derivatives", "[cuda][fem][stencil]")
{
    const Patch     patch = make_patch();
    constexpr Float eps   = 1e-6;
    for(int s = 0; s < 2; ++s)
    {
        const auto system = assemble(patch, s, false);
        CHECK(system.hessian.isApprox(system.hessian.transpose(), 1e-12));
        Vector12    finite_gradient;
        Matrix12x12 finite_hessian;
        for(int j = 0; j < 12; ++j)
        {
            Patch     plus = patch, minus = patch;
            const int vertex = patch.tets[s](j / 3);
            plus.x[vertex](j % 3) += eps;
            minus.x[vertex](j % 3) -= eps;
            finite_gradient(j) =
                (energy(plus, plus.x) - energy(minus, minus.x)) / (2 * eps);
            finite_hessian.col(j) =
                (assemble(plus, s, false).gradient - assemble(minus, s, false).gradient)
                / (2 * eps);
        }
        CHECK((finite_gradient - system.gradient).norm() < 1e-8);
        CHECK((finite_hessian - system.hessian).norm() < 1e-8);
    }
    // The neighbor's fourth vertex is outside stencil 0, but moving it changes
    // stencil 0's shared-vertex gradient. An element-only implementation fails.
    Patch changed = patch;
    changed.x[4](0) += 0.2;
    CHECK((assemble(changed, 0).gradient - assemble(patch, 0).gradient).norm() > 1e-3);

    Vector15 g;
    Matrix15 H;
    assemble_global(patch, g, H);
    const auto restricted = assemble(patch, 0);
    CHECK(restricted.gradient.isApprox(g.head<12>(), 1e-12));
    CHECK(restricted.hessian.isApprox(H.topLeftCorner<12, 12>(), 1e-12));
}

TEST_CASE("Local stencil restricts vertex edge face and disjoint contributions", "[cuda][fem][stencil]")
{
    const Vector4i stencil{10, 20, 30, 40};
    const Vector4i elements[] = {Vector4i{10, 1, 2, 3},
                                 Vector4i{1, 20, 10, 3},
                                 Vector4i{40, 20, 1, 30},
                                 Vector4i{1, 2, 3, 4}};
    const Vector12 g          = Vector12::LinSpaced(1, 12);
    Matrix12x12    H;
    for(int i = 0; i < 12; ++i)
        for(int j = 0; j < 12; ++j)
            H(i, j) = (i + 1) * (j + 1);
    for(const auto& element : elements)
    {
        Matrix12x12 selection = Matrix12x12::Zero();
        for(int i = 0; i < 4; ++i)
            for(int j = 0; j < 4; ++j)
                if(stencil(i) == element(j))
                    selection.block<3, 3>(3 * j, 3 * i).setIdentity();
        fem::LocalStencilSystem system;
        system.add_element(stencil, element, g, H);
        CHECK((system.gradient - selection.transpose() * g).norm() == 0);
        CHECK((system.hessian - selection.transpose() * H * selection).norm() == 0);
    }
}

TEST_CASE("Local stencil SNK stress derivative remains unprojected under compression",
          "[cuda][fem][stencil]")
{
    const Patch patch = make_patch();
    Matrix3x3   states[3];
    states[0].setIdentity();
    states[1] << 1.2, 0.8, -0.1, 0.1, 0.7, 0.3, 0.2, 0.0, 1.1;
    states[2] = Vector3{0.3, 0.4, 0.5}.asDiagonal();
    Matrix3x3 direction;
    direction << 0.3, -0.2, 0.1, 0.4, 0.6, -0.1, -0.2, 0.3, 0.8;
    constexpr Float eps = 1e-6;
    for(const auto& F : states)
    {
        Matrix3x3       plus, minus;
        const Matrix3x3 F_plus  = F + eps * direction;
        const Matrix3x3 F_minus = F - eps * direction;
        snk1::dEdF(plus, patch.mu, patch.lambda, F_plus);
        snk1::dEdF(minus, patch.mu, patch.lambda, F_minus);
        const Matrix3x3 exact =
            fem::snk_stress_direction(F, direction, patch.mu, patch.lambda);
        CHECK(((plus - minus) / (2 * eps) - exact).norm() < 1e-6);
        Float before, after;
        snk1::E(before, patch.mu, patch.lambda, F);
        snk1::E(after, patch.mu, patch.lambda, F_plus);
        CHECK(std::abs(fem::snk_stencil_energy_change(
                           F, eps * direction, patch.mu, patch.lambda)
                       - (after - before))
              < 1e-12);
    }
    // A tiny shear has energy mu*s^2/2, even when adding it to the nonzero
    // rest energy rounds back to the same floating-point number.
    Matrix3x3 tiny        = Matrix3x3::Zero();
    tiny(0, 1)            = 1e-10;
    const Float increment = fem::snk_stencil_energy_change(
        Matrix3x3::Identity(), tiny, patch.mu, patch.lambda);
    CHECK(increment > 0);
    CHECK(std::abs(increment / (0.5 * patch.mu * 1e-20) - 1) < 1e-12);
    Vector12    gradient;
    Matrix12x12 hessian;
    fem::snk_stencil_element(states[2],
                             patch.inverse[0],
                             patch.mu,
                             patch.lambda,
                             patch.dt * patch.dt * patch.volume[0],
                             gradient,
                             hessian);
    Eigen::SelfAdjointEigenSolver<Matrix12x12> eigen(hessian);
    REQUIRE(eigen.info() == Eigen::Success);
    CHECK(eigen.eigenvalues().minCoeff() < -1e-3);
}

TEST_CASE("Local Newton CG handles curvature constraints and invalid input", "[cuda][fem][stencil]")
{
    SECTION("positive definite matches dense solve")
    {
        const auto system = assemble(make_patch(), 0);
        const auto result = fem::local_stencil_direction(system);
        const Vector12 reference = system.hessian.ldlt().solve(-system.gradient);
        REQUIRE(result.valid);
        CHECK(result.iterations <= 12);
        CHECK(result.converged);
        CHECK((result.displacement - reference).norm() < 1e-9);
        CHECK(result.displacement.head<3>().norm() == 0);
    }
    SECTION("nonpositive curvature uses a descent fallback")
    {
        fem::LocalStencilSystem system;
        system.gradient.setOnes();
        system.hessian    = -Matrix12x12::Identity();
        const auto result = fem::local_stencil_direction(system);
        REQUIRE(result.valid);
        CHECK(result.nonpositive_curvature);
        CHECK(result.gradient_fallback);
        CHECK_FALSE(result.converged);
        CHECK(system.gradient.dot(result.displacement) < 0);
    }
    SECTION("later negative curvature retains the accumulated step")
    {
        fem::LocalStencilSystem system;
        system.hessian.setIdentity();
        system.hessian(0, 0) = -1;
        system.hessian(1, 1) = 2;
        system.gradient(0) = system.gradient(1) = 1;
        const auto result = fem::local_stencil_direction(system);
        REQUIRE(result.valid);
        CHECK(result.nonpositive_curvature);
        CHECK(result.iterations == 1);
        CHECK_FALSE(result.gradient_fallback);
        CHECK(system.gradient.dot(result.displacement) < 0);
    }
    SECTION("zero residual and all fixed vertices")
    {
        auto system = assemble(make_patch(), 0);
        for(int i = 0; i < 4; ++i)
            system.fix_vertex(i);
        const auto result = fem::local_stencil_direction(system);
        CHECK(result.valid);
        CHECK(result.converged);
        CHECK(result.iterations == 0);
        CHECK(result.displacement.norm() == 0);
    }
    SECTION("nonfinite input is rejected")
    {
        auto system          = assemble(make_patch(), 0);
        system.hessian(2, 4) = std::numeric_limits<Float>::quiet_NaN();
        CHECK_FALSE(fem::local_stencil_direction(system).valid);
    }
}

TEST_CASE("Local stencil sub-tolerance residual leaves the state unchanged", "[cuda][fem][stencil]")
{
    fem::LocalStencilSystem system;
    system.hessian.setIdentity();
    system.gradient.setConstant(1e-15);
    const auto result = fem::local_stencil_direction(system);
    CHECK(result.valid);
    CHECK(result.converged);
    CHECK(result.displacement.norm() == 0);
    CHECK_FALSE(result.gradient_fallback);
}

struct ImpossibleEnergy
{
    __host__ __device__ Float operator()(const Vector12& dx) const
    {
        return dx.squaredNorm() == 0 ? 0 : std::numeric_limits<Float>::infinity();
    }
};

TEST_CASE("Local stencil backtracking and GPU execution", "[cuda][fem][stencil]")
{
    const Patch patch    = make_patch();
    const auto  system   = assemble(patch, 0);
    const auto  rejected = fem::local_stencil_step(system, ImpossibleEnergy{});
    CHECK_FALSE(rejected.accepted);
    CHECK(rejected.trials == 32);
    CHECK(rejected.displacement.norm() == 0);

    // Underestimate curvature deliberately: the search must backtrack rather
    // than accepting a finite but energy-increasing full step.
    auto underestimated = system;
    underestimated.hessian *= 0.01;
    const auto damped = fem::local_stencil_step(underestimated, PatchEnergy{&patch, 0});
    REQUIRE(damped.accepted);
    CHECK(damped.alpha < 1);
    CHECK(damped.energy_after < damped.energy_before);

    std::vector<Patch> host_patches(17, patch);
    for(int i = 0; i < 17; ++i)
        host_patches[i].x[4](0) += 0.003 * i;
    ct::DeviceBuffer<Patch> patches;
    patches.copy_from(host_patches);
    ct::DeviceBuffer<fem::LocalStencilStep> results;
    results.resize_discard(host_patches.size());
    stencil_step_kernel<<<2, 16>>>(patches.cview(), results.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    std::vector<fem::LocalStencilStep> host_results;
    results.copy_to(host_results);
    for(int i = 0; i < 17; ++i)
    {
        CAPTURE(i);
        const auto reference =
            fem::local_stencil_step(assemble(host_patches[i], 0),
                                    PatchEnergy{&host_patches[i], 0});
        REQUIRE(host_results[i].accepted);
        CHECK(host_results[i].energy_after < host_results[i].energy_before);
        CHECK((host_results[i].displacement - reference.displacement).norm() < 1e-10);
        CHECK(host_results[i].displacement.head<3>().norm() == 0);
    }
}

TEST_CASE("Local stencil two tetrahedron convergence experiment", "[cuda][fem][stencil]")
{
    Patch         local = make_patch(), global = local;
    const Vector3 fixed          = local.x[0];
    const Float   initial_energy = energy(local, local.x);
    Vector15      g;
    Matrix15      H;
    int           sweeps = 0;
    // Sequential sweeps are a tiny correctness oracle, NOT a production GPU
    // schedule or the paper's hybrid/complementary-coloring algorithm.
    for(; sweeps < 200; ++sweeps)
    {
        assemble_global(local, g, H);
        if(g.norm() < 1e-8)
            break;
        const Float before = energy(local, local.x);
        for(int s = 0; s < 2; ++s)
        {
            const auto step =
                fem::local_stencil_step(assemble(local, s), PatchEnergy{&local, s});
            if(step.accepted)
                for(int i = 0; i < 4; ++i)
                    local.x[local.tets[s](i)] += step.displacement.segment<3>(3 * i);
        }
        REQUIRE(energy(local, local.x) <= before + 1e-14);
    }
    REQUIRE(sweeps < 200);
    const Float local_residual = g.norm();
    int         newton_steps   = 0;
    for(; newton_steps < 50; ++newton_steps)
    {
        assemble_global(global, g, H);
        if(g.norm() < 1e-10)
            break;
        const Vector15 dx = H.ldlt().solve(-g);
        REQUIRE(g.dot(dx) < 0);
        const Float before   = energy(global, global.x);
        bool        accepted = false;
        Float       alpha    = 1;
        for(int trial = 0; trial < 32; ++trial, alpha *= 0.5)
        {
            Patch candidate = global;
            for(int i = 0; i < 5; ++i)
                candidate.x[i] += alpha * dx.segment<3>(3 * i);
            if(energy(candidate, candidate.x) <= before + 1e-4 * alpha * g.dot(dx))
            {
                global   = candidate;
                accepted = true;
                break;
            }
        }
        REQUIRE(accepted);
    }
    REQUIRE(newton_steps < 50);
    Float max_error = 0;
    for(int i = 0; i < 5; ++i)
        max_error = std::max(max_error, (local.x[i] - global.x[i]).norm());
    CHECK(max_error < 1e-7);
    CHECK((local.x[0] - fixed).norm() == 0);
    CHECK(energy(local, local.x) < initial_energy);
    std::cout << "Stencil stage 1: sweeps=" << sweeps
              << ", dense Newton steps=" << newton_steps << ", residual=" << local_residual
              << ", max vertex error=" << max_error << ", energy=" << initial_energy
              << " -> " << energy(local, local.x) << std::endl;
}

__global__ void stencil_geometry_margin_kernel(ct::CBufferView<Matrix3x3> matrices,
                                               ct::BufferView<Float> margins)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < matrices.size())
        margins(i) = fem::stencil_determinant_margin(matrices(i));
}

TEST_CASE("SoS geometry certificate rejects recorded collapsed budget endpoints",
          "[cuda][fem][stencil][budget_geometry]")
{
    // Dense bunny budget-32 run, element 55700, frames 56 and 57. The saved
    // doubles are nearly coplanar; determinant signs depend on evaluation order.
    // Never require a particular rounded sign in this regression.
    const Vector3 rest[4] = {
        Vector3{0.04370379338511453, 0.11331910285641839, 0.07790146884294581},
        Vector3{0.051940859689853194, 0.11891081772588113, 0.08452404458698012},
        Vector3{0.06203060730326919, 0.118966780080935, 0.0785201724284856},
        Vector3{0.05149253731343283, 0.12577859899776986, 0.08192779661087403}};
    const Vector3 endpoints[2][4] = {
        {Vector3{0.04329690039549844, 0.02247075852493404, 0.043125525305798426},
         Vector3{0.04926252159101436, 0.025847258845919548, 0.0429589364786687},
         Vector3{0.05467457917213012, 0.025615982073660733, 0.04089815356912179},
         Vector3{0.045988011701871334, 0.026186996885935794, 0.044321610791604626}},
        {Vector3{0.04329690039549843, 0.022470758524934045, 0.04312552530579843},
         Vector3{0.04926252159101436, 0.025847258845919548, 0.0429589364786687},
         Vector3{0.05467457917213011, 0.025615982073660736, 0.0408981535691218},
         Vector3{0.045988011701871334, 0.026186996885935794, 0.044321610791604626}}};
    Matrix3x3 Dm;
    for(int j = 0; j < 3; ++j)
        Dm.col(j) = rest[j + 1] - rest[0];
    const Matrix3x3        inverse = Dm.inverse();
    std::vector<Matrix3x3> matrices;
    for(const auto& x : endpoints)
    {
        Matrix3x3 Ds;
        for(int j = 0; j < 3; ++j)
            Ds.col(j) = x[j + 1] - x[0];
        matrices.push_back(Ds * inverse);
    }
    matrices.push_back(Matrix3x3::Identity());
    matrices.push_back(1e-6 * Matrix3x3::Identity());
    matrices.push_back(-Matrix3x3::Identity());
    matrices.push_back(Matrix3x3::Zero());
    Matrix3x3 nonfinite = Matrix3x3::Identity();
    nonfinite(0, 0)     = std::numeric_limits<Float>::infinity();
    matrices.push_back(nonfinite);
    ct::DeviceBuffer<Matrix3x3> device_matrices;
    device_matrices.copy_from(matrices);
    ct::DeviceBuffer<Float> device_margins;
    device_margins.resize_discard(matrices.size());
    stencil_geometry_margin_kernel<<<1, 32>>>(device_matrices.cview(),
                                              device_margins.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    std::vector<Float> margins;
    device_margins.copy_to(margins);
    for(int i = 0; i < matrices.size(); ++i)
    {
        CAPTURE(i);
        const bool expected_valid = i == 2 || i == 3;
        CHECK((fem::stencil_determinant_margin(matrices[i]) > 0) == expected_valid);
        CHECK((margins[i] > 0) == expected_valid);
    }
}

TEST_CASE("Local inversion certification retains interior and tangential roots", "[cuda][fem][stencil]")
{
    const Matrix3x3 F  = Matrix3x3::Identity();
    Matrix3x3       dF = Matrix3x3::Zero();
    dF.diagonal() << .01, -.02, .03;
    CHECK(fem::stencil_inversion_cap(F, dF) == 1);
    // Both endpoints are positive; the path still touches det=0 at t=1/2.
    dF.diagonal() << -2, -2, 0;
    const Float tangent = fem::stencil_inversion_cap(F, dF);
    CHECK(tangent > .39999);
    CHECK(tangent < .4);
    // A negative interior interval lies between roots 1/4 and 1/2.
    dF.diagonal() << -4, -2, 0;
    const Float interior = fem::stencil_inversion_cap(F, dF);
    CHECK(interior > .19999);
    CHECK(interior <= .2);
    CHECK(fem::stencil_inversion_cap(Matrix3x3::Zero(), dF) == 0);
}

__global__ void csfd_element_check_kernel(ct::BufferView<Vector12> result)
{
    Matrix3x3 F, inverse;
    F << 1.1, .2, -.1, .05, .8, .12, .15, -.07, 1.2;
    inverse << 1.3, .2, 0, 0, 1.1, -.1, .1, 0, .9;
    Vector12 direction;
    for(int i = 0; i < 12; ++i)
        direction(i) = (i - 5.0) * .037;
    Vector12    g;
    Matrix12x12 H;
    fem::snk_stencil_element(F, inverse, 80, 120, .013, g, H);
    result(0) = H * direction;
    result(1) = fem::snk_stencil_product_csfd(F, inverse, 80, 120, .013, direction);
    // Translation is a null direction of an isolated elastic element.
    for(int a = 0; a < 4; ++a)
        direction.segment<3>(3 * a) = Vector3(.2, -.3, .4);
    result(2) = fem::snk_stencil_product_csfd(F, inverse, 80, 120, .013, direction);
}

TEST_CASE("CSFD stress product matches the independent analytic element Hessian", "[cuda][fem][stencil]")
{
    ct::DeviceBuffer<Vector12> result(3);
    csfd_element_check_kernel<<<1, 1>>>(result.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    Vector12 host[3];
    result.view().copy_to(host);
    CHECK((host[0] - host[1]).norm() < 1e-12 * host[0].norm());
    CHECK(host[2].norm() < 1e-13);
}

TEST_CASE("SoS customized barrier and conservative primitive history", "[cuda][fem][stencil]")
{
    for(Float thickness : {Float{0}, Float{.003}})
    {
        const Float h = .005, k = 1e8, c = .5 * h;
        Float       native_derivative;
        sym::codim_ipc_contact::dKappaBarrierdD(
            native_derivative, k, (c + thickness) * (c + thickness), h, thickness);
        native_derivative *= 2 * (c + thickness);
        const Float converted =
            fem::sos_barrier(c, {h, fem::sos_distance_stiffness(k, h, thickness)})
                .first;
        CHECK(std::abs(converted / native_derivative - 1) < 1e-12);
    }
    const auto parameters = fem::SoSBarrierParameters::from_reference(.01, .001, 10);
    CHECK(parameters.kappa == 1);
    CHECK(fem::sos_barrier(.01, parameters).energy == 0);
    CHECK(fem::sos_barrier(.02, parameters).first == 0);
    const Float d = .004, step = 1e-7;
    const auto  value = fem::sos_barrier(d, parameters);
    const auto  plus  = fem::sos_barrier(d + step, parameters);
    const auto  minus = fem::sos_barrier(d - step, parameters);
    CHECK(std::abs((plus.energy - minus.energy) / (2 * step) - value.first) < 1e-10);
    CHECK(std::abs((plus.first - minus.first) / (2 * step) - value.second) < 1e-7);

    std::vector<fem::SoSBarrierRecord> records(130);
    for(int i = 0; i < records.size(); ++i)
    {
        records[i].key        = fem::sos_pair_key(0, i % 65, 7);
        records[i].parameters = parameters;
    }
    ct::DeviceBuffer<fem::SoSBarrierRecord> input;
    input.copy_from(records);
    fem::SoSBarrierHistory history;
    history.append(input.cview());
    CHECK(history.records().size() == 65);
    records.resize(258);
    for(int i = 0; i < records.size(); ++i)
    {
        records[i].key        = fem::sos_pair_key(0, i, 7);
        records[i].parameters = {.02, 2};
    }
    input.copy_from(records);
    history.append(input.cview());
    REQUIRE(history.records().size() == 258);
    std::vector<fem::SoSBarrierRecord> saved(258);
    history.records().copy_to(saved.data());
    bool retained = true;
    for(const auto& record : saved)
    {
        const auto vertex = (record.key >> 31) & 0x7fffffff;
        retained &= record.parameters.activation_distance == (vertex < 65 ? .01 : .02);
    }
    CHECK(retained);
    CHECK(fem::sos_pair_key(1, 3, 7) == fem::sos_pair_key(1, 7, 3));
    CHECK(fem::sos_pair_key(0, 3, 7) != fem::sos_pair_key(1, 3, 7));
    history.reset();
    CHECK(history.records().size() == 0);
    records.assign(3, fem::SoSBarrierRecord{});
    input.copy_from(records);
    history.append(input.cview());
    CHECK(history.records().size() == 0);
}

__global__ void sos_derivative_consistency_kernel(ct::BufferView<Vector3> errors)
{
    const int kind = threadIdx.x;
    if(kind >= 5)
        return;
    StencilBarrierTerm term;
    term.customized   = true;
    term.type         = kind == 0 ? 0 : kind == 3 ? 4 : 1;
    term.d_hat        = .3;
    term.kappa_dt2    = 7;
    term.thickness    = .01;
    term.plane_normal = Vector3(0, 0, 1);
    Vector3 p[4] = {Vector3(.2, .3, .1), Vector3(0, 0, 0), Vector3(1, 0, 0), Vector3(0, 1, 0)};
    if(kind == 1 || kind == 2 || kind == 4)
    {
        p[0] = Vector3(-.5, 0, 0);
        p[1] = Vector3(.5, 0, 0);
        p[2] = kind == 1 ? Vector3(0, -.5, .15) : Vector3(.7, 0, .1);
        p[3] = kind == 1 ? Vector3(0, .5, .15) : Vector3(1.4, 0, .1);
    }
    if(kind == 4)
    {
        p[2]               = Vector3(.7, .002, .1);
        p[3]               = Vector3(1.4, .009, .1);
        term.mollifier_eps = 1e-3;
    }
    Vector12 direction;
    for(int i = 0; i < 12; ++i)
        direction(i) = .013 * (i - 5);
    Vector12    g, gp, gm;
    Matrix12x12 H, unused;
    fem::sos_contact_derivatives(term, p, g, H);
    constexpr Float step = 1e-6;
    Vector3         plus[4], minus[4];
    for(int a = 0; a < 4; ++a)
    {
        plus[a]  = p[a] + step * direction.segment<3>(3 * a);
        minus[a] = p[a] - step * direction.segment<3>(3 * a);
    }
    fem::sos_contact_derivatives(term, plus, gp, unused);
    fem::sos_contact_derivatives(term, minus, gm, unused);
    const Float ep =
        fem::sos_contact_energy(term, plus, fem::sos_distance_squared(term, plus));
    const Float em =
        fem::sos_contact_energy(term, minus, fem::sos_distance_squared(term, minus));
    errors(kind) = Vector3(std::abs((ep - em) / (2 * step) - g.dot(direction)),
                           ((gp - gm) / (2 * step) - H * direction).norm(),
                           (H - H.transpose()).norm());
}
TEST_CASE("SoS full-primitive barrier uses the complete distance derivative chain", "[cuda][fem][stencil]")
{
    ct::DeviceBuffer<Vector3> errors(5);
    sos_derivative_consistency_kernel<<<1, 5>>>(errors.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    Vector3 host[5];
    errors.view().copy_to(host);
    for(int i = 0; i < 5; ++i)
    {
        INFO("primitive branch " << i);
        CHECK(host[i](0) < 1e-8);
        CHECK(host[i](1) < 1e-7);
        CHECK(host[i](2) < 1e-10);
    }
}
TEST_CASE("Packed stencil CG preserves independent neighbor directions", "[cuda][fem][stencil]")
{
    const auto                           patch = make_patch();
    std::vector<Vector4i>                elements{patch.tets[0], patch.tets[1]};
    std::vector<Vector4i>                contacts{Vector4i(0, 2, 3, 4)};
    std::vector<fem::StencilElasticTerm> terms(2);
    std::vector<Matrix12x12>             eh(2), ch(1);
    std::vector<Vector12>                eg(2);
    Vector12                             cg;
    for(int a = 0; a < 12; ++a)
        cg(a) = .002 * (a - 4);
    ch[0] = .3 * Matrix12x12::Identity() + cg * cg.transpose();
    for(int e = 0; e < 2; ++e)
    {
        terms[e].type    = 10;
        terms[e].inverse = patch.inverse[e];
        terms[e].weight  = patch.volume[e];
        terms[e].mu      = patch.mu;
        terms[e].lambda  = patch.lambda;
        fem::snk_stencil_element(deformation(patch, e, patch.x),
                                 patch.inverse[e],
                                 patch.mu,
                                 patch.lambda,
                                 patch.dt * patch.dt * patch.volume[e],
                                 eg[e],
                                 eh[e]);
    }
    ct::DeviceBuffer<Vector4i>                de, dc, ds;
    ct::DeviceBuffer<fem::StencilElasticTerm> dt;
    ct::DeviceBuffer<Vector3>                 dx, dg, dgc;
    ct::DeviceBuffer<Float>                   dm, slopes;
    ct::DeviceBuffer<IndexT>                  df;
    ct::DeviceBuffer<Matrix12x12>             dh, dch;
    ct::DeviceBuffer<Vector12>                directions;
    ct::DeviceBuffer<Vector4>                 diagnostics;
    de.copy_from(elements);
    dc.copy_from(contacts);
    dt.copy_from(terms);
    // Packed SNK products must not read the unassembled tetrahedral blocks.
    dh.copy_from(std::vector<Matrix12x12>(
        2, Matrix12x12::Constant(std::numeric_limits<Float>::quiet_NaN())));
    dch.copy_from(ch);
    dx.copy_from(std::vector<Vector3>(patch.x, patch.x + 5));
    dm.copy_from(std::vector<Float>(patch.mass, patch.mass + 5));
    df.copy_from(std::vector<IndexT>(patch.fixed, patch.fixed + 5));
    fem::LocalStencilPackedSolver solver;
    for(bool warm : {false, true})
    {
        std::vector<Vector4i> supports = elements;
        if(warm)
        {
            supports.clear();
            for(int v = 0; v < 5; ++v)
                supports.emplace_back(v, -1, -1, -1);
        }
        supports.push_back(contacts[0]);
        supports.push_back(Vector4i::Constant(-1));
        ds.copy_from(supports);
        std::vector<Vector3> gradient(5), contact_gradient(5, Vector3::Zero());
        for(int v = 0; v < 5; ++v)
            gradient[v] = patch.mass[v] * (patch.x[v] - patch.predicted[v]);
        if(!warm)
            for(int e = 0; e < 2; ++e)
                for(int a = 0; a < 4; ++a)
                    gradient[elements[e](a)] += eg[e].segment<3>(3 * a);
        for(int a = 0; a < 4; ++a)
            contact_gradient[contacts[0](a)] += cg.segment<3>(3 * a);
        dg.copy_from(gradient);
        dgc.copy_from(contact_gradient);
        directions.resize_discard(supports.size());
        slopes.resize_discard(supports.size());
        diagnostics.resize_discard(supports.size());
        fem::PackedStencilContext context;
        context.elements          = de.cview();
        context.contacts          = dc.cview();
        context.stencils          = ds.cview();
        context.terms             = dt.cview();
        context.xs                = dx.cview();
        context.elastic_gradients = dg.cview();
        context.contact_gradients = dgc.cview();
        context.masses            = dm.cview();
        context.fixed             = df.cview();
        context.elastic_hessians  = dh.cview();
        context.contact_hessians  = dch.cview();
        context.dt                = warm ? 0 : patch.dt;
        solver.solve(context, directions.view(), slopes.view(), diagnostics.view());
        std::vector<Vector12> result(supports.size());
        directions.cview().copy_to(result.data());
        Float error = 0;
        for(SizeT k = 0; k < supports.size(); ++k)
        {
            const auto              ids = supports[k];
            fem::LocalStencilSystem system;
            for(int a = 0; a < 4; ++a)
                if(ids(a) >= 0)
                    system.add_inertia(a,
                                       patch.mass[ids(a)],
                                       patch.x[ids(a)] - patch.predicted[ids(a)]);
            if(!warm)
                for(int e = 0; e < 2; ++e)
                    system.add_element(ids, elements[e], eg[e], eh[e]);
            system.add_element(ids, contacts[0], cg, ch[0]);
            for(int a = 0; a < 4; ++a)
                if(ids(a) < 0 || patch.fixed[ids(a)])
                    system.fix_vertex(a);
            const auto expected = fem::local_stencil_direction(system);
            REQUIRE(result[k].allFinite());
            error = std::max(error, (result[k] - expected.displacement).norm());
        }
        INFO("warm=" << warm << ", packed direction error=" << error);
        CHECK(error < 1e-8);
    }
    const auto expanded = fem::SoSBarrierParameters::from_reference(.0001, .001, 10, true);
    CHECK(expanded.activation_distance == .001);
    CHECK(expanded.kappa == 10);
}
}  // namespace local_stencil_test
