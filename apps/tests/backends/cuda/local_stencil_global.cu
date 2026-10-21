#include <finite_element/local_stencil_global.h>
#include <app/app.h>
#include <Eigen/Cholesky>
#include <vector>

using namespace uipc;
namespace ct  = uipc::backend::cuda_tool;
namespace fem = uipc::backend::cuda::fem;

TEST_CASE("Global stencil correction counts unique primitives and fixes constrained DOFs",
          "[cuda][fem][stencil][global_correction]")
{
    // Two overlapping elastic supports, a three-vertex shell-like support,
    // and a one-vertex plane contact. Independent dense assembly is the oracle.
    std::vector<Vector4i> ids{{0, 1, 2, 3}, {1, 2, 3, 4}, {2, 4, 5, -1}};
    std::vector<Vector4i> contacts{{5, -1, -1, -1}};
    std::vector<Matrix12x12> hessians(3), contact_hessians(1, Matrix12x12::Zero());
    for(int e = 0; e < 3; ++e)
    {
        Matrix12x12 B;
        for(int i = 0; i < 12; ++i)
            for(int j = 0; j < 12; ++j)
                B(i, j) = std::sin((i + 1) * (j + 2) + e * .7);
        hessians[e] = B.transpose() * B;
    }
    contact_hessians[0](1, 1) = 13;
    std::vector<Float>    mass{1, 2, 3, 4, 5, 6};
    std::vector<IndexT>   fixed{1, 0, 0, 0, 0, 0};
    Matrix<Float, 18, 18> H = Matrix<Float, 18, 18>::Zero();
    for(int v = 0; v < 6; ++v)
        H.block<3, 3>(3 * v, 3 * v) = mass[v] * Matrix3x3::Identity();
    auto assemble = [&](const Vector4i& s, const Matrix12x12& h)
    {
        for(int a = 0; a < 4; ++a)
            for(int b = 0; b < 4; ++b)
                if(s(a) >= 0 && s(b) >= 0)
                    H.block<3, 3>(3 * s(a), 3 * s(b)) += h.block<3, 3>(3 * a, 3 * b);
    };
    for(int e = 0; e < 3; ++e)
        assemble(ids[e], hessians[e]);
    assemble(contacts[0], contact_hessians[0]);
    H.topRows<3>().setZero();
    H.leftCols<3>().setZero();
    std::vector<Vector3> input(6), gradient(6);
    for(int v = 0; v < 6; ++v)
        input[v] = Vector3{.1 + v, -.3 * v, .7};
    const Eigen::Map<const Vector<Float, 18>> flat(input[0].data());
    const Vector<Float, 18>                   expected = H * flat;
    for(int v = 0; v < 6; ++v)
        gradient[v] = -expected.segment<3>(3 * v);
    ct::DeviceBuffer<Vector4i>    de, dc;
    ct::DeviceBuffer<Matrix12x12> dh, dch;
    ct::DeviceBuffer<Float>       dm;
    ct::DeviceBuffer<IndexT>      df;
    ct::DeviceBuffer<Vector3>     di, dg, result(6);
    de.copy_from(ids);
    dc.copy_from(contacts);
    dh.copy_from(hessians);
    dch.copy_from(contact_hessians);
    dm.copy_from(mass);
    df.copy_from(fixed);
    di.copy_from(input);
    dg.copy_from(gradient);
    fem::GlobalStencilContext c{
        de.cview(), dc.cview(), dh.cview(), dch.cview(), dm.cview(), df.cview()};
    fem::GlobalStencilSolver solver;
    solver.product(c, di.cview(), result.view());
    std::vector<Vector3> host(6);
    result.view().copy_to(host.data());
    for(int v = 0; v < 6; ++v)
        REQUIRE((host[v] - expected.segment<3>(3 * v)).norm() < 1e-11);
    const auto solved = solver.solve(c, dg.cview(), result.view(), 100, 1e-11);
    result.view().copy_to(host.data());
    REQUIRE(solved.relative_residual < 1e-10);
    REQUIRE_FALSE(solved.negative_curvature);
    REQUIRE_FALSE(solved.fallback);
    REQUIRE(host[0].norm() == 0);
    for(int v = 1; v < 6; ++v)
        REQUIRE((host[v] - input[v]).norm() < 1e-9);
    const auto truncated = solver.solve(c, dg.cview(), result.view(), 1, 1e-12);
    REQUIRE(truncated.iterations == 1);
    REQUIRE(truncated.relative_residual > 1e-12);
    REQUIRE(truncated.slope < 0);
}

TEST_CASE("Global stencil negative curvature retains a descent fallback",
          "[cuda][fem][stencil][global_correction]")
{
    ct::DeviceBuffer<Vector4i>    de, dc;
    ct::DeviceBuffer<Matrix12x12> dh, dch;
    ct::DeviceBuffer<Float>       dm;
    ct::DeviceBuffer<IndexT>      df;
    ct::DeviceBuffer<Vector3>     dg, result(1);
    dc.copy_from(std::vector<Vector4i>{{0, -1, -1, -1}});
    Matrix12x12 H = Matrix12x12::Zero();
    H(0, 0)       = -2;
    dch.copy_from(std::vector<Matrix12x12>{H});
    dm.copy_from(std::vector<Float>{1});
    df.copy_from(std::vector<IndexT>{0});
    dg.copy_from(std::vector<Vector3>{Vector3{1, 0, 0}});
    fem::GlobalStencilContext c{
        de.cview(), dc.cview(), dh.cview(), dch.cview(), dm.cview(), df.cview()};
    fem::GlobalStencilSolver solver;
    const auto solved = solver.solve(c, dg.cview(), result.view(), 10, 1e-6);
    REQUIRE(solved.negative_curvature);
    REQUIRE(solved.fallback);
    REQUIRE(solved.slope < 0);
    REQUIRE(std::isfinite(solved.relative_residual));
}
