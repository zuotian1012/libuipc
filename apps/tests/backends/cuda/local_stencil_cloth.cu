#include <app/app.h>
#include <finite_element/local_stencil_elastic.h>

namespace uipc::backend::cuda
{
TEST_CASE("local stencil cloth derivatives match the incident energy", "[local_stencil][cloth]")
{
    // One integration-level material contract: local E, g and raw H must
    // describe the same objective for all supported shell energy families.
    for(IndexT type : {819, 11, 17})
    {
        fem::StencilElasticTerm term;
        term.type        = type;
        term.mu          = 3;
        term.lambda      = 8;
        term.strain_rate = 10;
        term.weight      = .4;
        term.rest_area2  = 1;
        term.length      = 1;
        term.height      = .3;
        term.angle       = .2;
        Vector3 p[4] = {{.1, .2, .1}, {1.2, .1, .2}, {.15, 1.05, -.1}, {1.1, 1.2, .3}};
        const Float dt = .07, eps = 1e-6;
        Vector12    g;
        Matrix12x12 H;
        fem::stencil_shell_derivatives(term, p, dt, g, H);
        Vector12    fd_g = Vector12::Zero();
        Matrix12x12 fd_H = Matrix12x12::Zero();
        const int   n    = type == 17 ? 12 : 9;
        for(int j = 0; j < n; ++j)
        {
            const Float value = p[j / 3](j % 3);
            p[j / 3](j % 3)   = value + eps;
            const Float plus  = fem::stencil_shell_energy(term, p) * dt * dt;
            Vector12    gp, gm;
            Matrix12x12 unused;
            fem::stencil_shell_derivatives(term, p, dt, gp, unused);
            p[j / 3](j % 3)   = value - eps;
            const Float minus = fem::stencil_shell_energy(term, p) * dt * dt;
            fem::stencil_shell_derivatives(term, p, dt, gm, unused);
            p[j / 3](j % 3) = value;
            fd_g(j)         = (plus - minus) / (2 * eps);
            fd_H.col(j)     = (gp - gm) / (2 * eps);
        }
        CAPTURE(type);
        CHECK((g - fd_g).norm() < 1e-7 * (1 + g.norm()));
        CHECK((H - fd_H).norm() < 1e-6 * (1 + H.norm()));
        CHECK((H - H.transpose()).norm() < 1e-12 * (1 + H.norm()));
    }
}
}  // namespace uipc::backend::cuda
