#pragma once
#include <type_define.h>

namespace uipc::backend::cuda
{
struct StencilBarrierTerm
{
    bool    customized = false;
    IndexT  type       = 0;  // PT, EE, PE, PP, PH
    Float   kappa_dt2 = 0, d_hat = 0, thickness = 0;
    Float   mollifier_eps  = 0;
    Vector3 plane_position = Vector3::Zero(), plane_normal = Vector3::Zero();
};

}  // namespace uipc::backend::cuda
