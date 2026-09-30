#pragma once
#include <type_define.h>
namespace uipc::backend::cuda::fem
{
// Mirrors the constitutions' rest metrics and material weights. Topology is
// stored separately; triangles use a -1 fourth index, never a duplicate DOF.
struct StencilElasticTerm
{
    IndexT    type    = 10;  // SNK1=10, NH shell=11, BW shell=819, bending=17.
    Matrix3x3 inverse = Matrix3x3::Identity();
    Matrix2x2 shell_inverse = Matrix2x2::Identity();
    Float     weight = 1, mu = 0, lambda = 0, strain_rate = 100;
    Float     length = 0, height = 0, angle = 0;
    Float     rest_area2 = 0;
};

}  // namespace uipc::backend::cuda::fem
