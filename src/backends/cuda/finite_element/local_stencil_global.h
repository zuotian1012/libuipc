#pragma once
#include <type_define.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda::fem
{
// Unique primitive Hessians, already scaled to the physical time-step energy.
// These are NOT the overlapping restricted matrices of the local solves.
struct GlobalStencilContext
{
    cuda_tool::CBufferView<Vector4i>    elements, contacts;
    cuda_tool::CBufferView<Matrix12x12> elastic_hessians, contact_hessians;
    cuda_tool::CBufferView<Float>       masses;
    cuda_tool::CBufferView<IndexT>      fixed;
};

struct GlobalStencilResult
{
    IndexT iterations        = 0;
    Float  relative_residual = 0, slope = 0;
    bool   negative_curvature = false, fallback = false;
};

// Experimental global truncated PCG correction for the SAME stencil objective.
// All vector/matrix work stays on the GPU; CUB reductions synchronize scalars.
// A positive diagonal preconditioner does not project the physical Hessian.
class GlobalStencilSolver
{
  public:
    void                product(const GlobalStencilContext&     context,
                                cuda_tool::CBufferView<Vector3> input,
                                cuda_tool::BufferView<Vector3>  output);
    GlobalStencilResult solve(const GlobalStencilContext&     context,
                              cuda_tool::CBufferView<Vector3> gradient,
                              cuda_tool::BufferView<Vector3>  direction,
                              IndexT                          max_iterations,
                              Float                           tolerance);

  private:
    Float dot(cuda_tool::CBufferView<Vector3> a, cuda_tool::CBufferView<Vector3> b);
    cuda_tool::DeviceBuffer<Vector3> m_diagonal, m_r, m_z, m_p, m_ap;
    cuda_tool::DeviceBuffer<Float>   m_values, m_scalar;
};
}  // namespace uipc::backend::cuda::fem
