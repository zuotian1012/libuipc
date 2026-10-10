#pragma once
#include <finite_element/local_stencil_elastic_data.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda::fem
{
struct PackedStencilContext
{
    cuda_tool::CBufferView<Vector4i>           elements, contacts, stencils;
    cuda_tool::CBufferView<StencilElasticTerm> terms;
    cuda_tool::CBufferView<Vector3> xs, elastic_gradients, contact_gradients;
    cuda_tool::CBufferView<Float>   masses;
    cuda_tool::CBufferView<IndexT>  fixed, colors;
    cuda_tool::CBufferView<Matrix12x12> elastic_hessians, contact_hessians;
    Float                               dt    = 0;
    IndexT                              color = 0;
};

struct PackedStencilPair
{
    IndexT   target = -1, term = -1;
    Vector4i local = Vector4i::Constant(-1);
};

struct PackedStencilCG
{
    Vector12 x  = Vector12::Zero(), r, p, g;
    Float    r2 = 0, tolerance = 0;
    IndexT   iterations = 0, active = 0, negative = 0, fallback = 0;
};

// Section-7.1 directional layout: each vertex has one separate column per
// incident solve stencil. Slot/target identities survive redistribution; a
// column index at two different vertices never implies the same direction.
// Energy/stencil incidence pairs distribute HVP work over GPU threads instead
// of traversing all neighbors inside each CG thread. Atomic reductions retain
// normal parallel rounding behavior, without fixed-order accumulation.
class LocalStencilPackedSolver
{
  public:
    void solve(const PackedStencilContext&     context,
               cuda_tool::BufferView<Vector12> directions,
               cuda_tool::BufferView<Float>    slopes,
               cuda_tool::BufferView<Vector4>  diagnostics);

  private:
    void   build(const PackedStencilContext& context);
    void   product(const PackedStencilContext& context, bool final);
    IndexT m_stride = 0;
    cuda_tool::DeviceBuffer<IndexT> m_counts, m_refs, m_pair_counts, m_pair_offsets, m_scalar;
    cuda_tool::DeviceBuffer<Vector4i>          m_slots;
    cuda_tool::DeviceBuffer<Vector3>           m_p;
    cuda_tool::DeviceBuffer<Vector12>          m_products;
    cuda_tool::DeviceBuffer<PackedStencilPair> m_pairs;
    cuda_tool::DeviceBuffer<PackedStencilCG>   m_cg;
};
}  // namespace uipc::backend::cuda::fem
