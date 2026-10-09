#pragma once
#include <finite_element/sos_barrier.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda::fem
{
// Keys identify full surface primitives, never their current closest feature.
// PT: (global vertex, surface triangle); EE: sorted surface edge indices;
// PH: (global vertex, plane). Two kind bits plus two nonnegative int31 indices.
UIPC_GENERIC inline U64 sos_pair_key(IndexT kind, IndexT first, IndexT second)
{
    assert(kind >= 0 && kind < 3 && first >= 0 && second >= 0);
    if(kind == 1 && first > second)
    {
        const auto temporary = first;
        first                = second;
        second               = temporary;
    }
    return (static_cast<U64>(kind) << 62) | (static_cast<U64>(first) << 31)
           | static_cast<U64>(second);
}

struct SoSBarrierRecord
{
    U64                  key      = ~U64{0};
    Vector4i             vertices = Vector4i::Constant(-1);
    SoSBarrierParameters parameters;
    Float                thickness     = 0;
    Float                mollifier_eps = 0;
    Vector3 plane_position = Vector3::Zero(), plane_normal = Vector3::Zero();
};

// Append-only within a physical time step. Stable record indices and immutable
// coefficients survive rehash/growth; full-key comparisons resolve collisions.
// Reset at each physical step, never at a local or outer iteration. Candidates
// are distinct input storage and may contain duplicates or inactive old pairs.
class SoSBarrierHistory
{
  public:
    void reset();
    void append(cuda_tool::CBufferView<SoSBarrierRecord> candidates);
    cuda_tool::CBufferView<SoSBarrierRecord> records() const
    {
        return m_records.cview();
    }

  private:
    cuda_tool::DeviceBuffer<SoSBarrierRecord> m_records;
    cuda_tool::DeviceBuffer<U64>              m_hash;
    cuda_tool::DeviceBuffer<IndexT>           m_count;
};
}  // namespace uipc::backend::cuda::fem
