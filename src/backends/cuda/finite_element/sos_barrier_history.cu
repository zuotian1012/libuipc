#include <finite_element/sos_barrier_history.h>
#include <kernel_cout.h>
#include <limits>
#include <stdexcept>

namespace uipc::backend::cuda::fem
{
namespace
{
    namespace ct = cuda_tool;
    __device__ U64 pair_hash(U64 key)
    {
        key ^= key >> 30;
        key *= 0xbf58476d1ce4e5b9ULL;
        key ^= key >> 27;
        key *= 0x94d049bb133111ebULL;
        return key ^ (key >> 31);
    }
    __device__ bool insert_key(ct::BufferView<U64> table, U64 key)
    {
        const U64 mask = table.size() - 1;
        U64       slot = pair_hash(key) & mask;
        for(SizeT probe = 0; probe < table.size(); ++probe)
        {
            auto* address = reinterpret_cast<unsigned long long*>(&table(slot));
            const U64 previous =
                atomicCAS(address, ~0ULL, static_cast<unsigned long long>(key));
            if(previous == key)
                return false;
            if(previous == ~U64{0})
                return true;
            slot = (slot + 1) & mask;
        }
        UIPC_KERNEL_ASSERT(false, "SoS history hash capacity exhausted");
        return false;
    }
    __global__ void sos_history_rehash_kernel(ct::CBufferView<SoSBarrierRecord> records,
                                              ct::BufferView<U64> table,
                                              IndexT              count)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i < count)
            insert_key(table, records(i).key);
    }
    __global__ void sos_history_insert_kernel(ct::CBufferView<SoSBarrierRecord> candidates,
                                              ct::BufferView<SoSBarrierRecord> records,
                                              ct::BufferView<U64>    table,
                                              ct::BufferView<IndexT> count)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= candidates.size())
            return;
        const auto candidate = candidates(i);
        UIPC_KERNEL_ASSERT((candidate.key >> 62) < 3, "Invalid SoS primitive key");
        if(insert_key(table, candidate.key))
        {
            const int index = atomicAdd(&count(0), 1);
            UIPC_KERNEL_ASSERT(index < records.size(), "SoS history storage overflow");
            records(index) = candidate;
        }
    }
}  // namespace

void SoSBarrierHistory::reset()
{
    m_records.resize_discard(0);
}

void SoSBarrierHistory::append(cuda_tool::CBufferView<SoSBarrierRecord> candidates)
{
    if(candidates.size() == 0)
        return;
    const SizeT old_count = m_records.size(), bound = old_count + candidates.size();
    if(bound > static_cast<SizeT>(std::numeric_limits<IndexT>::max()) / 2)
        throw std::runtime_error("SoS barrier history exceeds index capacity");
    SizeT capacity = 16;
    while(capacity < 2 * bound)
        capacity *= 2;
    m_hash.resize_discard(capacity);
    CUDA_TOOL_CHECK(cudaMemset(m_hash.data(), 0xff, capacity * sizeof(U64)));
    // Existing records survive growth. Every appended slot is assigned by a
    // winning insertion thread before the final logical size becomes visible.
    m_records.resize_preserve(bound);
    m_count.resize_discard(1);
    IndexT count = static_cast<IndexT>(old_count);
    m_count.view().copy_from(&count);
    if(old_count)
        sos_history_rehash_kernel<<<(old_count + 127) / 128, 128>>>(
            m_records.cview(), m_hash.view(), count);
    sos_history_insert_kernel<<<(candidates.size() + 127) / 128, 128>>>(
        candidates, m_records.view(), m_hash.view(), m_count.view());
    CUDA_TOOL_CHECK(cudaGetLastError());
    m_count.view().copy_to(&count);
    m_records.resize_preserve(count);
}
}  // namespace uipc::backend::cuda::fem
