#pragma once
#include <finite_element/local_stencil.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda::fem
{
// A row adjacency over the upper-triangular contact Hessian. Negative entries
// encode transposed off-diagonal blocks as -(triplet_index + 1).
struct LocalStencilContactView
{
    cuda_tool::CBufferView<Vector3>         gradients;
    cuda_tool::CBufferView<IndexT>          offsets, entries;
    cuda_tool::CTripletMatrixView<Float, 3> hessians;
    IndexT                                  vertex_offset = 0;

    __device__ void add(LocalStencilSystem& system, const Vector4i& stencil) const
    {
        for(int a = 0; a < 4; ++a)
        {
            const int v = stencil(a);
            if(v < 0)
                continue;
            system.gradient.segment<3>(3 * a) += gradients(v);
            for(int k = offsets(v); k < offsets(v + 1); ++k)
            {
                const int  encoded   = entries(k);
                const bool transpose = encoded < 0;
                const int  index     = transpose ? -encoded - 1 : encoded;
                auto [i, j, H]       = hessians(index);
                const int column     = (transpose ? i : j) - vertex_offset;
                for(int b = 0; b < 4; ++b)
                    if(stencil(b) >= 0 && stencil(b) == column)
                    {
                        if(transpose)
                            system.hessian.block<3, 3>(3 * a, 3 * b) += H.transpose();
                        else
                            system.hessian.block<3, 3>(3 * a, 3 * b) += H;
                    }
            }
        }
    }
};
}  // namespace uipc::backend::cuda::fem
