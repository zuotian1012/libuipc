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
    cuda_tool::CBufferView<Vector4i>        frozen_stencils;
    cuda_tool::CBufferView<Matrix12x12>     frozen_hessians;
    bool                                    frozen = false;

    __device__ void add(LocalStencilSystem& system, const Vector4i& stencil) const
    {
        for(int a = 0; a < 4; ++a)
        {
            const int v = stencil(a);
            if(v < 0)
                continue;
            system.gradient.segment<3>(3 * a) += gradients(v);
            if(frozen)
            {
                for(int k = offsets(v); k < offsets(v + 1); ++k)
                {
                    const int  e   = entries(k);
                    const auto ids = frozen_stencils(e);
                    for(int i = 0; i < 4; ++i)
                        if(ids(i) == v)
                            for(int b = 0; b < 4; ++b)
                                if(stencil(b) >= 0)
                                    for(int j = 0; j < 4; ++j)
                                        if(ids(j) == stencil(b))
                                            system.hessian.block<3, 3>(3 * a, 3 * b) +=
                                                frozen_hessians(e).block<3, 3>(3 * i, 3 * j);
                }
                continue;
            }
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
