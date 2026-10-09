#pragma once
#include <finite_element/local_stencil_csfd.h>
#include <finite_element/local_stencil_contact.h>
#include <finite_element/local_stencil_elastic_data.h>

namespace uipc::backend::cuda::fem
{
// Direct restricted operator used to validate the matrix-free solve before
// packing simultaneous CG directions through the section-7.1 vertex layout.
// Analytic shell/contact operators remain explicit controls in this stage.
struct LocalStencilCSFDOperator
{
    Vector4i                                   stencil;
    cuda_tool::CBufferView<Vector4i>           elements;
    cuda_tool::CBufferView<StencilElasticTerm> terms;
    cuda_tool::CBufferView<Vector3>            xs;
    cuda_tool::CBufferView<Float>              masses;
    cuda_tool::CBufferView<IndexT>             fixed, offsets, incident;
    cuda_tool::CBufferView<Matrix12x12>        shell_hessians;
    LocalStencilContactView                    contact;
    Float                                      dt;

    __device__ Vector12 operator()(const Vector12& input) const
    {
        Vector12 direction = input;
        for(int a = 0; a < 4; ++a)
            if(stencil(a) < 0 || fixed(stencil(a)))
                direction.segment<3>(3 * a).setZero();
        Vector12 result = contact.product(stencil, direction);
        for(int a = 0; a < 4; ++a)
        {
            const int v = stencil(a);
            if(v < 0)
                continue;
            result.segment<3>(3 * a) += masses(v) * direction.segment<3>(3 * a);
            if(dt == 0)
                continue;  // Inertia-barrier warm start excludes every elastic term.
            for(int k = offsets(v); k < offsets(v + 1); ++k)
            {
                const int  e       = incident(k);
                const auto ids     = elements(e);
                bool       visited = false;
                for(int b = 0; b < a; ++b)
                    if(stencil(b) >= 0)
                        for(int j = 0; j < 4; ++j)
                            visited |= ids(j) == stencil(b);
                if(visited)
                    continue;
                Vector12 local = Vector12::Zero();
                for(int i = 0; i < 4; ++i)
                    if(ids(i) >= 0)
                        for(int b = 0; b < 4; ++b)
                            if(ids(i) == stencil(b))
                                local.segment<3>(3 * i) = direction.segment<3>(3 * b);
                Vector12   product;
                const auto term = terms(e);
                if(term.type == 10)
                    product = snk_stencil_product_csfd(
                        F(xs(ids(0)), xs(ids(1)), xs(ids(2)), xs(ids(3)), term.inverse),
                        term.inverse,
                        term.mu,
                        term.lambda,
                        term.weight * dt * dt,
                        local);
                else
                    product = shell_hessians(e) * local;
                for(int b = 0; b < 4; ++b)
                    if(stencil(b) >= 0)
                        for(int i = 0; i < 4; ++i)
                            if(ids(i) == stencil(b))
                                result.segment<3>(3 * b) += product.segment<3>(3 * i);
            }
        }
        for(int a = 0; a < 4; ++a)
            if(stencil(a) < 0 || fixed(stencil(a)))
                result.segment<3>(3 * a) = input.segment<3>(3 * a);
        return result;
    }
};
}  // namespace uipc::backend::cuda::fem
