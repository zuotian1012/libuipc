#pragma once
#include <sim_system.h>
#include <cuda_tool/cuda_tool.h>

namespace uipc::backend::cuda
{
class FiniteElementMethod;

// Non-contact, synchronous additive stencil Newton-CG. One call produces a
// mesh-wide descent direction; the engine owns the outer nonlinear iteration
// and global energy line search. No global linear solve is performed.
class FEMLocalStencilSolver final : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    void solve();
    bool converged() const;
    bool valid_configuration();

  private:
    void do_build() override;
    void init();

    FiniteElementMethod*                 m_fem = nullptr;
    cuda_tool::DeviceBuffer<Float>       m_mu, m_lambda;
    cuda_tool::DeviceBuffer<IndexT>      m_vertex_offsets, m_vertex_tets;
    cuda_tool::DeviceBuffer<IndexT>      m_stencil_offsets, m_stencil_neighbors;
    cuda_tool::DeviceBuffer<Vector12>    m_element_gradients, m_directions;
    cuda_tool::DeviceBuffer<Matrix12x12> m_element_hessians;
    cuda_tool::DeviceBuffer<Float>       m_residuals, m_determinants, m_scalar;
    Float                                m_weight    = 1;
    Float                                m_residual  = 0;
    Float                                m_tolerance = 0;
};
}  // namespace uipc::backend::cuda
