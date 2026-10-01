#pragma once
#include <sim_system.h>
#include <finite_element/local_stencil_elastic_data.h>
#include <cuda_tool/cuda_tool.h>
#include <finite_element/local_stencil_contact.h>

namespace uipc::backend::cuda
{
class FiniteElementMethod;
class FEMDyTopoEffectReceiver;
class FiniteElementVertexReporter;
class GlobalTrajectoryFilter;
class GlobalContactManager;
class GlobalVertexManager;
class HalfPlane;
class HalfPlaneVertexReporter;

struct StencilBarrierTerm
{
    IndexT  type      = 0;  // PT, EE, PE, PP, PH
    Float   kappa_dt2 = 0, d_hat = 0, thickness = 0;
    Vector3 plane_position = Vector3::Zero(), plane_normal = Vector3::Zero();
};

// Frictionless stencil Newton-CG. One call produces one complementary-color
// hybrid proposal (or the additive baseline); the engine accepts and commits it
// before the next color. Local solves and CCD/Armijo filtering are color-blind;
// priority selection and Jacobi averaging only compose the filtered directions.
// Their local certificates do not certify the combined proposal. The engine's
// global CCD/energy acceptance remains required. No global linear solve is performed.
class FEMLocalStencilSolver final : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    void solve();
    bool converged() const;
    bool valid_configuration();

  private:
    void                         do_build() override;
    void                         init();
    void                         init_elastic();
    void                         prepare_contacts();
    fem::LocalStencilContactView contact_view() const;
    void init_coloring(span<const Vector4i> elements, const vector<vector<IndexT>>& incident);
    void filter_local_steps();
    void prepare_barriers();

    FiniteElementMethod*                        m_fem               = nullptr;
    FEMDyTopoEffectReceiver*                    m_contact_receiver  = nullptr;
    FiniteElementVertexReporter*                m_vertex_reporter   = nullptr;
    GlobalTrajectoryFilter*                     m_trajectory_filter = nullptr;
    GlobalContactManager*                       m_contact_manager   = nullptr;
    GlobalVertexManager*                        m_vertices          = nullptr;
    HalfPlane*                                  m_planes            = nullptr;
    HalfPlaneVertexReporter*                    m_plane_reporter    = nullptr;
    bool                                        m_hybrid            = true;
    IndexT                                      m_color_limit       = 8;
    SizeT                                       m_sweep_index       = 0;
    vector<Vector2i>                            m_color_passes;
    cuda_tool::DeviceBuffer<IndexT>             m_colors, m_primary_owner;
    cuda_tool::DeviceBuffer<Vector3>            m_total_gradients;
    cuda_tool::DeviceBuffer<Float>              m_direction_slopes;
    cuda_tool::DeviceBuffer<Float>              m_global_slopes;
    cuda_tool::DeviceBuffer<IndexT>             m_local_statistics;
    cuda_tool::DeviceBuffer<StencilBarrierTerm> m_barrier_terms;
    cuda_tool::DeviceBuffer<IndexT>   m_barrier_counts, m_barrier_offsets;
    cuda_tool::DeviceBuffer<IndexT>   m_barrier_cursors, m_barrier_entries;
    cuda_tool::DeviceBuffer<Vector4i> m_contact_stencils;
    cuda_tool::DeviceBuffer<Vector3>  m_contact_gradients;
    cuda_tool::DeviceBuffer<IndexT>   m_contact_counts, m_contact_offsets;
    cuda_tool::DeviceBuffer<IndexT>   m_contact_cursors, m_contact_entries;
    cuda_tool::DeviceBuffer<IndexT>   m_overlap, m_max_overlap;
    cuda_tool::DeviceBuffer<Vector4i> m_elastic_stencils;
    cuda_tool::DeviceBuffer<fem::StencilElasticTerm> m_elastic_terms;
    cuda_tool::DeviceBuffer<IndexT>      m_vertex_offsets, m_vertex_elements;
    cuda_tool::DeviceBuffer<IndexT>      m_stencil_offsets, m_stencil_neighbors;
    cuda_tool::DeviceBuffer<Vector12>    m_element_gradients, m_directions;
    cuda_tool::DeviceBuffer<Matrix12x12> m_element_hessians;
    cuda_tool::DeviceBuffer<Float>       m_residuals, m_determinants, m_scalar;
    Float                                m_weight    = 1;
    Float                                m_residual  = 0;
    Float                                m_tolerance = 0;
};
}  // namespace uipc::backend::cuda
