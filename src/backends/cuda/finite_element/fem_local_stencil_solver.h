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

// Frictionless stencil Newton-CG. One call produces a single pass or a batch of
// complementary-color proposals. Inner updates use a frozen contact objective;
// the outer start is restored before engine acceptance of the accumulated step.
// Local solves and CCD/Armijo filtering are color-blind;
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
    void                         solve_sweep(bool first);
    void                         refresh_frozen_contacts();
    bool                         accept_inner_step();
    void                         cap_combined_elastic_step();
    fem::LocalStencilContactView contact_view() const;
    void init_coloring(span<const Vector4i> elements, const vector<vector<IndexT>>& incident);
    void prepare_gs_coloring();
    void filter_local_steps();
    void prepare_barriers();
    bool do_dump(DumpInfo& info) override;
    bool do_try_recover(RecoverInfo& info) override;
    void do_apply_recover(RecoverInfo& info) override;

    FiniteElementMethod*                        m_fem                = nullptr;
    FEMDyTopoEffectReceiver*                    m_contact_receiver   = nullptr;
    FiniteElementVertexReporter*                m_vertex_reporter    = nullptr;
    GlobalTrajectoryFilter*                     m_trajectory_filter  = nullptr;
    GlobalContactManager*                       m_contact_manager    = nullptr;
    GlobalVertexManager*                        m_vertices           = nullptr;
    HalfPlane*                                  m_planes             = nullptr;
    HalfPlaneVertexReporter*                    m_plane_reporter     = nullptr;
    bool                                        m_hybrid             = true;
    bool                                        m_jacobi             = false;
    bool                                        m_gs                 = false;
    bool                                        m_local_filter       = true;
    bool                                        m_inertial_guess     = false;
    bool                                        m_gradient_stopping  = false;
    bool                                        m_primary_coverage   = false;
    bool                                        m_restart_schedule   = false;
    IndexT                                      m_inner_sweeps       = 1;
    IndexT                                      m_inner_index        = 0;
    Float                                       m_gradient_tolerance = 1e-4;
    Float                                       m_gradient_norm      = 0;
    IndexT                                      m_color_limit        = 8;
    SizeT                                       m_sweep_index        = 0;
    SizeT                                       m_recovered_sweep    = 0;
    vector<Vector2i>                            m_color_passes;
    cuda_tool::DeviceBuffer<IndexT>             m_colors, m_primary_owner;
    cuda_tool::DeviceBuffer<IndexT>             m_gs_colors;
    vector<IndexT>                              m_gs_elastic_colors;
    vector<vector<IndexT>>                      m_gs_vertex_colors;
    IndexT                                      m_gs_color_count = 0;
    cuda_tool::DeviceBuffer<Vector3>            m_total_gradients;
    cuda_tool::DeviceBuffer<Matrix3x3>          m_vertex_hessians;
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
    cuda_tool::DeviceBuffer<Vector12>    m_element_gradients, m_directions;
    cuda_tool::DeviceBuffer<Matrix12x12> m_element_hessians;
    cuda_tool::DeviceBuffer<Float>       m_residuals, m_determinants, m_scalar;
    cuda_tool::DeviceBuffer<Float>       m_gradient_norms;
    cuda_tool::DeviceBuffer<Vector3>     m_inertial_proposal;
    cuda_tool::DeviceBuffer<Vector3> m_outer_start, m_outer_gradients, m_first_proposal;
    cuda_tool::DeviceBuffer<Vector12>    m_frozen_contact_gradients;
    cuda_tool::DeviceBuffer<Matrix12x12> m_frozen_contact_hessians;
    cuda_tool::DeviceBuffer<Float>       m_inner_values;
    cuda_tool::DeviceBuffer<Vector4>     m_cg_diagnostics;
    Float                                m_weight    = 1;
    Float                                m_residual  = 0;
    Float                                m_tolerance = 0;
};
}  // namespace uipc::backend::cuda
