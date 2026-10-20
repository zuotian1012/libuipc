#pragma once
#include <sim_system.h>
#include <finite_element/local_stencil_elastic_data.h>
#include <cuda_tool/cuda_tool.h>
#include <finite_element/local_stencil_contact.h>
#include <finite_element/local_stencil_barrier.h>
#include <finite_element/local_stencil_packed.h>
#include <finite_element/sos_barrier_history.h>
#include <utils/dump_utils.h>

namespace uipc::backend::cuda
{
class FiniteElementMethod;
class FEMDyTopoEffectReceiver;
class FiniteElementVertexReporter;
class GlobalTrajectoryFilter;
class GlobalContactManager;
class GlobalVertexManager;
class GlobalSimplicialSurfaceManager;
class HalfPlane;
class HalfPlaneVertexReporter;

// Frictionless stencil Newton-CG. One call produces a single pass or a batch of
// complementary-color proposals. Inner updates use a frozen contact objective;
// the outer start is restored before engine acceptance of the accumulated step.
// Local solves are color-blind. Frozen submission composes independently
// filtered directions; sequential submission searches the unowned Jacobi part
// at the committed GS state. The engine certifies the combined geometry and
// applies the selected energy policy. No global linear solve is performed.
class FEMLocalStencilSolver final : public SimSystem
{
  public:
    using SimSystem::SimSystem;
    void  solve();
    bool  converged() const;
    bool  valid_configuration();
    bool  customized_barriers() const { return m_custom_barriers; }
    bool  accepts_local_merges() const { return m_local_merge_acceptance; }
    bool  prepare_global_objective(Float alpha);
    Float objective_energy();
    // Refresh endpoint diagnostics without solving or submitting another step.
    Json certify_iteration_limit();
    // Nonempty only when submitting the post-warm feasible path from frame origin.
    cuda_tool::CBufferView<Vector3> warm_rebase_positions() const
    {
        return m_warm_finishing ? m_warm_global_origin.cview() :
                                  cuda_tool::CBufferView<Vector3>{};
    }

  private:
    void                         do_build() override;
    void                         init();
    void                         init_elastic();
    void                         prepare_contacts();
    void                         prepare_custom_contacts(Float alpha);
    void                         build_custom_barriers();
    bool                         begin_warm_start();
    bool                         warm_budget_exhausted() const;
    bool                         residual_converged() const;
    void                         solve_sweep(bool first);
    void                         refresh_frozen_contacts();
    bool                         accept_inner_step(const char* stage = "batch");
    void                         cap_combined_elastic_step();
    void                         project_warm_guess_locally();
    fem::LocalStencilContactView contact_view() const;
    void init_coloring(span<const Vector4i> elements, const vector<vector<IndexT>>& incident);
    void prepare_gs_coloring();
    void filter_local_steps(bool state_only = false, bool jacobi_submission = false);
    void submit_sequential_hybrid();
    void prepare_barriers();
    bool do_dump(DumpInfo& info) override;
    bool do_try_recover(RecoverInfo& info) override;
    void do_apply_recover(RecoverInfo& info) override;

    FiniteElementMethod*              m_fem               = nullptr;
    FEMDyTopoEffectReceiver*          m_contact_receiver  = nullptr;
    FiniteElementVertexReporter*      m_vertex_reporter   = nullptr;
    GlobalTrajectoryFilter*           m_trajectory_filter = nullptr;
    GlobalContactManager*             m_contact_manager   = nullptr;
    GlobalVertexManager*              m_vertices          = nullptr;
    HalfPlane*                        m_planes            = nullptr;
    HalfPlaneVertexReporter*          m_plane_reporter    = nullptr;
    bool                              m_hybrid            = true;
    bool                              m_sequential_hybrid = false;
    cuda_tool::DeviceBuffer<Vector12> m_unfiltered_directions;
    cuda_tool::DeviceBuffer<Vector3>  m_hybrid_start, m_hybrid_gradient;
    bool                              m_jacobi                 = false;
    bool                              m_gs                     = false;
    bool                              m_local_filter           = true;
    bool                              m_inertial_guess         = false;
    bool                              m_gradient_stopping      = false;
    bool                              m_primary_coverage       = false;
    bool                              m_restart_schedule       = false;
    bool                              m_convert_ipc_stiffness  = true;
    bool                              m_expand_barriers_only   = false;
    GlobalSimplicialSurfaceManager*   m_surface                = nullptr;
    Float                             m_warm_relaxation        = 1;
    bool                              m_local_merge_acceptance = false;
    bool                              m_custom_barriers        = false;
    bool                              m_history_changed        = false;
    SizeT                             m_history_frame          = ~SizeT{0};
    bool m_warm_enabled = false, m_warm_active = false;
    bool m_warm_post_inversion = false, m_warm_finishing = false;
    bool m_warm_needs_rebase     = false;
    bool m_warm_local_projection = false;
    cuda_tool::DeviceBuffer<Vector3>  m_warm_global_origin;
    SizeT                             m_warm_frame  = ~SizeT{0};
    Float                             m_previous_dt = 0, m_recovered_dt = 0;
    IndexT                            m_warm_iterations     = 0;
    IndexT                            m_warm_max_iterations = 0;
    cuda_tool::DeviceBuffer<Vector3>  m_previous_velocity;
    BufferDump                        m_velocity_dump;
    cuda_tool::DeviceBuffer<Vector4i> m_update_stencils;
    fem::SoSBarrierHistory            m_history;
    cuda_tool::DeviceBuffer<fem::SoSBarrierRecord> m_history_candidates;
    cuda_tool::DeviceBuffer<Vector3>               m_step_origin;
    cuda_tool::DeviceBuffer<Float>                 m_objective_values;
    bool                                           m_packed_product = false;
    fem::LocalStencilPackedSolver                  m_packed_solver;
    cuda_tool::DeviceBuffer<Vector4i>              m_packed_supports;
    bool                                           m_csfd_product       = false;
    IndexT                                         m_inner_sweeps       = 1;
    IndexT                                         m_inner_index        = 0;
    Float                                          m_gradient_tolerance = 1e-4;
    Float                                          m_gradient_norm      = 0;
    IndexT                                         m_color_limit        = 8;
    SizeT                                          m_sweep_index        = 0;
    SizeT                                          m_recovered_sweep    = 0;
    vector<Vector2i>                               m_color_passes;
    cuda_tool::DeviceBuffer<IndexT>                m_colors, m_primary_owner;
    cuda_tool::DeviceBuffer<IndexT>                m_gs_colors;
    vector<IndexT>                                 m_gs_elastic_colors;
    vector<vector<IndexT>>                         m_gs_vertex_colors;
    IndexT                                         m_gs_color_count = 0;
    cuda_tool::DeviceBuffer<Vector3>               m_total_gradients;
    cuda_tool::DeviceBuffer<Matrix3x3>             m_vertex_hessians;
    cuda_tool::DeviceBuffer<Float>                 m_direction_slopes;
    cuda_tool::DeviceBuffer<Float>                 m_global_slopes;
    cuda_tool::DeviceBuffer<IndexT>                m_local_statistics;
    cuda_tool::DeviceBuffer<StencilBarrierTerm>    m_barrier_terms;
    cuda_tool::DeviceBuffer<Matrix3x3>             m_filter_deformation;
    cuda_tool::DeviceBuffer<Float> m_filter_elastic_energy, m_filter_barrier_energy;
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
