# 25 - Complete paper algorithm implementation

The owner's 2026-10-08 request expands the scope to the complete algorithm in
Lan et al., DOI 10.1145/3592104, before another solver comparison. This is active
implementation work, not a completed reproduction. The prior native/stencil
baselines in documents 23-24 remain immutable controls. Implementation starts
from `b7290102`; unrelated worktree modifications must remain untouched.

Primary source: `https://wanghmin.github.io/publication/lan-2023-sos/Lan-2023-SOS.pdf`.
The local PDF/text are under `../tmp/pdfs/`. The authors' publication page does
not provide a directly linked reference implementation. Do not substitute a
later paper's differently scoped stencil implementation for this algorithm.

## Required implementation boundaries

1. Section 4 / Algorithm 1: exact restriction of every incident energy term,
   projection-free local Newton-CG, <=12 directions, negative-curvature handling.
   Replace stored local Hessians with a Hessian-vector interface and CSFD stress
   derivatives; keep the assembled path as a verification/control mode.
2. Section 5 / Algorithm 2: complementary coloring, priority submission of
   the current independent group, Jacobi averaging of remaining vertices.
   Resolve the text's primal-color requirement versus the weaker caption
   explicitly; do not silently change the meaning of an iteration/color sweep.
3. Section 6 / Algorithm 3: acceleration-based predictor; swept collision-pair
   prediction; per-pair activation distances and compensated stiffness; an
   inertia-plus-barrier Jacobi Newton-CG warm solve excluding elasticity.
   Inversion/CCD must still certify the warm motion. The elastic objective
   resumes after warm start; do not accept/reject a warm step against a different
   objective or use an undisclosed global Newton solve as a substitute.
4. Section 7.2: retain complete PT/EE primitive identities within each physical
   step, insert newly confirmed pairs, never retain duplicate reduced features.
   Keep inactive retained pairs available to local CCD. Reset history at the
   physical-step boundary. Pair-specific coefficients remain fixed within that
   step and are used consistently by gradients, HVPs and all energy searches.
5. Section 7.1: stencil/vertex adjacency-based directional products, persistent
   GPU buffers, explicit named kernels. Analytic gradient formulas must use
   non-conjugating complex arithmetic; real geometry determines piecewise
   closest-feature branches. Preserve the complete derivative chain of barriers.
6. Outer/global acceptance: discover new pairs over the actual proposed path,
   run full CCD at the required boundaries and certify final configurations.
   Remove duplicate work only when the replacement covers the same constraints.
7. Verification: derivative/operator consistency and targeted adversarial
   contact checks, followed by complete bunny and cloth trajectories. Then
   compare both solvers at a common independently evaluated accuracy target.
   Native displacement tolerance and mass-scaled gradient tolerance are not
   interchangeable. The paper's Fig. 9 L2 threshold does not specify Fig. 13.

The existing code already supplies elastic material handling, a projection-free
CG oracle, continuous inversion bounds, geometric CCD, coloring and diagnostic
artifacts. Reuse these utilities rather than replace the complete backend.
Original GPU/global solver behavior must remain available for comparison.

## Progress

- Source and current implementation reviewed; native baseline preserved.
- Foundation implemented: generic matrix-free local CG, a direct SNK1 CSFD
  stress/Hessian-vector operator, and the `hessian_product=assembled|csfd`
  control. Shell/contact products remain analytic. This direct implementation
  is not yet the packed directional-array execution of section 7.1.
- Distance-based customized barrier energy/derivatives and GPU full-key
  append-only history are implemented as building blocks, not yet wired to
  the physical-step lifecycle or warm-start objective.
- Release CUDA build passed. Stencil tests pass 169 assertions in 11 cases,
  including CSFD versus analytic element products and history growth/dedup.
- The CSFD cloth drape integration completed 120 frames (76 vertices, 72 cloth
  triangles, 48 tets, 15 inner sweeps, tolerance 0.01): minimum area ratio
  0.9668201, minimum det(F) 0.9127444, minimum ground gap 0.0043975 m,
  fixed error zero, at most 10 outer iterations, mean 564.97 ms/frame.
  Artifacts: `output/sos-paper-20261008/cloth-csfd`; log under `../tmp/`.
  This validates the current operator integration, not full paper completion
  or a speedup. Its tolerance differs from the older cloth regression.
- Next: connect full-primitive history and custom barriers to all objective
  consumers, implement inertia-barrier warm start, then packed products.
  No new full-paper trajectory or matched-accuracy speedup is validated.
