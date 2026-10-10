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

## Foundation validation (preserved checkpoint)

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

## Custom objective and warm-start integration (active)

The new opt-in controls are `barrier_model=sos`, `warm_start=1` and
`merge_acceptance=local`; legacy IPC/monotone behavior stays the default.
Full PT/EE keys are discovered through swept candidates and narrow-phase
confirmation; plane candidates respect contact/subscene masks. Pair coefficients
use physical-step-origin clearance, survive inactive periods and reset next
frame. Thickness subtraction is an explicit extension of the paper's distance.
New pairs invalidate an older residual certificate. Both local searches and the
engine evaluate the same custom objective.

Warm start now has an acceleration predictor, contact-stencil Jacobi solves and
singleton inertia solves only for vertices outside contact supports. Elastic
energy/HVPs are excluded; geometric inversion and CCD bounds remain. The warm
phase must not certify final-frame convergence. Previous velocity/dt are saved
with checkpoints so acceleration prediction survives recovery.

A necessary regression exposed a stationary rejection loop at drape frame 27:
all local searches accepted, but vertex-wise Jacobi averaging gave positive
global slope. Rejecting that merge repeated the identical state indefinitely.
Removing elastic supports from warm start alone did not fix it. The paper
specifies local searches; it does not specify the extra global monotonicity guard
used by the previous implementation. The explicit `merge_acceptance=local`
control therefore retains local Armijo searches and combined geometry/finite
energy certificates, but does not discard a valid merge solely for increasing
global energy. This also applies to inner batches; it is not a different residual
tolerance or a hidden global solve. The old monotone guard remains a control.
This is a demonstrated warm-start stall mechanism, not yet proof that it explains
all historical full-elastic slow convergence.

Release build and 182 stencil assertions passed at the initial integration checkpoint.
Two diagnostic warm-start drape runs were stopped after reproducing the same
frame-27 rejection loop; they are NOT completed 120-frame validations.
Subsequent verification and remaining failures are recorded below.

### Warm-start diagnosis on 2026-10-09

The local-acceptance path exits the stationary rejection loop, but unrelaxed
Jacobi still oscillates at drape frame 27. It hit the diagnostic 2048-iteration
limit. Audit dumps at passes 127/255 contain a positive-definite warm Hessian
(min eigenvalue about 0.001185, max about 281); local CG saw no negative
curvature/fallback. Independent dense restricted solves also have an unstable
vertex-averaged Jacobi iteration: spectral radius approximately 1.17/1.15.
This rules out CPU execution or solely CG breakdown as the warm stall cause.
Artifacts: `output/sos-paper-20261008/warm-audit/27`, `warm-linear-audit.json`,
`warm-active-linear-audit.json`; audit reconstruction script under `../tmp/`.

Of 181 retained pairs, only 67 have nonzero curvature at the audited iterate.
Separate active solver supports from retained CCD/objective history. Inactive
pairs no longer add CG solves/Jacobi weights, but remain in energy/CCD adjacency
and can reactivate. This alignment alone still hit a 512-pass diagnostic limit.

`warm_start_relaxation` is an explicit numerical control, default 1. It caps
local warm proposals before Armijo (not after the search); 0.5 is a stabilization
experiment motivated by the measured Jacobi spectrum. The paper does not
specify this parameter. Any successful under-relaxed run must be labeled as
such; it neither relaxes full-objective residual stopping nor establishes
full-paper performance. This verification is active.

### Coefficient interpretation correction

The 0.5 warm-relaxation diagnostic reached the strict warm residual at frame 27
in 1524 passes, then the full elastic phase remained slow/oscillatory. It was
stopped; it is not a complete trajectory.

A separate integration error was then identified: native libuipc `KappaBarrier`
actually dispatches to the squared-distance **log-squared** potential, while
Eq.11 is a distance **log** potential. Copying the raw numeric kappa into Eq.11
is dimensionally inconsistent. This error belongs to the new custom-barrier
integration, not the preserved earlier IPC-based SoS/global timing comparisons.

Add `barrier_stiffness=ipc|distance`. The default `ipc` calibrates normal force
at clearance c=h/2 against the actual native dispatcher (thickness included),
then applies kappa_i=kappa_distance*h/d_i. This is an explicit finite-reference
calibration, not identity of the energies, and still requires independent
accuracy assessment before comparison. `distance` preserves direct Eq.11 units.
A proposed cubic-activation match was discarded before validation because the
native dispatcher is log-squared (quartic activation), not classic log IPC.
The numerical check uses the real native derivative dispatcher as its oracle.
With the corrected calibration, unrelaxed warm Jacobi reaches the strict
frame-27 residual in 34 passes, instead of the prior failure or 1524-pass
under-relaxed diagnostic. The full elastic phase still oscillates; this does
not establish full-trajectory convergence.


### Contact consistency follow-up (2026-10-09)

Custom EE energies now reuse native IPC's rest-edge mollifier threshold
(coefficient 1e-3). The energy is m*b and the derivatives include both product
cross terms and b*H_m. Local/global energy consumers use the same helper;
penetration remains infeasible even when the mollifier is zero. This is an
explicit libuipc parallel-edge extension; the paper does not specify its formula.
The updated Release build passes 187 assertions in 12 cases, including a
near-parallel energy/gradient/HVP finite-difference check.

The mollified 15-inner-sweep drape still oscillated at first cloth/block impact
and was stopped. A single-sweep run also reached the strict 2048-iteration limit
at frame 27. Thus batching is not the sole cause. These are failed/partial runs,
not new complete 120-frame results. Artifacts are under
`output/sos-paper-20261008/cloth-warm-{mollified,single}`.

The monotone control exposed a distinct objective lifecycle defect: swept-pair
discovery appended customized barriers after computing the Newton direction.
Unlike ordinary IPC activation, a new customized barrier can already exert force
at the old iterate. An old direction is therefore not a direction for the new
objective. The engine now retains that iterate and recomputes after expansion;
the initial acceleration predictor is exempt because it deliberately precedes
the warm objective. New history also invalidates convergence as before.
Verification of this correction is active.

The cloth CLI exposes independent/primary color coverage and persists arguments
before running plus per-frame engine certificates. Primary coverage matches the
stronger section-5.2 prose (every stencil is primal in some scheme); independent
coverage preserves the older Fig.-5-caption interpretation as a control.
The Windows runtime aborts with 0xc0000409 when these strict failure paths throw;
logs and process exit status must be retained, not mistaken for successful exit.


### Verified guarded trajectory

After objective-expansion restart, the primary-coverage, single-sweep,
monotone-merge control completed all 120 drape frames with unrelaxed warm start,
custom barriers and CSFD. Runtime SHA256:
`19b2122e6d5229916d91e49aa89aa947f12e7b86055a00a4735fc2920cb9fa9c`.
At the unchanged 0.01 m/s tolerance: max 70 outer iterations, total advance
65.2209274 s, mean 543.5077 ms/frame. Minimum area ratio 0.9688841,
minimum det(F) 0.9078034, minimum cloth-ground clearance 0.00101217 m,
fixed-vertex error zero. Independent saved-surface checks passed all 121 states
(cloth self, block self and cloth/block), and all 120 engine certificates passed.
Artifacts: `cloth-warm-restart-monotone/{summary.json,positions.npz,
frame-stats.jsonl,independent-audit.json}` under the paper output directory.
These timings include diagnostic dumps and are not a performance comparison.

The otherwise identical local-merge run still reached 2048 iterations at frame
27. This isolates a remaining merged-update problem after repairing objective
expansion; the guarded success is not a full-paper-method success. Evidence:
`cloth-warm-restart-local`, `full-restart-local-audit/27`, and the matching log
under `../tmp/`. Next investigate separate GS/Jacobi submission and actual
submitted-displacement line-search consistency before packing/benchmarking.


### Separate sweep searches (cloth verified; complex contact open)

`hybrid_submission=sequential` is an explicit alternative to the frozen control.
All raw Newton directions are still solved before coloring. The independent
color is filtered and committed first. Jacobi submission then excludes committed
vertices, recomputes slopes at the updated GS state and performs its own local
CCD/inversion/Armijo checks. A component removed by priority is never checked as
though it were still being submitted. Each combined substep and the final outer
path keep geometric checks. The starting state and its residual are restored
before engine acceptance; neither intermediate residual is reported as final.
This implements the separate-search reading of section 5.1. It is a documented
interpretation, not confirmation from an author reference implementation.

The unrelaxed, local-merge, primary-coverage, single-sweep drape completed 120
frames at the unchanged 0.01 m/s tolerance: max 65 iterations, mean 668.0183
ms/frame (80.1622 s advance). Minimum area ratio 0.9671030, det(F) 0.9078034,
cloth-ground clearance 0.00114721 m, fixed error zero. All 121 independent saved
surface checks and 120 convergence certificates passed. Runtime SHA256:
`fa15f7ace9e825a625ddd24c34deeac2fc6b76f2ba718e55824fbb7301bde277`.
Artifacts: `cloth-warm-sequential`, including animation and independent audit.
This removes the extra monotone guard from the successful cloth control; it
is not a dense bunny/cloth result or a matched performance comparison.

A follow-up restores derivative audit buffers to the original iterate after
sequential submission (the live searches use the committed GS state). Build17
passed and the 187 assertions remain green; runtime SHA256:
`f76b2bdcdfd930b92ec1e4769ea8a338ad194718f73d34e449dd44b447a8b3be`.
The three example CLIs expose the opt-in controls and record actual settings.
The dense bunny/cloth manifest now hashes the loaded runtime, not a hardcoded
checkout DLL. Legacy controls and checkpoint defaults remain explicit.

### Mesh-U follow-up (2026-10-10)

The 1869-vertex / 7356-tet bunny with the 425-vertex mesh U completed only
19 of the requested 30 frames. At frame 20, warm start hit 4096 iterations with
velocity residual 23.8601 m/s. Full elasticity needed 1490--3907 outer iterations
on contact frames 12--19. The geometric gap fell from 1.39e-3 m at frame 12 to
2.61e-8 m at frame 19. This result does not validate the complex demo; no speedup
claim is warranted. Runtime and all settings are preserved in
`bunny-mesh-u-sequential/scene.json`, history and log under `../tmp/`.
Investigate frame-origin activation distances shrinking on persistent contact
and calibrated coefficient behavior before treating this as only HVP overhead.
The paper's wording on per-step activation and its coefficient compensation must
be distinguished from any additional lower-bound stabilization experiment.
