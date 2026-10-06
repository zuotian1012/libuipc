# 21 - Stencil stopping, coverage and outer acceptance

This stage follows the 2026-10-06 request to align three concrete execution
differences identified in [20](20-sos-convergence-audit.md). It does not add
matrix-free CSFD, custom barriers or the paper's inertia-barrier warm start.

## Explicit convergence experiment

The incremental potential is `J = 0.5*(x-x_tilde)^T*M*(x-x_tilde) + dt^2*V`,
including standard IPC contact in `V`. Fixed vertices are excluded from both
reported residuals:

- `stopping=velocity`: max vertex `norm(g_i)/(m_i*dt)`, in m/s, using the existing
  `newton/velocity_tol`. This remains the library and bunny-demo default.
- `stopping=gradient_l2`: `sqrt(sum_i dot(g_i,g_i))`, in kg*m, using the new
  `gradient_tol` (default 1e-4). It is absolute, without mass or initial-residual
  normalization. Both residuals are logged whichever one terminates the solve.

The paper explicitly specifies residual L2=1e-4 for its contact-free Fig. 9
benchmark, not a portable accuracy guarantee for every scene. Small-mass cloth
can satisfy an absolute gradient threshold while retaining significant velocity
residual. Do not count changed stopping accuracy as a scheduling speedup.

Only a residual at an outer iterate with freshly detected global contacts may
terminate a frame. The solver returns zero displacement for that certified
state. An inner residual can end a batch, but cannot certify the frame.

## Coverage interpretation

`color_coverage=independent` requires every elastic stencil to appear in a
non-grey independent color group in at least one scheme, following the Fig. 5
caption. `primary` preserves the old stronger color-zero requirement as an
explicit control. The latter is not silently described as necessary for all
stencils: independent groups remain valid regardless of their color index.

The conflict graph, least-available-color greedy rule, cumulative ranks and
coverage-stall handling are otherwise unchanged. Contact stencils remain grey.
On the unchanged dense bunny/cloth mesh, independent coverage with 15 colors
produces three schemes and 45 passes, versus the legacy 220 schemes and 1,760
passes with eight colors and primary coverage. This changes priority frequency;
it does not make every pass cheaper or imply a 39-fold timing improvement.

## Frozen-contact batches with global certification

`inner_sweeps` bounds the number of local color passes in one globally accepted
proposal. The library default one preserves per-pass global checks; the dense
demo requests 15. Additive sweeps require one.

1. Save the outer positions, contact stencils, barrier coefficients and adjacency.
2. Each inner pass recomputes elastic derivatives and the exact gradient/Hessian
   of the frozen contact terms, including current closest-feature flags. Inactive
   frozen terms retain their CCD constraints and have zero energy derivatives.
3. Solve and filter complete local directions; select current-color owners and
   average unowned vertices as before.
4. Certify the actual combined inner displacement: continuous elastic bounds,
   CCD on every frozen pair, and Armijo on the sum of all frozen objective terms.
   This total-objective check is distinct from per-stencil Armijo. Reject a
   failed combined step without moving positions, then advance the color.
5. Form the accumulated displacement, restore the outer positions, and hand the
   proposal to the unchanged engine's regular CCD/global energy acceptance.
   The accumulated straight path receives a separate inversion bound. If the
   net direction is non-descent at the outer start, use its saved first proposal.
   Exceptions restore the outer positions and clear the proposal.
6. The engine refreshes global contacts before the next batch. This catches
   newly encountered constraints that were absent from the frozen inner list.

Temporary inner iterates are proposals, not independently certified physical
states against unknown pairs. The outer check is never removed. The current
standard-IPC feature list is refreshed between batches: this is not the paper's
append-only full-primitive barrier history across outer iterations. Retaining
reduced PT/EE/PE/PP features indiscriminately would double-count energies after
closest-feature changes. That separate contact-representation change remains
outside this stage; do not claim a complete reproduction of section 7.2.

## Same-state experiments and recovery

The demo's `--fork-checkpoint <solver-directory> --checkpoint-frame N` copies
the saved physical state and full-precision prefix to a new output, verifies the
recovered positions exactly, and explicitly restarts the color schedule. Mesh,
materials, timestep and initial geometry must match. Solver controls and stopping
metrics may differ. The manifest records the source and previous parameters.
`--frames` is still an absolute target frame index, so `N+1` compares one step.

`newton/local_stencil/restart_schedule_on_recover=1` is enabled only for this
explicit fork. Ordinary resume requires the recorded parameters and restores the
stored color cursor. `new_advance_s` and `first_simulated_frame` distinguish the
new computation from inherited prefix timings. A forked trajectory must never be
presented as a complete trajectory generated by the new solver.

`history.csv` iterations now count global acceptance rounds. Color passes remain
visible in the batch log; they must be counted separately. Per-frame native
statistics, including the final global CCD/line-search values, are saved under
`frame-stats/`. The demo never overwrites an existing experiment implicitly.

## Validation and measured limits

CUDA release builds and the existing 150 assertions in eight stencil cases pass.
The 120-frame BW drape run with 15 inner passes and unchanged velocity tolerance
0.01 passes: minimum det(F) 0.912744, area ratio 0.965907, positive ground gap,
zero pin error. Independent surface checks over all 121 saved states find no
cloth/solid intersection and clearance beyond thickness 0.00426759 m.
Artifacts: `output/sos-alignment-20261006/drape-batched/`.

The dense same-state test forks frame 34 of the preserved baseline into frame 35.
Initial residuals agree exactly: velocity 6.903032383943498 m/s and gradient L2
0.00013362126952786078 kg*m. GPU: RTX 3060 Ti, CUDA 12.6, release, double precision.
The old long run was stopped at 37 converged frames, so these two new runs did
not compete with that simulation. The older GUI remains open. These are single
runs, not a repeated timing benchmark; the new schedules restart at pass zero.

| Run | Stop | Color passes | Global rounds | Advance time |
|---|---|---:|---:|---:|
| Previous frame 35 | velocity <= 0.01 | 14,227 | 14,227 | 593.909 s |
| Independent coverage, 15-pass batches | velocity <= 0.01 | 15,046 | 1,004 | 593.404 s |
| Same batches, absolute L2 experiment | gradient L2 <= 1e-4 | 361 | 25 | 14.365 s |

Strict batching ends at velocity residual 0.00975593 and gradient L2 5.98540e-8.
Its maximum position difference from the previous strict result is 2.89191e-7 m
(0.289 micrometers), and RMS difference is 8.31391e-8 m. It hits neither the
outer iteration limit nor the line-search limit. Global checks are much less
frequent, but extra local passes offset the saved work: **there is no meaningful
end-to-end speedup at unchanged accuracy in this example**. The earlier audit
overweighted scheduling as the explanation of the full runtime gap. These
structural corrections do not solve the remaining slow local relaxation.

The L2 experiment ends at gradient 9.85723e-5 but velocity residual 15.5432 m/s.
Its maximum cloth position difference from the previous strict result is
0.0185229 m (18.5 mm), with full-scene RMS 0.0027703 m. Do not present its
41-fold timing difference as an equal-accuracy speedup or make it the default.

Independent saved-frame checks find zero cloth/bunny triangle intersections for
both forks, with minimum surface separations 0.00134135 m (strict) and
0.00136875 m (L2), exceeding the 0.001 m cloth radius. This is a saved-frame
cross-body check, not a continuous/self-contact certificate.

Evidence: `output/sos-alignment-20261006/frame35-comparison.json`,
`frame35-batched-{strict,l2}/local_stencil/`,
`../tmp/sos-frame35-batched-{strict,l2}.log`, and
`../tmp/compare_sos_alignment.py`. The measured runtime binary SHA256 is
`f281522d8e8b1566c02ef3525da90856bd54ac2b635861659bcf3a4c6d7951be`.
The working tree includes unrelated preexisting diagnostics; only stencil files
and their direct configuration/example/documentation changes belong to this stage.

The full 120-frame dense SoS landing remains unverified. Do not restart the old
23-hour target and describe waiting as a convergence-rate fix. Remaining work
needs to address local relaxation efficiency and GPU Hessian-vector execution,
with an explicitly matched accuracy target. Paper-style L2 alone is insufficient
evidence for this mixed-mass cloth/solid scene.
