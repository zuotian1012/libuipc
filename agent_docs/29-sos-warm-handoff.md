# 29 - Full-objective warm handoff

The owner requested warm handoff improvements and small-case validation before
global error correction (2026-10-20). The new opt-in selector is
`newton/local_stencil/warm_start_handoff=legacy|full_objective`, default `legacy`.
Both the bunny/cloth demo and small diagnostic runner expose `--warm-handoff`.
This is an experimental initializer selection policy, not a claim of the
paper's exact algorithm or a new physical stopping criterion.

## State and objective contract

The auxiliary warm objective remains inertia plus contact. On completion,
`full_objective` always forms the candidate displacement from the saved physical
frame origin, even when the warm scratch path never needed inversion projection.
It restores FEM and collision geometry to that same origin and uses the existing
local projection (when selected), combined inversion bound and global CCD.

After contact history is expanded for the candidate path, both the origin and
trial energies use the same inertia + full elasticity + contact objective.
New history does not discard this initializer as if it were a Newton direction
from an obsolete objective. Geometry and full-energy backtracking still apply.
`merge_acceptance=local` cannot bypass full-energy decrease during handoff.

If no trial passes the initializer search, the engine restores zero displacement
from the physical origin, re-evaluates energy, and validates the origin. An
invalid origin still fails. Ordinary nonlinear line-search failures remain
strict. The handoff log records frame, outer round, accepted alpha and both
full energies. Alpha zero means the initializer was rejected. Candidate energy
decrease alone does not certify convergence or guarantee fewer later iterations.

The finishing flag remains active through engine acceptance, so an iteration
budget cannot publish a warm scratch/unfinished state. At the next solver call,
full derivatives and the selected residual are rebuilt. An expanded history
does not restart the guarded auxiliary solve, since its candidate was already
checked against that history. The unchanged stopping criterion certifies the
physical endpoint after full solving. Only then are velocities updated.

Legacy behavior remains selectable. Old demo manifests are interpreted as
`warm_handoff=legacy` when resuming; changing the handoff policy requires a new
run or an explicit checkpoint fork. No automatic native-solver fallback,
global correction or relaxed final residual is included in this change.

## Validation (2026-10-20)

The controlled impact uses a 0.2 m cube, density 1000 kg/m3, Young's modulus
1 MPa, Poisson ratio 0.3, dt 0.01 s, initial downward speed 0.8 m/s and
initial ground clearance 0.0007 m. Controls share packed Hessian products,
sequential submission, 15 inner sweeps, expanded custom barriers, IPC stiffness
calibration and the unchanged 0.01 m/s stopping threshold. All following impact
arms converged without budget truncation. Counts include initialization and
handoff outer rounds, not individual local CG iterations.

| Mesh | Warm disabled | Legacy warm | Full-objective warm | Handoff alpha |
|---|---:|---:|---:|---:|
| 27 vertices / 48 tetrahedra | 5 | 14 | 11 | 0.125 |
| 729 vertices / 3072 tetrahedra | 52 | 108 | 48 | 0.0625 |

The warm results agree between local and monotone merge policies in these
controls. Both `iterative` and `after` inversion ordering also converge in
11 rounds on the 27-vertex control. A contact-free falling cube with a distant
plane takes 12 cold rounds versus 4 guarded warm rounds; its handoff accepts
alpha 1. This checks that useful full warm motion is retained.

For the 729-vertex handoff, full energy decreases from
0.11753867881611728 to 0.11753484087791985. All 14 recorded guarded handoffs
have finite alpha in [0,1] and nonincreasing full energy. None exhausted
backtracking with the normal 32-trial limit. A separate 27-vertex control with
`--line-search-trials 1` deliberately exhausts initializer backtracking, accepts
alpha zero with unchanged origin energy, and then converges in 12 outer rounds.
This tests rejection without publishing the scratch candidate. A one-round
outer budget intentionally fails before warm completion and
publishes no physical frame. Its diagnostic worker exits with 3221226505;
the suite launcher itself returns zero, so inspect each `report.json` result.

### Continuous bunny/cloth control

Both runs start fresh with 1869 bunny vertices / 7356 tetrahedra and
81 cloth vertices / 128 triangles. They use drop height 0.004 m, cloth gap
0.02 m, dt 0.01 s, six frames, 64 outer rounds and `accept_last_valid`.
These are short near-impact controls, not a complete cloth landing.

| Quantity | Legacy | Full objective |
|---|---:|---:|
| Total advance time, seconds | 221.691 | 106.184 |
| Budget-truncated frames | 4 / 6 | 4 / 6 |
| Minimum det(F) across six endpoints | 0.91780 | 0.93930 |
| Frame 3 velocity residual, m/s | 0.11409 | 0.53841 |
| Frame 4 velocity residual, m/s | 0.21065 | 0.50641 |
| Frame 5 velocity residual, m/s | 0.30261 | 0.69847 |
| Frame 6 velocity residual, m/s | 0.42253 | 0.46748 |

All six endpoints have positive tetrahedral determinants, triangle areas and
ground gaps. Guarded alphas are 1, 1, 0.125, 0.25, 0.125, 0.125. However,
the guarded run has **larger residuals on every truncated frame** and the
trajectories diverge after impact. Lower runtime is not a matched-accuracy
speedup. This stage verifies the handoff contract and improves the converged
cube controls; it does not fix bunny/cloth convergence or establish long-run
stability. Keep the legacy default until stronger evidence supports changing it.

### Build and evidence

`xmake build -j 4 core cuda backend_cuda` succeeds. The existing CUDA
`[stencil]` suite passes 207 assertions in 13 cases; both modified Python
entry points pass `py_compile`. No full repository test-suite claim is made.

Local artifacts are under `output/sos-warm-handoff-20261020/`:
`manifest.json` records every command, return code and wall time;
each cube group has `report.json`; each bunny group has
`local_stencil/history.csv`; `verification.json` records energy and endpoint
checks. The isolated `runtime/` contains the rebuilt core/backend and compatible
Python bindings. Backend SHA-256:
`b6497866b21ef68ec801420065c30ab9fabc17ac2cab09d5ae72dc75a80db589`.
This build includes pre-existing uncommitted instrumentation; its source diff
is saved as `build-working.diff`. It is not a clean-checkout release benchmark.
The older comparison runtime was preserved.

From the repository root on the validation machine:

```powershell
$env:UIPC_LOCAL_PACKAGE = (Resolve-Path "output/sos-warm-handoff-20261020/runtime").Path
python python/examples/diagnose_local_stencil.py --resolution 8 --impact-speed .8 --rounds 128 --warm-handoff full_objective --cases ground-sos-monotone ground-warm-monotone ground-warm-local --output output/handoff-repeat
```

Use `--warm-handoff legacy` and a different output directory for the old rule.
To exercise rejected-initializer recovery, use resolution 2, the
`ground-warm-monotone` case and `--line-search-trials 1`; its result is recorded
under `rejected-initializer/report.json`.
Collaborators must build this revision and point `UIPC_LOCAL_PACKAGE` to their
own fresh package; the local artifact directory is not distributed by Git.
To reproduce the six-frame trajectory, change only the handoff selector and
output directory between the two arms:

```powershell
python python/examples/local_stencil_bunny_cloth.py --solver sos --cloth-resolution 8 --bunny-refinement 0 --drop-height .004 --cloth-gap .02 --frames 6 --hybrid-submission sequential --hessian-product packed --inner-sweeps 15 --max-iterations 64 --iteration-limit-policy accept_last_valid --barrier-model sos --barrier-activation expanded --barrier-stiffness ipc --merge-acceptance local --warm-start --warm-inversion after_local --warm-max-iterations 10 --warm-handoff full_objective --verbose --output output/bunny-handoff-repeat
```

## Next investigation: global error correction

Warm candidate selection addresses initializer quality, not the slow propagation
already observed without contact. Next freeze the same full-elastic state and
contact history, measure residual contraction per sweep, and compare a global
correction with local sweeps on that same objective. Compare accepted steps
after the same geometry checks and at the same stopping threshold. Separate
the cost of constructing a correction from its effect on residual reduction.
Do not infer a CG defect or paper convergence failure from the remaining
runtime gap alone. No global-correction algorithm is added in this stage.

The source inspection identifies the implementation boundary:

- `solve_sweep()` refreshes elastic derivatives, custom frozen-contact
  derivatives and the mass-scaled residual. A global correction must use these
  same terms, fixed-vertex mask, time scaling and current history.
- `LocalStencilCSFDOperator` in `local_stencil_operator.h` applies a restriction
  of the neighborhood Hessian to one support. Summing overlapping restricted
  matrices is not a unique global assembly; it counts shared primitive
  contributions repeatedly. Build the reference from each elastic/contact
  primitive once, plus the mass diagonal.
- The native `FEMLinearSubsystem` receives contact derivatives through the
  DyTopo reporter path, while the custom SoS history/barriers are built inside
  `FEMLocalStencilSolver`. Calling the native global solver directly therefore
  does not establish a correction for the same custom objective.
- Submit any eventual full-elastic correction through the existing inversion,
  CCD, contact-history refresh and objective acceptance pipeline. If history
  expands, rebuild its derivatives. Unlike an optional warm initializer, a
  correction is a direction computed for a particular current objective.

The earlier [frame-35 audit](22-sos-relaxation-diagnosis.md) already found
accurate local CG solves and a useful offline full-Hessian correction on its
older snapshot. It supports testing long-range error propagation first, but
does not validate a correction for the current custom-barrier trajectory.
Start with a small frozen-state reference before introducing another runtime
mode; retain the original SoS-only path as the comparison.
