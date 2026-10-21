# 30 - Same-objective global correction experiment

After the warm-handoff investigation, the owner authorized the next step:
investigate global error correction while keeping controlled small tests.
This adds an **experimental hybrid solver**, not a missing component claimed
to be part of the original SoS paper. The unmodified SoS path remains selectable.

## Algorithm and objective boundary

`newton/local_stencil/global_correction_interval=0` disables correction.
A positive N replaces every Nth full-elastic outer round of each frame with a
global truncated-PCG direction; warm rounds do not count. N=1 is a same-objective
global Newton control. N=4 uses three ordinary local batches before correction.
Inner sweeps retain their existing meaning on local rounds.

The correction refreshes the same inertia, elastic and frozen contact derivatives
as the stencil solver. Custom barriers use the current retained SoS history,
including its reference clearances and coefficients. Each primitive contributes
once to the global operator. Do not add overlapping local restriction matrices
or substitute native IPC contact derivatives.

`local_stencil_global.cu` implements the product and PCG with GPU kernels and CUB
reductions; host code receives only reduction scalars. Persistent buffers are
reused. The mass diagonal plus unique elastic/contact Hessians forms the raw
operator. Fixed coordinates are excluded from both rows and columns. A positive
coordinate-diagonal preconditioner uses absolute diagonal entries with a
mass-relative floor; this does not modify the Hessian being solved.

The linear iteration cap defaults to 256; relative unpreconditioned residual
target defaults to 1e-4. Negative curvature truncates CG; an unusable non-descent
result falls back to a preconditioned negative gradient. The final true linear
residual is recomputed and logged, including for truncated/fallback directions.
An inexact direction is not a nonlinear convergence certificate.

All corrections require combined inversion checks, global CCD and full-energy
backtracking, even with `merge_acceptance=local`. New history invalidates a
correction computed for the old objective and forces a refreshed correction.
There is no unconditional step acceptance or native-solver switch.

The physical stopping test remains the selected existing SoS test. With velocity
stopping this is `max_i ||gradient_i|| / (mass_i * dt) <= velocity_tol`.
Warm handoff, strict failures and explicit budget acceptance retain their previous
contracts. Per-frame `linear_solver_iterations` now includes correction PCG
iterations; it does not count the numerous independent local stencil CG solves.

## Code and validation tools

- `local_stencil_global.h/.cu`: unique-primitive GPU operator and truncated PCG.
- `fem_local_stencil_solver.h/.cu`: full-derivative refresh, scheduling, physical
  residual, descent direction and optional frozen-state dumps.
- `engine/advance_ipc.cu`: existing acceptance pipeline, correction diagnostics
  and linear iteration accounting.
- `python/examples/diagnose_local_stencil.py`: identical-control small tests,
  correction log parsing and optional `--global-audit`.
- `python/examples/audit_stencil_global.py`: independent SciPy sparse assembly
  and direct solve from the same frozen primitive data. This checks assembly and
  linear solving, not constitutive derivative correctness or nonlinear convergence.
- `apps/tests/backends/cuda/local_stencil_global.cu`: independent dense assembly
  with overlapping supports, a triangle sentinel, plane contact, fixed vertices,
  CG truncation and negative curvature.

The backend source belongs to the FEM glob in both CMake and XMake. Both test
manifests explicitly compile the standalone solver, matching the existing packed
solver pattern. No new third-party runtime dependency is introduced; SciPy is
only needed for the optional offline audit.

## Reproduction

Build a fresh core, CUDA backend and Python package. Point `UIPC_LOCAL_PACKAGE`
to that package. From the repository root:

```powershell
python python/examples/diagnose_local_stencil.py --resolution 2 --impact-speed .8 --rounds 128 --warm-handoff full_objective --global-correction-interval 4 --global-audit --cases beam-packed ground-sos-monotone ground-warm-monotone --output output/correction-small
python python/examples/audit_stencil_global.py output/correction-small/ground-sos-monotone/global-audit --max-relative-residual .00011
```

Use intervals 0, 4 and 1 in separate output directories for local-only, hybrid
and global-direction controls. Neither the linear tolerance nor the interval
changes the final physical stopping threshold. For conditioning experiments,
the additional selectors are `--global-correction-max-iter` and
`--global-correction-tolerance`.

The bunny/cloth demo exposes the same selectors with `--solver sos`.
Old resume manifests default to disabled correction; changing the method requires
a new run or explicit physical-checkpoint fork. Do not compare elapsed time alone
if either arm was truncated, used a different objective or reached a different
stopping criterion.

## Measured results (2026-10-20/21)

All 24 small arms converge without truncation at the existing 0.01 m/s
physical residual threshold. The 729-vertex / 3072-tetrahedron results are:

| Case | Local only: rounds / seconds | Interval 4: rounds / seconds | Interval 1: rounds / seconds |
|---|---:|---:|---:|
| Fixed-face beam, no contact | 46 / 6.014 | 5 / 0.441 | 3 / 0.027 |
| Cube impact, no warm start | 52 / 7.341 | 10 / 1.075 | 7 / 0.090 |
| Cube impact, guarded warm start | 48 / 5.525 | 17 / 0.959 | 14 / 0.122 |

Local and monotone merging agree in the tested warm impact arms. On the
27-vertex cases, interval 4 changes little: the beam finishes before the first
scheduled correction, and the guarded warm impact already needs only three
full-elastic rounds. Global directions are not universally fewer outer rounds:
the 27-vertex cold impact takes 8 interval-1 rounds versus 5 local-only rounds.
Each local round can contain 15 inner sweeps, so outer counts are not equal-cost
work units. Timings are single serial diagnostic runs, not repeated benchmarks.

Independent SciPy assembly/direct solving checks 19 frozen 27-vertex snapshots.
Maximum CPU relative linear residual is 9.93e-15; maximum GPU true residual is
9.97e-5, below the selected 1e-4 target. Maximum direction difference from the
direct solution is 0.5265%, illustrating that linear residual and direction error
are not interchangeable. No physical tolerance was changed. The CUDA stencil
suite passes 229 assertions in 15 cases, including the new coupled-operator and
negative-curvature controls.

### Bunny/cloth continuous run

The small scene uses bunny 1869 vertices / 7356 tetrahedra and cloth 81 vertices
/ 128 triangles, dt 0.01 s, drop height 0.004 m, cloth gap 0.02 m, guarded warm
handoff, sequential submission, 15 inner sweeps and correction interval 4.
The outer cap remains 64 and `iteration_limit_policy=error`.

- Six fresh frames all converge, with at most 20 rounds and 12.223 s total
  advance time. The previous same-scene pure-local warm-handoff run took
  106.184 s and truncated four frames; this is not an accuracy-matched speedup.
- A separate fresh **120-frame** run completes all frames strictly, with zero
  truncations. Total advance time is **249.711 s**, process time **252.672 s**,
  mean advance 2.081 s/frame and maximum 33 outer rounds.
- Minimum det(F) is 0.907988, minimum cloth area ratio 0.949130, minimum bunny
  ground gap 0.000120195 m and minimum cloth clearance above thickness
  0.000492389 m.
- Independent checks pass all **121 saved states**: determinants, areas, ground
  clearance, cross-body surfaces and nonincident self surfaces. No intersections
  are detected. Minimum tested cloth/bunny candidate distance is 0.00145377 m,
  above the 0.001 m clearance; cloth self distance is 0.00267937 m, above 0.002 m.
  These saved-state checks do not independently certify continuous CCD.
- All 209 logged correction acceptances have finite, nonincreasing full energy.
  There are 200 nonzero PCG calls, 30 negative-curvature truncations, three
  256-iteration cap hits and no gradient fallbacks. Some truncated linear
  residuals are large (maximum 968.83); they are not claimed converged linear
  solves. Geometry and energy acceptance still apply, and each physical frame
  must satisfy the unchanged nonlinear stopping test.

Full-trajectory stopping is certified by the backend's per-frame statistics;
the complete custom-history residual was not independently recomputed on CPU.
The older `validate_stencil_bunny_cloth.py` log parser assumes every stopping
certificate comes from a local batch, so it is not used as a residual validator
for this new mixed path. Its independent surface-check routines are reused by
the recorded validation script below. A larger dense 120-frame scene has not
been validated by this experiment.

### Artifact provenance and exact continuous command

Artifacts are local under `output/sos-global-correction-20261020/` (the directory
was created before the continuation on October 21). `manifest.json` records
commands, return codes, timings and runtime identity. `small-summary.json`,
`cpu-audit-summary.json`, per-arm reports and `stencil-tests-retry.log` record
the controls. `bunny-hybrid-strict120/local_stencil/` holds the trajectory,
frame statistics, checkpoints and summary; its parent contains
`independent-validation.json`. The validator is `verify_integration.py` and
its output is `integration-validation.log` in the artifact root.

Hardware: RTX 3060 Ti, 8 GiB, driver 610.88. Isolated backend SHA-256:
`ca8c6812b3edce5e5265c19dc7bc50d803eb95e903fc74cc436f4185775e8206`.
The build includes pre-existing uncommitted instrumentation; `build-working.diff`
records the tracked build changes. It is not a clean-checkout release benchmark.
XMake compilation succeeds; CMake source/test ownership is kept in sync but
CMake compilation was not run in this stage.

Use the newly built package, or on this machine select the isolated runtime:

```powershell
$env:UIPC_LOCAL_PACKAGE = (Resolve-Path "output/sos-global-correction-20261020/runtime").Path
python python/examples/local_stencil_bunny_cloth.py --solver sos --cloth-resolution 8 --bunny-refinement 0 --drop-height .004 --cloth-gap .02 --frames 120 --hybrid-submission sequential --hessian-product packed --inner-sweeps 15 --max-iterations 64 --iteration-limit-policy error --barrier-model sos --barrier-activation expanded --barrier-stiffness ipc --merge-acceptance local --warm-start --warm-inversion after_local --warm-max-iterations 10 --warm-handoff full_objective --global-correction-interval 4 --verbose --output output/bunny-global-correction
```

The evidence supports slow long-range error propagation as an important current
bottleneck: corrections help even without contact, with unchanged accuracy.
It does not prove the original paper's method fails or reproduce its speedup.
Next compare larger meshes and interval/frequency choices against the
same-objective global-direction control, retaining the pure SoS path separately.
