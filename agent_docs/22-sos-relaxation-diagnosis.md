# 22 - Slow local relaxation and full dense validation

The owner requested a diagnosis of the thousands of local passes and an actual
120-frame dense bunny/cloth validation. This stage keeps the existing physics,
mesh and mass-scaled gradient threshold 0.01. No global solve, coarse correction,
relaxed threshold or changed material is introduced into the simulation.

## Diagnosis

The existing strict frame-35 trace contains 15,046 local color passes. Its last
3,000 passes have zero CCD caps, zero inversion caps and zero rejected merged
steps. Only 3,030 individual local solves are rejected out of about 212 million
accepted solves in that tail. The gradient's fitted geometric contraction over
the last 4,000 passes is 0.99954193 per pass: about 0.046% decrease per pass.
This is slow relaxation, not global CCD locking or widespread failed line search.

A diagnostic fork starts at the same full-precision frame-34 checkpoint and
deliberately stops at 129 outer rounds. This bounded diagnostic is NOT a failed
120-frame validation or a completed new trajectory. It exports outer-start
snapshots at rounds 0 and 127 (1,905 preceding local passes).

At round 127:

- All 70,576 local solves have zero negative-curvature exits and zero gradient
  fallbacks. Their median relative linear residual is 7.36e-7; the 99th percentile
  is 8.89e-6. A few ill-conditioned solves are less accurate (maximum 0.0316).
- Independent assembly of 554 local restrictions (250 spread across elasticity
  plus all 304 contacts) finds positive-definite matrices in every sample. The
  median filtered direction's non-collinearity with the exact solve is 3.75e-9;
  the maximum is 0.0475. Median line-search scale is one. These data do not support
  blaming the overall tail on incorrect local linear solves.
- Bunny position error against the previous strict converged frame is at most
  0.259 mm. A mass-weighted affine fit explains 94.6% of its squared error. Cloth
  error is at most 1.451 mm, with an affine fit explaining 46.2%.
- An OFFLINE full raw-Hessian solve has relative linear residual 7.05e-13. Its
  correction differs from the remaining converged bunny correction by 0.322%
  in norm, and the cloth correction by 4.85%. This diagnostic is never submitted
  to the simulation and does not change its state.
- In contrast, one local color proposal has norm only 0.109% of the remaining
  bunny correction, and 0.616% of the remaining cloth correction.

The supported explanation is a slowly damped spatially smooth error in this
one-level local relaxation. Local Newton solves have second-order information,
but combining them does not reproduce a global Newton correction or imply
quadratic global convergence. At this snapshot the objective is already well
described by its global quadratic approximation; strong nonlinearity is not the
main explanation for the remaining thousands of local passes. This is evidence
for this scene and state, not proof of the only bottleneck in every frame.

The global solver's `MaxTranslationChecker` uses the maximum coordinate of the
Newton displacement, with threshold `velocity_tol * dt`. SoS uses
`max ||gradient_i|| / (mass_i * dt)`. Equal numeric tolerances have never certified
matched accuracy; the demo comparison already labels that distinction. Keep
the strict SoS criterion unchanged in this verification, and do not rename a
looser criterion as an algorithmic convergence fix.

## Diagnostic support

`UIPC_STENCIL_AUDIT` now samples only the first inner pass of a selected outer
round, rather than writing every inner Hessian. At round zero, each 128th outer
round and convergence, it saves contact topology/Hessians, elastic Hessians,
owners, and the existing state arrays. The `-cg.bin` array has four float64 values
per stencil: iteration count, negative-curvature flag, gradient-fallback flag and
relative linear residual. CG diagnostics are collected only at periodic audit
rounds; the convergence-only snapshot may have an empty CG array. This option is
off for the complete run. Eigen Hessian matrices are column-major; FEM ordering
in this scene is cloth followed by bunny, whereas saved demo positions are bunny
followed by cloth.

Evidence under `output/sos-relaxation-20261006/`:

- `log-analysis.json`, `spatial-audit.json`, `global-audit.json`.
- `frame35-convergence.png`.
- `diagnostic-frame35/` is the bounded checkpoint fork.
- Raw snapshots: `../tmp/sos-relaxation-audit/35/`.
- Reproduction scripts: `../tmp/analyze_sos_relaxation.py` and
  `../tmp/audit_sos_global_correction.py`; logs are in `../tmp/`.

## Complete dense run: status must be read from artifacts

The fresh run is `output/sos-relaxation-20261006/full-strict/`. It starts from
frame zero, uses no inherited prefix, and requests 120 steps at dt=0.01, unchanged
14,570 vertices / 43,710 DOFs, independent coloring and 15 inner passes. Inertial
initialization is the existing option, not the paper's inertia-barrier warm start.
The cap is 4,096 outer rounds (at most 61,440 local passes); this is a safety budget,
not a demonstrated convergence-rate improvement.

```powershell
python python/examples/local_stencil_bunny_cloth.py --solver sos --initial-guess inertial --stopping velocity --inner-sweeps 15 --color-limit 15 --color-coverage independent --frames 120 --max-iterations 4096 --verbose --output output/sos-relaxation-20261006/full-strict
python python/examples/validate_stencil_bunny_cloth.py output/sos-relaxation-20261006/full-strict --render
```

Redirect the simulation's standard output to `simulation.log` in that parent
directory; validation requires its per-frame residual certificates. On Windows,
the validator also accepts `--wait-pid <simulation PID>` for supervised completion.
`run-process.json` records the actual launch. The measured runtime DLL SHA256 is
`038ff89d8a13221f0f3acd2f629f1313e5474ed351f3e7f7364c628c7e63e85b`.

The validator writes `completion-status.json` while waiting. It requires exactly
120 consecutive converged frames, fresh-scene provenance, all per-frame strict
residual certificates and no iteration/line-search limit. It independently
recomputes saved det(F), cloth area ratios and ground gaps, checks all saved
cloth/bunny pairs and nonincident cloth/bunny self pairs using independent CPU
geometry, and requires cloth thickness clearances. Triangles sharing vertices
are excluded from self checks. These are saved-state checks, not independent CCD.
It then writes `validation.json` and optionally renders `bunny_cloth.gif` plus
initial/final images. Any incomplete simulation/check failure is recorded as
`incomplete`, never `complete`.

The fresh run completed on 2026-10-07 at 12:47 +08; independent validation and
rendering completed at 12:49. The completion artifacts were audited at 13:13:

- All 120 frames converged, with 121 full-precision saved states and no resumed
  prefix. All 120 native frame-stat records have no Newton/line-search limit.
- Maximum final mass-scaled residual across frames: 0.009995206569057707 m/s,
  below the unchanged 0.01 threshold. Final frame: 0.009663387906425291 m/s.
- Minimum det(F): 0.680176940712448; minimum cloth area ratio:
  0.8337471435385195. Minimum bunny/cloth ground gaps are respectively
  0.00004175183120161177 m and 0.0003965012068621854 m.
- All 121 independent saved-state cross-body and nonincident self-surface
  checks passed, including cloth thickness clearances; zero intersections.
- Total measured advance time: 49,860.944 s (13.85 h); mean 415.508 s/frame,
  median 386.217 s/frame. Maximum outer rounds: 1,348. Logs contain 1,305,012
  local-pass residual evaluations overall, maximum 20,198 in a frame; counts
  include convergence certification. This is completion evidence, not a speedup.
  Native logging is included and brief integration checks shared the GPU during
  part of this run, so these timings are not an isolated performance benchmark.
- `local_stencil/bunny_cloth.gif` contains 61 rendered samples of the trajectory
  (every other saved state); `initial.png` and `final.png` are also verified.

The strict full dense landing is now verified within the stated saved-state
check scope. Slow relaxation remains unresolved, and this does not establish
complete paper reproduction or paper-level performance. The single-bunny suite
in [23](23-single-bunny-ablation.md) automatically began after validation/rendering.

The thread follow-up `bunny-cloth` checks this run every 30 minutes, remains quiet
while it advances normally and reports completion or an actionable failure. Its
completion condition includes the independent report and rendering, not just a
process exit. Ordinary recovery of this same fresh-scene run is allowed; resume
manifests are included in the report. Append resumed simulation logs so all
per-frame residual certificates remain available.
