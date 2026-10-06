# 20 - SoS convergence diagnosis and controlled initialization

This work follows the request to obtain a complete dense bunny/cloth result,
retaining its mesh, materials, dt=0.01 and velocity residual tolerance 0.01.
It does not claim reproduction of the full paper or its performance.

## What the failure actually establishes

Figure 13 of Lan et al. reports completed Dragon-in-U ablations without warm
start (hybrid/regular CCD: 45.7 s; hybrid/local CCD: 32.4 s; adding warm start:
6.7 s). Missing warm start alone is not evidence of an implementation defect,
nor does this ablation guarantee convergence within our particular budget.
Figure 2 groups several color sweeps in an outer iteration. Our engine counts
one color pass as an iteration; do not compare those counts directly.

The original dense scene fails before contact at 2,048 passes. An independent
NumPy assembly of its SNK1 objective and gradient at five recorded iterates
agrees with the GPU gradient to approximately 6.4e-16 in L2. The objective
decreases. Gradient L2 falls from 3.018e-4 to 7.476e-5, while the maximum
mass-scaled velocity residual ends at 3.1767 m/s. The cloth is already below
the requested tolerance by pass 128; the slow component is the bunny's bulk
translation. At pass 2,048 its median position error against the exact free-fall
translation is about 0.289 mm. Twenty-four sampled local directions agree with
independent dense local solves to relative errors below 1e-6.

Increasing only the old binary's budget to 16,384 produced 19 reported converged
frames; frame 1 needed 11,423 passes and 963.85 s. Frame 20 then exhausted the
budget with residual 0.073299 m/s. These are diagnostic data, not a successful
120-frame result. The old convergence check was evaluated before committing
one last proposal, so its final saved state did not itself have a residual
certificate. OBJ exports also round positions to ten significant digits and
must not be used to certify a tight force residual.

Evidence: `output/sos-convergence-20261006/{baseline,strict-long}/`,
`../tmp/sos-audit-baseline/`, `../tmp/sos-strict-long.log`,
`../tmp/analyze_sos_audit.py`, and `../tmp/audit_bunny_objective.py`.

## Corrections and implementation

- When the current iterate satisfies the residual, return zero displacement.
  The engine therefore commits the state that was actually checked.
- Local Armijo compares energy increments. Inertia uses its quadratic increment;
  SNK1 uses the exact determinant-polynomial increment, avoiding cancellation
  against its nonzero rest-energy constant. Cloth/contact increments retain the
  same energies. Negative-curvature handling and the Armijo coefficient are unchanged.
- Compute the initial local feasibility cap and trial increment together, then
  reevaluate only the energy increment during backtracking. This preserves every
  incident term while removing repeated adjacency traversals.
- Cache elastic/inertial gradients and diagonal Hessian blocks per vertex.
  A local solve adds every incident off-diagonal block and the current contact
  restriction. This is shared assembly of the same local system, not a diagonal
  approximation or an added preconditioner.
- Continuously bound the **combined** displacement against tetrahedron inversion
  and projected triangle collapse before handing it to regular CCD/global energy
  acceptance. Local certificates do not transfer through priority selection or
  Jacobi averaging. This also certifies an optional inertial proposal.

## Optional initialization and reproducible runs

`newton/local_stencil/initial_guess` defaults to `previous`. `inertial` proposes
`x_tilde - x` once at the beginning of a frame if the full gradient makes it a
descent direction. Physical pins remain fixed. The proposal undergoes the same
combined inversion, global CCD and energy checks as a normal displacement.
Subsequent passes remain stencil Newton-CG. This is a separate initialization
experiment, not Algorithm 3's inertia-barrier solve, custom barriers or CSFD.
No global linear solve is used. The option is ignored by the global solver,
so enabling it does not constitute a matched-initialization timing comparison.

The demo exposes `--initial-guess previous|inertial` and records it, the actual
iteration budget, mesh hash and runtime DLL hash. It preserves full-precision
per-frame NPY files and native checkpoints as well as OBJ/CSV output. SoS
checkpoints include the color-sweep index. `--resume` requires identical mesh,
physics, solver, initialization and tolerance, and verifies recovered positions
against the saved NPY. Only the target frame count and iteration budget may change.
Use one solver per resume invocation; each resume creates its own manifest.

```powershell
python python/examples/local_stencil_bunny_cloth.py --solver sos --initial-guess inertial --frames 120 --max-iterations 65536 --output output/sos-convergence-run
python python/examples/local_stencil_bunny_cloth.py --solver sos --initial-guess inertial --frames 120 --max-iterations 65536 --resume --output output/sos-convergence-run
```

The longer budget is explicit; the original default remains 2,048. Do not call
a budget increase a convergence-rate improvement. Defaults remain `global` for
the demo and `previous` for local-stencil initialization.

Optional `UIPC_STENCIL_AUDIT=<directory>` exports binary diagnostic arrays at
pass zero, every 128 passes and convergence, grouped by frame. Indices are
int32 and scalar data are float64; Eigen matrices are column-major. These
snapshots are large and are not required for normal simulation.
`UIPC_STENCIL_PROFILE=1` synchronizes and times major phases of the first pass
of a frame. Instrumented timings must be labeled accordingly. Gradient L2 is
logged independently; it does not replace the stopping criterion.

## Validation status

The unfused increment/zero-final-step version passed 150 assertions in eight
existing stencil test cases, including a tiny-shear energy-increment regression.
A 120-frame mixed cloth/soft-block contact run completed with minimum det(F)
0.910410, area ratio 0.963851, positive ground gap and zero pin error.
The fused-filter/shared-assembly/combined-cap version passed the same 150
assertions. Twenty-four independently reassembled local systems agree with its
directions to maximum relative error 9.21e-7; the full-precision free-flight
iterate has residual 2.74e-11 m/s and its accepted final displacement is zero.
Evidence: `../tmp/sos-cached-direction-audit.txt` and `../tmp/sos-cached-tests.log`.

Its 120-frame mixed drape run at tolerance 0.01 completed with minimum det(F)
0.912755, area ratio 0.965876, ground gap 0.004390 m and zero pin error. Independent
triangle/segment checks over all 121 saved states find zero cloth/solid
intersections and minimum separation beyond cloth thickness 0.004270 m.
Artifacts: `output/sos-convergence-20261006/drape-cached/`. This is an integration
check, not the dense bunny result or a timing comparison with the earlier run
(which used tolerance 0.001 and a different concurrent workload).

The controlled inertial dense run completed eight free-flight frames with
two passes each, exact translation to roundoff, and successful native recovery
of frame 8. Frame 9 then converged through first ground contact in 13,142 passes,
with certified residual 0.00947570 m/s, minimum det(F) 0.950763 and ground gap
9.56943e-5 m. That frame ran on the earlier increment/zero-step binary and took
1,130.75 s with other GPU verification work during part of the run.

The final fused/shared-assembly binary, SHA256
`c4049d91f292f7ad5d72f498c05063a193f3f27ccba5a3deb5dced9c59b64edd`,
resumed frame 9 successfully and is targeting 120 frames with a 65,536-pass cap
at the unchanged physical parameters and residual tolerance. See
`output/sos-convergence-20261006/validated-inertial-v2/local_stencil/` and
`../tmp/sos-validated-inertial-v2-full.log`. Runtime transitions are recorded in
the resume manifests. Only inspect history/checkpoints for current progress:
the preexisting summary/trajectory files cover the last completed target (9
frames), not the running 120-frame target. A full landing is still pending.
The accompanying `../tmp/finish_sos_dense.py` waits for this simulation process,
requires 120 converged frames, checks every saved cloth/bunny surface pair using
bounded AABB batches, then invokes the demo's replay renderer. Its live status is
`output/sos-convergence-20261006/validated-inertial-v2/completion-status.json`;
its log is `../tmp/sos-dense-finish.log`. It records an incomplete status on a
failed run and does not relax tolerances or fabricate missing frames. The
simulation and this export helper continue independently of the chat turn.

An isolated first-pass diagnostic measured derivatives/contact 2.61 ms,
local directions 15.13 ms and local filtering 33.72 ms. Earlier unfused filtering
was about 69.52 ms. These synchronized single-pass measurements locate work;
they do not establish an end-to-end or paper-relative speedup.
