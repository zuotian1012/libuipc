# 23 - Single-bunny update and CCD ablations

The owner authorized a simpler single-bunny ablation on 2026-10-07, following
the comparison structure of Fig. 13 in Lan et al., *Second-order Stencil Descent
for Interior-point Hyperelasticity*. This is an implementation ablation, not a
reproduction of the paper's Dragon mesh, exact U geometry, Vivace implementation,
or published speedups. Results remain pending until the serial suite finishes.

## Implemented controls

`newton/local_stencil/sweep` now accepts `jacobi` and `gs` in addition to the
default `hybrid` and legacy `additive`. All share the existing unprojected local
Newton-CG objective and stopping criterion.

- Jacobi computes all stencil directions at the same iterate and averages
  incident displacements per vertex, including zero contributions in the count.
- GS uses an uncapped greedy vertex-conflict coloring of elastic stencils.
  Active contact stencils are recolored against elastic and contact neighbors at
  every outer iterate. Only the current color solves/submits directions. Each
  color is followed by global acceptance; there are no grey Jacobi submissions.
  This uses CPU greedy coloring, whose cost is included, not GPU Vivace.
- Hybrid retains complementary elastic coloring, grey contacts, primary copies
  and Jacobi averaging on unowned vertices.
- `globalization="regular"` skips per-stencil CCD/Armijo. `"local"` retains
  them. Both retain combined elastic inversion bounds and global CCD/energy
  certification. An uphill merged proposal is rejected, without hidden fallback
  or weight changes. A persistent Jacobi rejection can therefore fail explicitly.
- Batching above one pass requires hybrid/local. Defaults preserve the previous
  hybrid/local behavior. Legacy additive is unchanged.

These are stencil update modes, not alternative global linear-system solvers.
GS pass counts are not directly comparable to Jacobi full-stencil pass counts;
elapsed time and residuals are the primary comparison.

## Protocol

`python/examples/local_stencil_bunny_ablation.py` implements the following suite:

| Scene | Modes | Repetitions | Frames per run |
| --- | --- | --- | --- |
| Floor | Jacobi/regular, GS/regular, Hybrid/regular | 3 | 30 |
| Analytic U channel | Above three, Hybrid/local, Hybrid/local with 15 inner passes | 3 | 30 |

The 24 runs are serial. Mode order rotates between repetitions. Each uses the
same base Stanford bunny: 1,869 vertices, 7,356 tetrahedra, height 0.45 m,
initial floor gap 0.12 m, SNK1 Young's modulus 1 MPa, Poisson ratio 0.3,
density 1000 kg/m^3, dt 0.01 s, gravity 9.8 m/s^2, standard IPC contact with
0.5 mm activation distance, coefficient 1e8 and no friction. The channel is
three analytic half planes: a floor and two walls tilted 25 degrees outward
from vertical, bottom width 0.65 times bunny width (0.187678 m).

All modes use inertial initialization (not paper inertia-barrier warm start),
strict `max ||g_i||/(m_i*dt) <= 0.01 m/s`, and a 65,536 outer-iteration cap.
The cap allows observation of slow methods; it is not a convergence-rate fix.
Thirty frames cover impact and early post-impact motion, not a claim of settled
rest or the separate dense 120-frame validation. No tolerance is relaxed.

Each frame must pass native convergence without iteration/search limits,
positive determinant and collider-gap checks. All saved states additionally
undergo independent nonincident surface self-intersection checks. These are
saved-state checks, not an independent continuous-trajectory certificate.
Residual certificates are parsed from native logs before a run passes.

The suite records mesh/runtime hashes and rejects inconsistent physical controls.
Timing excludes initialization, retrieval/export and independent validation;
native solver logging is included. Contact-phase averages use the same frame
interval across modes within each scene. Plots show median and range across the
three repetitions; failed groups remain visibly failed. First-impact residual
curves and maximum/final RMS position differences are included. Hybrid/regular
repeat zero is only a cross-method reference, not an exact physical solution.
Matching the stopping criterion does not prove identical trajectory accuracy.

## Runtime and execution

The existing dense bunny/cloth run remains untouched. Checkout runtime backend
SHA256 is `038ff89d8a13221f0f3acd2f629f1313e5474ed351f3e7f7364c628c7e63e85b`.
The ablation uses a copied Python package and rebuilt core/backend under
`output/sos-ablation-20261007/runtime/uipc/`; backend SHA256 is
`fc99120208c9ce31d002a0bee275d5711bc6ea21dcd4c3ea609f1dd4b4cd1b4a`.
`UIPC_LOCAL_PACKAGE` selects this package through the existing example helper.
The default checkout-package lookup is unchanged.

```powershell
$env:UIPC_LOCAL_PACKAGE = (Resolve-Path output/sos-ablation-20261007/runtime).Path
python python/examples/local_stencil_bunny_ablation.py --suite `
  --output output/sos-ablation-20261007/benchmark `
  --wait-for output/sos-relaxation-20261006/full-strict
```

The suite waits on both existing simulation and verifier process handles, then
requires a successful dense completion status before starting timed runs. It
never stops/restarts the dense task. Consult `process.json`, `suite-status.json`
and logs in the benchmark directory for current state. Do not claim queued runs
are completed results. `--resume-suite` reuses only recorded results with
identical controls; it does not silently overwrite an interrupted case.

The suite worker was launched at 2026-10-07 09:38 +08 and confirmed in
`waiting_for_prior_run` state. The existing `bunny-cloth` thread heartbeat now
follows both tasks every 30 minutes; normal progress stays quiet. It must not
pause after dense completion while the ablations remain pending. Read process
files for current PIDs instead of relying on this launch-time record.

The prerequisite dense validation/rendering completed on 2026-10-07 at 12:49
+08, and the queued suite automatically started formal serial runs. At the
13:13 inspection, the first four floor runs passed (Jacobi/GS/Hybrid repeat zero
and GS repeat one); Hybrid repeat one was active. Full aggregate results remain
pending. Preserve the isolated timing run; do not launch competing GPU checks.

Important outputs: `results.json`, `comparison.json`, `matched-controls.json`,
`position-differences.json`, `timing-comparison.png`, `floor-convergence.png`,
`u-convergence.png`, and per-case logs, residuals, histories and saved positions.

## Verification at implementation time

Release CUDA builds passed, including the final contact-color empty-buffer guard.
Clang-format 18 checks and Python compilation passed. All five modes completed
a one-frame floor impact at 1 mm initial gap, with strict residual <= 0.01,
positive determinants/gaps and independent surface checks. Outer iterations:
Jacobi 1,280; GS 7,028; Hybrid/regular 1,223; Hybrid/local 1,223;
Hybrid/batched 84. The last value counts global rounds, not local passes.
Artifacts are under `output/sos-ablation-20261007/integration/`.
These checks shared the GPU with the dense run and their timings are excluded
from the formal benchmark. They do not establish full-suite performance or
complete U-channel convergence.

## Interpretation limits

Local CCD still has the outer global safety check. The batching comparison also
changes outer-loop frequency and frozen-contact reuse; it is a separate group,
not a pure CCD toggle. Append-only full-primitive barriers, paper warm start and
CSFD remain outside this change. Any observed ranking belongs to this implementation,
hardware, strict criterion and these scenes. The earlier smooth-error diagnosis
in [22](22-sos-relaxation-diagnosis.md) remains open; simplifying the scene does
not by itself fix that limitation.
