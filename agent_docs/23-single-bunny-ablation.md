# 23 - Single-bunny update and CCD ablations

The owner authorized a simpler single-bunny ablation on 2026-10-07, following
the comparison structure of Fig. 13 in Lan et al., *Second-order Stencil Descent
for Interior-point Hyperelasticity*. This is an implementation ablation, not a
reproduction of the paper's Dragon mesh, exact U geometry, Vivace implementation,
or published speedups. All 24 runs completed and passed on 2026-10-07; the
audited measurements and their limitations are recorded below.

The next authorized stage is [24](24-sos-contact-execution.md): shared local
evaluation state, a continuous-inversion fast certificate, and a triangle-mesh U
comparison. The original completed results in this document remain the baseline.

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

## Completed measurements (2026-10-07)

All 24 runs completed between 12:49:08 and 15:02:32 +08, following the successful
dense validation. The native log intervals are strictly serial. No failed case
was omitted, no tolerance was relaxed and no simulation was substituted.
`completion-audit.json` independently audits all 720 native final residual
certificates against raw logs, 720 native frame-stat records and 744 saved states.
All saved-state self-surface check reports passed; the completion audit also
recomputes determinants and analytic-plane gaps from the saved positions. Maximum
final residual is 0.009999786613958946 m/s; minimum det(F) is 0.6374903115278401;
minimum plane gap is 0.000031482235768917554 m. No Newton or line-search cap was
hit. Mesh, physical controls and runtime hash match. Total advance time across
the suite is 7,927.994 s (2.20 h), including native logging.

Contact-phase seconds per frame, median [minimum, maximum] of three repetitions:

| Update / globalization | Floor, frames 16-30 | U channel, frames 12-30 |
| --- | --- | --- |
| Jacobi / regular | 10.445 [10.186, 10.480] | 10.830 [10.821, 10.909] |
| GS / regular | 36.509 [36.450, 37.406] | 36.995 [36.848, 37.241] |
| Hybrid / regular | 10.319 [10.151, 10.419] | 10.734 [10.719, 10.755] |
| Hybrid / local | Not run | 20.027 [20.026, 20.065] |
| Hybrid / local, 15 inner passes | Not run | 15.017 [15.014, 15.039] |

The measurements do **not** reproduce the paper's acceleration. Hybrid/regular
and Jacobi are close (about 1% difference). The current GS baseline takes about
3.4-3.5 times as long as Hybrid/regular. Its one-color pass does less local solve
work and repeats global assembly/acceptance more frequently; this measurement
also includes CPU greedy coloring, so it is not a verdict on GPU Vivace.

U-channel evidence explains the local-filter overhead without attributing it to
a hypothetical instability. In repeat zero, Hybrid/regular and Hybrid/local
both have exactly 55,029 residual evaluations over the trajectory and 3,181 at
first contact. Their maximum trajectory difference is 4.60e-12 m. The local
variant records zero CCD caps and zero inversion caps, with 372 local Armijo
backtracks across about 405.6 million accepted stencil trials. It retains global
certification and is 1.866 times as expensive. Thus local filtering added work
without reducing the number of passes on this scene. The 15-pass batch reduces
time by about 25% relative to unbatched local filtering, but remains 1.399 times
the regular variant; its 55,179 residual evaluations do not indicate improved
relaxation. These counts include certification passes.

Against Hybrid/regular repeat zero, maximum differences over all vertices,
frames and repetitions are: floor Jacobi 0.01594 mm, floor GS 0.22056 mm;
U Jacobi 0.05169 mm, U GS 0.45607 mm, U local 4.60e-12 m, and U batched
0.004536 mm. These are cross-method differences, not errors against an exact
solution. The common stopping criterion therefore should not be described as
proof of equal physical error.

The U channel uses three analytic planes. It does not exercise the complex
triangle-mesh collision workload of the paper's Dragon in U comparison, and
regular plane CCD is a different workload. To investigate the missing paper
benefit next, separate the cost of local filtering from retained global checks,
and use the paper's contact representation/execution organization and a suitable
mesh-contact scene before making performance claims. The existing safeguards
must not simply be removed without replacing their correctness guarantee.

`timing-comparison.png` and the pass-based convergence plots were visually
checked. `convergence-time.png` additionally plots first-contact residuals
against actual native log timestamps, avoiding comparisons of unequal amounts
of work per pass. The audit is reproducible with
`python output/sos-ablation-20261007/benchmark/audit_completion.py` and does not
rerun the simulation. Both this suite and the dense verification are complete;
the thread heartbeat can now be paused after reporting these results.
