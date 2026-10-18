# 27 - Opt-in SoS iteration-budget acceptance

The owner requested a bounded approximate mode on 2026-10-18. The new scene
selector is `newton/local_stencil/iteration_limit_policy=error|accept_last_valid`,
default `error`. The bunny/cloth CLI exposes `--iteration-limit-policy` and uses
the existing `--max-iterations` outer-round budget. Inner color passes and local
CG steps are distinct counts, so this is not a wall-clock deadline.

At the limit, ordinary outer line search and continuous geometric checks have
already accepted the endpoint. The optional path requires completed animation
targets and an inactive warm phase, validates elastic geometry, refreshes
endpoint contacts, and evaluates full-elastic gradient norms and energy without
another direction solve or displacement submission. Nonfinite states fail.
Velocity is updated only after this budget certificate. Existing strict-mode
and stencil line-search failure behavior remains unchanged by default.

`frame_stats()` retains `converged=false`, `hit_newton_limit=true` and reports
`iteration_limit.accepted=true` plus fresh endpoint velocity residual, gradient
L2, energy and residual-threshold status. A residual that happens to pass on
this last update is reported separately; the budget path is not relabeled as
ordinary convergence. Completed approximate frames can be checkpointed and
resumed with the same explicit policy. Forks from historical reference runs
still require a converged prefix. Older journals get explicit false/NaN fields.

Warm scratch states cannot be published as physical frames. A budget smaller
than the warm handoff needs can still fail deliberately. The mode does not
bypass line-search failure or guarantee progress when accepted steps are zero.
It bounds rounds; it does not repair slow convergence or prove accuracy.

Build artifacts and
large logs remain under `output/sos-budget-20261018/` (not tracked). The working
tree also contains separate pre-existing instrumentation; its diff is preserved
in `before/working.diff` and must not be included in this feature commit.

## Bounded runtime results (2026-10-18)

Release build `xmake build -j 4 core cuda backend_cuda` passed. The isolated
runtime contains freshly built core/backend DLLs; backend SHA256:
`2890267b652c30383c3c3baaf6a35bd99f2183ddf16034105088ab6a9b09ce92`.
Compilation includes the recorded pre-existing working-tree changes; this is
not an independently rebuilt clean checkout of the published commit. Optional
audit environment switches were removed from comparison processes.
The existing `[stencil]` suite passed 193 assertions in 12 cases. Python syntax
and the actual strict/budget/warm-guard/resume integration paths were checked.

Small comparison: 1,869 bunny vertices / 7,356 tetrahedra; 81 cloth vertices /
128 triangles; dt=0.01 s; bunny drop height=0.004 m; cloth gap=0.02 m; 12 frames.
Local arms use sequential hybrid, packed products, 15 inner passes, IPC barriers,
monotone merging and no warm start. Other physical parameters use demo defaults.

| Arm | Completed frames | Budget-accepted frames | Mean advance ms/frame | Final bunny RMS difference from native | Final cloth RMS difference from native |
|---|---:|---:|---:|---:|---:|
| Strict local, 2 outer rounds | 0 | 0 | N/A | N/A | N/A |
| Budget local, 2 outer rounds | 12 | 12 | 843.68 | 7.85 mm | 10.74 mm |
| Budget local, 8 outer rounds | 12 | 12 | 3472.29 | 5.70 mm | 4.36 mm |
| Native global | 12 | 0 | 34.07 | reference | reference |

All completed runs retained positive determinant/area ratios and ground gaps.
Budget-2 final velocity residual is 4.1833 m/s; budget-8 is 3.0367 m/s, versus
the local threshold 0.01 m/s. Native is a comparison reference, not an exact
solution; its displacement stopping norm differs from the local mass-scaled
gradient norm. These are single-run diagnostics, not a matched-accuracy speedup
benchmark. More budget improved these endpoint differences, not a guarantee
of monotonic improvement for every frame or scene.

Dense continuation forked the converged build29 frame-38 checkpoint and
restarted the color schedule. Frames 39-40 each used 32 outer rounds. Settings:
12,169 bunny vertices / 58,848 tets; 2,401 cloth vertices / 4,608 triangles;
dt=0.01; independent coverage; 15 inner passes; packed/sequential; expanded
custom barriers; IPC stiffness calibration; local merge acceptance; inertial
initial proposal; warm start with `after_local` projection and a 10-pass budget.

- Frame 39: 73.023 s advance, endpoint velocity residual 15.4680 m/s.
- Frame 40: 62.286 s advance, endpoint velocity residual 29.0548 m/s.
- Both are budget-accepted, not converged. Total new advance: 135.309 s.
- Across these two frames: minimum det(F)=0.878976, area ratio=0.943439,
  bunny ground gap=2.97451e-5 m, cloth ground gap=0.00154370 m.

The summary's aggregate mean and max iterations include the historical prefix;
use the new-frame rows or `new_advance_s` for continuation timing. This crosses
the original failure point but is only two approximate continuation frames,
not a fresh or completed 120-frame dense validation. Large residuals and the
higher frame-40 residual remain an accuracy concern.

A one-round warm-initialization guard run exits without publishing a frame.
A copied budget-2 checkpoint resumes from frame 12 and completes frame 13 with
budget acceptance; the original comparison trajectory remains unchanged.

Artifacts in `output/sos-budget-20261018/`: `runtime-manifest.json` (commands,
hashes, exit codes), `comparison.json`, `resume-check.json`, per-frame journals,
and `budget-comparison.png` / `.gif`. The local runner is `run_comparison.py`.
These large artifacts are not in Git. Reproduce a small budget run with:

```shell
python python/examples/local_stencil_bunny_cloth.py --solver sos --cloth-resolution 8 --bunny-refinement 0 --drop-height .004 --cloth-gap .02 --frames 12 --hybrid-submission sequential --hessian-product packed --inner-sweeps 15 --max-iterations 8 --iteration-limit-policy accept_last_valid --output output/budget-example
```

Use a freshly built package, or explicitly select the isolated runtime through
`UIPC_LOCAL_PACKAGE` on this workstation.
