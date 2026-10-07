# 24 - Local filtering reuse and triangle-mesh contact comparison

The owner requested the next implementation step after the completed ablations
in [23](23-single-bunny-ablation.md). This stage measures local/global checking
costs, reduces repeated local evaluation work, and adds a true triangle-mesh U
channel. It preserves the objective, local Armijo condition, stopping threshold,
merged-step certification and final global CCD/energy acceptance.

## Changes and guarantees

`fem_local_stencil_linesearch.cu` now builds a per-iterate evaluation cache before
local filtering. It stores each tetrahedron's deformation gradient, each shell
or hinge's starting elastic energy, and each active barrier's starting energy.
All incident stencil trials reuse these values. The merged frozen-contact inner
search reuses the same cache before committing its accepted displacement.
The cache is rebuilt on every local pass, including every inner pass after a
position change; it never outlives its iterate. Buffers persist and are fully
rewritten after discard resize. Energy changes, gradients, directions and Armijo
inequalities are unchanged. Non-SNK shell/bending energies retain their existing
evaluation formulas.

The continuous tetrahedral inversion check additionally certifies a full unit
step when its determinant polynomial is provably positive on [0,1]:

`det(F+t*dF) >= c0 - (abs(c1)+abs(c2)+abs(c3))`.

The shortcut includes the existing positive determinant floor and a rounding
margin. When inconclusive, the existing derivative-root partition and bisection
remain intact, including tangential and interior zeros with positive endpoints.
This also benefits combined-step bounds; it does not replace them with endpoint
tests or skip an uncertified path.

New native timer scopes identify local filtering and merged inner acceptance.
The example's `--timers` flag saves `timer_frames.json`; it is off for formal
timing. The example now supports `--scenes`, `--modes`, `--scene mesh-u` and
`--collider-subdivisions`. Previous floor/U defaults remain unchanged.

## Profile evidence and limits

Release double-precision CUDA 12.6 build, NVIDIA RTX 3060 Ti, driver 591.86.
The worktree inherits unrelated profiling and demo edits from the preceding
experiments; they are not included in this task's commits. The before runtime
is the preserved doc-23 package, SHA256
`fc99120208c9ce31d002a0bee275d5711bc6ea21dcd4c3ea609f1dd4b4cd1b4a`.
The final isolated runtime under `output/sos-contact-20261007/runtime-final/`
has backend SHA256
`b3057d447eaf72c30651cfe976be52d093bd0fb72e413b455ab307158fd98998`.
The base commit is `c8d9a36d`; the implementation commit introducing this document
contains the after source. Build logs are `../tmp/sos-contact-*-build*.log`.

Short diagnostic runs use the unchanged analytic U scene through frame 13,
which includes contact at frames 12 and 13. Inclusive native wall scopes for
frame 13, in seconds:

| Version | Mode | Frame | Local solve | Outer line search |
| --- | --- | ---: | ---: | ---: |
| Before | Hybrid/regular | 13.551 | 6.180 | 4.318 |
| Before | Hybrid/local | 24.765 | 17.433 | 4.306 |
| Before | Hybrid/batched local | 18.400 | 17.297 | 0.294 |
| Cache only | Hybrid/local | 23.803 | 16.391 | 4.350 |
| Cache only | Hybrid/batched local | 17.129 | 16.040 | 0.294 |
| Cache + polynomial certificate | Hybrid/local | 21.827 | 14.711 | 4.192 |
| Cache + polynomial certificate | Hybrid/batched local | 15.586 | 14.488 | 0.294 |

These profiles localize the additional work: before changes, local filtering
adds about 11.25 s inside the local solve, while global checks are almost
unchanged. Batching already reduces global checking frequency substantially;
remaining local work dominates. Parent scopes include children and must not be
summed. Short instrumented runs, including final profiles overlapping CPU test
compilation, are diagnostics rather than repeated end-to-end speedup evidence.
The formal uninstrumented suite below is the performance gate.

Nsight Compute was attempted on `stencil_local_linesearch_kernel`, but NVIDIA
performance counters returned `ERR_NVGPUCTRPERM`. The diagnostic target was
stopped after that error; this was not a solver failure. No hardware occupancy
or kernel-efficiency measurement is claimed. The existing native stage timers
and controlled runtime comparisons remain available. Records:
`profile-comparison.json`, `profile-before/`, `profile-after/`, `profile-final/`
and `ncu-before/` under `output/sos-contact-20261007/`.

## Triangle-mesh U channel

The new connected fixed surface has 425 vertices and 768 triangles at default
subdivision 8. Its bottom width and outward 25-degree walls match the analytic
channel's geometric planes; wall height is 0.65 m and depth spans [-0.75,0.75] m.
It is represented by fully pinned Neo-Hookean shell triangles. Positive
constitutive thickness supplies mass; contact thickness is zero, matching the
analytic surface control. The example verifies that all collider vertices remain
fixed and exports `collider.obj` and collider arrays in `positions.npz`.

No analytic colliders are inserted in `mesh-u`; actual PT/EE contacts are used.
The 13-frame integration recorded up to six PT and four EE active pairs. Its
14 saved states passed both independent bunny self checks and bunny/collider
intersection checks. Plane-gap checks remain an additional conservative check
inside this channel. This is a finite piecewise-planar mesh channel, not the
paper's exact Dragon/U asset. It adds pinned elastic stencils and changes the
contact representation; plane-versus-mesh timing is not a pure CCD-cost toggle.
Within each scene all update variants retain identical geometry/material data.

## Verification and formal run

- Final release CUDA/backend-test builds and clang-format checks pass.
- 156 assertions in nine stencil/material test cases pass, including the
  continuous inversion certificate with interior and tangential zeros.
- Before/final analytic-U comparisons have bitwise identical saved positions
  at all 14 states for both local and batched modes; stopping criteria are unchanged.
- The existing cloth-on-solid drape completed 120 frames with 15 inner passes,
  positive geometry and fixed pins. All 121 saved states passed independent
  cloth self and cloth/solid intersection/thickness checks.
- Mesh integration is preserved under `mesh-integration-v2/`. The first setup
  attempt used the tetrahedral-only orientation-label API on a triangle surface;
  this initialization error was corrected, with its log preserved in
  `mesh-integration/`. Triangle winding is generated consistently in the mesh.

`integration-verification.json` records the saved-position and cloth checks.
The focused native-test log is `../tmp/sos-contact-tests.log`; its launch adds
the isolated native package to PATH for DLL resolution.

The formal suite completed on 2026-10-07 at approximately 23:06 local time,
starting with mesh U, then analytic U. Each scene ran
Hybrid/regular, Hybrid/local and Hybrid/batched local, three repetitions each,
30 frames each: 18 serial runs. Native Info logging remains enabled as in the
baseline; diagnostic timers were disabled. All 18 runs passed, and native log
intervals confirm serial execution. No simulation processes remain. The raw
case artifacts, runtime and baseline are preserved.

```powershell
$env:UIPC_LOCAL_PACKAGE = (Resolve-Path output/sos-contact-20261007/runtime-final).Path
python python/examples/local_stencil_bunny_ablation.py --suite `
  --scenes mesh-u u --modes hybrid hybrid-local hybrid-batched `
  --repeats 3 --frames 30 --output output/sos-contact-20261007/benchmark
```

`benchmark/audit_completion.py` independently checks the recorded results,
raw native residual logs, hashes, physical parameters, serial log intervals,
saved determinants/gaps and completed independent surface reports. Its
`completion-audit.json` confirms:

- All 540 final native residual certificates <= 0.01 m/s; maximum
  0.009993883681524759. No iteration or line-search limit was hit.
- All 558 saved bunny states passed self-surface checks; all 279 mesh-channel
  states passed bunny/collider intersection checks. Minimum det(F) is
  0.6939206328507838; minimum signed plane gap is 0.00003148223576886838 m.
- Collider geometry hashes match. The verified runner source checks fixed
  positions on every retrieved frame (1e-12 bound); independent per-frame
  collider snapshots were not stored, so this is an executed runtime guard,
  not a second independent pin-motion measurement.
- All scene/runtime/material/accuracy controls match within each scene;
  all before/after analytic-U physical controls match. Total measured advance
  time across the 18 runs is 6338.851 s (105.65 min), excluding postchecks.

### Repeated timing and trajectory results

Contact window is frames 12-30 for every run. Values below are median seconds
per contact frame and min-max across three repetitions. The before/after runs
were separated in time, not interleaved; timings include native logging.

| Analytic U mode | Before median [range] | After median [range] | Observed reduction |
| --- | ---: | ---: | ---: |
| Hybrid / regular CCD | 10.734 [10.719, 10.755] | 10.483 [10.482, 10.628] | 2.34% |
| Hybrid / local CCD | 20.027 [20.026, 20.065] | 17.817 [17.712, 18.044] | 11.04% |
| Hybrid / batched local CCD | 15.017 [15.014, 15.039] | 12.739 [12.727, 12.759] | 15.17% |

These changes reduce execution cost, not iteration count. Each analytic-U run
has exactly the same per-frame residual evaluation counts as its corresponding
baseline: 55,029 over 30 frames for regular/local, 55,179 for batched local.
Maximum before/after saved-position difference across all modes/repeats is
5.242e-12 m. The small regular-mode timing change should not be read as a
significant algorithmic improvement. Batched local still costs 1.215 times
regular CCD after optimization; unbatched local costs 1.700 times regular.

| Triangle-mesh U mode | Median [range], s/contact frame | Residual evaluations / 30 frames |
| --- | ---: | ---: |
| Hybrid / regular CCD | 17.866 [17.783, 18.439] | 54,188 |
| Hybrid / local CCD | 31.035 [30.910, 31.652] | 54,188 |
| Hybrid / batched local CCD | 20.783 [20.780, 20.844] | 54,504-54,549 |

The mesh case genuinely exercises local CCD: repeat 0 has 104 local collision
caps and two inversion caps in unbatched local, and 259/eight in batched local.
The trajectory filter records up to 48 PT and 147 EE candidates (candidate
counts, not simultaneous active barriers). Yet unbatched local does not reduce
the 54,188 residual evaluations. Batching cuts contact time about 33% versus
unbatched local, but is still 16.3% slower than regular CCD. These observations
do not establish a benefit comparable to the paper. Plane-versus-mesh timing
also includes pinned elastic stencils and a different contact representation.

Relative to mesh-U regular repeat 0 (a control, not exact ground truth), maximum
saved-position differences are 0.000757 mm for unbatched local and 0.696362 mm
for batched local. Batched final RMS differences are 0.3143-0.3146 mm. Thus all
modes meet the same stopping criterion, but the batched mesh trajectories are
not identical; do not confuse before/after cache equivalence with cross-mode
trajectory equivalence. Even the simple bunny continues to need roughly
2,500-3,500 local residual evaluations in a contact frame. Smaller outer
iteration counts in batched mode do not mean fewer local passes.

Reviewable artifacts under `output/sos-contact-20261007/benchmark/`:
`completion-audit.json`, `before-after.json`, `comparison.json`,
`position-differences.json`, `before-after.png`, `timing-comparison.png`,
`convergence-time.png`, `mesh-u-trajectory.png` and `mesh-u.gif`.
The mesh render uses saved batched-local repeat-0 geometry and transparent
colliders for visibility. The GIF displays all 31 states at quarter speed.
The audit and rendering scripts are retained next to these artifacts.

This verification stage is complete. Pause the shared heartbeat after reporting
the findings; do not automatically rerun the dense, ablation or contact suites.

## Remaining scope

This stage reduces repeated evaluation work. It does not implement append-only
full-primitive barrier history, paper warm start or CSFD, and does not remove
global acceptance. Existing local certificates cover retained contact stencils;
they cannot by themselves certify a merged path against newly encountered
primitives. Paper-level contact organization and slow global error relaxation
remain separate algorithmic work. No paper-level speedup is claimed here.
The next algorithmic decision should address contact-history organization and
the slowly decaying global error (including the paper's initialization/warm
start), then measure pass counts at the same strict stopping threshold. More
cache tuning alone cannot establish a convergence-rate improvement.
