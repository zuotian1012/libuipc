# 19 - Bunny and cloth solver experiment

`python/examples/local_stencil_bunny_cloth.py` creates a shared scene for the
experimental local stencil solver and the existing global fused-PCG solver.
It uses the tracked `assets/sim_data/tetmesh/bunny0.msh` asset (1,869 vertices,
7,356 tetrahedra) with optional conforming volume subdivision, without proxies or
new downloads. Negative tetrahedron ordering is corrected before constructing the simulation mesh. Boundary faces
are extracted for export/rendering; collision surface labels use libuipc's API.

## Controlled inputs

Both the bunny and cloth start at rest, unsupported and unpinned. Y is up, the
ground is `y=0`, and gravity is `(0,-9.8,0)` m/s^2. The bunny lands first; the cloth
then contacts the bunny and the floor. Landing and draping come from simulation.

| Input | Default |
|---|---|
| Bunny height / bottom clearance | 0.45 m / 0.04 m |
| Bunny material | SNK1, E=1 MPa, nu=0.3, density=1000 kg/m^3 |
| Bunny refinement | 1 level: 12,169 vertices, 58,848 tetrahedra, 8,608 surface triangles |
| Cloth size / initial clearance above ears | 0.85 m square / 0.12 m |
| Cloth grid | 48 x 48 cells, 2,401 vertices, 4,608 triangles |
| Cloth material | BW, stretch E=50 kPa, shear E=0.5 kPa, both nu=0.4 |
| Cloth bending | DiscreteShellBending, E=50 kPa, nu=0.4 |
| Cloth density / one-sided thickness | 200 kg/m^3 / 0.001 m |
| Contact | Standard IPC, d_hat=0.0005 m, friction disabled |
| Contact stiffness | Requested 1e8; existing adaptive corridor applies |
| Time integration | BDF1, dt=0.01 s, 120 frames |
| Nonlinear settings | Absolute velocity_tol=1e-2, max_iter=2048, no semi-implicit termination |
| Global linear solve | Existing default fused PCG / diagonal preconditioner |

Arguments expose bunny height/material stiffness, drop height, cloth size/gap/grid,
dt, frames, tolerance, iteration budget and hybrid/additive stencil selection.
Invalid geometry or a nonconverged frame stops the run rather than silently
including it in a successful comparison.

## Run and inspect

```text
python python/examples/local_stencil_bunny_cloth.py --solver global --render
python python/examples/local_stencil_bunny_cloth.py --solver sos --dt 0.001 --frames 3 --output output/bunny-cloth-sos-short
python python/examples/local_stencil_bunny_cloth.py --solver both --dt 0.001 --frames 3 --output output/bunny-cloth-both-short
python python/examples/local_stencil_bunny_cloth.py --replay --gui
python python/examples/local_stencil_bunny_cloth.py --solver both --replay --render --output output/bunny-cloth-both-short
```

`both` launches the two simulations **serially in separate processes**, keeping
the same scene parameters. `--replay` uses saved trajectories without running the
simulation. `sos` selects the implemented SoS subset, backed by
`newton/solver=local_stencil`; `local_stencil` remains a CLI alias. It does not
select a complete implementation of the paper. `global` selects the ordinary
IPC global Newton/fused-PCG path and is the default for this dense demo. Solver directories retain backend names
`local_stencil/` and `global/`.

The default output root is now `output/local-stencil-bunny-cloth-dense/`;
use distinct `--output` directories for changed parameters or additive sweeps.
Existing run data is not overwritten: use replay or select another output root.
To replay the older coarse experiment, add `--output output/local-stencil-bunny-cloth`.
The dense local run at dt=0.01 currently fails. The short switching examples above
use dt=0.001; they are not verified full-landing settings.

Each solver directory contains `scene.json` (parameters, full configuration,
asset/mesh fingerprint and runtime path), incrementally flushed `history.csv`,
OBJ frames, `positions.npz` and `summary.json`. The root comparison validates
matching mesh fingerprints, physical parameters and trajectory dimensions before
writing `comparison.json` and `comparison.png`. `--render` produces initial/final
images and a side-by-side `bunny_cloth.gif`. `--gui` opens a saved local-stencil
trajectory with playback and frame scrubbing.

## Mesh density controls

`--cloth-resolution N` (legacy alias `--resolution N`) specifies cells per edge:
`(N+1)^2` vertices and `2*N^2` triangles. `N` must be a positive integer.
`--bunny-refinement L` accepts 0, 1 or 2; every level splits **all simulation
tetrahedra into eight**, using one shared midpoint per unique edge. Four corner
tetrahedra and four tetrahedra along the central octahedron's shortest diagonal
form a conforming subdivision. Adjacent parents share the same face subdivision.
The original piecewise-linear shape and total volume are preserved. This increases
deformation/contact degrees of freedom, not scan detail or surface smoothness.

| Example scale | Cloth resolution | Bunny refinement | Cloth vertices / triangles | Bunny vertices / tetrahedra |
|---|---|---|---|---|
| Previous coarse scene | 12 | 0 | 169 / 288 | 1,869 / 7,356 |
| Current default | 48 | 1 | 2,401 / 4,608 | 12,169 / 58,848 |
| Larger optional scene | 96 | 2 | 9,409 / 18,432 | 87,489 / 470,784 |

The default has 43,710 scalar DOFs; the larger example has 290,694.
`--mesh-info` builds the real meshes and prints counts, total DOFs, volume and
minimum tetrahedron volume without initializing the CUDA engine or writing a run:

```text
python python/examples/local_stencil_bunny_cloth.py --mesh-info
python python/examples/local_stencil_bunny_cloth.py --cloth-resolution 96 --bunny-refinement 2 --mesh-info
python python/examples/local_stencil_bunny_cloth.py --solver both --cloth-resolution 96 --bunny-refinement 2 --output output/bunny-cloth-large
```

Both solvers get the same generated volume/surface meshes and physical settings;
the manifest records refinement, surface counts and DOFs alongside the mesh hash.
Higher refinement increases memory and solve cost substantially, especially for
the current local solver; the larger mesh count is not a completed simulation claim.
`--contact-distance` exposes IPC's activation distance in meters (finite, positive,
default 0.0005). The original 2.5 mm distance activated many close surface features
on the refined bunny already at rest. Reducing it to 0.5 mm eliminates those initial
active pairs in the level-1 scene. It does not disable contact or CCD.

## Dense validation and current solver limit (2026-09-30)

The [2026-10-01 hybrid correction](17-local-stencil-hybrid.md) removes the
color-dependent local constraints and non-paper secondary weighting described
in the implementation audit. Repeating the dense dt=0.01, E=1 MPa,
d_hat=0.5 mm, velocity_tol=0.01 case still exhausts 2,048 passes at frame 1;
final residual is 3.176705 m/s with no active contacts. The new run is preserved
at `output/stencil-hybrid-alignment/dense-dt001-guard/`; no successful dense SoS
landing or paired benchmark is claimed after this correction either.

Geometry checks for levels 1 and 2 verify positive child volume, eight children
per parent with volume ratio 1/8 (maximum relative error below 4.5e-14), all vertices
referenced, interior face incidence two and boundary triangle count quadrupling
at each level. The scaled bunny volume remains 0.0252192954321 m^3.

The denser mesh exposed limitations that the old coarse run did not:

- With the original 100 kPa stiffness, dt=0.01 and d_hat=2.5 mm, the global
  solver produced a negative tet determinant at frame 17. Preserved at
  `output/bunny-cloth-dense-validation/`.
- With 1 MPa stiffness and the same dt/d_hat, global completed 30 frames, with
  minimum det(F)=0.689930 and cloth area ratio=0.910586. Its saved render shows
  the cloth contacting and draping over the bunny. The local solver instead hit
  the 2,048-pass cap at frame 1. These are **not** a successful paired comparison.
  Artifacts: `output/bunny-cloth-dense-stiff-validation/`.
- The dt=0.0025 local attempt with d_hat=2.5 mm was manually stopped without
  completing a frame; `output/bunny-cloth-dense-smallstep-validation/` is incomplete.
- With d_hat=0.5 mm and dt=0.01, local still hit the 2,048-pass cap at frame 1,
  despite zero initial active simplex pairs. Reducing the barrier distance alone
  therefore does not resolve its convergence. Preserved at
  `output/bunny-cloth-dense-contact-validation/`.
- With d_hat=0.5 mm and dt=0.001, `--solver both --frames 3` completed in serial
  processes, verifying argument forwarding, backend selection, matching mesh hashes
  and comparison output. All three frames converged for both solvers. Local used
  1/189/220 color passes and retained minimum det(F)=0.999907; global used at most
  two Newton steps. This is only 0.003 seconds of pre-impact motion, **not** a
  validated full dense SoS landing. Artifacts:
  `output/bunny-cloth-dense-shortstep-validation/`.

The CLI switch changes the real backend solver. It is not a claim that the current
SoS subset converges at every mesh scale. Strict convergence/geometry checks remain
enabled; `both` stops when a run fails, without manufacturing a comparison report.
Do not use the above incomplete local runs for a performance ratio. Solver scaling
and convergence remain follow-up algorithm work.

The final dense **global** defaults (1 MPa, d_hat=0.5 mm, dt=0.01) completed the
full 120-frame landing. All frames reported convergence; minimum det(F)=0.680116,
minimum cloth area ratio=0.869073, minimum bunny ground gap=0.041756 mm and minimum
cloth clearance beyond radius=0.248522 mm. Total advance time was 58.240 seconds
(mean 485.331 ms, maximum 67 Newton iterations), a single observation, not a
matched-accuracy solver benchmark. Saved initial/final images and animation were
visually inspected. Artifacts: `output/local-stencil-bunny-cloth-dense/global/`.
No independent dense cloth/bunny intersection or continuous-path audit is claimed.

## Interpretation

Advance timing excludes trajectory export, rendering and retrieval; initialization
is recorded separately. Runs are single observations, with no claim of controlled
benchmark variance or paper-level performance. In particular, local stencil
termination uses the mass-scaled total gradient divided by dt, whereas the
global solver retains its displacement-based convergence criterion. Identical
numeric tolerance settings are **not** matched accuracy. One hybrid color pass
and one global Newton iteration are also different amounts of work.

The experiment reports minimum solid det(F), cloth area ratio and thickness-aware
ground clearances, convergence, per-frame solve/linear iteration counts, wall time,
and vertex trajectory differences. Ground clearance and positive element measures
alone are not a certificate of cloth/bunny nonintersection; regular IPC CCD still
governs contact. This is a reusable comparison scene for future algorithm stages,
not a completed reproduction of the paper's benchmark suite.

## Incomplete stress case

The initial 50 kPa bunny / 0.12 m drop / d_hat=0.005 / velocity_tol=1e-3
attempt reached the 2,048-pass limit at frame 26. Only its first 25 frames converged.
Its manifest, CSV and OBJ states are preserved under
`output/local-stencil-bunny-cloth/attempt-soft-incomplete/`; it is not a successful
comparison result. The original coarse baseline used 100 kPa bunny stiffness to reduce impact and
initial contact activation. The less strict tolerance is explicit, not an increase
in the iteration budget or a bypass of convergence/validity checks.

## Baseline observations (2026-09-30)

These are historical **coarse-mesh** results, with refinement 0, cloth resolution 12
and bunny E=100 kPa; they are not measurements of the new dense defaults. Both runs
completed 120 frames (1.2 seconds), all reported converged, using the
same initial mesh fingerprint and scene settings apart from the solver selector.

| Measurement | Local stencil hybrid | Global fused PCG / diag |
|---|---|---|
| Total advance time | 925.112 s | 8.844 s |
| Mean advance time | 7,709.268 ms | 73.701 ms |
| Maximum nonlinear iteration count | 659 color passes | 29 Newton iterations |
| Minimum solid det(F) | 0.060919 | 0.060092 |
| Minimum cloth area/rest area | 0.912245 | 0.888078 |
| Minimum bunny ground gap | 0.929663 mm | 0.928653 mm |
| Minimum cloth ground gap beyond radius | 2.020850 mm | 1.952108 mm |

The observed elapsed-time ratio is 104.60, with local stencil slower. This is the
current implementation and the stopping settings above, **not** a matched-accuracy
comparison or a claim about the complete SoS method. All stencil linear-solver
iteration counters are zero. GPU execution uses the existing locally built CUDA
runtime; no solver/backend code was changed for this scene.

Trajectories are not equivalent: the maximum bunny vertex difference is 15.968 mm
(final bunny RMS 2.675 mm); the cloth maximum is 575.826 mm (final cloth RMS
315.571 mm). The large cloth difference is retained and visible in the animation
and plots. Its cause is not established by this experiment; convergence-rule
alignment and objective/residual comparisons are needed before drawing accuracy
conclusions. Do not attribute this difference to atomic rounding without evidence.

An independent closest-feature/segment-triangle saved-frame check over all 121
states found no cloth/bunny surface intersections. Minimum surface clearances
beyond cloth radius were 1.802491 mm (stencil) and 1.816481 mm (global).
`surface-verification.json` records this check. It does not certify continuous
paths or cloth self-contact. Full artifacts, the comparison plot, and the
side-by-side GIF are under `output/local-stencil-bunny-cloth/`.
