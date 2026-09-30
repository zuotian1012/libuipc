# 19 - Bunny and cloth solver experiment

`python/examples/local_stencil_bunny_cloth.py` creates a shared scene for the
experimental local stencil solver and the existing global fused-PCG solver.
It uses the tracked `assets/sim_data/tetmesh/bunny0.msh` asset (1,869 vertices,
7,356 tetrahedra), without remeshing, proxies or new downloads. Negative tetrahedron
ordering is corrected before constructing the simulation mesh. Boundary faces
are extracted for export/rendering; collision surface labels use libuipc's API.

## Controlled inputs

Both the bunny and cloth start at rest, unsupported and unpinned. Y is up, the
ground is `y=0`, and gravity is `(0,-9.8,0)` m/s^2. The bunny lands first; the cloth
then contacts the bunny and the floor. Landing and draping come from simulation.

| Input | Default |
|---|---|
| Bunny height / bottom clearance | 0.45 m / 0.04 m |
| Bunny material | SNK1, E=100 kPa, nu=0.3, density=1000 kg/m^3 |
| Cloth size / initial clearance above ears | 0.85 m square / 0.12 m |
| Cloth grid | 12 x 12 cells, 169 vertices, 288 triangles |
| Cloth material | BW, stretch E=50 kPa, shear E=0.5 kPa, both nu=0.4 |
| Cloth bending | DiscreteShellBending, E=50 kPa, nu=0.4 |
| Cloth density / one-sided thickness | 200 kg/m^3 / 0.001 m |
| Contact | Standard IPC, d_hat=0.0025 m, friction disabled |
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
python python/examples/local_stencil_bunny_cloth.py --solver both --render
python python/examples/local_stencil_bunny_cloth.py --replay --gui
python python/examples/local_stencil_bunny_cloth.py --solver both --replay --render
```

`both` launches the two simulations **serially in separate processes**, keeping
the same scene parameters. `--replay` uses saved trajectories without running the
simulation. The default output root is `output/local-stencil-bunny-cloth/`;
use distinct `--output` directories for changed parameters or additive sweeps.
Existing run data is not overwritten: use replay or select another output root.

Each solver directory contains `scene.json` (parameters, full configuration,
asset/mesh fingerprint and runtime path), incrementally flushed `history.csv`,
OBJ frames, `positions.npz` and `summary.json`. The root comparison validates
matching mesh fingerprints, physical parameters and trajectory dimensions before
writing `comparison.json` and `comparison.png`. `--render` produces initial/final
images and a side-by-side `bunny_cloth.gif`. `--gui` opens a saved local-stencil
trajectory with playback and frame scrubbing.

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
comparison result. The baseline uses the parameters above to reduce impact and
initial contact activation. The less strict tolerance is explicit, not an increase
in the iteration budget or a bypass of convergence/validity checks.

## Baseline observations (2026-09-30)

Both runs completed 120 frames (1.2 seconds), all reported converged, using the
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
