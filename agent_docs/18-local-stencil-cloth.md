# 18 - Cloth in the local stencil solver

The scene-selectable solver now supports triangle cloth as well as SNK1 solids,
including mixed cloth/solid scenes. This extends the scheduling and local step
filtering in [17](17-local-stencil-hybrid.md); it does not add another solver mode.

## Supported energies and assembly

- `StrainLimitingBaraffWitkinShell` (UID 819): stretch, shear and the existing
  cubic extension penalty, with independent stretch/shear material attributes.
- `NeoHookeanShell` (UID 11): the existing membrane energy and analytic derivatives.
- Optional `DiscreteShellBending` (UID 17): four-vertex interior-edge hinges,
  with the same sorted edge/opposite-vertex convention and rest dihedral as the
  standard backend. Boundary edges have no hinge; nonmanifold edges are rejected.
- `StableNeoHookean` (UID 10) solids remain supported in the same scene.

`fem_local_stencil_elastic.cu` initializes a unified elastic topology and material
table. Each membrane triangle contributes a three-vertex stencil (fourth slot
is -1); each bending edge and tetrahedron contributes four vertices. Padded slots
are excluded from adjacency, restriction, coloring and all vertex accesses.
All elastic stencils enter the shared-vertex conflict graph and complementary
primary-coverage construction. Active contact stencils stay grey.

Both elastic and contact solves restrict the complete union of incident membrane,
bending, solid, inertia and contact terms. No term is counted twice within a local
objective. Pins are eliminated after assembling these contributions. The existing
hybrid priority/secondary rules, mass-scaled residual and global acceptance apply.

`local_stencil_elastic.h` reuses the constitutions' energy and gradient functions.
NH and bending reuse their raw analytic Hessians. BW's standard Hessian helper
is a PSD approximation, so the local path evaluates the exact radial stretch
and bilinear shear Hessian explicitly; Newton-CG handles its negative curvature.
The standard global solver's material implementations are unchanged.

Rest weights match the standard backend: BW uses triangle area (its stiffness
attributes already contain the applicable thickness factors), NH uses area times
full thickness `2r`, and bending uses the neutral outer multiplier from the shared
reference helper. All terms receive `dt^2` exactly once. This is not CSFD/HVP.

## Geometry and contact filtering

Each incident shell triangle gets a continuous local bound on projected area
along the trial displacement. Its projection normal is the **current** normal,
not the rest normal: a sheet can turn over across successive accepted steps.
The projected area is quadratic, evaluated using the existing cubic determinant
root-bracketing helper with a fixed third column. A local trial cannot cross the
first zero; the same 0.8 safety factor applies. Both triangles of a bending hinge
are checked. Global acceptance also rejects degenerate endpoint shell triangles.

As before, independent local bounds do not certify the combined simultaneous
motion. Regular trajectory CCD and global energy acceptance still follow every
combined proposal. A separate continuous elastic cap now checks the actual
combined motion, using the same determinant/projected-area root bounds.
CCD uses the existing thickness-aware IPC infrastructure, including
cloth/cloth and cloth/solid contacts and its normal self-contact candidate rules.

The mixed drape demo exposed an existing local barrier evaluation defect: raw
PT/EE/PE distance routines measure supporting features, whereas the global IPC
objective uses closest-feature flags. Parallel EE pairs or a change of closest
feature during a trial could consequently give a false infinite local energy.
Local distance guards and PT energy now recompute the same closest-feature flags
as the global energy, preserving the frozen pair list and standard EE mollifier.

## Use

```python
config["newton"]["solver"] = "local_stencil"
config["newton"]["local_stencil"]["sweep"] = "hybrid"
config["contact"]["friction"]["enable"] = False
```

Create a triangle mesh, apply one supported membrane followed by optional
`DiscreteShellBending`, and advance the ordinary World. Existing restrictions
remain: IPC, BDF1, static topology, fixed vertex pins, no differentiable simulation,
and zero relative velocity tolerance. Friction, ABD/rod coupling, plastic bending,
stitching/soft constraints and other constitutions are not enabled by this change.
Use positive finite membrane coefficients, nonnegative strain rate/bending
stiffness, nondegenerate rest elements, positive vertex masses and initially
separated contact geometry. Material/topology changes require a new World.

```text
python python/examples/local_stencil_cloth.py --scene drape --render
python python/examples/local_stencil_cloth.py --scene pinned --gui
python python/examples/local_stencil_cloth.py --scene layers --render
python python/examples/local_stencil_cloth.py --scene pinned --membrane neo --render
```

The drape scene combines a falling sheet and a deformable block pinned at its
bottom. Layers uses two falling sheets to exercise cloth/cloth contact. Pinned
starts a horizontal sheet with two pinned corners and lets it swing onto the
floor. `--resolution`, `--frames`, `--dt`, `--bending-scale`, `--solver global`,
`--sweep additive` and `--verbose` are available. The GUI replays saved frames.
Outputs under `output/local-stencil-cloth/<scene>/<membrane>/<solver-sweep>/`
include OBJ frames, NPZ positions, CSV diagnostics, summary JSON and optional
`contact.gif`/`contact.png`. These are coarse integration scenes, not paper demos
or garment-scale performance benchmarks.

## Validation

Four complete 120-frame cloth runs, resolution 6, `dt=0.01`, absolute velocity
tolerance `1e-3`, all frames converged and all pins exact:

| Scene | Membrane | Vertices | Minimum area/rest area | Minimum cloth ground gap beyond radius | Maximum color passes/frame |
|---|---|---|---|---|---|
| Pinned corners | BW | 49 | 0.985925 | 0.00413446 m | 106 |
| Drape over soft block | BW | 76 | 0.964228 | 0.00437690 m | 171 |
| Two cloth layers | BW | 98 | 0.999831 | 0.00407439 m | 126 |
| Pinned corners | NH | 49 | 0.980044 | 0.00403913 m | 30 |

The mixed run's solid minimum det(F) is 0.910233. The existing two-soft-body
regression also completed 120 converged frames (minimum det(F) 0.424257).
All runs use the actual GPU stencil path, with zero global linear iterations.
An independent saved-frame surface-distance/intersection check finds no
cloth/block or inter-layer intersections across all 121 states. Minimum clearances
beyond combined thickness are 4.26302 mm and 4.52450 mm respectively. This is a
discrete trajectory check, not a continuous contact certificate.

Verbose local-filter counters show 24 CCD caps / 1 element cap / 8,341 backtracks
in drape (one local search exhausted and correctly contributed zero), and 79 CCD
caps / 458 backtracks in layers (no exhausted searches). Thus these runs exercise
local filtering as well as regular CCD. Results live in each run's summary/history
and `output/local-stencil-cloth/verification.json`. Recorded times are uncontrolled
observations, including verbose output and some concurrent CPU compilation;
no paper performance or reference-trajectory equivalence is claimed.

The focused material contract in `apps/tests/backends/cuda/local_stencil_cloth.cu`
compares energy/gradient/Hessian by centered differences for all three shell
energy families (9 assertions passed). The existing stencil checks also pass
(136 assertions in 7 cases). XMake release/CUDA 12.6 builds the backend and tests;
CMake and XMake already glob both new backend and test sources. CMake was not
independently rebuilt for this stage. The checkout-local Python backend DLL is
synchronized; the machine's installed Python package is not changed.
