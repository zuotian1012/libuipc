# 17 - Complementary-color hybrid stencil updates

This stage replaces the default additive update of the [stencil solver](16-local-stencil-solver.md)
with complementary-color hybrid sweeps and per-stencil step filtering. The
reference is Sections 4.2, 5.1 and 5.2 of
[Lan et al., Second-order Stencil Descent for Interior-point Hyperelasticity](https://wanghmin.github.io/publication/lan-2023-sos/Lan-2023-SOS.pdf).

## Scheduling and conflicts

Elastic tetrahedra conflict when they share a vertex. Initialization greedily
assigns the least available color, up to `newton/local_stencil/color_limit`
(default 8); excess tetrahedra are grey. The first scheme uses descending graph
degree, subsequent schemes use cumulative received color ranks. Grey has a
finite rank penalty larger than all finite colors accumulated during the bounded
construction, avoiding floating infinity ties. If a rank-only scheme makes no
new primary coverage, the next scheme prioritizes uncovered stencils, then the
same cumulative rank. This explicit coverage completion avoids cycling on
symmetric meshes while preserving the conflict graph and grey policy; it is an
implementation addition to Algorithm 2's ranking heuristic. Schemes are generated until every
elastic stencil has appeared in primary color zero. Construction fails explicitly
if this coverage cannot be obtained; it never silently omits stencils.

The solver cycles through nonempty colors of the complementary schemes. Each
engine nonlinear iteration performs one hybrid sweep, not an entire cycle:

1. Identify the unique current-color stencil owning each of its vertices.
   A device assertion rejects overlapping owners in an independent group.
2. Solve all elastic and active contact stencils in parallel. Current-color
   stencils have priority. Other stencils hold owned vertices fixed in their
   local systems, so their directions cannot overwrite priority updates.
3. Independently filter each direction with local CCD, inversion bounds and a
   local Armijo line search on the complete incident objective.
4. Copy current-color directions to their owned vertices. Sum and average the
   other stencil directions on unowned vertices. Contact stencils always remain
   grey; changing contacts never require recoloring the elastic graph.
5. Reuse regular trajectory CCD and the engine's global energy acceptance for
   the combined proposal, then commit positions. The next color reads those
   accepted positions and freshly computed derivatives.

This is a buffered, synchronous implementation of priority-preserving hybrid
submission. Kernels never write positions while another stencil reads them.
Parallel atomics are retained for secondary contact accumulation. If per-vertex
averaging makes the aggregate direction non-descending, a uniform secondary
weight is used while preserving priority values. This is an additional descent
safeguard, not a claim that arbitrary per-vertex averaging guarantees descent.

## Local step filtering

A GPU-built vertex-to-barrier incidence list freezes the current standard IPC
barrier set for each sweep. A stencil visits every incident tetrahedron and
barrier once, including terms whose other vertices lie outside that stencil.
Non-stencil vertices stay fixed for that local trial.

Local CCD checks only those existing PT/EE/PE/PP/PH pairs, without a new broad
phase. It reuses libuipc's conservative advancement CCD with a 20% gap margin
and an analytic half-plane bound. It does **not** implement the paper's Yuksel
polynomial CCD routine. Inversion filtering brackets the earliest zero of each
incident tetrahedron's cubic determinant, checking derivative extrema as well
as endpoints, and keeps 80% of the safe interval. It applies to contact stencil
motions too. A finite-energy Armijo check then backtracks the complete local
inertia, SNK1 and standard IPC energy, with no change to the barrier list. A
failed local search contributes zero displacement.

Local checks cannot discover previously unknown contacts or certify the sum of
multiple stencil motions. Regular CCD and global energy acceptance therefore
remain required; they are not substitutes for the new per-stencil checks.
The combined proposal still has an endpoint determinant check. A continuous
inversion guarantee for the *combined* simultaneous proposal is not claimed.

## Select and run

```python
config["newton"]["solver"] = "local_stencil"
config["newton"]["local_stencil"]["sweep"] = "hybrid"  # default
config["newton"]["local_stencil"]["color_limit"] = 8
config["contact"]["friction"]["enable"] = False
```

`sweep="additive"` retains the previous baseline, without local CCD/line search.
Both keys are set before initialization, unitless, and ignored by the global
solver. The color limit must be an integer in [1,64]. Existing SNK1/BDF1/static
topology and non-differentiable-simulation restrictions remain.

```text
python python/examples/local_stencil_contact.py --scene ground --sweep hybrid --render
python python/examples/local_stencil_contact.py --scene stack --sweep hybrid --render
python python/examples/local_stencil_beam.py --sweep hybrid --frames 120
```

Output directories use `local_stencil-hybrid` or `local_stencil-additive`, keeping
prior-stage artifacts separate. The contact example's `--verbose` logs each
scheme/color and counts local CCD caps, inversion caps, backtracks, accepted and
rejected directions. These diagnostics distinguish local filtering from the
existing global filter. Neither CSFD, customized paper barriers nor warm start
are included in this stage. CMake and XMake both glob the new FEM `.cu` sources.

## Integration evidence

CUDA 12.6 / XMake release, 120 frames per scene, all frames converged:

| Scene | Minimum det(F) | Minimum ground gap | Mean frame time | Maximum color passes/frame |
|---|---|---|---|---|
| Cube ground impact | 0.671934 | 0.00544614 m | 24.0 ms | 12 |
| Two-body impact | 0.424263 | 0.00660704 m | 227.7 ms | 170 |
| Cantilever | 0.784888 | N/A | 488.8 ms | See exported CSV |

These timings are observations, not controlled benchmarks. Contact runs include
verbose logging, and a color pass has different work from the old additive sweep.
Both contact meshes use 40 schemes / 320 passes to provide primary coverage;
the coverage-first completion is intentionally conservative and not optimized.
Beam fixed-vertex error is zero. Across the two-body run, local filtering records
33 CCD caps, 33 inversion caps, 1,932 backtracks and no exhausted local searches.
Ground records seven backtracks; it does not need a local CCD cap in this run.
The two bodies have no overlapping tetrahedral pair in any of 121 saved states,
as checked by separating axes. This finite-frame check is not a continuous
collision certificate.

The corresponding global reference differs by up to 1.394 mm (ground) and
12.870 mm (two-body) across saved vertices. Different stopping criteria and
nonlinear paths mean this does not establish numerical equivalence. Artifacts:
`output/local-stencil-contact/hybrid-diagnostics.json`, `hybrid-comparison.json`,
and each `local_stencil-hybrid` directory's OBJ/NPZ/CSV/summary and GIF files.
