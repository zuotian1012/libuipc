# 17 - Complementary-color hybrid stencil updates

Cloth support is now added in [18](18-local-stencil-cloth.md): the same conflict
and priority rules apply to membrane triangles and bending hinges as well as
tetrahedra. The tetrahedral measurements below are historical stage evidence.

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

1. Solve all elastic and active contact stencils in parallel at the same frozen
   iterate. Only physical pins and padded slots are fixed. Colors/priority owners
   are not inputs to the local solver: every stencil retains its complete coupling.
2. Independently filter each complete direction with local CCD, inversion bounds
   and local Armijo search on the complete incident objective at that iterate.
   The direction, directional slope, CCD path and energy trial use identical DOFs.
3. Identify the unique current-color stencil owning each of its vertices.
   A device assertion rejects overlapping owners in an independent group.
4. Copy current-color directions to their owned vertices. Sum and average the
   other stencil directions on unowned vertices. Contact stencils always remain
   grey; changing contacts never require recoloring the elastic graph.
5. Reuse regular trajectory CCD and the engine's global energy acceptance for
   the combined proposal, then commit positions. The next color reads those
   accepted positions and freshly computed derivatives.

This is a buffered, synchronous implementation of priority-preserving hybrid
submission. Kernels never write positions while another stencil reads them.
Parallel atomics are retained for secondary contact accumulation. An owned vertex
receives exactly its owner's locally filtered displacement. An unowned vertex
receives the sum of incident filtered elastic/contact displacements divided by
its elastic/contact stencil multiplicity. Rejected directions count as zero
contributions without changing that divisor. Contact scatter and Jacobi averaging
never overwrite the primary displacement. There is no extra secondary reweighting.

As an explicit integration safeguard (not an additional paper merge rule), the
complete proposal's `g.dot(dx)` is checked before global acceptance. A non-descent
proposal is rejected in full: positions remain unchanged and the next call tries
the next color. No Jacobi divisor or priority value is retuned to force descent,
and rejection does not satisfy convergence, which still uses the full residual.
This handles the fact that a scalar line search cannot repair an uphill merged
direction. Nonfinite slopes remain errors. Strict global acceptance/iteration
limits are retained for other failures.

This aligns the displacement composition in Section 5.1. The paper's description
does not provide an explicit acceptance algorithm for a clipped secondary local
subproblem; this implementation does not invent one. Priority and averaging act
on already filtered full directions in the global displacement buffer. They do
not re-solve a constrained local problem or mutate positions during local searches.

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

The Armijo slope is `g_s.dot(p_s)` from the same incident Newton-CG system and
frozen state used by `E_s(x + alpha*p_s)`. Nonfinite or nonnegative slopes are
rejected. No priority mask is applied to `p_s` inside this local certificate.
After priority composition/averaging, the proposal generally differs from every
individual local trial. Therefore local descent and CCD certificates **do not**
certify the merged displacement; regular global filtering remains mandatory.

Local checks cannot discover previously unknown contacts or certify the sum of
multiple stencil motions. Regular CCD and global energy acceptance therefore
remain required; they are not substitutes for the new per-stencil checks.
The combined proposal now receives its own continuous elastic feasibility cap:
tetrahedron determinant roots and projected shell-triangle collapse are checked
along the actual merged path. Endpoint validity remains an additional guard.
See [the convergence audit](20-sos-convergence-audit.md) for validation status.

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

## Alignment verification (2026-10-01)

The first literal full-direction/priority/average implementation exposed a
global line-search failure in the two-solid contact regression near frame 31.
Locally accepted directions can form an uphill merged proposal. The final
integration safeguard checks the actual merged slope and rejects that proposal
without changing weights, positions or the stopping criterion. The 120-frame
stack run exercises four such rejections; one has `g.dot(dx)=1.809506e-4` with
residual 37.015743 m/s, well above its stopping tolerance. Subsequent colors
continue and converge. The unguarded failed attempts are retained as diagnostics.

Final 120-frame regressions, with original scene settings:

| Scene | Minimum det(F) | Minimum cloth area ratio | Maximum color passes/frame | Rejected merged proposals |
|---|---|---|---|---|
| BW cloth drape on SNK1 block | 0.910216 | 0.964452 | 168 | 0 |
| Pinned BW cloth | N/A | 0.985934 | 98 | 0 |
| Two soft solids / ground | 0.424180 | N/A | 137 | 4 |

All completed frames reported convergence, with positive ground clearances and
exact fixed vertices in the two cloth runs. These checks do not certify continuous
combined element validity or surface nonintersection independently of backend CCD.
The dense bunny/cloth case at dt=0.01, E=1 MPa, d_hat=0.5 mm and tolerance=0.01
still exhausts 2,048 passes on frame 1 (final residual 3.176705 m/s). It triggers
zero merged-proposal rejections; this particular failure is not caused by the
new rejection safeguard. No active contacts remain at the final iterate. Keep
`dense-dt001-guard` as a failed run, not a successful animation or timing result.

CUDA 12.6/XMake release backend and test build pass, as do 145 assertions in the
eight existing stencil/cloth derivative cases. clang-format 18.1.8 checks pass.
Only existing backend source files changed, so CMake/XMake source lists and
dependencies remain equivalent. No new micro-test suite was introduced.
Artifacts and logs are under `output/stencil-hybrid-alignment/` and
`../tmp/stencil-hybrid-alignment-*.log`; `verification.json` records summaries,
proposal rejection counts and binary hashes. No controlled speedup comparison is claimed.

## Historical integration evidence

The measurements below precede the 2026-10-01 hybrid alignment. That correction
removes color-dependent artificial constraints from local Newton-CG and removes
the extra uniform-secondary reweighting. Do not treat the historical timings as
measurements of the corrected update rule.

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
