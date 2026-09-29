# 16 - Local stencil mesh solver

This stage connects the [local Newton-CG building block](15-local-stencil-stage1.md)
to the CUDA engine and provides a complete non-contact cantilever simulation.
It is an incremental reproduction of second-order stencil descent, not the
paper's full SoS algorithm or its benchmark scenes.

## Algorithm

For BDF1, minimize `E(x) = sum_i m_i/2 ||x_i-x_tilde_i||^2 + dt^2 sum_e V_e psi(F_e)`.
Each outer iteration does the following on the GPU:

1. Evaluate each tetrahedron's SNK1 gradient and unprojected analytic Hessian.
2. Assign one thread to each tetrahedral stencil (12 position coordinates).
   Restrict inertia and **all** incident tetrahedral energies to those coordinates.
   Apply fixed constraints after assembly, then use truncated Newton-CG with
   negative-curvature handling and a steepest-descent fallback.
3. Gather overlapping local directions into vertex displacements using one
   uniform factor `1 / maximum_vertex_stencil_degree`. A uniform positive factor
   preserves descent of the summed restricted directions; arbitrary per-vertex
   averaging does not provide that guarantee. All stencils use the same iterate.
4. Reuse the engine's global energy backtracking. Require positive finite
   `det(F)` at accepted trial endpoints. Restore the last accepted positions and
   fail explicitly if backtracking is exhausted.
5. Stop when the maximum free-vertex `||gradient_i|| / (m_i dt)` is at most
   `newton/velocity_tol`, measured in m/s. This differs from the default global
   solver's displacement-based stopping criterion. `newton/max_iter` counts
   complete stencil sweeps; global linear solver iterations remain zero.

The incidence and stencil-neighbor lists are built once. Element derivatives are
shared by overlapping stencils, and CUB reductions reuse cuda_tool workspace.
CMake and XMake both already glob the new finite-element source files.

## Run

Build the core and CUDA backend and refresh the checkout's Python runtime
(`xmake build pyuipc`). From the repository root:

```text
python python/examples/local_stencil_beam.py --frames 120
python python/examples/local_stencil_beam.py --gui
python python/examples/local_stencil_beam.py --solver global --frames 120
```

The example prefers `build/python/src` when present. It creates a 1 x 0.25 x
0.25 m beam, fixes the x=0 end, and advances it under gravity using
`World.advance()`. Defaults: 153 vertices, 384 tetrahedra, E=500 kPa, nu=0.3,
rho=1000 kg/m^3, dt=0.01 s. It exports OBJ frames, `positions.npz`, `history.csv`
and `summary.json` beneath `output/local-stencil-beam/<solver>`.

For another scene, set `config["newton"]["solver"] = "local_stencil"`, disable
contact, and use only StableNeoHookean tetrahedra with optional fixed vertices.
Only BDF1, static topology and non-differentiable simulation are supported;
incompatible configurations fail explicitly. Relative velocity tolerance must
be zero. Semi-implicit early termination is bypassed for this solver.

## Limits and next stages

Integration run (CUDA 12.6 / XMake release): the default beam completed 120
converged frames with zero fixed-end displacement and minimum det(F)=0.784865.
A matching global-solver run has maximum vertex difference 4.10436e-5 m across
the trajectory. The solvers use different termination criteria. The initial
additive implementation took about 0.86 s/frame versus 0.007 s/frame for the
global solver on this small scene; these are indicative timings, not a controlled
benchmark or a paper performance comparison. Output includes a rendered
`local_stencil/beam.gif` and trajectory data under `output/local-stencil-beam/`.

This is synchronous **additive** stencil descent with global line search. It
does not implement the paper's complementary coloring, hybrid Gauss-Seidel /
Jacobi schedule, CSFD, local line searches, contact barriers or warm start.
Endpoint determinant checks do not certify the entire update segment against
element inversion. They do not replace the paper's continuous inversion search.
The dense 12x12 analytic Hessians prioritize a readable complete implementation;
performance optimization and paper demo matching are separate future stages.
