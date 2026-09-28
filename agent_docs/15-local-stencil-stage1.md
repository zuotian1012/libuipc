# Local stencil second-order solve: stage 1

This is an independently testable backend building block, not a new Scene
configuration or a replacement for `World.advance()`. Scope was explicitly
limited to the local solver on 2026-09-28. Contact, coloring, hybrid sweeps,
warm starts, CSFD and large paper demos are deferred.

## Implemented contract

- `src/backends/cuda/finite_element/local_stencil.h` restricts incident
  element gradients and Hessians to a four-vertex, 12-DOF stencil; adds lumped
  inertia once per local vertex; and eliminates fixed-vertex increments.
- The caller must supply **all** incident energy contributions exactly once,
  with valid, unique vertex IDs. Shared vertices, edges and faces all count.
  Do not sum the resulting overlapping stencil energies as a global objective.
  Vertex-local indices must be in `[0, 4)`. Accumulate first, constrain last.
- Newton-CG starts from zero and takes at most 12 iterations. It uses the
  unprojected Hessian. Nonpositive curvature terminates CG, as allowed in the
  alternative discussed in section 4.1 of Lan et al., SIGGRAPH 2023
  ([paper](https://wanghmin.github.io/publication/lan-2023-sos/Lan-2023-SOS.pdf)).
  If no descent step has accumulated, the direction falls back to `-gradient`.
  This is not Algorithm 1's sigma-based curvature replacement.
- Relative/absolute CG residual tolerances default to `1e-10`/`1e-12`.
  The explicit residual reports convergence separately from validity; reaching
  the iteration budget is not an exact-solve certificate.
- Armijo backtracking uses coefficient `1e-4`, initial alpha 1, halving, and
  at most 32 trials. Failure returns `accepted=false` and zero displacement.
  A stationary stencil also returns no accepted update; consult the direction's
  `converged` flag. The energy functor freezes all external DOFs and returns
  infinity for inadmissible trials. Callers may apply only accepted updates.
- `local_stencil_snk.h` reuses libuipc's SNK1 energy/stress and supplies their
  **unprojected** analytic directional stress derivative. It constructs the
  explicit 12-by-12 element Hessian using the existing shape-gradient utility.
  Energy scaling is `rest_volume * dt^2`, with separate lumped inertia.

The first implementation deliberately stores local matrices to make derivatives
and neighbor gathering inspectable. It does not reproduce the paper's matrix-free
HVP/CSFD implementation or its GPU scheduling. It uses libuipc's current SNK1
material as a controlled baseline, not a claim of matching all paper materials.
The production projected Hessian and existing solver paths are unchanged.

## Verification and runnable experiment

`apps/tests/backends/cuda/local_stencil.cu` is registered by the existing
recursive source globs in **both** the CMake and XMake CUDA test targets.
No source-manifest or build-option additions are needed for these header-only
backend utilities. XMake compilation/execution is checked locally; CMake source
inclusion is inspected, not a second configured build.

```powershell
xmake build -j 4 backend_cuda
# This checkout's executable needs its existing packaged TBB/spdlog DLLs.
# Add the directory to this process only; do not replace installed libraries.
$env:PATH = (Resolve-Path 'build/python/src/uipc/_native').Path + ';' + $env:PATH
& 'build/windows/x64/release/uipc_test_backend_cuda.exe' '[stencil]'
```

The test suite checks full-energy finite-difference gradients and Hessians,
neighbor gathering and restriction against a global system, fixed vertices,
positive-definite CG against LDLT, first/later negative curvature, invalid
numeric input, tiny residuals, failed/backtracked searches, and CPU/GPU agreement
for 17 independent stencil proposals. Rest, shear and strong compression check
the directional stress derivative; compression must retain negative eigenvalues.

The coupled experiment uses two tetrahedra and five vertices with one pinned
vertex. It minimizes a **single non-contact incremental potential**, rather than
advancing a full animated Scene. Sequential host stencil sweeps are used solely
as a small correctness oracle. A dense global Newton solve of the same energy
provides the reference. There is no production serialization or global solver
change. CUDA runs the assembly, CG and line search for independent proposals;
it does not run the multi-sweep simulation schedule.

Validated on 2026-09-28 with XMake release, CUDA 12.6 and an RTX 3060 Ti:
7 test cases / 136 assertions passed. The patch required 3 local sweeps versus
4 dense Newton iterations; final free-gradient norm was `1.92238e-13`, maximum
vertex difference `2.66579e-13`, and energy decreased from `0.111828` to
`0.0867332`. These counts compare different amounts of work, not performance.
The three added C++/CUDA files passed clang-format 18.1.8's dry-run check.

The experiment checks monotonic energy decrease, a free-gradient norm below
`1e-8`, fixed-point preservation and maximum vertex error below `1e-7` relative
to the dense solve. Its energy callback rejects nonpositive endpoint determinants
only; **it does not certify continuous inversion freedom or collision freedom**.

## Next bounded step

After this stage is reviewed, add a small non-contact mesh driver and a clearly
defined update schedule, reusing these derivatives and solver diagnostics. Then
validate a beam against the same material/energy/tolerances. Contact and the other
SoS components should remain separate stages.
