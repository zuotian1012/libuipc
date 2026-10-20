# Rules — Owner-Mandated Working Agreements

Rules the project owner (the user) has explicitly laid down. They override
habit and generic best practice. When one applies to your task, follow it;
when a rule is marked task-scoped, do not generalize it.

Standing rule for this file: **whenever the owner sets or changes a rule,
record it here in the same commit.**

## Change discipline

Task-scoped update (2026-10-20): the owner now explicitly requests small-scale
tests to isolate the SoS runtime/convergence problems. Prioritize controlled
diagnostics with independent references before more full dense runs; this
supersedes the earlier implementation-first preference for this investigation.

1. **Commit to the current working branch only.** Never open a PR to (or
   push to) the default branch (`main`) unless the owner explicitly says
   so. (Set 2026-08-23; current working branch at the time:
   `refactor-main`.)
2. **Commit in stages, push without asking.** During multi-step work,
   commit each completed stage separately (clear rollback points) and push
   directly — no per-commit confirmation. (Set 2026-08-23.)
3. **Minimal diffs**: make the smallest change that achieves the goal;
   reuse the project's existing algorithm/utility code instead of
   introducing parallel implementations; no opportunistic refactors,
   renames, or cleanups outside the task's scope.
4. **Keep agent_docs current in the same change.** Any update that changes
   structure, workflow, conventions, build, or behavior must update the
   matching doc (and `handoff.md` / `09-known-issues-and-roadmap.md` when
   it closes or opens work) **in the same commit**.
5. **Commit sample changes directly to `libuipc-samples/main`.** Do not use an
   intermediate samples branch unless the owner explicitly requests one. When
   the shared submodule worktree contains unrelated local edits, use a clean
   worktree to update `main` and never include those edits. (Set 2026-09-07.)

## Language

4. **English only in repo artifacts**: documentation, code comments, commit
   messages. (The conversation language stays whatever the owner uses.)
   Exception (2026-10-17): the owner requests mixed Chinese/English in the
   `sos` branch's project homepage and SoS onboarding guide for internal
   collaborators. This applies to `README.md` and `README_SOS.md`; other
   artifacts retain the English convention.

## Documentation scope

5. **Demo showcases belong on the project homepage, not in the technical
   documentation.** Keep video/demo catalogues in the root README or project
   website; keep `docs/` focused on tutorials, reference, specifications,
   build guidance, and development documentation. The documentation must
   expose a global link back to the project repository. (Set 2026-08-24.)
   Scene-configuration reference material must state every registered key's
   default, unit, effective-value rule, and valid domain/selector where one is
   defined. Representative simulation tutorials must cover rigid bodies, FEM,
   cloth, rigid-soft coupling, and contact in both C++ and Python, verified
   against `libuipc-samples`, public APIs/bindings, and the consuming backend
   implementation. Educational examples are documentation; showcase/demo
   catalogues remain homepage-only. (Expanded 2026-08-24.)

## Build system

6. **clang-format-18 before every C++ commit/PR**: run
   `python scripts/format_changed.py` (formats the C++ files changed vs the
   base ref, CI's exact selection) or `--check` for a dry run. CI rejects
   non-conforming diffs; the pinned local formatter is
   `output/venv_clangfmt/Scripts/clang-format.exe` (18.1.8, same major as
   CI's clang-format-18).
7. **CMake and XMake stay in sync**: every build-affecting change (sources,
   flags, dependencies, pins) must be applied to both build systems, even
   if only one can be verified locally.
8. **No ccache.** It was integrated once and the owner had it reverted; do
   not re-introduce it (or similar compiler-cache layers) without an
   explicit request.

- **Use only the embedded C++ METIS target.** Mesh partitioning must use
  `src/geometry/metis/` through `uipc_metis`; do not restore separate
  `external/METIS` or `external/GKlib` source trees or targets. Keep the CMake
  and XMake source boundary equivalent. (Set 2026-09-03.)

## Geometry

- **Native tetrahedralization belongs in `src/geometry/tetrahedralization/`.**
  Provide strict surface preservation and quality-oriented modes. Strict mode
  must keep original vertices, coordinates and triangle connectivity, with no
  boundary subdivision. Construct/retain a conservative valid mesh when quality
  optimization cannot improve it; do not return a changed boundary or fail merely
  because optimization did not succeed. (Set and clarified 2026-09-05.)

## CUDA backend

9. **Raw kernels only**: business GPU code is written as named
   `__global__` functions with `<<<>>>` launches — no lambda kernels, no
   ParallelFor-style wrappers.
10. **`cuda_tool` is the only device utility layer and must not contain
   muda** (no vendored copy, no submodule, no compat header). Keep
   cuda_tool minimal: no primitive without an in-tree user.
11. **Eigen stays** as the host/device small-matrix dependency. (A
   replacement experiment — geigen under `src/math/` — was tried and
   deliberately reverted; do not resurrect it without an explicit request.)

12. **Use CUB, not Thrust, for new or modified CUDA bulk algorithms.** CUB
   temporary storage must be persistent and reused (normally through
   `cuda_tool`'s per-stream workspace); never allocate/free CUB scratch on
   every call. (Set 2026-08-25.)

13. **Make CUDA buffer growth semantics explicit.** Keep `resize()` for
   value-initialized state. Use `resize_discard()` only for output that the
   following kernel/copy/CUB call completely regenerates, and use
   `resize_preserve()` only when the old logical range must survive but the new
   range may remain uninitialized. Growth capacity must be based on the latest
   requirement with deliberate headroom; do not copy or initialize dead output
   ranges merely to emulate `std::vector`. (Set 2026-08-30.)

14. **Profile heterogeneous kernel fusion/splitting end to end.** Static
    register or stack reductions do not prove a speedup. In particular, keep
    IPC simplex PT/EE/PE/PP Hessian assembly in one launch unless both the
    kernel profile and enclosing DyTopo timer prove a split is faster: rare
    PT/EE threads are expensive and overlap the dominant PE population in the
    fused launch. Compile-time specialization of a uniform mode such as
    `gradient_only` is preferred when it removes dead work without serializing
    stencil families. (Set 2026-08-30.)

15. **Use pipeline-specific parallel-EE handling.** AL-IPC never classifies
    parallel EE pairs: both normal and frictional contact use the shared
    negative disabled threshold, making `need_mollify()` false and routing all
    AL pairs through their ordinary EE paths. Standard IPC uses the positive
    `1e-3` coefficient. Its normal-contact path evaluates the complete
    mollified energy, gradient, and Hessian for detected parallel pairs; its
    friction path skips detected parallel pairs. Do not make `need_mollify()`
    globally constant. (Set 2026-09-03; refined 2026-09-03.)

16. **Do not serialize or impose fixed-order floating-point accumulation merely
    to eliminate run-to-run rounding differences.** Preserve the existing parallel
    architecture and efficiency. Small numerical differences caused solely by
    rounding under different valid atomic-add accumulation orders are expected,
    not numerical correctness defects. Distinguish permitted atomic arrival-order
    variation from violations of required parallel execution dependencies. A
    correctness fix must address a demonstrated algorithmic error or an actual
    concurrency/synchronization defect (for example, lost contributions, data
    races, missing synchronization, out-of-bounds access or uninitialized reads).
    Do not add sorting, fixed-order reductions or serialization merely to obtain
    bitwise reproducibility without an explicit requirement. (Set and clarified
    2026-09-11.)

## Task-scoped (recorded for context, not general policy)

- SoS investigation order (2026-10-20): modify and validate warm-start handoff
  first, then investigate global error correction. Preserve controlled small
  comparisons and distinguish initializer quality from final convergence.

- SoS iteration budget (2026-10-18): add an opt-in mode accepting the last
  checked full-elastic endpoint at the outer iteration limit, then run bounded
  comparisons. Preserve default strict behavior and all geometry/line-search
  failure handling; label truncated frames as unconverged and record residuals.

- SoS collaboration (2026-10-17): the owner requested a `sos` branch in the
  existing GitHub repository for sharing the implementation with collaborators.
  Publish the committed SoS history and onboarding documentation there; preserve
  `main` and unrelated uncommitted experiments.

- Complete SoS reproduction (2026-10-08): the owner now requests the complete
  paper algorithm before another performance comparison. This supersedes the
  earlier staged exclusions of custom barriers, conservative history, warm start
  and CSFD. Preserve previous controls; implement a coherent objective across
  derivatives, warm start, local/global acceptance and convergence evaluation.
  Do not declare completion from partial modules or changed tolerances.

- SoS contact execution follow-up (2026-10-07): continue from the ablation
  results by measuring local/global checking costs, reducing repeated work and
  comparing a triangle-mesh U channel. Preserve strict accuracy, safety checks
  and existing baseline artifacts; distinguish implementation gains from paper
  algorithm alignment.

- Single-bunny ablations (2026-10-07): execute the authorized floor/U-channel
  comparisons of Jacobi, GS and hybrid stencil updates with regular/local CCD.
  Keep physical controls and stopping criteria matched, record failures honestly,
  and preserve the ongoing dense 120-frame validation.

- SoS relaxation diagnosis (2026-10-06): locate why local updates need thousands
  of passes and actually verify the complete dense 120-frame bunny/cloth landing.
  A started run or a partial trajectory is not a completed validation.

- SoS execution alignment (2026-10-06): align explicit stopping metrics,
  complementary coverage and local/global iteration organization. Keep the
  existing dense checkpoints as controls and compare the same contact frame;
  distinguish scheduling gains from changed accuracy. Do not substitute a
  larger iteration budget or an unlabeled relaxed tolerance for the fix.

- SoS hybrid alignment (2026-10-01): align local direction computation, priority
  submission and Jacobi merging with the paper, and audit local line-search
  consistency. Keep direction solves independent of coloring; verify against the
  previously failing dense scene without relaxing convergence or changing its physics.

- Bunny/cloth scale study (2026-09-30): increase both cloth and bunny simulation
  mesh density and expose command-line SoS/ordinary-solver selection. Keep the
  implemented SoS subset explicitly distinguished from the complete paper method.

- SoS cloth extension (2026-09-30): complete cloth membrane and bending support
  in the existing local-stencil solver so cloth demos can support the subsequent
  algorithm stages. Preserve the implementation-first workflow, using complete
  simulations for integration verification rather than expanding many small tests.

- SoS colored updates (2026-09-30): implement complementary elastic-stencil
  coloring, priority-preserving hybrid sweeps, per-stencil CCD and local line
  searches following the paper's conflict policy. Keep the implementation-first
  workflow and validate with complete simulations. Warm start remains separate.

- SoS implementation priority (2026-09-29): complete the usable algorithm and
  scene-to-animation workflow before expanding small isolated tests. Keep only
  necessary integration verification. The next authorized stage adds frictionless
  contact assembly, contact stencils, CCD and dynamic contact adjacency, starting
  with soft-body ground impact and then body-body contact. Coloring and warm start
  remain later work. (Scope expanded by the owner on 2026-09-29.)

- SoS reproduction, stage 1 (2026-09-28): implement the local stencil
  second-order solver incrementally. Do not bundle the paper's contact,
  complementary coloring, hybrid sweep, warm-start or other complex algorithms
  into this stage. Start with an independently testable non-contact building block.

- 2026-09-11 project review: inspect implementation, frontend design, public APIs
  and documentation; update the documentation and record findings, but do not
  implement fixes or optimizations during this audit. Distinguish confirmed issues
  from risks/proposals and keep unrelated work unchanged.

- Dining-scene reconstruction: ground is `z = 0`; tabletop objects may begin
  separated and suspended, with final stacking and tablecloth drape obtained
  through libuipc gravity/contact simulation, not manually posed final geometry.
  (Set 2026-09-05.)

- Blender integration: use `spiriMirror` for the new extension's copyright
  attribution and maintainer. (Set 2026-09-05.)

- Blender cloth parameters: stretch, shear and bending each have independent
  Young's modulus and Poisson ratio controls. Preserve the previous shared
  Poisson value when loading old scenes; changing one channel must not change
  either of the others. Keep the existing full-thickness `2r` material formulas.
  (Set 2026-09-10.)

- During the Stiff-GIPC performance-alignment work: "when a design choice
  is uncertain, follow Stiff-GIPC's algorithm design" — that applied to
  that effort (and its evidence lives in `handoff.md`), not to unrelated
  features.
- During CI watch duty: "you confirm all operations yourself" — scoped to
  that autonomous monitoring task; normal confirmation rules apply
  elsewhere.
