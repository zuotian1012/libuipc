# SoS reproduction on libuipc

Collaborative research implementation of
[Lan et al., DOI 10.1145/3592104](https://dl.acm.org/doi/10.1145/3592104)
on libuipc's CUDA backend. This is not the authors' reference implementation.

Repository: [zuotian1012/libuipc, sos branch](https://github.com/zuotian1012/libuipc/tree/sos).
See [README.md](README.md) for the original library overview.

## Status

Implemented components include local Newton-CG, complementary coloring,
GS/Jacobi hybrid submission, local CCD and line searches, cloth membrane and
bending, custom distance barriers, conservative contact history, inertia/barrier
warm start, and packed GPU Hessian-vector products. Their presence does not
certify complete paper alignment or a speedup.

The recorded checkpoint `69340034` passed 202 stencil assertions and a 120-frame
cloth trajectory. Its subsequent fresh dense bunny/cloth build29 run completed
38 frames, then hit the 4096-outer-round cap in frame 39. It did not complete
the requested 120-frame dense validation. Earlier trajectories used different
configurations and do not validate this configuration.

Open work includes slow propagation of global deformation error, merged-update
oscillation, contact-history restarts, warm-start handoff, and comparable
stopping accuracy. The local solver is frictionless and FEM-focused; it does
not inherit all native libuipc rigid-body and friction features. No
paper-equivalent speedup is established.

## Clone and build

```shell
git clone --branch sos --recurse-submodules https://github.com/zuotian1012/libuipc.git
cd libuipc
git rev-parse HEAD
```

Keep the full repository: the stencil sources depend on the engine, materials,
configuration and contact infrastructure. Preserve the original license and
attribution files; assets and dependencies retain their own provenance.

**Build this branch from source. Installing PyPI `pyuipc` alone does not install
these changes.** The local SoS experiments used Windows, CUDA 12.6 and an RTX
3060 Ti. A clean installation on every platform has not been verified.

For Windows, install Visual Studio C++ Build Tools, CUDA Toolkit, a compatible
NVIDIA driver, Python 3.11 and XMake >=3.0.5; use a developer shell. Source-build
documentation specifies CUDA >=12.4. Ensure `python` and `python3` select the
same Python 3.11 installation for configuration and execution.

```shell
python --version
python3 --version
xmake --version
nvcc --version
python -m pip install numpy matplotlib "polyscope>=2.5,<3" huggingface_hub
xmake f -m release --backend_cuda=true --pybind=true --python_system=true --python_version=3.11.x
xmake build -j 4 pyuipc
```

The Python target builds its backend dependencies and copies the package to
`build/python/src/uipc`. See [XMake](docs/build_install/xmake.md),
[Windows/CMake](docs/build_install/windows.md) and
[Linux/CMake](docs/build_install/linux.md) for build details. Use the same Python
minor version for the compiled extension and runtime.

The examples prefer the locally built package. Verify the import path from the
repository root:

```shell
python -c "import sys; sys.path.insert(0, 'python/examples'); import local_stencil_beam; import uipc; print(uipc.__file__)"
python python/examples/local_stencil_bunny_cloth.py --help
```

The printed path should be inside this checkout's `build/python/src/uipc`.
`UIPC_LOCAL_PACKAGE` can select another built package root containing `uipc/`;
avoid leaving it pointed at an old experimental runtime. After C++ changes,
rebuild and refresh the packaged native libraries before measuring results.

## First runs

Run from the repository root, using a new output directory for each experiment.
These short commands are onboarding checks, not completed landing experiments.

```shell
# Small non-contact integration check.
python python/examples/local_stencil_beam.py --solver local_stencil --nx 4 --frames 3 --output output/onboarding-beam

# Inspect mesh sizes without running a simulation.
python python/examples/local_stencil_bunny_cloth.py --cloth-resolution 8 --bunny-refinement 0 --mesh-info

# Native and stencil controls on the same small physical scene.
python python/examples/local_stencil_bunny_cloth.py --solver global --cloth-resolution 8 --bunny-refinement 0 --dt .001 --frames 3 --output output/onboarding-global
python python/examples/local_stencil_bunny_cloth.py --solver sos --cloth-resolution 8 --bunny-refinement 0 --dt .001 --frames 3 --output output/onboarding-sos
```

`--solver sos` selects the local solver; it does not enable all paper-related
options. Control defaults include assembled products, IPC barriers and no warm
start. An explicit experimental configuration is:

```shell
python python/examples/local_stencil_bunny_cloth.py --solver sos --cloth-resolution 8 --bunny-refinement 0 --dt .001 --frames 3 --hybrid-submission sequential --hessian-product packed --barrier-model sos --barrier-activation expanded --barrier-stiffness ipc --merge-acceptance local --warm-start --warm-inversion after_local --warm-max-iterations 10 --output output/onboarding-sos-custom
```

Expanded activation, IPC-based stiffness calibration and a bounded warm
initializer are implementation choices, not an author-exact preset. Native
and custom-barrier runs also use different contact objectives; timing alone
is not a fair algorithm comparison.

Replay a completed trajectory:

```shell
python python/examples/local_stencil_bunny_cloth.py --solver sos --replay --gui --output output/onboarding-sos
```

Bunny/cloth outputs live under `global/` or `local_stencil/` inside the output
root. Depending on the example, outputs include `history.csv`, `summary.json`,
`positions.npz`, scene configuration, checkpoints and OBJ frames. An
iteration-limit exit or a partial trajectory is not successful completion.

## Code map and reading order

Core files below are in `src/backends/cuda/finite_element/` unless stated otherwise.

| Responsibility | Files |
|---|---|
| Sweep loop, hybrid submission, warm lifecycle, stopping | `fem_local_stencil_solver.h/.cu` |
| Local Newton-CG and constraints | `local_stencil.h` |
| Packed GPU products and CG | `local_stencil_packed.h/.cu` |
| Direct operator and CSFD derivatives | `local_stencil_operator.h`, `local_stencil_csfd.h` |
| Complementary coloring and GS groups | `fem_local_stencil_coloring.cu` |
| Local CCD, line search, merged acceptance, energy | `fem_local_stencil_linesearch.cu` |
| Solid/cloth data and derivatives | `fem_local_stencil_elastic.cu`, `local_stencil_elastic*.h`, `local_stencil_snk.h` |
| Continuous inversion and degeneracy bounds | `local_stencil_inversion.h` |
| Contact assembly and local restriction | `fem_local_stencil_contact.cu`, `local_stencil_contact.h` |
| Barrier terms and objective | `fem_local_stencil_barriers.cu`, `local_stencil_barrier.h`, `sos_barrier.h`, `sos_contact.h` |
| Contact history and time-step integration | `fem_local_stencil_history.cu`, `sos_barrier_history.h/.cu` |
| Engine integration | `src/backends/cuda/engine/advance_ipc.cu` |
| Configuration registration | `src/core/core/scene_default_config.cpp` |

Read the scene script, then follow `solve()` -> `solve_sweep()` ->
`submit_sequential_hybrid()`. Track positions, raw local directions, submitted
directions and residuals separately. A local CG result is a proposal, not an
accepted vertex update. Study GPU launch details after this control flow.

Entry points:

- [Bunny and cloth](python/examples/local_stencil_bunny_cloth.py).
- [Cloth](python/examples/local_stencil_cloth.py).
- [Single-bunny ablations](python/examples/local_stencil_bunny_ablation.py).
- [Trajectory validation](python/examples/validate_stencil_bunny_cloth.py).
- [Configuration reference](docs/specification/scene_config.md).
- [Paper-alignment record](agent_docs/25-sos-paper-completion.md).

Documents 15-24 in `agent_docs/` describe historical stages, not one uniform
current configuration. Output paths mentioned in research notes refer to local
artifacts; they are not automatically available in a Git clone.

## Collaboration and experiment protocol

Create a focused topic branch from `sos`:

```shell
git switch sos
git pull --ff-only origin sos
git switch -c fix/local-linesearch
```

Submit pull requests targeting `sos`, describing the problem, changed rule,
exact command and validation result. Contributors with repository write access
can push topic branches; others can work through their own fork. Coordinate
edits to the shared integration points `fem_local_stencil_solver.cu` and
`advance_ipc.cu`. Useful task boundaries are local operators, submission/line
search, contact history/warm start, and experiments/documentation.

For comparisons, record the commit and any uncommitted diff, full command,
GPU/driver/toolkit, loaded runtime, physical parameters, energy model, stopping
metric and threshold, completed frames, iteration counts and timing. Separate
local CG iterations, color passes and outer rounds. Native Newton displacement
tolerance and SoS mass-scaled gradient tolerance are not equivalent accuracy
criteria even when their numeric thresholds match.

`build/` and `output/` are ignored. Share needed checkpoints and large result
bundles separately with their configuration and commit identifier. Stage only
intended source/documentation files, rather than using `git add .` on a working
tree containing unrelated experiments.

Targeted tests live in `apps/tests/backends/cuda/local_stencil.cu` and
`local_stencil_cloth.cu`. Validate the changed behavior and a relevant integration
scene. Follow [repository agreements](agent_docs/rule.md): English repository
artifacts, clang-format 18 for C++, and consistent CMake/XMake source ownership.
