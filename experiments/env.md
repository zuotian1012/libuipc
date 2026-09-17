# Contact bottleneck probe environment (2026-09-16)

- Repository baseline: `a037f41927ac46eec9aec064ffe92fb215a182bb` plus the research-only instrumentation in this working tree.
- Working tree before the probe: clean. The probe itself changes `implementation.md`, one simulation case, the simplex trajectory filter, and one runner script.
- GPU: NVIDIA GeForce RTX 3060 Ti, 8192 MiB; driver 591.86; Windows display GPU.
- CUDA toolkit: 12.6, `nvcc` V12.6.85.
- OS: Windows NT 10.0.26200.0.
- Compiler/build: XMake 3.1.1, release mode, CUDA backend, `UIPC_RUNTIME_CHECK=1`.
- Python: 3.11.8.
- Contact model: IPC, friction disabled; 8x8 and 16x16 quad grids triangulated into 128 and 512 triangles per sheet; 1, 2, or 4 parallel sheets.
- Simulation: 40 frames, gravity `(0,-9.8,0)`, ground at `y=0` only when contact is enabled; all other solver settings are identical within each matched control/contact pair.
- Repetition: three fresh processes per condition, frames 6–40 used for per-run median frame time; the reported median is the median of the three run medians.
- Correctness checks: valid world, completed and converged frame stats, no strict-mode exception, minimum `y >= -1e-5` for contact runs, matching-vertex layer gap `>= -1e-5`, and vertical slab separation (minimum height of each upper sheet minus maximum height of the lower sheet) `>= -1e-5` for multi-layer contact runs. Slab separation proves these horizontal sheets do not intersect, but these checks do not replace a general triangle-intersection test for other geometries.
- The final 36-run throughput batch used per-process frame-data files (`UIPC_PROBE_CSV_PATH`) to avoid interleaving CSV rows with CUDA console logging.
- Runtime dependency note: the prebuilt XMake executable required `spdlog.dll` from XMake's package cache and `tbb12.dll` from `C:/Program Files/Adobe/Adobe Photoshop 2026` on this host. They were supplied through the process `PATH`; no DLL was copied into the repository. This is a portability caveat for reproduction.
- Stage timing: separate runs with `UIPC_PROBE_TIMERS=1`; synchronized timers change wall time. Pair-count and pair-churn diagnostics are also separate from throughput measurements.
- Sample submodule: large `libuipc-samples` benchmark assets are absent locally, so the preliminary study uses an in-repository procedural scene.

Rebuild:

```powershell
xmake build -j4 sim_case
```

Throughput probe, supplying the two runtime directories above through `--runtime-dir`:

```powershell
python scripts/run_contact_density_probe.py --grid 8 --out experiments/probe --runtime-dir <spdlog-bin> --runtime-dir <tbb-bin>
python scripts/run_contact_density_probe.py --grid 16 --out experiments/probe_grid16 --runtime-dir <spdlog-bin> --runtime-dir <tbb-bin>
```

Raw per-frame CSV, run summaries, and original logs are in the corresponding output directories.

Follow-up diagnostics use the same binary and runtime directories:

```powershell
python scripts/run_contact_refresh_diagnostic.py --runtime-dir <spdlog-bin> --runtime-dir <tbb-bin>
python scripts/run_toi_speed_sweep.py --runtime-dir <spdlog-bin> --runtime-dir <tbb-bin>
python scripts/run_toi_speed_sweep.py --layers 4 --speeds 0 2 --frames 20 --timers --out experiments/next_step/impact_stack_speed_sweep --runtime-dir <spdlog-bin> --runtime-dir <tbb-bin>
python scripts/run_toi_speed_sweep.py --layers 4 --speeds 0 2 --frames 20 --timers --no-toi-hist --out experiments/next_step/impact_stack_stage_sweep --runtime-dir <spdlog-bin> --runtime-dir <tbb-bin>
```

The first runner sets `UIPC_PROBE_PAIR_CHURN`, `UIPC_PROBE_DCD_CORE_TIMES`,
`UIPC_PROBE_TOI_HIST`, and `UIPC_PROBE_TIMERS`, then analyzes the full-baseline
log and raw timers. This is a synchronized, host-copy diagnostic, not a
throughput run. `UIPC_PROBE_INITIAL_DOWN_SPEED` sets the optional initial
downward cloth velocity in m/s. See `next_step/2026-09-16-followup.md` for
the exact cases and conclusions.
