"""Serial single-bunny update/CCD ablations with unchanged solve accuracy.

Use --suite for floor and analytic U-channel runs, or --case for one run.
Use --case --mode global for the existing global Newton solver; its native
displacement stopping rule differs from the stencil gradient criterion.
Set UIPC_LOCAL_PACKAGE to an isolated built Python runtime when another
simulation is using the checkout runtime. Timings include native logging.
"""
from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODES = {"jacobi": ("jacobi", "regular", 1),
         "gs": ("gs", "regular", 1),
         "hybrid": ("hybrid", "regular", 1),
         "hybrid-local": ("hybrid", "local", 1),
         "hybrid-batched": ("hybrid", "local", 15)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def channel_mesh(width, angle, subdivisions):
    """Connected zero-contact-thickness triangle surface of the analytic channel."""
    height, depth = .65, .75
    corner = width / 2 + height * np.tan(angle)
    knots = np.array([[-corner, height], [-width / 2, 0.],
                      [width / 2, 0.], [corner, height]])
    profile = np.concatenate([np.linspace(a, b, subdivisions + 1)[:-1]
                              for a, b in zip(knots[:-1], knots[1:])] + [knots[-1:]])
    zs = np.linspace(-depth, depth, 2 * subdivisions + 1)
    x = np.array([[p[0], p[1], z] for p in profile for z in zs])
    faces = []
    for i in range(len(profile) - 1):
        for j in range(len(zs) - 1):
            a = i * len(zs) + j
            b, c, d = a + len(zs), a + 1, a + len(zs) + 1
            faces.extend([(a, c, b), (c, d, b)])
    return x, np.asarray(faces, dtype=np.int32)


def case(args):
    from local_stencil_bunny_cloth import boundary_faces, refine_bunny, write_obj
    from uipc import Scene, Engine, World, Logger, Timer, builtin, view
    from uipc.geometry import SimplicialComplexIO, tetmesh, trimesh, label_surface, label_triangle_orient, halfplane
    from uipc.constitution import StableNeoHookean, ElasticModuli, NeoHookeanShell, ElasticModuli2D
    from validate_stencil_bunny_cloth import check_surfaces

    out = args.output.resolve()
    if (out / "scene.json").exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    mesh = SimplicialComplexIO().read(str(ROOT / "assets/sim_data/tetmesh/bunny0.msh"))
    x = np.asarray(mesh.positions().view()).reshape(-1, 3).copy()
    tets = np.asarray(mesh.tetrahedra().topo().view()).reshape(-1, 4).copy()
    x, tets = refine_bunny(x, tets, args.bunny_refinement)
    x -= (x.min(axis=0) + x.max(axis=0)) / 2
    x *= args.bunny_height / np.ptp(x[:, 1])
    x[:, 1] += args.drop_height - x[:, 1].min()
    tet = x[tets]
    negative = np.linalg.det((tet[:, 1:] - tet[:, :1]).transpose(0, 2, 1)) < 0
    tets[negative, 1], tets[negative, 2] = tets[negative, 2].copy(), tets[negative, 1].copy()
    faces = boundary_faces(tets)
    tet = x[tets]
    rest_det = np.linalg.det((tet[:, 1:] - tet[:, :1]).transpose(0, 2, 1))
    if not np.isfinite(rest_det).all() or rest_det.min() <= 0:
        raise RuntimeError("Invalid rest tetrahedra")
    planes = [(np.zeros(3), np.array([0., 1., 0.]))]
    width = .65 * np.ptp(x[:, 0])
    angle = np.deg2rad(25.)
    if args.scene in {"u", "mesh-u"}:
        planes += [(np.array([-width / 2, 0., 0.]), np.array([np.cos(angle), np.sin(angle), 0.])),
                   (np.array([width / 2, 0., 0.]), np.array([-np.cos(angle), np.sin(angle), 0.]))]
    def gap(positions):
        return min(float(((positions - p) @ n).min()) for p, n in planes)
    if gap(x) <= args.contact_distance:
        raise RuntimeError("Initial bunny must be separated from every collider")
    sweep, globalization, inner = MODES.get(args.mode, MODES["hybrid"])
    config = Scene.default_config()
    config["dt"] = args.dt
    config["gravity"] = [[0.], [-9.8], [0.]]
    config["newton"]["solver"] = "global" if args.mode == "global" else "local_stencil"
    local = config["newton"]["local_stencil"]
    local.update(sweep=sweep, globalization=globalization, inner_sweeps=inner,
                 initial_guess="inertial", stopping="velocity", color_limit=15,
                 color_coverage="independent")
    config["newton"]["semi_implicit"]["enable"] = False
    config["newton"]["velocity_tol"] = args.tolerance
    config["newton"]["max_iter"] = args.max_iterations
    config["line_search"]["max_iter"] = 32
    config["contact"]["friction"]["enable"] = False
    config["contact"]["d_hat"] = args.contact_distance
    config["extras"]["strict_mode"]["enable"] = True
    Logger.set_level(Logger.Level.Info)
    if args.timers:
        Timer.enable_all()
    scene = Scene(config)
    scene.contact_tabular().default_model(0., 1e8)
    contact = scene.contact_tabular().default_element()
    solid = tetmesh(x, tets)
    label_surface(solid)
    label_triangle_orient(solid)
    StableNeoHookean().apply_to(solid, ElasticModuli.youngs_poisson(args.bunny_young, .3), 1000.)
    contact.apply_to(solid)
    slot, _ = scene.objects().create("Stanford bunny").geometries().create(solid)
    collider_slot = None
    collider_x, collider_faces = np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    if args.scene == "mesh-u":
        collider_x, collider_faces = channel_mesh(width, angle, args.collider_subdivisions)
        collider = trimesh(collider_x, collider_faces)
        label_surface(collider)
        NeoHookeanShell().apply_to(collider, ElasticModuli2D.youngs_poisson(1e6, .3), 1000., .001)
        view(collider.vertices().find(builtin.is_fixed))[:] = 1
        # Constitutive shell thickness supplies positive mass; the fixed contact
        # surface itself is zero-thickness, like the analytic-plane control.
        view(collider.vertices().find(builtin.thickness))[:] = 0.
        contact.apply_to(collider)
        collider_slot, _ = scene.objects().create("Fixed triangle U channel").geometries().create(collider)
        write_obj(out / "collider.obj", collider_x, collider_faces)
    else:
        for i, (point, normal) in enumerate(planes):
            collider = halfplane(point, normal)
            contact.apply_to(collider)
            scene.objects().create(f"Collider {i}").geometries().create(collider)
    runtime = Path(os.environ.get("UIPC_LOCAL_PACKAGE", ROOT / "build/python/src")) / "uipc/_native"
    manifest = dict(scene=args.scene, mode=args.mode, config=config,
                    vertices=len(x), tetrahedra=len(tets), surface_triangles=len(faces),
                    mesh_sha256=hashlib.sha256(x.tobytes() + tets.tobytes()).hexdigest(),
                    collider_vertices=len(collider_x), collider_triangles=len(collider_faces),
                    collider_sha256=hashlib.sha256(collider_x.tobytes() + collider_faces.tobytes()).hexdigest(),
                    runtime=str(runtime.resolve()),
                    runtime_sha256=hashlib.sha256((runtime / "uipc_backend_cuda.dll").read_bytes()).hexdigest(),
                    planes=[dict(point=p.tolist(), normal=n.tolist()) for p, n in planes],
                    channel_bottom_width=width, channel_wall_angle_degrees=25.,
                    stopping=("Native global Newton displacement tolerance plus CCD; not equivalent to the stencil gradient criterion"
                              if args.mode == "global" else "max free-vertex ||gradient||/(mass*dt), identical in every stencil mode"),
                    gs_scope="Full elastic and dynamic contact vertex-conflict coloring; CPU greedy, not GPU Vivace",
                    local_scope="Local CCD/Armijo with retained global certification; no append-only barrier history or paper warm start",
                    parameters={k: v for k, v in vars(args).items() if k not in {"output", "suite", "case", "wait_for", "repeats", "resume_suite"}})
    write_json(out / "scene.json", manifest)
    engine = Engine("cuda", str(out / "engine"))
    world = World(engine)
    start = time.perf_counter()
    world.init(scene)
    init_ms = 1000 * (time.perf_counter() - start)
    if not world.is_valid():
        raise RuntimeError("World initialization failed")
    timer_frames = []
    if args.timers:
        Timer.report_as_json()
    positions, rows, stats_all = [x.copy()], [], []
    np.save(out / "frames/0000.npy", x)
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for frame in range(1, args.frames + 1):
            start = time.perf_counter()
            world.advance()
            elapsed = 1000 * (time.perf_counter() - start)
            if args.timers:
                timer_frames.append(Timer.report_as_json())
            world.retrieve()
            stats = engine.frame_stats()
            current = np.asarray(slot.geometry().positions().view()).reshape(-1, 3).copy()
            if collider_slot is not None:
                fixed = np.asarray(collider_slot.geometry().positions().view()).reshape(-1, 3)
                if not np.isfinite(fixed).all() or np.max(np.abs(fixed - collider_x)) > 1e-12:
                    raise RuntimeError(f"Fixed collider moved at frame {frame}")
            np.save(out / "frames" / f"{frame:04d}.npy", current)
            tet = current[tets]
            determinants = np.linalg.det((tet[:, 1:] - tet[:, :1]).transpose(0, 2, 1)) / rest_det
            row = dict(frame=frame, advance_ms=elapsed, outer_iterations=stats["newton_iterations"],
                       converged=stats["converged"], min_det_F=float(determinants.min()), min_gap=gap(current))
            if not world.is_valid() or not np.isfinite(current).all() or row["min_det_F"] <= 0 or row["min_gap"] <= 0:
                raise RuntimeError(f"Invalid geometry at frame {frame}: {row}")
            if not stats["converged"] or stats["hit_newton_limit"] or stats["hit_line_search_limit"]:
                raise RuntimeError(f"Uncertified frame {frame}: {stats}")
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            stats_all.append(stats)
            positions.append(current)
    np.savez_compressed(out / "positions.npz", positions=np.asarray(positions), faces=faces,
                        tetrahedra=tets, dt=args.dt, collider_positions=collider_x, collider_faces=collider_faces)
    write_obj(out / "final.obj", positions[-1], faces)
    write_json(out / "frame-stats.json", stats_all)
    if args.timers:
        write_json(out / "timer_frames.json", timer_frames)
    # Independent saved-state self-intersection check, excluded from advance time.
    checks = [check_surfaces(p, faces, faces, 0., same=True) for p in positions]
    write_json(out / "surface-verification.json", checks)
    if collider_slot is not None:
        cross = [check_surfaces(np.concatenate([p, collider_x]), faces,
                                collider_faces + len(x), 0.) for p in positions]
        write_json(out / "collider-verification.json", cross)
    contact_start = next((r["frame"] for r in rows if r["min_gap"] < args.contact_distance), None)
    if contact_start is None:
        raise RuntimeError("No collider contact occurred; this is not a valid landing benchmark")
    contact_rows = rows[contact_start - 1:]
    summary = dict(passed=True, frames=args.frames, mode=args.mode, scene=args.scene,
                   init_ms=init_ms, total_advance_s=sum(r["advance_ms"] for r in rows) / 1000,
                   first_contact_frame=contact_start,
                   contact_mean_ms=float(np.mean([r["advance_ms"] for r in contact_rows])),
                   contact_median_ms=float(np.median([r["advance_ms"] for r in contact_rows])),
                   max_outer_iterations=max(r["outer_iterations"] for r in rows),
                   min_det_F=min(r["min_det_F"] for r in rows), min_gap=min(r["min_gap"] for r in rows))
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def parse_trace(folder, frames, tolerance):
    current = None
    traces = {}
    for line in (folder / "simulation.log").open(errors="replace"):
        if ">>> Begin Frame:" in line:
            current = int(line.split("Begin Frame:")[1])
            traces[current] = dict(velocity=[], gradient_l2=[])
        if current is not None:
            if "Local stencil velocity residual:" in line:
                traces[current]["velocity"].append(float(line.split("residual:")[1].split()[0]))
            elif "Local stencil gradient L2:" in line:
                traces[current]["gradient_l2"].append(float(line.split("L2:")[1]))
    for frame in range(1, frames + 1):
        values = traces.get(frame, {}).get("velocity", [])
        if not values or not np.isfinite(values[-1]) or values[-1] > tolerance:
            raise RuntimeError(f"Missing/invalid final residual at frame {frame}")
    write_json(folder / "convergence.json", traces)
    return {str(frame): len(value["velocity"]) for frame, value in traces.items()}


def wait_for_prior(root, prior):
    # Use a Windows synchronization handle to distinguish a slow live run from
    # a stale status file. Never kill or restart the earlier experiment here.
    record = json.loads((prior / "run-process.json").read_text())
    if os.name != "nt":
        raise RuntimeError("--wait-for currently requires Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    # Wait for simulation and verifier/rendering, so no other task uses the GPU.
    for key in ["process_id", "verifier_pid"]:
        handle = kernel.OpenProcess(0x00100000, False, int(record[key]))
        if not handle:
            if ctypes.get_last_error() == 87:  # Process already exited.
                continue
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            while True:
                write_json(root / "suite-status.json", dict(state="waiting_for_prior_run", prior=str(prior), process_id=record[key], updated_unix=time.time()))
                result = kernel.WaitForSingleObject(handle, 15000)
                if result == 0:
                    break
                if result != 258:
                    raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)
    prior_status = json.loads((prior / "completion-status.json").read_text())
    if prior_status["state"] != "complete":
        raise RuntimeError(f"Prior validation needs attention before timing ablations: {prior_status}")


def suite(args):
    root = args.output.resolve()
    if root.exists() and (root / "suite.json").exists() and not args.resume_suite:
        raise FileExistsError("Use a new output or explicit --resume-suite")
    root.mkdir(parents=True, exist_ok=True)
    selections = {s: args.modes or (["jacobi", "gs", "hybrid"] if s == "floor" else list(MODES))
                  for s in args.scenes}
    total_runs = sum(map(len, selections.values())) * args.repeats
    spec = dict(frames=args.frames, repeats=args.repeats, bunny_refinement=args.bunny_refinement,
                bunny_height=args.bunny_height, drop_height=args.drop_height, dt=args.dt,
                tolerance=args.tolerance, bunny_young=args.bunny_young, contact_distance=args.contact_distance,
                max_iterations=args.max_iterations, modes=MODES, scenes=args.scenes,
                selected_modes=selections, collider_subdivisions=args.collider_subdivisions, timers=args.timers,
                note="Serial runs; all regular-CCD update modes share the same local Newton-CG and strict gradient criterion. Local variants retain global certification. No paper speedup is presumed.")
    if args.resume_suite:
        previous = json.loads((root / "suite.json").read_text())
        previous.setdefault("scenes", ["floor", "u"])
        previous.setdefault("selected_modes", {"floor": ["jacobi", "gs", "hybrid"], "u": list(MODES)})
        previous.setdefault("collider_subdivisions", 8)
        previous.setdefault("timers", False)
        if json.loads(json.dumps(spec)) != previous:
            raise ValueError("Suite resume requires identical controls")
    else:
        write_json(root / "suite.json", spec)
    if args.wait_for:
        wait_for_prior(root, args.wait_for.resolve())
    results = []
    for scene, modes in selections.items():
        for repeat in range(args.repeats):
            # Rotate order across repetitions to reduce systematic order bias.
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order:
                folder = root / scene / mode / f"repeat-{repeat}"
                if (folder / "result.json").exists() and args.resume_suite:
                    results.append(json.loads((folder / "result.json").read_text()))
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(Path(__file__).resolve()), "--case", "--scene", scene, "--mode", mode, "--output", str(folder)]
                for name in ["frames", "bunny_refinement", "bunny_height", "drop_height", "dt", "tolerance", "bunny_young", "contact_distance", "max_iterations", "collider_subdivisions"]:
                    command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
                if args.timers:
                    command.append("--timers")
                write_json(root / "suite-status.json", dict(state="running", scene=scene, mode=mode, repeat=repeat, completed_runs=len(results), total_runs=total_runs, updated_unix=time.time()))
                with (folder / "simulation.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
                    process = subprocess.run(command, stdout=stdout, stderr=stderr)
                result = dict(scene=scene, mode=mode, repeat=repeat, returncode=process.returncode, folder=str(folder), passed=False)
                if process.returncode == 0:
                    try:
                        result["color_passes_per_frame"] = parse_trace(folder, args.frames, args.tolerance)
                        result.update(json.loads((folder / "summary.json").read_text()))
                    except Exception as error:
                        result["error"] = str(error)
                else:
                    result["error"] = (folder / "stderr.log").read_text(errors="replace")[-2500:]
                write_json(folder / "result.json", result)
                results.append(result)
                write_json(root / "results.json", results)
    summarize(root, results)
    write_json(root / "suite-status.json", dict(state="complete" if all(r["passed"] for r in results) else "completed_with_failures", completed_runs=len(results), total_runs=total_runs, updated_unix=time.time()))


def summarize(root, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    specification = json.loads((root / "suite.json").read_text())
    scenes = specification.get("scenes", ["floor", "u"])
    # Fail rather than draw a timing comparison of different physical problems.
    controls = {}
    for result in results:
        path = Path(result["folder"]) / "scene.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        config = copy.deepcopy(manifest["config"])
        for key in ["sweep", "globalization", "inner_sweeps"]:
            del config["newton"]["local_stencil"][key]
        control = dict(config=config, mesh_sha256=manifest["mesh_sha256"],
                       runtime_sha256=manifest["runtime_sha256"], planes=manifest["planes"],
                       collider_sha256=manifest.get("collider_sha256"))
        scene = result["scene"]
        if scene in controls and control != controls[scene]:
            raise RuntimeError(f"Mismatched benchmark controls in {path}")
        controls[scene] = control
    write_json(root / "matched-controls.json", controls)
    comparison = []
    fig, axes = plt.subplots(1, len(scenes), figsize=(6 * len(scenes), 5), layout="constrained")
    for ax, scene in zip(np.atleast_1d(axes), scenes):
        modes = specification.get("selected_modes", {}).get(scene, ["jacobi", "gs", "hybrid"] if scene == "floor" else list(MODES))
        valid_scene = [r for r in results if r["scene"] == scene and r["passed"]]
        contact_frame = min((r["first_contact_frame"] for r in valid_scene), default=None)
        for index, mode in enumerate(modes):
            group = [r for r in results if r["scene"] == scene and r["mode"] == mode]
            good = [r for r in group if r["passed"]]
            record = dict(scene=scene, mode=mode, passed_runs=len(good), total_runs=len(group),
                          common_contact_start_frame=contact_frame)
            if len(good) == len(group) and good:
                times = []
                for run in good:
                    with (Path(run["folder"]) / "history.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    times.append(np.mean([float(row["advance_ms"]) for row in rows
                                          if int(row["frame"]) >= contact_frame]) / 1000)
                times = np.array(times)
                record.update(contact_mean_s_median=float(np.median(times)), contact_mean_s_min=float(times.min()), contact_mean_s_max=float(times.max()))
                ax.bar(index, np.median(times), color="#49789e")
                ax.vlines(index, times.min(), times.max(), color="black")
                ax.text(index, np.median(times), f"{np.median(times):.2f}", ha="center", va="bottom")
            else:
                ax.text(index, .03, "FAILED", transform=ax.get_xaxis_transform(), ha="center", color="#b4342e")
            comparison.append(record)
        ax.set(xticks=range(len(modes)), xticklabels=modes, ylabel="Contact-phase time per frame (s)", title=scene)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Single bunny: measured ablations (median and range across repetitions)")
    fig.savefig(root / "timing-comparison.png", dpi=170)
    plt.close(fig)
    write_json(root / "comparison.json", comparison)
    # Distinguish cross-method differences from an exact physical reference.
    differences = []
    for scene in scenes:
        if not any(r["scene"] == scene and r["mode"] == "hybrid" and r["repeat"] == 0
                   and r["passed"] for r in results):
            continue
        reference = root / scene / "hybrid/repeat-0/positions.npz"
        if not reference.exists():
            continue
        a = np.load(reference)["positions"]
        for r in results:
            if r["scene"] == scene and r["passed"]:
                b = np.load(Path(r["folder"]) / "positions.npz")["positions"]
                error = np.linalg.norm(a - b, axis=2)
                differences.append(dict(scene=scene, mode=r["mode"], repeat=r["repeat"], maximum_position_difference_m=float(error.max()), final_rms_difference_m=float(np.sqrt(np.mean(error[-1] ** 2)))))
    write_json(root / "position-differences.json", dict(reference="hybrid regular CCD repeat 0; not exact ground truth", values=differences))
    for scene in scenes:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        reference_run = next((r for r in results if r["scene"] == scene and r["mode"] == "hybrid" and r["repeat"] == 0 and r["passed"]), None)
        impact_frame = reference_run["first_contact_frame"] if reference_run else None
        for r in results:
            if r["scene"] != scene or r["repeat"] != 0 or not r["passed"]:
                continue
            trace = json.loads((Path(r["folder"]) / "convergence.json").read_text())
            # Use first impact, never choose a frame to favor a particular mode.
            frame = impact_frame if impact_frame is not None else r["first_contact_frame"]
            residuals = trace[str(frame)]["velocity"]
            ax.semilogy(range(len(residuals)), residuals, label=f"{r['mode']} (frame {frame})")
        tolerance = json.loads((root / "suite.json").read_text())["tolerance"]
        ax.axhline(tolerance, color="black", ls="--", lw=.8)
        ax.set(xlabel="Local passes (GS updates one color; Jacobi updates all stencils)", ylabel="Mass-scaled gradient residual (m/s)", title=f"{scene}: first-contact convergence")
        ax.legend(fontsize=8)
        fig.savefig(root / f"{scene}-convergence.png", dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--suite", action="store_true")
    group.add_argument("--case", action="store_true")
    parser.add_argument("--scene", choices=["floor", "u", "mesh-u"], default="floor")
    parser.add_argument("--scenes", nargs="+", choices=["floor", "u", "mesh-u"], default=["floor", "u"])
    parser.add_argument("--modes", nargs="+", choices=list(MODES))
    parser.add_argument("--collider-subdivisions", type=int, default=8)
    parser.add_argument("--mode", choices=[*MODES, "global"], default="hybrid",
                        help="Single-case solver; global uses native displacement stopping and is excluded from stencil-only suites")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bunny-refinement", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--bunny-height", type=float, default=.45)
    parser.add_argument("--drop-height", type=float, default=.12)
    parser.add_argument("--bunny-young", type=float, default=1e6)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--tolerance", type=float, default=.01)
    parser.add_argument("--contact-distance", type=float, default=.0005)
    parser.add_argument("--max-iterations", type=int, default=65536)
    parser.add_argument("--wait-for", type=Path, help="Wait for an existing supervised dense run before timing")
    parser.add_argument("--resume-suite", action="store_true")
    parser.add_argument("--timers", action="store_true", help="Save native per-frame timer trees for profiling")
    args = parser.parse_args()
    for key in ["frames", "repeats", "bunny_height", "drop_height", "bunny_young", "dt", "tolerance", "contact_distance", "max_iterations", "collider_subdivisions"]:
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    try:
        suite(args) if args.suite else case(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / ("suite-status.json" if args.suite else "failure.json"), dict(state="failed", reason=str(error), updated_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
