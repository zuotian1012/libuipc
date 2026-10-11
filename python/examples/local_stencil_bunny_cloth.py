"""A deformable Stanford bunny and a cloth sheet falling onto the ground.

Uses the repository's bunny0.msh tetrahedral asset and the checkout-local CUDA
runtime. Both bodies are free; the cloth starts above the bunny without overlap.
    python python/examples/local_stencil_bunny_cloth.py --solver global --render
    python python/examples/local_stencil_bunny_cloth.py --solver sos --dt .001 --frames 3 --output output/dense-sos-short
    python python/examples/local_stencil_bunny_cloth.py --solver both --dt .001 --frames 3 --output output/dense-both-short
    python python/examples/local_stencil_bunny_cloth.py --cloth-resolution 96 --bunny-refinement 2 --mesh-info
    python python/examples/local_stencil_bunny_cloth.py --replay --gui

sos is the implemented SoS subset. Global acceptance rounds can contain several
frozen-barrier color passes. Strict velocity stopping remains the default;
gradient_l2 is an explicitly separate convergence experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

from local_stencil_beam import ROOT, write_obj
from local_stencil_cloth import sheet_mesh
import numpy as np
import uipc
from uipc import Engine, World, Scene, Logger
from uipc.geometry import SimplicialComplexIO, tetmesh, trimesh, label_surface, label_triangle_orient, ground
from uipc.constitution import StableNeoHookean, ElasticModuli, ElasticModuli2D, StrainLimitingBaraffWitkinShell, DiscreteShellBending


def boundary_faces(tets):
    faces = np.concatenate([tets[:, p] for p in [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]])
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[first[counts == 1]]


def refine_bunny(vertices, tets, levels):
    """Conforming 1-to-8 subdivision of the existing simulation tetrahedra.

    One midpoint per shared edge prevents cracks. Split each central octahedron
    along its shortest diagonal. This preserves the original piecewise-linear
    boundary and volume; it does not add scan detail or perform surface smoothing.
    """
    edge_pairs = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    diagonals = np.array([(0, 5), (1, 4), (2, 3)])
    rings = np.array([(1, 2, 4, 3), (0, 2, 5, 3), (0, 1, 5, 4)])
    for _ in range(levels):
        edges, inverse = np.unique(np.sort(tets[:, edge_pairs].reshape(-1, 2), axis=1),
                                   axis=0, return_inverse=True)
        mid = (inverse.reshape(-1, 6) + len(vertices)).astype(np.int32)
        vertices = np.concatenate([vertices, vertices[edges].mean(axis=1)])
        corners = [np.column_stack([tets[:, i], mid[:, ids]])
                   for i, ids in enumerate([(0, 1, 2), (0, 3, 4), (1, 3, 5), (2, 4, 5)])]
        lengths = np.sum((vertices[mid[:, diagonals[:, 0]]] - vertices[mid[:, diagonals[:, 1]]])**2, axis=2)
        split = lengths.argmin(axis=1)
        rows = np.arange(len(tets))[:, None]
        axis = mid[rows, diagonals[split]]
        ring = mid[rows, rings[split]]
        center = [np.column_stack([axis, ring[:, i], ring[:, (i+1) % 4]]) for i in range(4)]
        tets = np.concatenate(corners + center).astype(np.int32)
    return vertices, tets


def geometry(args):
    mesh = SimplicialComplexIO().read(str(ROOT / "assets/sim_data/tetmesh/bunny0.msh"))
    bunny = np.asarray(mesh.positions().view()).reshape(-1, 3).copy()
    tets = np.asarray(mesh.tetrahedra().topo().view()).reshape(-1, 4).copy()
    bunny, tets = refine_bunny(bunny, tets, args.bunny_refinement)
    bunny -= (bunny.min(axis=0) + bunny.max(axis=0)) / 2
    bunny *= args.bunny_height / np.ptp(bunny[:, 1])
    bunny[:, 1] += args.drop_height - bunny[:, 1].min()
    pts = bunny[tets]
    negative = np.linalg.det((pts[:, 1:] - pts[:, :1]).transpose(0, 2, 1)) < 0
    tets[negative, 1], tets[negative, 2] = tets[negative, 2].copy(), tets[negative, 1].copy()
    cloth, triangles = sheet_mesh(args.resolution)
    cloth[:, [0, 2]] *= args.cloth_size / .8
    cloth[:, 1] = bunny[:, 1].max() + args.cloth_gap
    return bunny, tets, boundary_faces(tets), cloth, triangles


def mesh_info(meshes):
    bunny, tets, bunny_faces, cloth, cloth_faces = meshes
    pts = bunny[tets]
    volumes = np.linalg.det((pts[:, 1:] - pts[:, :1]).transpose(0, 2, 1)) / 6
    if not np.isfinite(volumes).all() or volumes.min() <= 0:
        raise ValueError("Bunny contains invalid or degenerate tetrahedra")
    return dict(bunny_vertices=len(bunny), bunny_tetrahedra=len(tets),
                bunny_surface_triangles=len(bunny_faces), cloth_vertices=len(cloth),
                cloth_triangles=len(cloth_faces), total_dofs=3*(len(bunny)+len(cloth)),
                bunny_volume_m3=float(volumes.sum()), min_tet_volume_m3=float(volumes.min()))


def run(args, output):
    previous = None
    source = None
    if args.fork_checkpoint:
        source = args.fork_checkpoint.resolve()
        previous = json.loads((source / "scene.json").read_text(encoding="utf-8"))
    if args.resume:
        previous = json.loads((output / "scene.json").read_text(encoding="utf-8"))
    elif (output / "history.csv").exists() or (output / "summary.json").exists():
        raise FileExistsError(f"Run data already exists in {output}; use --replay or a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    (output / "frame-stats").mkdir(exist_ok=True)
    Logger.set_level(Logger.Level.Info if args.verbose else Logger.Level.Warn)
    bunny, tets, bunny_faces, cloth, cloth_faces = geometry(args)
    info = mesh_info((bunny, tets, bunny_faces, cloth, cloth_faces))
    print(json.dumps(dict(solver=args.solver, **info), indent=2), flush=True)
    rest = np.concatenate([bunny, cloth])
    cloth_faces = cloth_faces + len(bunny)
    faces = np.concatenate([bunny_faces, cloth_faces])
    fingerprint = hashlib.sha256(rest.tobytes() + tets.tobytes() + faces.tobytes()).hexdigest()
    config = Scene.default_config()
    config["dt"] = args.dt
    config["gravity"] = [[0.], [-9.8], [0.]]
    config["newton"]["solver"] = args.solver
    config["newton"]["local_stencil"]["sweep"] = args.sweep
    config["newton"]["local_stencil"]["initial_guess"] = args.initial_guess
    config["newton"]["local_stencil"]["stopping"] = args.stopping
    config["newton"]["local_stencil"]["gradient_tol"] = args.gradient_tolerance
    config["newton"]["local_stencil"]["color_limit"] = args.color_limit
    config["newton"]["local_stencil"]["color_coverage"] = args.color_coverage
    config["newton"]["local_stencil"]["inner_sweeps"] = args.inner_sweeps
    for key in ["hessian_product", "barrier_model", "barrier_activation", "barrier_stiffness", "merge_acceptance", "hybrid_submission"]:
        config["newton"]["local_stencil"][key] = getattr(args, key)
    config["newton"]["local_stencil"]["warm_start"] = int(args.warm_start)
    config["newton"]["local_stencil"]["warm_start_inversion"] = args.warm_inversion
    config["newton"]["local_stencil"]["warm_start_relaxation"] = args.warm_relaxation
    config["newton"]["local_stencil"]["warm_start_max_iter"] = args.warm_max_iterations
    config["newton"]["local_stencil"]["restart_schedule_on_recover"] = int(source is not None)
    config["newton"]["semi_implicit"]["enable"] = False
    config["newton"]["velocity_tol"] = args.tolerance
    config["newton"]["max_iter"] = args.max_iterations
    config["line_search"]["max_iter"] = 32
    config["contact"]["friction"]["enable"] = False
    config["contact"]["d_hat"] = args.contact_distance
    config["extras"]["strict_mode"]["enable"] = True
    scene = Scene(config)
    scene.contact_tabular().default_model(0., 1e8)
    contact = scene.contact_tabular().default_element()
    solid = tetmesh(bunny, tets)
    label_surface(solid)
    label_triangle_orient(solid)
    StableNeoHookean().apply_to(solid, ElasticModuli.youngs_poisson(args.bunny_young, .3), 1000.)
    contact.apply_to(solid)
    bunny_slot, _ = scene.objects().create("Stanford bunny").geometries().create(solid)
    sheet = trimesh(cloth, cloth_faces - len(bunny))
    label_surface(sheet)
    StrainLimitingBaraffWitkinShell().apply_to(
        sheet, stretch_moduli=ElasticModuli2D.youngs_poisson(5e4, .4),
        shear_moduli=ElasticModuli2D.youngs_poisson(5e2, .4),
        mass_density=200., thickness=.001, strain_rate=100.)
    DiscreteShellBending().apply_to(sheet, 5e4, .4)
    contact.apply_to(sheet)
    cloth_slot, _ = scene.objects().create("Cloth").geometries().create(sheet)
    floor = ground(0.)
    contact.apply_to(floor)
    scene.objects().create("Ground").geometries().create(floor)
    parameters = {k: v for k, v in vars(args).items() if k not in {"output", "gui", "render", "replay", "resume", "verbose", "mesh_info", "fork_checkpoint", "checkpoint_frame"}}
    manifest = dict(parameters=parameters, config=config, mesh_sha256=fingerprint,
                    bunny_asset="assets/sim_data/tetmesh/bunny0.msh", **info,
                    refinement="Conforming edge-midpoint 1-to-8; original piecewise-linear shape preserved",
                    solver_scope="sos selects the implemented local_stencil subset, not the complete paper method",
                    initialization_scope="warm_start selects inertia-barrier initialization; initial_guess is the legacy inertial-proposal control",
                    stopping_scope="gradient_l2 uses the absolute free-DOF gradient of J=0.5*(x-x_tilde)^T*M*(x-x_tilde)+dt^2*V (kg*m); 1e-4 is a paper-style experiment, not certified matched accuracy with Fig.13",
                    iteration_scope="iterations counts global acceptance rounds; each contains up to inner_sweeps local color passes; a fresh outer residual certifies the accepted state",
                    units="meter, kilogram, second", friction=False,
                    runtime=str(Path(uipc.__file__).resolve().parent / "_native"))
    binary = Path(manifest["runtime"]) / "uipc_backend_cuda.dll"
    if binary.exists():
        manifest["runtime_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
    if source:
        physics_keys = {"resolution", "bunny_refinement", "dt", "bunny_height", "drop_height", "cloth_size", "cloth_gap", "bunny_young", "contact_distance"}
        if previous["mesh_sha256"] != fingerprint or any(previous["parameters"][k] != parameters[k] for k in physics_keys):
            raise ValueError("Checkpoint fork requires identical mesh, material, timestep and geometry")
        with (source / "history.csv").open(newline="", encoding="utf-8") as stream:
            prefix = [r for r in csv.DictReader(stream) if int(r["frame"]) <= args.checkpoint_frame]
        if len(prefix) != args.checkpoint_frame or any(r["converged"] != "True" for r in prefix):
            raise ValueError("Source must contain all converged frames through checkpoint-frame")
        manifest["checkpoint_origin"] = dict(path=str(source), frame=args.checkpoint_frame,
                                             parameters=previous["parameters"], schedule_restarted=True)
        dump_source = source / "engine/dump"
        for file in dump_source.rglob("*"):
            if file.is_file() and (file.name.endswith(f".{args.checkpoint_frame}") or file.name.endswith(f".{args.checkpoint_frame}.json")):
                target = output / "engine/dump" / file.relative_to(dump_source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
        for frame in range(args.checkpoint_frame + 1):
            shutil.copy2(source / "frames" / f"{frame:04d}.npy", output / "frames" / f"{frame:04d}.npy")
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=prefix[0].keys())
            writer.writeheader()
            writer.writerows(prefix)
        (output / "scene.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    elif previous:
        physics = lambda p: {k: v for k, v in p.items() if k not in {"frames", "max_iterations"}}
        previous_parameters = dict(previous["parameters"])
        # Older manifests predate the explicit execution controls. Recover their
        # actual settings, not today's demo defaults, before checking equality.
        for key, value in dict(stopping="velocity", gradient_tolerance=1e-4,
                               color_limit=8, color_coverage="primary", inner_sweeps=1,
                               hessian_product="assembled", barrier_model="ipc", barrier_activation="reference", barrier_stiffness="ipc",
                               merge_acceptance="monotone", hybrid_submission="frozen", warm_start=False, warm_inversion="iterative", warm_relaxation=1., warm_max_iterations=0).items():
            previous_parameters.setdefault(key, value)
        if previous["mesh_sha256"] != fingerprint or physics(previous_parameters) != physics(parameters):
            raise ValueError("Resume requires identical mesh, physics, solver, initialization and tolerance")
        (output / f"resume-{time.time_ns()}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        (output / "scene.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    engine = Engine("cuda", str(output / "engine"))
    world = World(engine)
    start = time.perf_counter()
    world.init(scene)
    init_ms = (time.perf_counter() - start) * 1000
    if not world.is_valid():
        raise RuntimeError("Bunny/cloth World initialization failed")
    pts = rest[tets]
    reference_det = np.linalg.det((pts[:, 1:] - pts[:, :1]).transpose(0, 2, 1))
    tri = rest[cloth_faces]
    reference_area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    positions = [rest]
    history = []
    if args.resume or source:
        with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
            for raw in csv.DictReader(stream):
                row = {key: (value == "True" if key == "converged" else
                             int(value) if key in {"frame", "iterations", "linear_iterations"} else float(value))
                       for key, value in raw.items()}
                if row["frame"] != len(history)+1 or not row["converged"]:
                    raise ValueError("Resume history must contain contiguous converged frames")
                history.append(row)
                positions.append(np.load(output / "frames" / f"{row['frame']:04d}.npy"))
        if not history or args.frames <= len(history):
            raise ValueError("Resume requires a checkpoint and a larger target frame count")
        if not world.recover(len(history)):
            raise RuntimeError("Could not recover the last recorded frame")
        world.retrieve()
        recovered = np.concatenate([np.asarray(slot.geometry().positions().view()).reshape(-1, 3)
                                    for slot in (bunny_slot, cloth_slot)])
        if not np.array_equal(recovered, positions[-1]):
            raise RuntimeError("Recovered positions do not match the saved full-precision frame")
    else:
        write_obj(output / "frames/0000.obj", rest, faces)
        np.save(output / "frames/0000.npy", rest)
    print(f"{args.solver}: {len(bunny)} bunny vertices, {len(tets)} tets, "
          f"{len(cloth)} cloth vertices, {len(cloth_faces)} cloth triangles; init={init_ms:.0f} ms", flush=True)
    # Write diagnostics incrementally so an interrupted experiment retains data.
    first_simulated_frame = len(history) + 1
    with (output / "history.csv").open("a" if args.resume or source else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys()) if history else None
        for frame in range(len(history)+1, args.frames + 1):
            start = time.perf_counter()
            world.advance()
            advance_ms = 1000 * (time.perf_counter() - start)
            world.retrieve()
            if not world.is_valid():
                raise RuntimeError(f"World failed at frame {frame}; see {output / 'engine'}")
            x = np.concatenate([np.asarray(slot.geometry().positions().view()).reshape(-1, 3).copy()
                                for slot in (bunny_slot, cloth_slot)])
            stats = engine.frame_stats()
            pts = x[tets]
            determinant = np.linalg.det((pts[:, 1:] - pts[:, :1]).transpose(0, 2, 1)) / reference_det
            tri = x[cloth_faces]
            area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) / reference_area
            row = dict(frame=frame, time_s=frame*args.dt, advance_ms=advance_ms,
                       iterations=stats["newton_iterations"], linear_iterations=stats["linear_solver_iterations"],
                       converged=stats["converged"], min_det_F=float(determinant.min()),
                       min_area_ratio=float(area.min()), bunny_ground_gap=float(x[:len(bunny), 1].min()),
                       cloth_ground_gap=float(x[len(bunny):, 1].min()-.001))
            # Preserve full precision even if a later strict-mode frame aborts.
            np.save(output / "frames" / f"{frame:04d}.npy", x)
            write_obj(output / "frames" / f"{frame:04d}.obj", x, faces)
            if not np.isfinite(x).all() or min(row[k] for k in ["min_det_F", "min_area_ratio", "bunny_ground_gap", "cloth_ground_gap"]) <= 0:
                raise RuntimeError(f"Invalid geometry at frame {frame}: {row}")
            if not stats["converged"]:
                raise RuntimeError(f"Frame {frame} did not converge: {row}")
            if not world.dump():
                raise RuntimeError(f"Could not checkpoint converged frame {frame}")
            (output / "frame-stats" / f"{frame:04d}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
            # History is the resume journal: only publish a frame after its
            # geometry, convergence and checkpoint have all been validated.
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            history.append(row)
            positions.append(x)
            if frame == 1 or frame % 10 == 0:
                print(f"{args.solver}: frame={frame}/{args.frames}, outer_iterations={row['iterations']}, "
                      f"advance={advance_ms:.1f} ms, det={row['min_det_F']:.4f}", flush=True)
    np.savez_compressed(output / "positions.npz", positions=np.asarray(positions), bunny_faces=bunny_faces,
                        cloth_faces=cloth_faces, tetrahedra=tets, bunny_vertices=len(bunny), dt=args.dt)
    times = [r["advance_ms"] for r in history]
    summary = dict(solver=args.solver, frames=args.frames, mesh_sha256=fingerprint, init_ms=init_ms,
                   first_simulated_frame=first_simulated_frame,
                   new_advance_s=sum(r["advance_ms"] for r in history if r["frame"] >= first_simulated_frame)/1000,
                   trajectory_origin="checkpoint prefix plus current run" if source or previous and "checkpoint_origin" in previous else "original scene",
                   total_advance_s=sum(times)/1000, mean_advance_ms=float(np.mean(times)),
                   median_advance_ms=float(np.median(times)), max_iterations=max(r["iterations"] for r in history),
                   all_converged=all(r["converged"] for r in history),
                   min_det_F=min(r["min_det_F"] for r in history), min_area_ratio=min(r["min_area_ratio"] for r in history),
                   min_bunny_ground_gap=min(r["bunny_ground_gap"] for r in history),
                   min_cloth_ground_gap=min(r["cloth_ground_gap"] for r in history))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def render_runs(paths, destination):
    os.environ.setdefault("MPLCONFIGDIR", str(destination / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    data = [np.load(path / "positions.npz") for path in paths]
    fig = plt.figure(figsize=(7 * len(paths), 6), facecolor="#f5f5f3")
    meshes = []
    lower = np.minimum.reduce([d["positions"].min(axis=(0, 1)) for d in data])-.08
    upper = np.maximum.reduce([d["positions"].max(axis=(0, 1)) for d in data])+.08
    for i, (path, d) in enumerate(zip(paths, data)):
        ax = fig.add_subplot(1, len(paths), i+1, projection="3d", facecolor="#f5f5f3")
        local = []
        for name, color, edge, width in [("bunny_faces", "#dc9863", "#90674e", .10),
                                          ("cloth_faces", "#648fc6", "#354f75", .25)]:
            mesh = Poly3DCollection(d["positions"][0][d[name]][:, :, [0, 2, 1]],
                                    facecolors=color, edgecolors=edge, linewidths=width, alpha=1.)
            ax.add_collection3d(mesh)
            local.append((mesh, name))
        floor = [[[lower[0], lower[2], 0], [upper[0], lower[2], 0],
                  [upper[0], upper[2], 0], [lower[0], upper[2], 0]]]
        ax.add_collection3d(Poly3DCollection(floor, facecolors="#d8d8d3", alpha=.3))
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[2], upper[2]), zlim=(0, upper[1]),
               xlabel="x (m)", ylabel="z (m)", zlabel="y (m)")
        ax.set_box_aspect((upper[0]-lower[0], upper[2]-lower[2], upper[1]))
        ax.view_init(elev=18, azim=-65)
        meshes.append((ax, local, path.name))

    def update(frame):
        for d, (ax, local, title) in zip(data, meshes):
            for mesh, name in local:
                mesh.set_verts(d["positions"][frame][d[name]][:, :, [0, 2, 1]])
            ax.set_title(f"Bunny + cloth | {title}\nt = {frame*float(d['dt']):.2f} s")
    frames = len(data[0]["positions"])
    update(0)
    fig.savefig(destination / "initial.png", dpi=120)
    indices = sorted(set(range(0, frames, 2)) | {frames-1})
    animation = FuncAnimation(fig, update, frames=indices)
    fps = max(1, round(.5 / float(data[0]["dt"])))
    animation.save(destination / "bunny_cloth.gif", writer=PillowWriter(fps=fps), dpi=90)
    update(frames-1)
    fig.savefig(destination / "final.png", dpi=120)
    plt.close(fig)


def replay_gui(path):
    import polyscope as ps
    import polyscope.imgui as ui
    data = np.load(path / "positions.npz")
    x = data["positions"]
    ps.init()
    ps.set_ground_plane_height(0.)
    bunny = ps.register_surface_mesh("Stanford bunny", x[0], data["bunny_faces"], color=(.86, .60, .39))
    cloth = ps.register_surface_mesh("Cloth", x[0], data["cloth_faces"], color=(.39, .56, .78))
    state = {"frame": 0, "playing": True, "last": time.perf_counter()}

    def callback():
        _, state["playing"] = ui.Checkbox("Play", state["playing"])
        _, state["frame"] = ui.SliderInt("Frame", state["frame"], 0, len(x)-1)
        if state["playing"] and time.perf_counter()-state["last"] >= float(data["dt"]):
            state["frame"] = (state["frame"]+1) % len(x)
            state["last"] = time.perf_counter()
        bunny.update_vertex_positions(x[state["frame"]])
        cloth.update_vertex_positions(x[state["frame"]])
    ps.set_user_callback(callback)
    ps.show()


def compare(root):
    paths = [root / name for name in ["local_stencil", "global"]]
    summaries = [json.loads((p / "summary.json").read_text()) for p in paths]
    if summaries[0]["mesh_sha256"] != summaries[1]["mesh_sha256"]:
        raise RuntimeError("Comparison requires identical starting meshes")
    manifests = [json.loads((p / "scene.json").read_text()) for p in paths]
    parameters = [{k: v for k, v in m["parameters"].items() if k not in {"solver", "sweep"}}
                  for m in manifests]
    if parameters[0] != parameters[1]:
        raise RuntimeError("Comparison requires identical physical parameters and frame settings")
    configs = [m["config"] for m in manifests]
    for config in configs:
        config["newton"].pop("solver", None)
    if configs[0] != configs[1]:
        raise RuntimeError("Comparison requires matching scene configuration apart from the solver selector")
    a, b = [np.load(p / "positions.npz")["positions"] for p in paths]
    if a.shape != b.shape:
        raise RuntimeError("Comparison requires matching trajectory dimensions")
    error = np.linalg.norm(a-b, axis=-1)
    bunny_count = manifests[0]["bunny_vertices"]
    result = dict(runs=summaries, max_vertex_difference_m=float(error.max()),
                  final_rms_vertex_difference_m=float(np.sqrt(np.mean(error[-1]**2))),
                  bunny_max_difference_m=float(error[:, :bunny_count].max()),
                  bunny_final_rms_difference_m=float(np.sqrt(np.mean(error[-1, :bunny_count]**2))),
                  cloth_max_difference_m=float(error[:, bunny_count:].max()),
                  cloth_final_rms_difference_m=float(np.sqrt(np.mean(error[-1, bunny_count:]**2))),
                  stencil_over_global_time=summaries[0]["total_advance_s"]/summaries[1]["total_advance_s"],
                  note="Single serial runs, not a controlled benchmark. Stencil uses a mass-scaled gradient residual; global uses its existing displacement stopping rule. Equal tolerance settings do not imply equal solve accuracy or equivalent iterations.")
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for path in paths:
        with (path / "history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        frame = [int(r["frame"]) for r in history]
        axes[0].plot(frame, [float(r["advance_ms"]) for r in history], label=path.name)
        axes[1].plot(frame, [int(r["iterations"]) for r in history], label=path.name)
    axes[0].set(ylabel="Advance time (ms)", yscale="log")
    axes[1].set(ylabel="Outer iterations / Newton iterations", yscale="log")
    axes[2].plot(np.arange(len(error)), error[:, :bunny_count].max(axis=1)*1000, label="Bunny", color="#c98145")
    axes[2].plot(np.arange(len(error)), error[:, bunny_count:].max(axis=1)*1000, label="Cloth", color="#7657a8")
    axes[2].set(ylabel="Max vertex difference (mm)", xlabel="Frame")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    fig.suptitle("Bunny + cloth: preliminary solver comparison\nDifferent stopping rules; iteration counts are not equivalent work")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(root / "comparison.png", dpi=140)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["sos", "global", "both", "local_stencil"], default="global",
                        help="sos: current SoS subset (local_stencil alias); global: standard IPC Newton solver (default); both: serial comparison")
    parser.add_argument("--sweep", choices=["hybrid", "additive"], default="hybrid")
    parser.add_argument("--initial-guess", choices=["previous", "inertial"], default="previous",
                        help="SoS only: previous keeps the baseline; inertial tries a CCD/energy-filtered predictor, without the paper's inertia-barrier warm start")
    parser.add_argument("--stopping", choices=["velocity", "gradient_l2"], default="velocity",
                        help="SoS stopping norm; velocity retains the strict legacy residual")
    parser.add_argument("--gradient-tolerance", type=float, default=1e-4,
                        help="Absolute free-DOF gradient L2 tolerance in kg*m; not a velocity tolerance")
    parser.add_argument("--color-limit", type=int, default=15)
    parser.add_argument("--color-coverage", choices=["independent", "primary"], default="independent")
    parser.add_argument("--inner-sweeps", type=int, default=15,
                        help="SoS local color passes per global CCD/energy acceptance; use 1 for the original outer loop")
    parser.add_argument("--hessian-product", choices=["assembled", "csfd", "packed"], default="assembled")
    parser.add_argument("--barrier-model", choices=["ipc", "sos"], default="ipc")
    parser.add_argument("--barrier-activation", choices=["reference", "expanded"], default="reference")
    parser.add_argument("--barrier-stiffness", choices=["ipc", "distance"], default="ipc")
    parser.add_argument("--merge-acceptance", choices=["monotone", "local"], default="monotone")
    parser.add_argument("--hybrid-submission", choices=["frozen", "sequential"], default="frozen")
    parser.add_argument("--warm-relaxation", type=float, default=1., help="Explicit warm Jacobi damping control; 1 preserves the unrelaxed path")
    parser.add_argument("--warm-inversion", choices=["iterative", "after", "after_local"], default="iterative")
    parser.add_argument("--warm-max-iterations", type=int, default=0, help="Auxiliary warm-pass budget; 0 requires warm residual convergence; final tolerance is unchanged")
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--cloth-resolution", "--resolution", dest="resolution", type=int, default=48,
                        help="Cloth cells per edge: (N+1)^2 vertices and 2*N^2 triangles (default: 48)")
    parser.add_argument("--bunny-refinement", type=int, choices=[0, 1, 2], default=1,
                        help="Conforming volumetric subdivision levels; each multiplies tets by 8 (default: 1)")
    parser.add_argument("--mesh-info", action="store_true", help="Print actual mesh counts/volume and exit without simulation")
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--max-iterations", type=int, default=2048)
    parser.add_argument("--bunny-height", type=float, default=.45)
    parser.add_argument("--drop-height", type=float, default=.04)
    parser.add_argument("--cloth-size", type=float, default=.85)
    parser.add_argument("--cloth-gap", type=float, default=.12)
    parser.add_argument("--bunny-young", type=float, default=1e6,
                        help="Bunny Young's modulus in Pa (default: 1 MPa for the refined mesh)")
    parser.add_argument("--contact-distance", type=float, default=.0005,
                        help="IPC barrier activation distance in meters (default: 0.5 mm)")
    parser.add_argument("--output", type=Path, default=ROOT / "output/local-stencil-bunny-cloth-dense")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Read existing trajectories without simulating")
    parser.add_argument("--resume", action="store_true", help="Recover the last converged checkpoint; only frame count and iteration budget may change")
    parser.add_argument("--fork-checkpoint", type=Path, help="Existing solver output directory: fork physical state into a new run for same-frame algorithm comparisons")
    parser.add_argument("--checkpoint-frame", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.warm_max_iterations < 0:
        parser.error("warm-max-iterations must be nonnegative")
    if args.solver in {"global", "both"} and (args.barrier_model != "ipc" or args.warm_start or args.warm_max_iterations != 0 or args.warm_inversion != "iterative" or args.warm_relaxation != 1
            or args.hessian_product != "assembled" or args.merge_acceptance != "monotone"
            or args.hybrid_submission != "frozen"):
        parser.error("Custom SoS controls require --solver sos; compare separate runs with explicitly matched accuracy")
    if args.solver == "sos":
        args.solver = "local_stencil"
    if args.resume and (args.solver == "both" or args.replay):
        parser.error("Use --resume with one solver, without --replay")
    if args.fork_checkpoint and (args.resume or args.replay or args.solver == "both" or not 0 < args.checkpoint_frame < args.frames):
        parser.error("Checkpoint fork requires one solver, a new output, and 0 < checkpoint-frame < frames")
    if not np.isfinite(args.warm_relaxation) or not 0 < args.warm_relaxation <= 1:
        parser.error("warm-relaxation must be in (0, 1]")
    if not 1 <= args.color_limit <= 64 or not 1 <= args.inner_sweeps <= 256:
        parser.error("color-limit must be in [1,64], inner-sweeps in [1,256]")
    if args.sweep == "additive" and args.inner_sweeps != 1:
        parser.error("Use --inner-sweeps 1 with additive sweeps")
    for name in ["frames", "resolution", "dt", "tolerance", "gradient_tolerance", "max_iterations", "bunny_height", "drop_height", "cloth_size", "cloth_gap", "bunny_young", "contact_distance"]:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.mesh_info:
        print(json.dumps(mesh_info(geometry(args)), indent=2))
        return
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.solver == "both":
        if not args.replay:
            for solver in ["local_stencil", "global"]:
                command = [sys.executable, str(Path(__file__).resolve())]
                for k, v in vars(args).items():
                    if k in {"render", "gui", "replay", "solver"} or v is False or v is None:
                        continue
                    command.append("--" + k.replace("_", "-"))
                    if v is not True:
                        command.append(str(v))
                subprocess.run(command + ["--solver", solver], check=True)
        paths = compare(root)
        if args.render:
            render_runs(paths, root)
        if args.gui:
            replay_gui(paths[0])
    else:
        path = root / args.solver
        if not args.replay:
            run(args, path)
        if args.render:
            render_runs([path], path)
        if args.gui:
            replay_gui(path)


if __name__ == "__main__":
    main()
