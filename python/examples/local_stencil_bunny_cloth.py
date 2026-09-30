"""A deformable Stanford bunny and a cloth sheet falling onto the ground.

Uses the repository's bunny0.msh tetrahedral asset and the checkout-local CUDA
runtime. Both bodies are free; the cloth starts above the bunny without overlap.
    python python/examples/local_stencil_bunny_cloth.py --render
    python python/examples/local_stencil_bunny_cloth.py --solver both --render
    python python/examples/local_stencil_bunny_cloth.py --replay --gui
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from local_stencil_beam import ROOT, write_obj
from local_stencil_cloth import sheet_mesh
import numpy as np
from uipc import Engine, World, Scene, Logger
from uipc.geometry import SimplicialComplexIO, tetmesh, trimesh, label_surface, label_triangle_orient, ground
from uipc.constitution import StableNeoHookean, ElasticModuli, ElasticModuli2D, StrainLimitingBaraffWitkinShell, DiscreteShellBending


def boundary_faces(tets):
    faces = np.concatenate([tets[:, p] for p in [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]])
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    return faces[first[counts == 1]]


def geometry(args):
    mesh = SimplicialComplexIO().read(str(ROOT / "assets/sim_data/tetmesh/bunny0.msh"))
    bunny = np.asarray(mesh.positions().view()).reshape(-1, 3).copy()
    tets = np.asarray(mesh.tetrahedra().topo().view()).reshape(-1, 4).copy()
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


def run(args, output):
    if (output / "history.csv").exists() or (output / "summary.json").exists():
        raise FileExistsError(f"Run data already exists in {output}; use --replay or a new --output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    Logger.set_level(Logger.Level.Info if args.verbose else Logger.Level.Warn)
    bunny, tets, bunny_faces, cloth, cloth_faces = geometry(args)
    rest = np.concatenate([bunny, cloth])
    cloth_faces = cloth_faces + len(bunny)
    faces = np.concatenate([bunny_faces, cloth_faces])
    fingerprint = hashlib.sha256(rest.tobytes() + tets.tobytes() + faces.tobytes()).hexdigest()
    config = Scene.default_config()
    config["dt"] = args.dt
    config["gravity"] = [[0.], [-9.8], [0.]]
    config["newton"]["solver"] = args.solver
    config["newton"]["local_stencil"]["sweep"] = args.sweep
    config["newton"]["semi_implicit"]["enable"] = False
    config["newton"]["velocity_tol"] = args.tolerance
    config["newton"]["max_iter"] = args.max_iterations
    config["line_search"]["max_iter"] = 32
    config["contact"]["friction"]["enable"] = False
    config["contact"]["d_hat"] = .0025
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
    parameters = {k: v for k, v in vars(args).items() if k not in {"output", "gui", "render", "replay", "verbose"}}
    manifest = dict(parameters=parameters, config=config, mesh_sha256=fingerprint,
                    bunny_asset="assets/sim_data/tetmesh/bunny0.msh", bunny_vertices=len(bunny),
                    bunny_tetrahedra=len(tets), cloth_vertices=len(cloth), cloth_triangles=len(cloth_faces),
                    units="meter, kilogram, second", friction=False,
                    runtime=str(ROOT / "build/python/src/uipc/_native"))
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
    write_obj(output / "frames/0000.obj", rest, faces)
    print(f"{args.solver}: {len(bunny)} bunny vertices, {len(tets)} tets, "
          f"{len(cloth_faces)} cloth triangles; init={init_ms:.0f} ms", flush=True)
    # Write diagnostics incrementally so an interrupted experiment retains data.
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = None
        for frame in range(1, args.frames + 1):
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
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=row)
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            history.append(row)
            positions.append(x)
            write_obj(output / "frames" / f"{frame:04d}.obj", x, faces)
            if not np.isfinite(x).all() or min(row[k] for k in ["min_det_F", "min_area_ratio", "bunny_ground_gap", "cloth_ground_gap"]) <= 0:
                raise RuntimeError(f"Invalid geometry at frame {frame}: {row}")
            if not stats["converged"]:
                raise RuntimeError(f"Frame {frame} did not converge: {row}")
            if frame == 1 or frame % 10 == 0:
                print(f"{args.solver}: frame={frame}/{args.frames}, passes={row['iterations']}, "
                      f"advance={advance_ms:.1f} ms, det={row['min_det_F']:.4f}", flush=True)
    np.savez_compressed(output / "positions.npz", positions=np.asarray(positions), bunny_faces=bunny_faces,
                        cloth_faces=cloth_faces, tetrahedra=tets, bunny_vertices=len(bunny), dt=args.dt)
    times = [r["advance_ms"] for r in history]
    summary = dict(solver=args.solver, frames=args.frames, mesh_sha256=fingerprint, init_ms=init_ms,
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
    axes[1].set(ylabel="Color passes / Newton iterations", yscale="log")
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
    parser.add_argument("--solver", choices=["local_stencil", "global", "both"], default="local_stencil")
    parser.add_argument("--sweep", choices=["hybrid", "additive"], default="hybrid")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--tolerance", type=float, default=1e-2)
    parser.add_argument("--max-iterations", type=int, default=2048)
    parser.add_argument("--bunny-height", type=float, default=.45)
    parser.add_argument("--drop-height", type=float, default=.04)
    parser.add_argument("--cloth-size", type=float, default=.85)
    parser.add_argument("--cloth-gap", type=float, default=.12)
    parser.add_argument("--bunny-young", type=float, default=1e5)
    parser.add_argument("--output", type=Path, default=ROOT / "output/local-stencil-bunny-cloth")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Read existing trajectories without simulating")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    for name in ["frames", "resolution", "dt", "tolerance", "max_iterations", "bunny_height", "drop_height", "cloth_size", "cloth_gap", "bunny_young"]:
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.solver == "both":
        if not args.replay:
            for solver in ["local_stencil", "global"]:
                command = [sys.executable, str(Path(__file__).resolve())]
                for k, v in vars(args).items():
                    if k in {"render", "gui", "replay", "solver"} or v is False:
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
