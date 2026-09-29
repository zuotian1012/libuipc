"""Non-contact cantilever driven by World.advance() and GPU local stencils.

From the repository root, after building the local runtime:
    python python/examples/local_stencil_beam.py --frames 120
    python python/examples/local_stencil_beam.py --solver global --frames 120
    python python/examples/local_stencil_beam.py --gui

Outputs: OBJ animation frames, positions.npz, history.csv, and summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGE = ROOT / "build/python/src"
DLL_HANDLES = []
if (LOCAL_PACKAGE / "uipc/_native").is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGE))
    if os.name == "nt":
        DLL_HANDLES.append(os.add_dll_directory(str(LOCAL_PACKAGE / "uipc/_native")))

import numpy as np
from uipc import Engine, World, Scene, Logger, view, builtin
from uipc.geometry import tetmesh, label_surface, label_triangle_orient
from uipc.constitution import StableNeoHookean, ElasticModuli


def beam_mesh(nx: int, ny: int, nz: int):
    vertices = np.array([
        (i / nx, 0.25 * (j / ny - 0.5), 0.25 * (k / nz - 0.5))
        for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1)
    ], dtype=np.float64)

    def index(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    cells = []
    pattern = ((0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
               (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                corners = [index(i, j, k), index(i+1, j, k),
                           index(i+1, j+1, k), index(i, j+1, k),
                           index(i, j, k+1), index(i+1, j, k+1),
                           index(i+1, j+1, k+1), index(i, j+1, k+1)]
                cells.extend([[corners[v] for v in tet] for tet in pattern])
    cells = np.asarray(cells, dtype=np.int32)
    X = vertices[cells]
    determinants = np.linalg.det((X[:, 1:] - X[:, :1]).transpose(0, 2, 1))
    negative = determinants < 0
    cells[negative, 1], cells[negative, 2] = cells[negative, 2].copy(), cells[negative, 1].copy()
    faces = {}
    for tet in cells:
        for face in ((tet[1], tet[2], tet[3]), (tet[0], tet[3], tet[2]),
                     (tet[0], tet[1], tet[3]), (tet[0], tet[2], tet[1])):
            key = tuple(sorted(face))
            faces[key] = None if key in faces else face
    boundary = np.asarray([face for face in faces.values() if face is not None], dtype=np.int32)
    return vertices, cells, boundary


def write_obj(path, positions, faces):
    with path.open("w", encoding="utf-8") as stream:
        for x, y, z in positions:
            stream.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")
        for a, b, c in faces + 1:
            stream.write(f"f {a} {b} {c}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["local_stencil", "global"], default="local_stencil")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--young", type=float, default=500000.0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if min(args.nx, args.ny, args.nz, args.frames, args.max_iterations) <= 0:
        parser.error("mesh dimensions, frame count and iteration budget must be positive")
    output = (args.output or ROOT / "output/local-stencil-beam" / args.solver).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    Logger.set_level(Logger.Level.Warn)
    config = Scene.default_config()
    if "solver" not in config["newton"]:
        raise RuntimeError("Build the updated libuipc core/backend and refresh build/python/src/uipc/_native first.")
    config["newton"]["solver"] = args.solver
    config["newton"]["semi_implicit"]["enable"] = False
    config["newton"]["velocity_tol"] = args.tolerance
    config["newton"]["max_iter"] = args.max_iterations
    config["line_search"]["max_iter"] = 32
    config["contact"]["enable"] = False
    config["contact"]["friction"]["enable"] = False
    config["extras"]["strict_mode"]["enable"] = True
    config["dt"] = args.dt
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    scene = Scene(config)
    rest, tetrahedra, faces = beam_mesh(args.nx, args.ny, args.nz)
    mesh = tetmesh(rest, tetrahedra)
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(args.young, 0.3), 1000.0)
    fixed = view(mesh.vertices().find(builtin.is_fixed)).reshape(-1)
    fixed[:] = (rest[:, 0] == 0).astype(np.int32)
    slot, _ = scene.objects().create("cantilever").geometries().create(mesh)
    engine = Engine("cuda", str(output / "engine"))
    world = World(engine)
    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("World initialization failed; inspect engine diagnostics")
    reference_det = np.linalg.det((rest[tetrahedra][:, 1:] - rest[tetrahedra][:, :1]).transpose(0, 2, 1))
    trajectories = [rest.copy()]
    history = []
    write_obj(output / "frames/0000.obj", rest, faces)
    frame = 0

    def step():
        nonlocal frame
        begin = time.perf_counter()
        world.advance()
        world.retrieve()
        elapsed_ms = 1000 * (time.perf_counter() - begin)
        if not world.is_valid():
            raise RuntimeError(f"Simulation failed at frame {frame + 1}")
        frame += 1
        positions = np.asarray(slot.geometry().positions().view()).reshape(-1, 3).copy()
        stats = engine.frame_stats()
        X = positions[tetrahedra]
        det_ratio = np.linalg.det((X[:, 1:] - X[:, :1]).transpose(0, 2, 1)) / reference_det
        if not np.isfinite(positions).all() or det_ratio.min() <= 0:
            raise RuntimeError(f"Invalid geometry at frame {frame}")
        if not stats.get("converged", False):
            raise RuntimeError(f"Frame {frame} did not converge: {stats}")
        trajectories.append(positions)
        history.append({"frame": frame, "time_s": frame * args.dt,
                        "tip_y": float(positions[rest[:, 0] == 1, 1].mean()),
                        "min_det_F": float(det_ratio.min()),
                        "fixed_error": float(np.max(np.linalg.norm(positions[rest[:, 0] == 0] - rest[rest[:, 0] == 0], axis=1))),
                        "elapsed_ms": elapsed_ms,
                        "iterations": stats["newton_iterations"],
                        "linear_iterations": stats["linear_solver_iterations"]})
        write_obj(output / "frames" / f"{frame:04d}.obj", positions, faces)
        if frame == 1 or frame % 10 == 0:
            print(f"{args.solver}: frame={frame}, tip_y={history[-1]['tip_y']:.5f}, "
                  f"iterations={history[-1]['iterations']}, ms={elapsed_ms:.1f}", flush=True)
        return positions

    if args.gui:
        import polyscope as ps
        import polyscope.imgui as ui
        ps.init()
        ps.set_up_dir("y_up")
        surface = ps.register_surface_mesh("cantilever", rest, faces)
        surface.set_edge_width(1.0)
        running = False

        def callback():
            nonlocal running
            if ui.Button("Run / pause"):
                running = not running
            ui.TextUnformatted(f"Solver: {args.solver} | Frame: {frame}/{args.frames}")
            if running and frame < args.frames:
                surface.update_vertex_positions(step())
        ps.set_user_callback(callback)
        ps.show()
    else:
        for _ in range(args.frames):
            step()

    if history:
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    np.savez_compressed(output / "positions.npz", positions=np.asarray(trajectories),
                        tetrahedra=tetrahedra, faces=faces, dt=args.dt)
    summary = {"solver": args.solver, "vertices": len(rest), "tetrahedra": len(tetrahedra),
               "frames": frame, "dt": args.dt, "young": args.young,
               "max_fixed_error": max((row["fixed_error"] for row in history), default=0),
               "min_det_F": min((row["min_det_F"] for row in history), default=1),
               "mean_ms": float(np.mean([row["elapsed_ms"] for row in history])) if history else 0,
               "last_frame": history[-1] if history else None}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
