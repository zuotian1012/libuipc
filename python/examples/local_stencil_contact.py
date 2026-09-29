"""Frictionless tetrahedral contact with the additive local stencil solver.

Run from the repository root after building the local Python runtime:
    python python/examples/local_stencil_contact.py --render
    python python/examples/local_stencil_contact.py --scene stack --render
    python python/examples/local_stencil_contact.py --solver global

The ground case drops one soft cube. The stack case drops a second tilted cube
onto it. Both use the normal World.advance(), IPC energy and CCD pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

# Reuse the mesh builder and the checkout-local runtime selection.
from local_stencil_beam import ROOT, beam_mesh, write_obj
import numpy as np
from uipc import Engine, World, Scene, Logger
from uipc.geometry import tetmesh, label_surface, label_triangle_orient, ground
from uipc.constitution import StableNeoHookean, ElasticModuli


def render(output, trajectories, faces, dt):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    colors = np.where(trajectories[0][faces][:, :, 1].mean(axis=1) > .5,
                      "#e4a360", "#5f9dd3")
    mesh = Poly3DCollection(trajectories[0][faces][:, :, [0, 2, 1]],
                           facecolors=colors, edgecolor="#23465b", linewidths=0.4)
    ax.add_collection3d(mesh)
    lower = trajectories.min(axis=(0, 1)) - .06
    upper = trajectories.max(axis=(0, 1)) + .06
    xmin, xmax = min(-.3, lower[0]), max(.3, upper[0])
    zmin, zmax = min(-.25, lower[2]), max(.25, upper[2])
    floor = np.array([[[xmin, zmin, 0], [xmax, zmin, 0],
                       [xmax, zmax, 0], [xmin, zmax, 0]]])
    ax.add_collection3d(Poly3DCollection(floor, facecolor="#d7dbe2", alpha=.5))
    ymax = max(.6, float(trajectories[:, :, 1].max()) + .05)
    ax.set(xlim=(xmin, xmax), ylim=(zmin, zmax), zlim=(0, ymax),
           xlabel="x (m)", ylabel="z (m)", zlabel="y (m)")
    ax.set_box_aspect((xmax - xmin, zmax - zmin, ymax))
    ax.view_init(elev=17, azim=-63)
    title = ax.set_title("")

    def update(i):
        mesh.set_verts(trajectories[i][faces][:, :, [0, 2, 1]])
        title.set_text(f"Frictionless contact | t = {i * dt:.2f} s")
        return mesh, title

    indices = list(range(0, len(trajectories), 2))
    animation = FuncAnimation(fig, update, frames=indices)
    animation.save(output / "contact.gif", writer=PillowWriter(fps=25), dpi=100)
    update(len(trajectories) - 1)
    fig.savefig(output / "contact.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["ground", "stack"], default="ground")
    parser.add_argument("--solver", choices=["local_stencil", "global"], default="local_stencil")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if min(args.frames, args.resolution, args.max_iterations) < 1 or args.dt <= 0:
        parser.error("frame count, resolution, iterations and dt must be positive")
    output = (args.output or ROOT / "output/local-stencil-contact" / args.scene / args.solver).resolve()
    (output / "frames").mkdir(parents=True, exist_ok=True)
    Logger.set_level(Logger.Level.Warn)
    config = Scene.default_config()
    config["newton"]["solver"] = args.solver
    config["newton"]["semi_implicit"]["enable"] = False
    config["newton"]["velocity_tol"] = args.tolerance
    config["newton"]["max_iter"] = args.max_iterations
    config["line_search"]["max_iter"] = 32
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = False
    config["extras"]["strict_mode"]["enable"] = True
    config["dt"] = args.dt
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    scene = Scene(config)
    scene.contact_tabular().default_model(0.0, 1e8)
    element = scene.contact_tabular().default_element()
    slots, vertices, cells, boundaries = [], [], [], []
    for body in range(2 if args.scene == "stack" else 1):
        X, T, F = beam_mesh(args.resolution, args.resolution, args.resolution)
        X *= [.2, .8, .8]
        X[:, 0] -= .1
        if body:
            angle = .2
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                 [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            X = X @ rotation.T
        X[:, 1] += .35 + .32 * body
        mesh = tetmesh(X, T)
        label_surface(mesh)
        label_triangle_orient(mesh)
        StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(2e4, .3), 1000.)
        element.apply_to(mesh)
        slot, _ = scene.objects().create(f"cube_{body}").geometries().create(mesh)
        slots.append(slot)
        offset = sum(len(v) for v in vertices)
        vertices.append(X)
        cells.append(T + offset)
        boundaries.append(F + offset)
    plane = ground(0.)
    element.apply_to(plane)
    scene.objects().create("ground").geometries().create(plane)
    rest, tets, faces = np.concatenate(vertices), np.concatenate(cells), np.concatenate(boundaries)
    X = rest[tets]
    reference = np.linalg.det((X[:, 1:] - X[:, :1]).transpose(0, 2, 1))
    engine = Engine("cuda", str(output / "engine"))
    world = World(engine)
    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("World initialization failed")
    trajectories, history = [rest.copy()], []
    write_obj(output / "frames/0000.obj", rest, faces)
    for frame in range(1, args.frames + 1):
        begin = time.perf_counter()
        world.advance()
        world.retrieve()
        elapsed_ms = 1000 * (time.perf_counter() - begin)
        if not world.is_valid():
            raise RuntimeError(f"Simulation failed at frame {frame}")
        positions = np.concatenate([np.asarray(s.geometry().positions().view()).reshape(-1, 3).copy() for s in slots])
        stats = engine.frame_stats()
        X = positions[tets]
        determinants = np.linalg.det((X[:, 1:] - X[:, :1]).transpose(0, 2, 1)) / reference
        if not np.isfinite(positions).all() or determinants.min() <= 0 or positions[:, 1].min() <= 0:
            raise RuntimeError(f"Invalid geometry or ground penetration at frame {frame}")
        if not stats.get("converged", False):
            raise RuntimeError(f"Frame {frame} did not converge: {stats}")
        trajectories.append(positions)
        history.append(dict(frame=frame, time_s=frame * args.dt,
                            min_ground_gap=float(positions[:, 1].min()),
                            min_det_F=float(determinants.min()), elapsed_ms=elapsed_ms,
                            iterations=stats["newton_iterations"],
                            linear_iterations=stats["linear_solver_iterations"]))
        write_obj(output / "frames" / f"{frame:04d}.obj", positions, faces)
        if frame == 1 or frame % 10 == 0:
            print(f"{args.scene}/{args.solver}: frame={frame}, gap={history[-1]['min_ground_gap']:.6g}, "
                  f"iterations={history[-1]['iterations']}, ms={elapsed_ms:.1f}", flush=True)
    trajectories = np.asarray(trajectories)
    np.savez_compressed(output / "positions.npz", positions=trajectories, tetrahedra=tets,
                        faces=faces, dt=args.dt)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    summary = dict(scene=args.scene, solver=args.solver, frames=args.frames,
                   vertices=len(rest), tetrahedra=len(tets),
                   min_ground_gap=min(h["min_ground_gap"] for h in history),
                   min_det_F=min(h["min_det_F"] for h in history),
                   mean_ms=float(np.mean([h["elapsed_ms"] for h in history])),
                   max_iterations=max(h["iterations"] for h in history),
                   last_frame=history[-1])
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.render:
        render(output, trajectories, faces, args.dt)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
