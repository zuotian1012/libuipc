"""Cloth membrane, bending and frictionless contact with local stencil descent.

Run from the repository root after building the checkout-local CUDA runtime:
    python python/examples/local_stencil_cloth.py --scene drape --render
    python python/examples/local_stencil_cloth.py --scene pinned --gui
    python python/examples/local_stencil_cloth.py --membrane neo --render

Both cloth constitutions reuse the ordinary public Scene/World API. The drape
scene drops a sheet on a deformable SNK1 block attached at its bottom vertices.
The layers scene exercises cloth-cloth contact as two sheets land on the floor.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from local_stencil_beam import ROOT, beam_mesh, write_obj
from local_stencil_contact import render
import numpy as np
from uipc import Engine, World, Scene, Logger, view, builtin
from uipc.geometry import trimesh, tetmesh, label_surface, label_triangle_orient, ground
from uipc.constitution import (
    StrainLimitingBaraffWitkinShell, NeoHookeanShell, DiscreteShellBending,
    StableNeoHookean, ElasticModuli, ElasticModuli2D,
)


def sheet_mesh(n):
    vertices = np.array([(0.8 * (i / n - .5), .55, .8 * (j / n - .5))
                         for i in range(n + 1) for j in range(n + 1)])
    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b, c, d = a + n + 1, a + 1, a + n + 2
            faces.extend([(a, c, b), (b, c, d)])
    return vertices, np.asarray(faces, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["pinned", "drape", "layers"], default="drape")
    parser.add_argument("--membrane", choices=["bw", "neo"], default="bw")
    parser.add_argument("--solver", choices=["local_stencil", "global"], default="local_stencil")
    parser.add_argument("--sweep", choices=["hybrid", "additive"], default="hybrid")
    parser.add_argument("--inner-sweeps", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--dt", type=float, default=.01)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-iterations", type=int, default=2048)
    parser.add_argument("--bending-scale", type=float, default=1.)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if min(args.frames, args.resolution, args.max_iterations) < 1 or args.dt <= 0 or args.tolerance <= 0 or args.bending_scale < 0:
        parser.error("counts, dt and tolerance must be positive; bending scale must be nonnegative")
    run = f"{args.solver}-{args.sweep}" if args.solver == "local_stencil" else args.solver
    output = (args.output or ROOT / "output/local-stencil-cloth" / args.scene / args.membrane / run).resolve()
    (output / "frames").mkdir(parents=True, exist_ok=True)
    Logger.set_level(Logger.Level.Info if args.verbose else Logger.Level.Warn)
    config = Scene.default_config()
    config["newton"]["solver"] = args.solver
    config["newton"]["local_stencil"]["sweep"] = args.sweep
    config["newton"]["local_stencil"]["inner_sweeps"] = args.inner_sweeps
    config["newton"]["semi_implicit"]["enable"] = False
    config["newton"]["velocity_tol"] = args.tolerance
    config["newton"]["max_iter"] = args.max_iterations
    config["line_search"]["max_iter"] = 32
    config["contact"]["enable"] = True
    config["contact"]["friction"]["enable"] = False
    config["contact"]["d_hat"] = .005
    config["extras"]["strict_mode"]["enable"] = True
    config["dt"] = args.dt
    config["gravity"] = [[0.0], [-9.8], [0.0]]
    scene = Scene(config)
    scene.contact_tabular().default_model(0., 1e8)
    contact = scene.contact_tabular().default_element()
    slots, rest_parts, face_parts, cloth_parts, fixed_parts = [], [], [], [], []
    radius = .001
    for layer in range(2 if args.scene == "layers" else 1):
        X, F = sheet_mesh(args.resolution)
        if args.scene == "layers":
            X[:, 1] += layer * .10
            X[:, 0] += layer * .02
        cloth = trimesh(X, F)
        label_surface(cloth)
        if args.membrane == "bw":
            StrainLimitingBaraffWitkinShell().apply_to(
                cloth, stretch_moduli=ElasticModuli2D.youngs_poisson(5e4, .4),
                shear_moduli=ElasticModuli2D.youngs_poisson(5e2, .4),
                mass_density=200., thickness=radius, strain_rate=100.)
        else:
            NeoHookeanShell().apply_to(cloth, ElasticModuli2D.youngs_poisson(5e4, .4), 200., radius)
        DiscreteShellBending().apply_to(cloth, 5e4 * args.bending_scale, .4)
        pins = np.array([0, args.resolution], dtype=int) if args.scene == "pinned" else np.array([], dtype=int)
        view(cloth.vertices().find(builtin.is_fixed)).reshape(-1)[pins] = 1
        contact.apply_to(cloth)
        slot, _ = scene.objects().create(f"cloth_{layer}").geometries().create(cloth)
        offset = sum(map(len, rest_parts))
        slots.append(slot)
        rest_parts.append(X)
        face_parts.append(F + offset)
        cloth_parts.append(F + offset)
        fixed_parts.extend((pins + offset).tolist())
    cloth_vertices = sum(map(len, rest_parts))
    tets = np.empty((0, 4), dtype=np.int32)
    if args.scene == "drape":
        X, T, F = beam_mesh(2, 2, 2)
        X *= [.3, .8, 1.2]
        X[:, 0] -= .15
        X[:, 1] += .102
        block = tetmesh(X, T)
        label_surface(block)
        label_triangle_orient(block)
        StableNeoHookean().apply_to(block, ElasticModuli.youngs_poisson(2e4, .3), 1000.)
        pins = np.flatnonzero(np.isclose(X[:, 1], X[:, 1].min()))
        view(block.vertices().find(builtin.is_fixed)).reshape(-1)[pins] = 1
        contact.apply_to(block)
        slot, _ = scene.objects().create("soft_support").geometries().create(block)
        offset = sum(map(len, rest_parts))
        slots.append(slot)
        rest_parts.append(X)
        face_parts.append(F + offset)
        fixed_parts.extend((pins + offset).tolist())
        tets = T + offset
    floor = ground(0.)
    contact.apply_to(floor)
    scene.objects().create("floor").geometries().create(floor)
    rest, faces, cloth_faces = np.concatenate(rest_parts), np.concatenate(face_parts), np.concatenate(cloth_parts)
    fixed_ids = np.asarray(fixed_parts, dtype=int)
    tri = rest[cloth_faces]
    rest_areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if len(tets):
        pts = rest[tets]
        rest_det = np.linalg.det((pts[:, 1:] - pts[:, :1]).transpose(0, 2, 1))
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
        elapsed = 1000 * (time.perf_counter() - begin)
        if not world.is_valid():
            raise RuntimeError(f"Simulation failed at frame {frame}")
        x = np.concatenate([np.asarray(s.geometry().positions().view()).reshape(-1, 3).copy() for s in slots])
        stats = engine.frame_stats()
        tri = x[cloth_faces]
        area_ratio = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / rest_areas
        pin_error = float(np.linalg.norm(x[fixed_ids] - rest[fixed_ids], axis=1).max()) if len(fixed_ids) else 0.
        gap = float(x[:cloth_vertices, 1].min() - radius)
        det = 1.
        if len(tets):
            pts = x[tets]
            det = float((np.linalg.det((pts[:, 1:] - pts[:, :1]).transpose(0, 2, 1)) / rest_det).min())
        if not np.isfinite(x).all() or area_ratio.min() <= 1e-12 or gap <= 0 or pin_error > 1e-12 or det <= 0:
            raise RuntimeError(f"Invalid geometry at frame {frame}: gap={gap}, area={area_ratio.min()}, fixed={pin_error}, det={det}")
        if not stats.get("converged", False):
            raise RuntimeError(f"Frame {frame} did not converge: {stats}")
        trajectories.append(x)
        history.append(dict(frame=frame, time_s=frame * args.dt, elapsed_ms=elapsed,
                            min_area_ratio=float(area_ratio.min()), min_ground_gap=gap,
                            min_det_F=det, fixed_error=pin_error,
                            iterations=stats["newton_iterations"], linear_iterations=stats["linear_solver_iterations"]))
        write_obj(output / "frames" / f"{frame:04d}.obj", x, faces)
        if frame == 1 or frame % 10 == 0:
            print(f"{args.scene}/{args.membrane}: frame={frame}, area={area_ratio.min():.5f}, "
                  f"gap={gap:.6g}, iterations={history[-1]['iterations']}, ms={elapsed:.1f}", flush=True)
    trajectories = np.asarray(trajectories)
    np.savez_compressed(output / "positions.npz", positions=trajectories, faces=faces,
                        cloth_faces=cloth_faces, tetrahedra=tets, fixed_ids=fixed_ids, dt=args.dt)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    summary = dict(scene=args.scene, membrane=args.membrane, solver=args.solver, sweep=args.sweep,
                   frames=args.frames, vertices=len(rest), cloth_triangles=len(cloth_faces),
                   tetrahedra=len(tets), bending_scale=args.bending_scale,
                   min_area_ratio=min(h["min_area_ratio"] for h in history),
                   min_ground_gap=min(h["min_ground_gap"] for h in history),
                   min_det_F=min(h["min_det_F"] for h in history),
                   fixed_error=max(h["fixed_error"] for h in history),
                   max_iterations=max(h["iterations"] for h in history),
                   mean_ms=float(np.mean([h["elapsed_ms"] for h in history])))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.render:
        render(output, trajectories, faces, args.dt)
    print(json.dumps(summary, indent=2))
    if args.gui:
        import polyscope as ps
        import polyscope.imgui as ui
        ps.init()
        mesh = ps.register_surface_mesh("Cloth and support", trajectories[0], faces)
        state = {"frame": 0, "play": True}

        def callback():
            _, state["play"] = ui.Checkbox("Play", state["play"])
            _, state["frame"] = ui.SliderInt("Frame", state["frame"], 0, args.frames)
            mesh.update_vertex_positions(trajectories[state["frame"]])
            if state["play"]:
                state["frame"] = (state["frame"] + 1) % (args.frames + 1)
        ps.set_user_callback(callback)
        ps.show()


if __name__ == "__main__":
    main()
