"""Independently validate a complete bunny/cloth run and optionally render it.

The saved-surface checks include cross-body and nonincident self pairs. They
are independent of the CUDA collision filters, but are not continuous CCD.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree


def dot(a, b):
    return np.sum(a * b, axis=-1)


def point_edge(p, a, b):
    edge = b - a
    t = np.clip(dot(p - a, edge) / dot(edge, edge), 0, 1)
    delta = p - a - t[:, None] * edge
    return dot(delta, delta)


def point_triangle(p, triangle):
    a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    u, v, w = b - a, c - a, p - a
    uu, uv, vv = dot(u, u), dot(u, v), dot(v, v)
    wu, wv = dot(w, u), dot(w, v)
    denominator = uu * vv - uv * uv
    s = (vv * wu - uv * wv) / denominator
    t = (uu * wv - uv * wu) / denominator
    normal = np.cross(u, v)
    plane = dot(w, normal) ** 2 / dot(normal, normal)
    edges = np.minimum.reduce([point_edge(p, a, b), point_edge(p, b, c), point_edge(p, c, a)])
    return np.where((s >= 0) & (t >= 0) & (s + t <= 1), plane, edges)


def edge_edge(a, b, c, d):
    u, v, w = b - a, d - c, a - c
    aa, bb, cc, dd, ee = dot(u, u), dot(u, v), dot(v, v), dot(u, w), dot(v, w)
    denominator = aa * cc - bb * bb
    nonparallel = denominator > 1e-12 * aa * cc
    safe = np.where(nonparallel, denominator, 1)
    s, t = (bb * ee - cc * dd) / safe, (aa * ee - bb * dd) / safe
    delta = w + s[:, None] * u - t[:, None] * v
    ends = np.minimum.reduce([point_edge(a, c, d), point_edge(b, c, d),
                              point_edge(c, a, b), point_edge(d, a, b)])
    return np.where(nonparallel & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1),
                    np.minimum(dot(delta, delta), ends), ends)


def segment_triangle(a, b, triangle):
    u, v = triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
    d, w = b - a, a - triangle[:, 0]
    h = np.cross(d, v)
    determinant = dot(u, h)
    nonparallel = abs(determinant) > 1e-12 * np.sqrt(dot(u, u) * dot(d, d) * dot(v, v))
    safe = np.where(nonparallel, determinant, 1)
    s = dot(w, h) / safe
    cross = np.cross(w, u)
    t, r = dot(d, cross) / safe, dot(v, cross) / safe
    return nonparallel & (s >= 0) & (t >= 0) & (s + t <= 1) & (r >= 0) & (r <= 1)


def check_surfaces(x, first, second, clearance, same=False):
    a, b = x[first], x[second]
    ca, cb = a.mean(axis=1), b.mean(axis=1)
    ra = np.linalg.norm(a - ca[:, None], axis=2).max(axis=1)
    rb = np.linalg.norm(b - cb[:, None], axis=2).max(axis=1)
    tree = cKDTree(cb)
    margin = max(clearance * 1.05, 1e-10)
    amin, amax, bmin, bmax = a.min(axis=1), a.max(axis=1), b.min(axis=1), b.max(axis=1)
    minimum, intersections, pairs = np.inf, 0, 0
    for start in range(0, len(a), 128):
        nearby = tree.query_ball_point(ca[start:start + 128], ra[start:start + 128] + rb.max() + margin)
        lengths = np.array([len(ids) for ids in nearby])
        if not lengths.sum():
            continue
        i = np.repeat(np.arange(start, start + len(nearby)), lengths)
        j = np.concatenate([ids for ids in nearby if len(ids)])
        keep = np.all((amin[i] <= bmax[j] + margin) & (amax[i] >= bmin[j] - margin), axis=1)
        if same:
            keep &= i < j
            keep &= ~np.any(first[i, :, None] == second[j, None, :], axis=(1, 2))
        i, j = i[keep], j[keep]
        if not len(i):
            continue
        p, q = a[i], b[j]
        distance, hit = np.full(len(i), np.inf), np.zeros(len(i), dtype=bool)
        for k in range(3):
            distance = np.minimum(distance, point_triangle(p[:, k], q))
            distance = np.minimum(distance, point_triangle(q[:, k], p))
            hit |= segment_triangle(p[:, k], p[:, (k + 1) % 3], q)
            hit |= segment_triangle(q[:, k], q[:, (k + 1) % 3], p)
            for l in range(3):
                distance = np.minimum(distance, edge_edge(p[:, k], p[:, (k + 1) % 3],
                                                          q[:, l], q[:, (l + 1) % 3]))
        if not np.isfinite(distance).all():
            raise RuntimeError("Nonfinite independent surface distance")
        minimum = min(minimum, float(np.sqrt(distance.min())))
        intersections += int(hit.sum())
        pairs += len(i)
    if intersections or minimum <= max(clearance, 1e-12):
        raise RuntimeError(f"Surface check failed: clearance={clearance}, distance={minimum}, intersections={intersections}")
    return dict(candidate_pairs=pairs, intersections=intersections,
                minimum_candidate_distance_m=None if np.isinf(minimum) else minimum,
                certified_clearance_m=clearance, candidate_margin_m=margin)


def history(run):
    path = run / "history.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def update(root, state, **extra):
    rows = history(root / "local_stencil")
    data = dict(state=state, completed_frames=len(rows), target_frames=120,
                updated_unix=time.time(), **extra)
    log = root / "simulation.log"
    if state == "simulating" and log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 16384))
            tail = stream.read().decode(errors="replace")
        batches = re.findall(r"batch: outer=(\d+), color_passes=\d+, accepted_inner=\d+, velocity_residual=([^,]+)", tail)
        if batches:
            data["current_outer_iteration"] = int(batches[-1][0])
            data["current_outer_velocity_residual"] = float(batches[-1][1])
    temp = root / "completion-status.tmp"
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(root / "completion-status.json")


def wait_for_process(root, pid):
    if os.name != "nt":
        raise RuntimeError("--wait-pid currently requires Windows; validation itself is portable")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        while True:
            update(root, "simulating", simulation_pid=pid)
            result = kernel.WaitForSingleObject(handle, 15000)
            if result == 0:
                return
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def validate(root):
    run = root / "local_stencil"
    rows = history(run)
    manifest = json.loads((run / "scene.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if len(rows) != 120 or [int(r["frame"]) for r in rows] != list(range(1, 121)):
        raise RuntimeError("Expected 120 consecutive frames; partial runs cannot pass")
    if "checkpoint_origin" in manifest:
        raise RuntimeError("Expected a fresh-scene run, not an inherited checkpoint prefix")
    if not summary["all_converged"] or any(r["converged"] != "True" for r in rows):
        raise RuntimeError("A simulation frame failed convergence")
    parameters = manifest["parameters"]
    if parameters["stopping"] != "velocity" or parameters["tolerance"] != .01:
        raise RuntimeError("Expected the unchanged strict velocity residual threshold 0.01")
    certificates = {}
    current = None
    batch = re.compile(r"batch: outer=(\d+), color_passes=(\d+), accepted_inner=(\d+), velocity_residual=([^,]+), gradient_l2=([^\s]+)")
    for line in (root / "simulation.log").open(errors="replace"):
        if ">>> Begin Frame:" in line:
            current = int(line.split("Begin Frame:")[1])
            certificates[current] = dict(color_passes=0, global_rounds=0)
        match = batch.search(line)
        if match and current is not None:
            record = certificates[current]
            record["color_passes"] += int(match[2])
            record["global_rounds"] += 1
            record["velocity_residual"] = float(match[4])
            record["gradient_l2"] = float(match[5])
    if set(certificates) != set(range(1, 121)):
        raise RuntimeError("Missing per-frame residual certificates in the simulation log")
    data = np.load(run / "positions.npz")
    if len(data["positions"]) != 121:
        raise RuntimeError("Expected initial state plus 120 full-precision saved states")
    checks = []
    tets, cloth_faces = data["tetrahedra"], data["cloth_faces"]
    rest = data["positions"][0]
    rest_tets, rest_tri = rest[tets], rest[cloth_faces]
    rest_det = np.linalg.det((rest_tets[:, 1:] - rest_tets[:, :1]).transpose(0, 2, 1))
    rest_area = np.linalg.norm(np.cross(rest_tri[:, 1] - rest_tri[:, 0], rest_tri[:, 2] - rest_tri[:, 0]), axis=1)
    bunny_count = int(data["bunny_vertices"])
    for frame, x in enumerate(data["positions"]):
        if not np.array_equal(x, np.load(run / "frames" / f"{frame:04d}.npy")) or not np.isfinite(x).all():
            raise RuntimeError(f"Inconsistent/nonfinite saved positions at frame {frame}")
        if frame:
            c = certificates[frame]
            if not np.isfinite(c["velocity_residual"]) or c["velocity_residual"] > .01:
                raise RuntimeError(f"Strict residual failed at frame {frame}: {c}")
            stats = json.loads((run / "frame-stats" / f"{frame:04d}.json").read_text())
            if not stats["converged"] or stats["hit_newton_limit"] or stats["hit_line_search_limit"]:
                raise RuntimeError(f"Native certification failed at frame {frame}")
            for key in ["min_det_F", "min_area_ratio", "bunny_ground_gap", "cloth_ground_gap"]:
                if not float(rows[frame - 1][key]) > 0:
                    raise RuntimeError(f"Invalid {key} at frame {frame}")
        record = dict(frame=frame)
        tet, tri = x[tets], x[cloth_faces]
        record["min_det_F"] = float((np.linalg.det((tet[:, 1:] - tet[:, :1]).transpose(0, 2, 1)) / rest_det).min())
        record["min_area_ratio"] = float((np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / rest_area).min())
        record["bunny_ground_gap"] = float(x[:bunny_count, 1].min())
        record["cloth_ground_gap"] = float(x[bunny_count:, 1].min() - .001)
        if any(not np.isfinite(value) or value <= 0 for key, value in record.items() if key != "frame"):
            raise RuntimeError(f"Independent saved-state geometry check failed: {record}")
        for name, a, b, gap, same in [
            ("cloth_bunny", data["cloth_faces"], data["bunny_faces"], .001, False),
            ("cloth_self", data["cloth_faces"], data["cloth_faces"], .002, True),
            ("bunny_self", data["bunny_faces"], data["bunny_faces"], 0., True),
        ]:
            record[name] = check_surfaces(x, a, b, gap, same)
        checks.append(record)
        update(root, "checking_saved_surfaces", checked_states=frame + 1)
    report = dict(passed=True, saved_states=121, frame_certificates=certificates,
                  surfaces=checks, summary=summary,
                  resume_manifests=[json.loads(p.read_text()) for p in sorted(run.glob("resume-*.json"))],
                  scope="Saved-state checks include cross-body and nonincident self pairs. Incident self triangles are excluded. This does not independently certify continuous trajectories.")
    (root / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Demo output parent containing local_stencil/")
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.wait_pid:
            wait_for_process(root, args.wait_pid)
        validate(root)
        if args.render:
            update(root, "rendering")
            subprocess.run([sys.executable, str(Path(__file__).with_name("local_stencil_bunny_cloth.py")),
                            "--solver", "sos", "--replay", "--render", "--output", str(root)], check=True)
        update(root, "complete", validation=str(root / "validation.json"))
    except Exception as error:
        update(root, "incomplete", reason=str(error))
        raise


if __name__ == "__main__":
    main()
