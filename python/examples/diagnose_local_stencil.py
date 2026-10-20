"""Bounded small-mesh SoS diagnostics; failures and truncations remain explicit.

Set UIPC_LOCAL_PACKAGE to a freshly built package, then run:
    python python/examples/diagnose_local_stencil.py --output output/sos-small-check
Each arm runs in a separate process, serially on the GPU, with a time limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CASES = {
    "beam-global": (False, "global", "assembled", "ipc", "monotone", False),
    "beam-assembled": (False, "local_stencil", "assembled", "ipc", "monotone", False),
    "beam-packed": (False, "local_stencil", "packed", "ipc", "monotone", False),
    "ground-global": (True, "global", "assembled", "ipc", "monotone", False),
    "ground-ipc": (True, "local_stencil", "packed", "ipc", "monotone", False),
    "ground-sos-monotone": (True, "local_stencil", "packed", "sos", "monotone", False),
    "ground-sos-local": (True, "local_stencil", "packed", "sos", "local", False),
    "ground-warm-monotone": (True, "local_stencil", "packed", "sos", "monotone", True),
    "ground-warm-local": (True, "local_stencil", "packed", "sos", "local", True),
}


def elastic_certificate(rest, x, tets, fixed):
    """Independent CPU energy/gradient and dense Newton reference, no contact."""
    import numpy as np
    dt, rho, young, poisson = .01, 1000., 1e6, .3
    mu = young / (2*(1+poisson))
    lam = young*poisson / ((1+poisson)*(1-2*poisson))
    # StableNeoHookean.apply_to stores converted coefficients on tetrahedra.
    lam, mu = lam + 5*mu/6, 4*mu/3
    p = rest[tets]
    Dm = (p[:, 1:]-p[:, :1]).transpose(0, 2, 1)
    inverse = np.linalg.inv(Dm)
    volume = np.linalg.det(Dm)/6
    mass = np.zeros(len(rest))
    for j in range(4):
        np.add.at(mass, tets[:, j], rho*volume/4)
    predicted = rest + [0, -9.8*dt*dt, 0]

    def evaluate(y):
        q = y[tets]
        F = (q[:, 1:]-q[:, :1]).transpose(0, 2, 1) @ inverse
        J = np.linalg.det(F)
        Ic = (F*F).sum(axis=(1, 2))
        energy = (.5*mass[:, None]*(y-predicted)**2).sum()
        # The active constitution is SNK1, not the legacy logarithmic variant.
        energy += dt*dt*np.dot(volume, .5*lam*(J-1-mu/lam)**2+.5*mu*(Ic-3))
        cofactor = np.stack([np.cross(F[:, :, 1], F[:, :, 2]),
                             np.cross(F[:, :, 2], F[:, :, 0]),
                             np.cross(F[:, :, 0], F[:, :, 1])], axis=2)
        P = mu*F + (lam*(J-1)-mu)[:, None, None]*cofactor
        dDs = dt*dt*volume[:, None, None]*(P @ inverse.transpose(0, 2, 1))
        gradient = mass[:, None]*(y-predicted)
        for j in range(3):
            np.add.at(gradient, tets[:, j+1], dDs[:, :, j])
        np.add.at(gradient, tets[:, 0], -dDs.sum(axis=2))
        return float(energy), gradient

    energy, gradient = evaluate(x)
    ids = np.flatnonzero(np.repeat(~fixed, 3))
    free_gradient = gradient.ravel()[ids]
    eps = 1e-6
    numeric = []
    for i in ids:
        plus, minus = x.copy(), x.copy()
        plus.flat[i] += eps
        minus.flat[i] -= eps
        numeric.append((evaluate(plus)[0]-evaluate(minus)[0])/(2*eps))
    residual = lambda g: float(np.max(np.linalg.norm(g[~fixed], axis=1)/(mass[~fixed]*dt)))
    y = rest.copy()
    for iteration in range(30):
        E, g = evaluate(y)
        if residual(g) <= 1e-8:
            break
        H = np.empty((len(ids), len(ids)))
        for j, i in enumerate(ids):
            plus, minus = y.copy(), y.copy()
            plus.flat[i] += eps
            minus.flat[i] -= eps
            H[:, j] = ((evaluate(plus)[1]-evaluate(minus)[1])/(2*eps)).ravel()[ids]
        direction = np.linalg.solve(H, -g.ravel()[ids])
        for trial in range(32):
            candidate = y.copy()
            candidate.ravel()[ids] += 2.**(-trial)*direction
            if evaluate(candidate)[0] <= E + 1e-4*2.**(-trial)*g.ravel()[ids].dot(direction):
                y = candidate
                break
        else:
            break
    return dict(energy=energy, independent_velocity_residual=residual(gradient),
                finite_difference_gradient_relative_error=float(np.linalg.norm(numeric-free_gradient)/max(np.linalg.norm(free_gradient), 1e-15)),
                dense_reference_residual=residual(evaluate(y)[1]),
                max_vertex_difference_from_dense_m=float(np.linalg.norm(x-y, axis=1).max()))


def worker(args):
    from local_stencil_beam import beam_mesh
    import numpy as np
    import uipc
    from uipc import Engine, World, Scene, Logger, view, builtin
    from uipc.geometry import tetmesh, label_surface, label_triangle_orient, ground
    from uipc.constitution import StableNeoHookean, ElasticModuli

    contact, solver, product, barrier, merge, warm = CASES[args.case]
    args.output.mkdir(parents=True, exist_ok=False)
    Logger.set_level(Logger.Level.Info)
    config = Scene.default_config()
    config["dt"] = .01
    config["gravity"] = [[0.], [-9.8], [0.]]
    config["newton"]["solver"] = solver
    config["newton"]["semi_implicit"]["enable"] = False
    config["newton"]["velocity_tol"] = .01
    config["newton"]["max_iter"] = args.rounds
    config["line_search"]["max_iter"] = 32
    config["contact"]["enable"] = contact
    config["contact"]["friction"]["enable"] = False
    config["contact"]["d_hat"] = .0005
    config["extras"]["strict_mode"]["enable"] = True
    if solver == "local_stencil":
        config["newton"]["local_stencil"].update(
            inner_sweeps=15, hessian_product=product, barrier_model=barrier,
            barrier_activation="expanded", barrier_stiffness="ipc",
            merge_acceptance=merge, hybrid_submission="sequential",
            color_coverage="independent", initial_guess="inertial",
            warm_start=int(warm), warm_start_inversion="after_local",
            warm_start_max_iter=10, iteration_limit_policy="accept_last_valid")
    scene = Scene(config)
    scene.contact_tabular().default_model(0., 1e8)
    element = scene.contact_tabular().default_element()
    rest, tets, faces = beam_mesh(args.resolution, args.resolution, args.resolution)
    rest *= [.2, .8, .8]
    rest[:, 1] += .1007 if contact else .3
    mesh = tetmesh(rest, tets)
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e6, .3), 1000.)
    fixed = (rest[:, 0] == 0) if not contact else np.zeros(len(rest), dtype=bool)
    view(mesh.vertices().find(builtin.is_fixed)).reshape(-1)[:] = fixed
    if contact:
        view(mesh.vertices().find(builtin.velocity)).reshape(-1, 3)[:, 1] = -args.impact_speed
    element.apply_to(mesh)
    slot, _ = scene.objects().create("tiny cube").geometries().create(mesh)
    if contact:
        plane = ground(0.)
        element.apply_to(plane)
        scene.objects().create("ground").geometries().create(plane)
    native = Path(uipc.__file__).resolve().parent / "_native/uipc_backend_cuda.dll"
    manifest = dict(case=args.case, config=config, vertices=len(rest), tets=len(tets),
                    impact_speed=args.impact_speed if contact else 0.,
                    backend_sha256=hashlib.sha256(native.read_bytes()).hexdigest())
    (args.output / "scene.json").write_text(json.dumps(manifest, indent=2))
    engine = Engine("cuda", str(args.output / "engine"))
    world = World(engine)
    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("Initialization failed")
    initial = rest[tets]
    reference = np.linalg.det((initial[:, 1:] - initial[:, :1]).transpose(0, 2, 1))
    start = time.perf_counter()
    world.advance()
    world.retrieve()
    elapsed = time.perf_counter() - start
    if not world.is_valid():
        raise RuntimeError("Advance failed")
    x = np.asarray(slot.geometry().positions().view()).reshape(-1, 3).copy()
    stats = engine.frame_stats()
    pts = x[tets]
    ratios = np.linalg.det((pts[:, 1:] - pts[:, :1]).transpose(0, 2, 1)) / reference
    np.savez(args.output / "state.npz", rest=rest, positions=x, tets=tets, fixed=fixed)
    report = dict(case=args.case, elapsed_s=elapsed, stats=stats,
                  min_det_F=float(ratios.min()), ground_gap=float(x[:, 1].min()),
                  fixed_error=float(np.linalg.norm(x[fixed]-rest[fixed], axis=1).max()) if fixed.any() else 0.)
    if not np.isfinite(x).all() or ratios.min() <= 0 or contact and x[:, 1].min() <= 0:
        raise RuntimeError("Invalid endpoint geometry")
    if not contact and args.resolution <= 2:
        report["elastic_certificate"] = elastic_certificate(rest, x, tets, fixed)
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        if solver == "local_stencil" and stats["converged"]:
            if report["elastic_certificate"]["independent_velocity_residual"] > .010001:
                raise RuntimeError("Engine convergence disagrees with independent CPU residual")
    (args.output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def log_metrics(path):
    stages, residuals = {}, []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"Local stencil batch: outer=(\d+), color_passes=(\d+), accepted_inner=(\d+), velocity_residual=([^,]+)", line)
        if match:
            residuals.append(dict(outer=int(match[1]), color_passes=int(match[2]),
                                  accepted_inner=int(match[3]), residual=float(match[4])))
        marker = "SoS energy audit: "
        if marker not in line:
            continue
        fields = dict(item.split("=", 1) for item in line.split(marker)[1].split(", "))
        if fields["accepted"] != "true":
            continue
        key = fields["stage"] + ("-warm" if fields["warm"] == "true" else "-elastic")
        row = stages.setdefault(key, dict(accepted=0, uphill=0, max_change=0.))
        change = float(fields["change"])
        row["accepted"] += 1
        row["uphill"] += int(change > 1e-16)
        row["max_change"] = max(row["max_change"], change)
    return dict(stage_energy=stages, full_elastic_outer_trace=residuals,
                color_passes=sum(row["color_passes"] for row in residuals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=90.)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--impact-speed", type=float, default=0.)
    args = parser.parse_args()
    if args.rounds <= 0 or args.timeout <= 0 or args.resolution <= 0 or args.impact_speed < 0:
        parser.error("rounds, timeout and resolution must be positive; impact-speed must be nonnegative")
    if args.case:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    runtime=os.environ.get("UIPC_LOCAL_PACKAGE", "build/python/src"),
                    scope="Single-frame diagnostics, not a timing benchmark or full-trajectory validation. Native and local stopping rules differ.")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("UIPC_STENCIL_", "UIPC_SHIRT_", "UIPC_SPATIAL_", "UIPC_LOCAL_CCD_", "UIPC_LOCAL_STEP_", "UIPC_STEP_BRANCH_")):
            env.pop(key)
    env["UIPC_STENCIL_ENERGY_AUDIT"] = "1"
    results = {}
    for name in args.cases:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--case", name,
                   "--rounds", str(args.rounds), "--resolution", str(args.resolution),
                   "--impact-speed", str(args.impact_speed),
                   "--output", str((args.output/name).resolve())]
        log = args.output / (name + ".log")
        start = time.perf_counter()
        with log.open("w", encoding="utf-8") as stream:
            try:
                code = subprocess.run(command, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        result_path = args.output / name / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        results[name] = dict(command=command, returncode=code,
                             process_s=time.perf_counter()-start, result=result,
                             **log_metrics(log))
        (args.output/"report.json").write_text(json.dumps(results, indent=2))
        print(name, "exit=", code, "result=", result, flush=True)


if __name__ == "__main__":
    main()
