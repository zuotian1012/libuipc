"""Independent sparse CPU check of frozen global-correction snapshots.

Requires NumPy and SciPy. Export small scenes with diagnose_local_stencil.py
--global-audit, then pass the case's global-audit directory here. This checks
assembly and linear solving at the identical frozen state; it is not a nonlinear
convergence certificate or an independent validation of constitutive derivatives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def audit(prefix: Path):
    def read(suffix, dtype=np.float64):
        return np.fromfile(str(prefix) + suffix + ".bin", dtype=dtype)

    mass = read("-mass")
    fixed = read("-fixed", np.int32).astype(bool)
    gradient, direction = read("-gradient"), read("-dx")
    n = 3 * len(mass)
    matrix = diags(np.repeat(mass, 3), format="csr")
    for topology, hessians in (("-elements", "-hessians"),
                              ("-contacts", "-contact-hessians")):
        ids = read(topology, np.int32).reshape(-1, 4)
        # Each Eigen matrix is column-major; primitive records are consecutive.
        blocks = read(hessians).reshape(-1, 12, 12).transpose(0, 2, 1)
        if len(ids) != len(blocks) or np.any(ids < -1) or np.any(ids >= len(mass)):
            raise ValueError("Invalid frozen primitive data")
        dofs = (3 * ids[..., None] + np.arange(3)).reshape(-1, 12)
        rows = np.broadcast_to(dofs[:, :, None], blocks.shape)
        cols = np.broadcast_to(dofs[:, None, :], blocks.shape)
        valid = (rows >= 0) & (cols >= 0)
        matrix += coo_matrix((blocks[valid], (rows[valid], cols[valid])),
                             shape=(n, n)).tocsr()
    free = np.flatnonzero(~np.repeat(fixed, 3))
    H = matrix[free][:, free]
    g, dx = gradient[free], direction[free]
    reference = spsolve(H, -g)
    denominator = max(np.linalg.norm(g), np.finfo(float).tiny)
    result = dict(
        snapshot=prefix.name, free_dofs=len(free),
        cpu_linear_residual=float(np.linalg.norm(H @ reference + g) / denominator),
        gpu_linear_residual=float(np.linalg.norm(H @ dx + g) / denominator),
        relative_direction_error=float(np.linalg.norm(dx - reference) /
                                       max(np.linalg.norm(reference), np.finfo(float).tiny)),
        fixed_displacement_max=float(np.abs(direction[np.repeat(fixed, 3)]).max(initial=0)),
        slope=float(g @ dx),
    )
    if not all(np.isfinite(v) for k, v in result.items() if k != "snapshot"):
        raise RuntimeError(f"Nonfinite correction audit: {result}")
    if result["fixed_displacement_max"] != 0:
        raise RuntimeError(f"Correction moved fixed vertices: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--max-relative-residual", type=float, default=0,
                        help="Optional assertion for converged positive-curvature controls; 0 reports only")
    args = parser.parse_args()
    files = sorted(args.directory.glob("*-gradient.bin"))
    if not files:
        parser.error("No global-correction snapshots found")
    records = [audit(Path(str(path).removesuffix("-gradient.bin"))) for path in files]
    target = args.directory / "cpu-audit.json"
    target.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))
    if args.max_relative_residual > 0 and any(
            r["gpu_linear_residual"] > args.max_relative_residual for r in records):
        raise RuntimeError("A frozen global direction exceeds the requested linear residual")


if __name__ == "__main__":
    main()
