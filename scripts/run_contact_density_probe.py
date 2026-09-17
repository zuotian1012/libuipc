"""Run the small contact-density research probe in fresh processes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "layers",
    "contact",
    "grid",
    "frame",
    "frame_ms",
    "newton",
    "linear_iterations",
    "line_search_trials",
    "last_alpha",
    "last_ccd_toi",
    "min_y",
    "min_gap",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/windows/x64/release/uipc_test_sim_case.exe")
    parser.add_argument("--runtime-dir", type=Path, action="append", default=[])
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--contacts", type=int, nargs="+", choices=[0, 1], default=[0, 1])
    parser.add_argument("--initial-down-speed", type=float, default=0.0)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/probe")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)

    env = os.environ.copy()
    for key in (
        "UIPC_PROBE_PAIR_COUNTS",
        "UIPC_PROBE_PAIR_CHURN",
        "UIPC_PROBE_TOI_HIST",
        "UIPC_PROBE_DCD_CORE_TIMES",
        "UIPC_PROBE_TIMER_PATH",
        "UIPC_PROBE_TIMER_RAW_PATH",
    ):
        env.pop(key, None)
    env["UIPC_PROBE_TIMERS"] = "0"
    env["PATH"] = os.pathsep.join(str(p.resolve()) for p in args.runtime_dir) + os.pathsep + env.get("PATH", "")
    rows: list[dict[str, float | int]] = []
    runs: list[dict[str, object]] = []
    for layers in args.layers:
        for contact in args.contacts:
            for repeat in range(1, args.repeats + 1):
                run_env = env.copy()
                run_env.update(
                    UIPC_PROBE_LAYERS=str(layers),
                    UIPC_PROBE_CONTACT=str(contact),
                    UIPC_PROBE_FRAMES=str(args.frames),
                    UIPC_PROBE_GRID=str(args.grid),
                    UIPC_PROBE_INITIAL_DOWN_SPEED=str(args.initial_down_speed),
                )
                name = f"layers{layers}_contact{contact}_repeat{repeat}"
                frame_path = args.out / "logs" / f"{name}.frames.txt"
                run_env["UIPC_PROBE_CSV_PATH"] = str(frame_path.resolve())
                result = subprocess.run(
                    [str(args.binary.resolve()), "94_contact_density_probe", "--reporter", "compact"],
                    cwd=ROOT,
                    env=run_env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                (args.out / "logs" / f"{name}.txt").write_text(result.stdout, encoding="utf-8")
                parsed = []
                frame_lines = frame_path.read_text(encoding="utf-8").splitlines() if frame_path.exists() else []
                for line in frame_lines:
                    if not line.startswith("PROBE_FRAME,"):
                        continue
                    values = line.split(",")[1:]
                    if len(values) != len(FIELDS):
                        raise RuntimeError(f"Malformed probe row in {name}: {line}")
                    row = dict(zip(FIELDS, values, strict=True))
                    parsed.append(
                        {
                            "run": name,
                            "layers": int(row["layers"]),
                            "contact": int(row["contact"]),
                            "grid": int(row["grid"]),
                            "frame": int(row["frame"]),
                            "frame_ms": float(row["frame_ms"]),
                            "newton": int(row["newton"]),
                            "linear_iterations": int(row["linear_iterations"]),
                            "line_search_trials": int(row["line_search_trials"]),
                            "last_alpha": float(row["last_alpha"]),
                            "last_ccd_toi": float(row["last_ccd_toi"]),
                            "min_y": float(row["min_y"]),
                            "min_gap": float(row["min_gap"]),
                        }
                    )
                valid = result.returncode == 0 and len(parsed) == args.frames
                if valid:
                    rows.extend(parsed)
                runs.append({"run": name, "exit_code": result.returncode, "frames": len(parsed), "valid": valid})
                print(f"{name}: exit={result.returncode} frames={len(parsed)} valid={valid}", flush=True)

    if rows:
        with (args.out / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summaries = []
    for layers in args.layers:
        for contact in args.contacts:
            run_medians = []
            run_newton = []
            run_linear = []
            for repeat in range(1, args.repeats + 1):
                name = f"layers{layers}_contact{contact}_repeat{repeat}"
                measured = [r for r in rows if r["run"] == name and r["frame"] > 5]
                if measured:
                    run_medians.append(statistics.median(r["frame_ms"] for r in measured))
                    run_newton.append(statistics.median(r["newton"] for r in measured))
                    run_linear.append(statistics.median(r["linear_iterations"] for r in measured))
            summaries.append(
                {
                    "layers": layers,
                    "contact": contact,
                    "valid_runs": len(run_medians),
                    "median_frame_ms": statistics.median(run_medians) if run_medians else None,
                    "range_frame_ms": [min(run_medians), max(run_medians)] if run_medians else None,
                    "median_newton": statistics.median(run_newton) if run_newton else None,
                    "median_linear_iterations": statistics.median(run_linear) if run_linear else None,
                }
            )
    report = {"configuration": {"frames": args.frames, "repeats": args.repeats, "layers": args.layers, "contacts": args.contacts, "grid": args.grid, "initial_down_speed": args.initial_down_speed}, "runs": runs, "summary": summaries}
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    if not all(run["valid"] for run in runs):
        raise SystemExit("At least one run failed; see logs and summary.json")


if __name__ == "__main__":
    main()
