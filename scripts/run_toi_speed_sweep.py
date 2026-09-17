"""Run repeated cloth-impact TOI diagnostics without changing the solver."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOI = re.compile(
    r"PROBE_TOI,frame=(\d+),newton=(\d+),type=(\w+),count=(\d+),"
    r"min=([^,]+),lt_01=(\d+),lt_05=(\d+),lt_09=(\d+),lt_1=(\d+),nonfinite=(\d+)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/windows/x64/release/uipc_test_sim_case.exe")
    parser.add_argument("--runtime-dir", type=Path, action="append", default=[])
    parser.add_argument("--speeds", type=float, nargs="+", default=[0, 1, 2, 4, 8])
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timers", action="store_true")
    parser.add_argument("--no-toi-hist", action="store_true", help="Time stages without host-side TOI logging")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/next_step/toi_speed_sweep")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(str(p.resolve()) for p in args.runtime_dir) + os.pathsep + env.get("PATH", "")
    for key in (
        "UIPC_PROBE_TIMERS",
        "UIPC_PROBE_TOI_HIST",
        "UIPC_PROBE_PAIR_CHURN",
        "UIPC_PROBE_DCD_CORE_TIMES",
        "UIPC_PROBE_PAIR_COUNTS",
        "UIPC_PROBE_TIMER_RAW_PATH",
        "UIPC_PROBE_TIMER_PATH",
    ):
        env.pop(key, None)
    env.update(
        UIPC_PROBE_LAYERS=str(args.layers),
        UIPC_PROBE_CONTACT="1",
        UIPC_PROBE_GRID=str(args.grid),
        UIPC_PROBE_FRAMES=str(args.frames),
    )
    if not args.no_toi_hist:
        env["UIPC_PROBE_TOI_HIST"] = "1"
    runs = []
    for speed in args.speeds:
        for repeat in range(1, args.repeats + 1):
            name = f"speed{speed:g}_repeat{repeat}"
            run_env = env.copy()
            run_env["UIPC_PROBE_INITIAL_DOWN_SPEED"] = str(speed)
            frame_path = args.out / f"{name}.frames.txt"
            run_env["UIPC_PROBE_CSV_PATH"] = str(frame_path.resolve())
            if args.timers:
                run_env["UIPC_PROBE_TIMERS"] = "1"
                run_env["UIPC_PROBE_TIMER_PATH"] = str((args.out / f"{name}_timers.json").resolve())
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
            (args.out / f"{name}.log").write_text(result.stdout.rstrip() + "\n", encoding="utf-8")
            frame_lines = [
                line
                for line in (frame_path.read_text(encoding="utf-8").splitlines() if frame_path.exists() else [])
                if line.startswith("PROBE_FRAME,")
            ]
            toi = [match.groups() for match in TOI.finditer(result.stdout)]
            ph_toi = [row for row in toi if row[2] == "PH"]
            valid = result.returncode == 0 and len(frame_lines) == args.frames
            hits = [row for row in ph_toi if int(row[8]) > 0]
            toi_by_type = {}
            for kind in ("PH", "PP", "PE", "PT", "EE"):
                records = [row for row in toi if row[2] == kind]
                toi_by_type[kind] = {
                    "evaluations": sum(int(row[3]) for row in records),
                    "min_toi": min((float(row[4]) for row in records), default=None),
                    "evaluations_below_1": sum(int(row[8]) for row in records),
                    "calls_with_toi_below_1": sum(int(row[8]) > 0 for row in records),
                }
            run = {
                "name": name,
                "speed_m_per_s": speed,
                "valid": valid,
                "frames": len(frame_lines),
                "last_frame_min_y": float(frame_lines[-1].split(",")[-2]) if frame_lines else None,
            }
            run.update(
                toi_calls=len(ph_toi) if not args.no_toi_hist else None,
                min_ph_toi=min((float(row[4]) for row in ph_toi), default=None) if not args.no_toi_hist else None,
                first_ph_hit_frame=min((int(row[0]) for row in hits), default=None) if not args.no_toi_hist else None,
                ph_evaluations_below_1=sum(int(row[8]) for row in ph_toi) if not args.no_toi_hist else None,
                toi_by_type=toi_by_type if not args.no_toi_hist else None,
            )
            if args.timers and valid:
                def stages(node: dict, result: dict[str, float]) -> None:
                    result[node["name"]] = result.get(node["name"], 0.0) + 1000.0 * node["duration"]
                    for child in node.get("children", []):
                        stages(child, result)

                stage_ms: dict[str, float] = {}
                stages(json.loads(Path(run_env["UIPC_PROBE_TIMER_PATH"]).read_text(encoding="utf-8")), stage_ms)
                run["stage_ms"] = {
                    key: stage_ms.get(key, 0.0)
                    for key in ("Pipeline", "Detect DCD Candidates", "Detect Trajectory Candidates", "Filter CCD TOI", "Compute DyTopo Effect", "FusedPCG")
                }
            runs.append(run)
            print(json.dumps(run), flush=True)
    (args.out / "summary.json").write_text(json.dumps(runs, indent=2) + "\n", encoding="utf-8")
    if not all(run["valid"] for run in runs):
        raise SystemExit("At least one run failed; see preserved logs and summary.json")


if __name__ == "__main__":
    main()
