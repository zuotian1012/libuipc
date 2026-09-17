"""Repeat full-baseline DCD and TOI diagnostics in fresh processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/windows/x64/release/uipc_test_sim_case.exe")
    parser.add_argument("--runtime-dir", type=Path, action="append", default=[])
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/next_step/reproduction")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.pop("UIPC_PROBE_PAIR_COUNTS", None)
    env["PATH"] = os.pathsep.join(str(p.resolve()) for p in args.runtime_dir) + os.pathsep + env.get("PATH", "")
    env.update(
        UIPC_PROBE_LAYERS=str(args.layers),
        UIPC_PROBE_CONTACT="1",
        UIPC_PROBE_GRID=str(args.grid),
        UIPC_PROBE_FRAMES=str(args.frames),
        UIPC_PROBE_TIMERS="1",
        UIPC_PROBE_PAIR_CHURN="1",
        UIPC_PROBE_DCD_CORE_TIMES="1",
        UIPC_PROBE_TOI_HIST="1",
        UIPC_PROBE_INITIAL_DOWN_SPEED="0",
    )
    for repeat in range(1, args.repeats + 1):
        prefix = args.out / f"repeat{repeat}"
        run_env = env.copy()
        frame_path = prefix.with_name(prefix.name + "_frames.txt")
        run_env["UIPC_PROBE_CSV_PATH"] = str(frame_path.resolve())
        run_env["UIPC_PROBE_TIMER_RAW_PATH"] = str(prefix.with_name(prefix.name + "_timers_raw.json").resolve())
        run_env["UIPC_PROBE_TIMER_PATH"] = str(prefix.with_name(prefix.name + "_timers_merged.json").resolve())
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
        log = prefix.with_suffix(".log")
        log.write_text(result.stdout.rstrip() + "\n", encoding="utf-8")
        frames = sum(
            line.startswith("PROBE_FRAME,")
            for line in (frame_path.read_text(encoding="utf-8").splitlines() if frame_path.exists() else [])
        )
        if result.returncode or frames != args.frames:
            raise SystemExit(f"Repeat {repeat} failed: exit={result.returncode}, frames={frames}; see {log}")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/analyze_contact_refresh_oracle.py"),
                "--log", str(log),
                "--timers", run_env["UIPC_PROBE_TIMER_RAW_PATH"],
                "--out", str(prefix.with_name(prefix.name + "_summary.json")),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
