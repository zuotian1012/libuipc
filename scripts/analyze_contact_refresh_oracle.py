"""Classify full-baseline DCD calls by whether they found new active pairs.

This is an offline diagnostic. A zero-new call is not a certificate that
collision detection could have been skipped in a production simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


CHURN = re.compile(
    r"PROBE_CHURN,scope=(\w+),frame=(\d+),newton=(\d+),"
    r"previous=(\d+),current=(\d+),new=(\d+)"
)
TOI = re.compile(
    r"PROBE_TOI,frame=(\d+),newton=(\d+),type=(\w+),count=(\d+),"
    r"min=([^,]+),lt_01=(\d+),lt_05=(\d+),lt_09=(\d+),lt_1=(\d+),nonfinite=(\d+)"
)


def child(node: dict, name: str) -> dict | None:
    return next((item for item in node.get("children", []) if item["name"] == name), None)


def dcd_cost(node: dict) -> float:
    names = {"Probe DCD Detect Core", "Probe DCD Filter Core"}
    parts = [item for item in node["children"] if item["name"] in names]
    if len(parts) != 2:
        raise ValueError(f"Missing DCD core timers: {node}")
    return 1000.0 * sum(item["duration"] for item in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--timers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    log_text = args.log.read_text(encoding="utf-8")
    churn: dict[tuple[int, str, int], dict[str, int]] = {}
    for match in CHURN.finditer(log_text):
        scope, frame, newton, previous, current, new = match.groups()
        key = (int(frame), scope, int(newton) if scope == "within" else 0)
        if key in churn:
            raise ValueError(f"Duplicate churn record: {key}")
        churn[key] = {"previous": int(previous), "current": int(current), "new": int(new)}

    timers = json.loads(args.timers.read_text(encoding="utf-8"))
    rows = []
    pipeline_ms = 0.0
    for frame, pipeline in enumerate(timers, start=1):
        if pipeline["name"] != "Pipeline":
            raise ValueError(f"Unexpected root timer: {pipeline['name']}")
        pipeline_ms += 1000.0 * pipeline["duration"]
        simulation = child(pipeline, "Simulation")
        if simulation is None:
            raise ValueError(f"Missing Simulation timer in frame {frame}")
        initial = child(simulation, "Detect DCD Candidates")
        if initial is None:
            raise ValueError(f"Missing initial DCD timer in frame {frame}")
        calls = [("initial", 0, initial)]
        for iteration, item in enumerate(
            item for item in simulation["children"] if item["name"] == "Newton Iteration"
        ):
            dcd = child(item, "Detect DCD Candidates")
            if dcd is not None:
                calls.append(("within", iteration, dcd))
        for scope, iteration, node in calls:
            key = (frame, "across" if scope == "initial" else "within", iteration)
            pair_change = churn.get(key)
            if frame > 1 and pair_change is None:
                raise ValueError(f"Missing churn record: {key}")
            if frame == 1 and scope == "within" and pair_change is None:
                raise ValueError(f"Missing within-frame churn record: {key}")
            rows.append(
                {
                    "frame": frame,
                    "scope": scope,
                    "newton": iteration,
                    "core_ms": dcd_cost(node),
                    "previous": pair_change["previous"] if pair_change else None,
                    "current": pair_change["current"] if pair_change else None,
                    "new": pair_change["new"] if pair_change else None,
                }
            )

    def stats(selected: list[dict]) -> dict:
        labeled = [row for row in selected if row["new"] is not None]
        zero = [row for row in labeled if row["new"] == 0]
        zero_nonempty = [row for row in zero if row["current"] > 0]
        return {
            "calls": len(selected),
            "labeled_calls": len(labeled),
            "core_ms": sum(row["core_ms"] for row in selected),
            "zero_new_calls": len(zero),
            "zero_new_core_ms": sum(row["core_ms"] for row in zero),
            "zero_new_nonempty_calls": len(zero_nonempty),
            "zero_new_nonempty_core_ms": sum(row["core_ms"] for row in zero_nonempty),
        }

    groups = {
        "all": rows,
        "within": [row for row in rows if row["scope"] == "within"],
        "within_frames_6_40": [row for row in rows if row["scope"] == "within" and row["frame"] >= 6],
        "initial": [row for row in rows if row["scope"] == "initial"],
        "settled_frames_33_40": [row for row in rows if row["frame"] >= 33],
    }
    report = {
        "pipeline_ms_diagnostic": pipeline_ms,
        "scope_note": "Zero-new labels come from full baseline active-pair sets; they do not authorize skipping DCD.",
        "groups": {name: stats(group) for name, group in groups.items()},
    }
    toi_by_type: dict[str, dict] = {}
    for match in TOI.finditer(log_text):
        frame, newton, kind, count, minimum, lt_01, lt_05, lt_09, lt_1, nonfinite = match.groups()
        group = toi_by_type.setdefault(
            kind,
            {
                "calls": 0,
                "evaluations": 0,
                "min_toi": 1.1,
                "lt_01": 0,
                "lt_05": 0,
                "lt_09": 0,
                "lt_1": 0,
                "nonfinite": 0,
                "calls_with_toi_below_1": 0,
                "first_hit_frame": None,
            },
        )
        group["calls"] += 1
        group["evaluations"] += int(count)
        group["min_toi"] = min(group["min_toi"], float(minimum))
        for key, value in (
            ("lt_01", lt_01),
            ("lt_05", lt_05),
            ("lt_09", lt_09),
            ("lt_1", lt_1),
            ("nonfinite", nonfinite),
        ):
            group[key] += int(value)
        if int(lt_1):
            group["calls_with_toi_below_1"] += 1
            if group["first_hit_frame"] is None:
                group["first_hit_frame"] = int(frame)
    report["toi_by_type"] = toi_by_type
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
