#!/usr/bin/env python3
"""Benchmark counterfactual instant-I/O factor and I/O/compute overlap for Kangaroo workflows.

Runs the same workflow command in two modes:
- normal (dataset-backed I/O as-is)
- preload_inputs (input chunks preloaded before runtime/execute_plan)

The workflow must emit task events via KANGAROO_EVENT_LOG and include
runtime/execute_plan spans (e.g. scripts/plotfile_slice.py, plotfile_projection.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple


def _parse_runtime_execute_plan(event_log: Path) -> float:
    spans: List[Tuple[float, float]] = []
    with event_log.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{event_log}:{lineno}: invalid JSON: {exc}") from exc
            if event.get("type") != "task":
                continue
            if event.get("name") != "runtime/execute_plan":
                continue
            if str(event.get("status", "")) not in {"end", "error"}:
                continue
            try:
                start = float(event.get("start", event.get("ts", 0.0)))
                end = float(event.get("end", start))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{event_log}:{lineno}: non-numeric start/end") from exc
            if end < start:
                start, end = end, start
            if end > start:
                spans.append((start, end))
    if not spans:
        raise ValueError(f"No completed runtime/execute_plan span found in {event_log}")
    start = min(s for s, _ in spans)
    end = max(e for _, e in spans)
    return end - start


def _parse_named_span(event_log: Path, span_name: str) -> float:
    spans: List[Tuple[float, float]] = []
    with event_log.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{event_log}:{lineno}: invalid JSON: {exc}") from exc
            if event.get("type") != "task":
                continue
            if event.get("name") != span_name:
                continue
            if str(event.get("status", "")) not in {"end", "error"}:
                continue
            try:
                start = float(event.get("start", event.get("ts", 0.0)))
                end = float(event.get("end", start))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{event_log}:{lineno}: non-numeric start/end") from exc
            if end < start:
                start, end = end, start
            if end > start:
                spans.append((start, end))
    if not spans:
        return 0.0
    start = min(s for s, _ in spans)
    end = max(e for _, e in spans)
    return end - start


def _run_once(command: List[str], mode: str, run_idx: int, keep_logs: bool, out_dir: Path) -> Tuple[float, float]:
    log_path = out_dir / f"events_{mode}_{run_idx}.jsonl"
    env = os.environ.copy()
    env["KANGAROO_EVENT_LOG"] = str(log_path)
    env["KANGAROO_IO_MODE"] = mode

    proc = subprocess.run(command, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Workflow command failed\n"
            f"mode={mode} run={run_idx} returncode={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    runtime_s = _parse_runtime_execute_plan(log_path)
    preload_s = _parse_named_span(log_path, "setup/preload_inputs") if mode == "preload_inputs" else 0.0
    if not keep_logs:
        log_path.unlink(missing_ok=True)
    return runtime_s, preload_s


def _run_mode(
    command: List[str],
    mode: str,
    warmups: int,
    runs: int,
    keep_logs: bool,
    out_dir: Path,
) -> Tuple[List[float], List[float]]:
    runtime_samples: List[float] = []
    preload_samples: List[float] = []
    for i in range(warmups):
        _run_once(command, mode, -1 - i, keep_logs, out_dir)

    for i in range(runs):
        runtime_s, preload_s = _run_once(command, mode, i, keep_logs, out_dir)
        runtime_samples.append(runtime_s)
        preload_samples.append(preload_s)
        print(
            f"mode={mode} run={i+1}/{runs} runtime_execute_plan_s={runtime_s:.6f}"
            + (f" preload_inputs_s={preload_s:.6f}" if mode == "preload_inputs" else "")
        )
    return runtime_samples, preload_samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark counterfactual instant-I/O factor and overlap-vs-serialized-I/O factor"
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of measured runs per mode")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per mode")
    parser.add_argument(
        "--keep-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep per-run JSONL logs under --out-dir",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for logs (default: temporary dir)",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Workflow command after --, e.g. -- pixi run python scripts/plotfile_slice.py /path/to/plt",
    )
    args = parser.parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmups < 0:
        raise ValueError("--warmups must be >= 0")

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Missing workflow command. Pass it after --")

    if args.out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="kangaroo-io-bench-"))
    else:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    normal_runtime, _ = _run_mode(command, "normal", args.warmups, args.runs, args.keep_logs, out_dir)
    preload_runtime, preload_setup = _run_mode(
        command, "preload_inputs", args.warmups, args.runs, args.keep_logs, out_dir
    )

    med_normal = statistics.median(normal_runtime)
    med_preload_runtime = statistics.median(preload_runtime)
    med_preload_setup = statistics.median(preload_setup)
    speedup = math.inf if med_preload_runtime == 0.0 else med_normal / med_preload_runtime
    overlap_factor = (
        math.inf
        if med_normal == 0.0
        else (med_preload_setup + med_preload_runtime) / med_normal
    )

    print(f"normal_median_runtime_execute_plan_s={med_normal:.6f}")
    print(f"preload_median_runtime_execute_plan_s={med_preload_runtime:.6f}")
    print(f"preload_median_setup_preload_inputs_s={med_preload_setup:.6f}")
    print(
        f"counterfactual_instant_io_factor_x={speedup:.6f}"
        if math.isfinite(speedup)
        else "counterfactual_instant_io_factor_x=inf"
    )
    print(
        f"overlap_vs_serialized_io_factor_x={overlap_factor:.6f}"
        if math.isfinite(overlap_factor)
        else "overlap_vs_serialized_io_factor_x=inf"
    )
    print(f"logs_dir={out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
