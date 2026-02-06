#!/usr/bin/env python3
"""Estimate I/O slowdown from Kangaroo workflow event logs.

Metrics:
- Conservative lower bound: remove only globally idle I/O wait wall-time.
- Scheduler replay: set per-task wait_inputs/wait_outputs to zero and replay DAG
  with observed locality/worker capacities.
- Critical-path upper bound: same durations, infinite workers.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


@dataclass
class Interval:
    name: str
    start: float
    end: float


@dataclass
class RawTaskEvent:
    event_id: str
    name: str
    start: float
    end: float
    locality: int
    worker: str


@dataclass
class TaskNode:
    key: Tuple[int, int, int, int]
    stage_idx: int
    tmpl_idx: int
    block_id: int
    locality: int
    worker: str
    observed_duration: float
    cf_duration: float
    wait_inputs: float
    wait_outputs: float
    kernel: float


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip_range(start: float, end: float, t0: float, t1: float) -> Optional[Tuple[float, float]]:
    s = max(start, t0)
    e = min(end, t1)
    if e <= s:
        return None
    return s, e


def _parse_base_id(event_id: str) -> Optional[Tuple[int, int, int, int]]:
    parts = event_id.split(":")
    if len(parts) < 4:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None


def _load_task_events(path: Path) -> List[RawTaskEvent]:
    out: List[RawTaskEvent] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if event.get("type") != "task":
                continue
            status = str(event.get("status", ""))
            if status not in {"end", "error"}:
                continue
            try:
                start = float(event.get("start", event.get("ts", 0.0)))
                end = float(event.get("end", start))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: non-numeric start/end") from exc
            if end < start:
                start, end = end, start
            if end == start:
                continue
            out.append(
                RawTaskEvent(
                    event_id=str(event.get("id", "")),
                    name=str(event.get("name", "")),
                    start=start,
                    end=end,
                    locality=_safe_int(event.get("locality"), -1),
                    worker=str(event.get("worker", "")),
                )
            )
    if not out:
        raise ValueError(f"No completed task events found in {path}")
    return out


def _analysis_window(events: List[RawTaskEvent]) -> Tuple[float, float]:
    runtime_spans = [ev for ev in events if ev.name == "runtime/execute_plan"]
    if runtime_spans:
        span = max(runtime_spans, key=lambda ev: ev.end - ev.start)
        return span.start, span.end
    starts = [ev.start for ev in events]
    ends = [ev.end for ev in events]
    return min(starts), max(ends)


def _union_duration(ranges: Iterable[Tuple[float, float]]) -> float:
    pts = sorted((s, e) for s, e in ranges if e > s)
    if not pts:
        return 0.0
    total = 0.0
    cur_s, cur_e = pts[0]
    for s, e in pts[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def _blocked_time(io_waits: List[Tuple[float, float]], kernels: List[Tuple[float, float]]) -> float:
    markers: List[Tuple[float, int, int]] = []
    for s, e in io_waits:
        if e > s:
            markers.append((s, +1, 0))
            markers.append((e, -1, 0))
    for s, e in kernels:
        if e > s:
            markers.append((s, 0, +1))
            markers.append((e, 0, -1))
    if not markers:
        return 0.0

    markers.sort(key=lambda item: (item[0], 0 if (item[1] < 0 or item[2] < 0) else 1))

    blocked = 0.0
    wait_active = 0
    kernel_active = 0
    prev_t = markers[0][0]
    idx = 0
    n = len(markers)
    while idx < n:
        t = markers[idx][0]
        if t > prev_t and wait_active > 0 and kernel_active == 0:
            blocked += t - prev_t
        while idx < n and markers[idx][0] == t:
            _, dw, dk = markers[idx]
            wait_active += dw
            kernel_active += dk
            idx += 1
        prev_t = t
    return blocked


def _load_plan(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"{path}: plan JSON missing top-level 'stages' list")
    return payload


def _domain_blocks(domain: Dict[str, Any] | None) -> Optional[List[int]]:
    if not domain:
        return None
    blocks = domain.get("blocks")
    if isinstance(blocks, list) and blocks:
        return [int(b) for b in blocks]
    return None


def _build_nodes(events: List[RawTaskEvent], t0: float, t1: float) -> Tuple[Dict[Tuple[int, int, int, int], TaskNode], Dict[int, set[str]]]:
    agg: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    workers_by_loc: Dict[int, set[str]] = defaultdict(set)

    for ev in events:
        clipped = _clip_range(ev.start, ev.end, t0, t1)
        if clipped is None:
            continue
        base = _parse_base_id(ev.event_id)
        if base is None:
            continue
        s, e = clipped
        dur = e - s
        if dur <= 0.0:
            continue
        plan_id, stage_idx, tmpl_idx, block_id = base
        entry = agg.setdefault(
            base,
            {
                "stage_idx": stage_idx,
                "tmpl_idx": tmpl_idx,
                "block_id": block_id,
                "locality": ev.locality,
                "worker": ev.worker,
                "observed": 0.0,
                "wait_inputs": 0.0,
                "wait_outputs": 0.0,
                "kernel": 0.0,
            },
        )

        if ev.locality >= 0:
            entry["locality"] = ev.locality
        if ev.worker:
            entry["worker"] = ev.worker

        if ev.event_id.count(":") == 3:
            entry["observed"] += dur
        if ev.name.endswith("/wait_inputs"):
            entry["wait_inputs"] += dur
        elif ev.name.endswith("/wait_outputs"):
            entry["wait_outputs"] += dur
        elif ev.name.endswith("/kernel"):
            entry["kernel"] += dur

    nodes: Dict[Tuple[int, int, int, int], TaskNode] = {}
    for key, data in agg.items():
        observed = float(data["observed"])
        wait_inputs = float(data["wait_inputs"])
        wait_outputs = float(data["wait_outputs"])
        kernel = float(data["kernel"])

        if observed <= 0.0:
            observed = wait_inputs + wait_outputs + kernel
        io_wait = wait_inputs + wait_outputs
        cf_duration = max(observed - io_wait, kernel, 0.0)

        node = TaskNode(
            key=key,
            stage_idx=int(data["stage_idx"]),
            tmpl_idx=int(data["tmpl_idx"]),
            block_id=int(data["block_id"]),
            locality=int(data["locality"]),
            worker=str(data["worker"]),
            observed_duration=observed,
            cf_duration=cf_duration,
            wait_inputs=wait_inputs,
            wait_outputs=wait_outputs,
            kernel=kernel,
        )
        nodes[key] = node
        if node.locality >= 0 and node.worker:
            workers_by_loc[node.locality].add(node.worker)

    if not nodes:
        raise ValueError("No runtime task nodes found in selected analysis window")
    return nodes, workers_by_loc


def _build_edges(plan: Dict[str, Any], nodes: Dict[Tuple[int, int, int, int], TaskNode]) -> Dict[Tuple[int, int, int, int], set[Tuple[int, int, int, int]]]:
    stages = plan["stages"]

    stage_parents: Dict[int, List[int]] = {}
    for s_idx, stage in enumerate(stages):
        after = stage.get("after") or []
        stage_parents[s_idx] = [int(v) for v in after]

    stage_template_nodes: Dict[int, Dict[int, List[Tuple[int, int, int, int]]]] = defaultdict(lambda: defaultdict(list))
    stage_nodes: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
    for key, node in nodes.items():
        stage_nodes[node.stage_idx].append(key)
        stage_template_nodes[node.stage_idx][node.tmpl_idx].append(key)

    parents_of: Dict[Tuple[int, int, int, int], set[Tuple[int, int, int, int]]] = defaultdict(set)

    for child_stage_idx, parent_stages in stage_parents.items():
        if not parent_stages:
            continue
        child_stage = stages[child_stage_idx]
        child_plane = str(child_stage.get("plane", ""))
        child_templates = child_stage.get("templates") or []
        child_tasks = stage_nodes.get(child_stage_idx, [])

        for parent_stage_idx in parent_stages:
            parent_stage = stages[parent_stage_idx]
            parent_plane = str(parent_stage.get("plane", ""))
            parent_templates = parent_stage.get("templates") or []
            parent_tasks = stage_nodes.get(parent_stage_idx, [])

            data_edges = 0
            if (
                child_plane == "chunk"
                and parent_plane == "chunk"
                and child_templates
                and parent_templates
            ):
                output_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
                for p_tmpl_idx, p_tmpl in enumerate(parent_templates):
                    outputs = p_tmpl.get("outputs") or []
                    if not outputs:
                        continue
                    p_blocks = _domain_blocks(p_tmpl.get("domain") or {})
                    for out in outputs:
                        key_fv = (int(out.get("field", -1)), int(out.get("version", 0)))
                        for p_task_key in stage_template_nodes[parent_stage_idx].get(p_tmpl_idx, []):
                            p_block = nodes[p_task_key].block_id
                            output_map[key_fv].append(
                                {
                                    "task_key": p_task_key,
                                    "block_id": p_block,
                                    "blocks": p_blocks,
                                }
                            )

                for c_tmpl_idx, c_tmpl in enumerate(child_templates):
                    inputs = c_tmpl.get("inputs") or []
                    if not inputs:
                        continue
                    c_domain = c_tmpl.get("domain") or {}
                    c_task_keys = stage_template_nodes[child_stage_idx].get(c_tmpl_idx, [])

                    for inp in inputs:
                        key_fv = (int(inp.get("field", -1)), int(inp.get("version", 0)))
                        producers = output_map.get(key_fv)
                        if not producers:
                            continue

                        inp_domain = inp.get("domain")
                        same_as_task_domain = inp_domain is None or inp_domain == c_domain
                        in_domain = inp_domain or c_domain
                        in_blocks = _domain_blocks(in_domain)

                        for c_task_key in c_task_keys:
                            c_block = nodes[c_task_key].block_id
                            req_blocks = in_blocks
                            if same_as_task_domain:
                                req_blocks = [int(c_block)]
                            elif req_blocks is None:
                                req_blocks = [int(c_block)]

                            for prod in producers:
                                p_task_key = prod["task_key"]
                                p_block = prod["block_id"]
                                p_blocks = prod["blocks"]

                                if req_blocks is None:
                                    parents_of[c_task_key].add(p_task_key)
                                    data_edges += 1
                                    continue
                                if p_block is None:
                                    parents_of[c_task_key].add(p_task_key)
                                    data_edges += 1
                                    continue
                                if p_blocks is not None and p_block not in p_blocks:
                                    continue
                                if p_block in req_blocks:
                                    parents_of[c_task_key].add(p_task_key)
                                    data_edges += 1

            if data_edges > 0:
                continue

            for p_task_key in parent_tasks:
                for c_task_key in child_tasks:
                    parents_of[c_task_key].add(p_task_key)

    for key in nodes:
        parents_of.setdefault(key, set())

    return parents_of


def _locality_capacities(nodes: Dict[Tuple[int, int, int, int], TaskNode], workers_by_loc: Dict[int, set[str]]) -> Dict[int, int]:
    caps: Dict[int, int] = {}
    localities = {node.locality for node in nodes.values()}
    for loc in localities:
        if loc in workers_by_loc and workers_by_loc[loc]:
            caps[loc] = len(workers_by_loc[loc])
        else:
            caps[loc] = 1
    return caps


def _topological_order(keys: List[Tuple[int, int, int, int]], parents_of: Dict[Tuple[int, int, int, int], set[Tuple[int, int, int, int]]]) -> List[Tuple[int, int, int, int]]:
    indeg: Dict[Tuple[int, int, int, int], int] = {k: 0 for k in keys}
    children: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]] = defaultdict(list)
    for child, parents in parents_of.items():
        for parent in parents:
            indeg[child] += 1
            children[parent].append(child)

    q: Deque[Tuple[int, int, int, int]] = deque(sorted(k for k in keys if indeg[k] == 0))
    order: List[Tuple[int, int, int, int]] = []
    while q:
        k = q.popleft()
        order.append(k)
        for c in children.get(k, []):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)

    if len(order) != len(keys):
        raise ValueError("Task dependency graph contains a cycle")
    return order


def _critical_path_makespan(
    keys: List[Tuple[int, int, int, int]],
    parents_of: Dict[Tuple[int, int, int, int], set[Tuple[int, int, int, int]]],
    durations: Dict[Tuple[int, int, int, int], float],
) -> float:
    order = _topological_order(keys, parents_of)
    finish: Dict[Tuple[int, int, int, int], float] = {}
    for k in order:
        parent_finish = 0.0
        for p in parents_of[k]:
            parent_finish = max(parent_finish, finish[p])
        finish[k] = parent_finish + durations[k]
    return max(finish.values(), default=0.0)


def _replay_makespan(
    keys: List[Tuple[int, int, int, int]],
    parents_of: Dict[Tuple[int, int, int, int], set[Tuple[int, int, int, int]]],
    durations: Dict[Tuple[int, int, int, int], float],
    node_locality: Dict[Tuple[int, int, int, int], int],
    capacities: Dict[int, int],
) -> float:
    children: Dict[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]] = defaultdict(list)
    indeg: Dict[Tuple[int, int, int, int], int] = {k: 0 for k in keys}
    parent_ready: Dict[Tuple[int, int, int, int], float] = {k: 0.0 for k in keys}

    for child, parents in parents_of.items():
        for parent in parents:
            indeg[child] += 1
            children[parent].append(child)

    ready_queues: Dict[int, List[Tuple[float, Tuple[int, int, int, int]]]] = defaultdict(list)
    machine_heaps: Dict[int, List[float]] = {}
    for loc, cap in capacities.items():
        cap_n = max(1, int(cap))
        machine_heaps[loc] = [0.0 for _ in range(cap_n)]
        heapq.heapify(machine_heaps[loc])

    def ensure_locality(loc: int) -> None:
        if loc not in machine_heaps:
            machine_heaps[loc] = [0.0]
            heapq.heapify(machine_heaps[loc])

    for k in keys:
        if indeg[k] == 0:
            loc = node_locality[k]
            ensure_locality(loc)
            heapq.heappush(ready_queues[loc], (0.0, k))

    done = 0
    finish_time: Dict[Tuple[int, int, int, int], float] = {}

    while done < len(keys):
        best: Optional[Tuple[float, int, float, Tuple[int, int, int, int]]] = None
        for loc, q in ready_queues.items():
            if not q:
                continue
            ready_t, k = q[0]
            mach_free = machine_heaps[loc][0]
            start_t = max(ready_t, mach_free)
            cand = (start_t, loc, ready_t, k)
            if best is None or cand < best:
                best = cand

        if best is None:
            raise ValueError("Replay deadlocked (no ready tasks but unfinished nodes remain)")

        start_t, loc, ready_t, k = best
        heapq.heappop(ready_queues[loc])
        heapq.heappop(machine_heaps[loc])

        finish = start_t + durations[k]
        heapq.heappush(machine_heaps[loc], finish)
        finish_time[k] = finish
        done += 1

        for c in children.get(k, []):
            parent_ready[c] = max(parent_ready[c], finish)
            indeg[c] -= 1
            if indeg[c] == 0:
                c_loc = node_locality[c]
                ensure_locality(c_loc)
                heapq.heappush(ready_queues[c_loc], (parent_ready[c], c))

    return max(finish_time.values(), default=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate I/O slowdown from Kangaroo task logs")
    parser.add_argument("event_log", type=Path, help="Path to JSONL event log")
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Path to plan JSON emitted via KANGAROO_DASHBOARD_PLAN (enables scheduler replay metrics)",
    )
    args = parser.parse_args()

    events = _load_task_events(args.event_log)
    t0, t1 = _analysis_window(events)
    runtime_s = t1 - t0
    if runtime_s <= 0.0:
        raise ValueError("Analysis window has non-positive duration")

    clipped_intervals: List[Interval] = []
    for ev in events:
        clipped = _clip_range(ev.start, ev.end, t0, t1)
        if clipped is None:
            continue
        s, e = clipped
        clipped_intervals.append(Interval(name=ev.name, start=s, end=e))

    io_wait_ranges = [
        (iv.start, iv.end)
        for iv in clipped_intervals
        if iv.name.endswith("/wait_inputs") or iv.name.endswith("/wait_outputs")
    ]
    kernel_ranges = [(iv.start, iv.end) for iv in clipped_intervals if iv.name.endswith("/kernel")]

    io_wait_union_s = _union_duration(io_wait_ranges)
    io_blocked_s = _blocked_time(io_wait_ranges, kernel_ranges)

    non_io_runtime = max(runtime_s - io_blocked_s, 0.0)
    speedup_lb = math.inf if non_io_runtime == 0.0 else runtime_s / non_io_runtime

    print(f"analysis_window_start={t0:.6f}")
    print(f"analysis_window_end={t1:.6f}")
    print(f"runtime_s={runtime_s:.6f}")
    print(f"io_wait_union_s={io_wait_union_s:.6f}")
    print(f"io_blocked_s={io_blocked_s:.6f}")
    print(f"io_blocked_fraction={io_blocked_s / runtime_s:.6f}")
    print(
        f"instant_io_speedup_lb_x={speedup_lb:.6f}"
        if math.isfinite(speedup_lb)
        else "instant_io_speedup_lb_x=inf"
    )

    if args.plan is None:
        print("instant_io_speedup_sched_x=NA")
        print("instant_io_speedup_ub_x=NA")
        return 0

    plan = _load_plan(args.plan)
    nodes, workers_by_loc = _build_nodes(events, t0, t1)
    parents_of = _build_edges(plan, nodes)

    keys = sorted(nodes.keys())
    observed_durations = {k: nodes[k].observed_duration for k in keys}
    cf_durations = {k: nodes[k].cf_duration for k in keys}
    node_locality = {k: nodes[k].locality for k in keys}

    capacities = _locality_capacities(nodes, workers_by_loc)

    replay_observed = _replay_makespan(keys, parents_of, observed_durations, node_locality, capacities)
    replay_cf = _replay_makespan(keys, parents_of, cf_durations, node_locality, capacities)
    critical_cf = _critical_path_makespan(keys, parents_of, cf_durations)

    sched_speedup = math.inf if replay_cf == 0.0 else runtime_s / replay_cf
    ub_speedup = math.inf if critical_cf == 0.0 else runtime_s / critical_cf

    total_task_observed = sum(observed_durations.values())
    total_task_cf = sum(cf_durations.values())

    print(f"nodes={len(keys)}")
    print(f"localities={len(capacities)}")
    print("capacity_by_locality=" + json.dumps({str(k): v for k, v in sorted(capacities.items())}))
    print(f"task_observed_sum_s={total_task_observed:.6f}")
    print(f"task_counterfactual_sum_s={total_task_cf:.6f}")
    print(f"replay_observed_s={replay_observed:.6f}")
    print(f"replay_counterfactual_s={replay_cf:.6f}")
    print(f"critical_counterfactual_s={critical_cf:.6f}")
    print(
        f"instant_io_speedup_sched_x={sched_speedup:.6f}"
        if math.isfinite(sched_speedup)
        else "instant_io_speedup_sched_x=inf"
    )
    print(f"instant_io_speedup_ub_x={ub_speedup:.6f}" if math.isfinite(ub_speedup) else "instant_io_speedup_ub_x=inf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
