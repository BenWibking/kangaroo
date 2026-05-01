#!/usr/bin/env python3
"""Microbenchmark AMReX plotfile chunk reads through Kangaroo dataset paths."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis import Runtime
from analysis.dataset import open_dataset


DEFAULT_FIELDS = (
    "gasDensity",
    "x-GasMomentum",
    "y-GasMomentum",
    "z-GasMomentum",
    "gasEnergy",
    "scalar_0",
    "x-BField",
    "y-BField",
    "z-BField",
)


@dataclass(frozen=True)
class Trial:
    block_count: int
    ref_count: int
    concurrency: int
    repeat: int
    elapsed_s: float
    selected_bytes: int
    estimated_raw_bytes: int
    missing_chunks: int

    @property
    def selected_mib_s(self) -> float:
        if self.elapsed_s <= 0.0:
            return 0.0
        return (self.selected_bytes / (1024.0 * 1024.0)) / self.elapsed_s

    @property
    def estimated_raw_mib_s(self) -> float:
        if self.elapsed_s <= 0.0:
            return 0.0
        return (self.estimated_raw_bytes / (1024.0 * 1024.0)) / self.elapsed_s


def _parse_csv(value: str) -> list[str]:
    out = [item.strip() for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return out


def _parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer {item!r}") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError("counts must be positive")
        out.append(parsed)
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _resolve_level(meta: dict[str, Any], level: int | None) -> int:
    finest = int(meta["finest_level"])
    if level is None:
        return finest
    if level < 0 or level > finest:
        raise ValueError(f"level {level} is outside available range 0..{finest}")
    return int(level)


def _resolve_fields(ds: Any, meta: dict[str, Any], names: list[str]) -> tuple[list[int], list[int]]:
    var_names = list(meta.get("var_names", []))
    component_by_name = {str(name): idx for idx, name in enumerate(var_names)}
    missing = [name for name in names if name not in component_by_name]
    if missing:
        available = ", ".join(var_names)
        raise ValueError(f"unknown field(s): {', '.join(missing)}; available: {available}")
    field_ids = [int(ds.field_id(name)) for name in names]
    components = [int(component_by_name[name]) for name in names]
    return field_ids, components


def _build_refs(
    *,
    step: int,
    level: int,
    field_ids: list[int],
    version: int,
    block_start: int,
    block_count: int,
    field_major: bool,
) -> list[tuple[int, int, int, int, int]]:
    blocks = range(block_start, block_start + block_count)
    if field_major:
        return [
            (step, level, field_id, version, block)
            for field_id in field_ids
            for block in blocks
        ]
    return [
        (step, level, field_id, version, block)
        for block in blocks
        for field_id in field_ids
    ]


def _build_ref_batches(
    *,
    step: int,
    level: int,
    field_ids: list[int],
    version: int,
    block_start: int,
    block_count: int,
    concurrency: int,
    field_major: bool,
) -> list[list[tuple[int, int, int, int, int]]]:
    workers = min(max(1, concurrency), block_count)
    batches: list[list[tuple[int, int, int, int, int]]] = []
    next_block = block_start
    for worker_idx in range(workers):
        remaining_blocks = block_count - (next_block - block_start)
        remaining_workers = workers - worker_idx
        this_count = (remaining_blocks + remaining_workers - 1) // remaining_workers
        batches.append(
            _build_refs(
                step=step,
                level=level,
                field_ids=field_ids,
                version=version,
                block_start=next_block,
                block_count=this_count,
                field_major=field_major,
            )
        )
        next_block += this_count
    return batches


def _estimate_raw_bytes(
    *,
    selected_bytes: int,
    block_count: int,
    field_count: int,
    components: list[int],
) -> int:
    if selected_bytes <= 0 or block_count <= 0 or field_count <= 0:
        return 0
    selected_components = len(set(components))
    if selected_components <= 0:
        return selected_bytes
    component_span = max(components) - min(components) + 1
    bytes_per_component_per_block = selected_bytes / float(block_count * field_count)
    return int(round(bytes_per_component_per_block * block_count * component_span))


def _run_trial(
    *,
    dataset_handle: Any,
    ref_batches: list[list[tuple[int, int, int, int, int]]],
    block_count: int,
    concurrency: int,
    repeat: int,
    field_count: int,
    components: list[int],
    via_data_service: bool,
) -> Trial:
    def read_sizes(refs: list[tuple[int, int, int, int, int]]) -> list[int]:
        if via_data_service:
            sizes = dataset_handle.read_chunks_ref_sizes_data_service(refs)
        else:
            sizes = dataset_handle.read_chunks_ref_sizes(refs)
        return [int(size) for size in sizes]

    start = time.perf_counter()
    if len(ref_batches) == 1:
        sizes = read_sizes(ref_batches[0])
    else:
        sizes = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(ref_batches)) as pool:
            futures = [pool.submit(read_sizes, refs) for refs in ref_batches]
            for future in concurrent.futures.as_completed(futures):
                sizes.extend(future.result())
    elapsed = time.perf_counter() - start
    selected_bytes = sum(sizes)
    missing = sum(1 for size in sizes if size == 0)
    estimated_raw_bytes = _estimate_raw_bytes(
        selected_bytes=selected_bytes,
        block_count=block_count,
        field_count=field_count,
        components=components,
    )
    return Trial(
        block_count=block_count,
        ref_count=sum(len(refs) for refs in ref_batches),
        concurrency=min(max(1, concurrency), block_count),
        repeat=repeat,
        elapsed_s=elapsed,
        selected_bytes=selected_bytes,
        estimated_raw_bytes=estimated_raw_bytes,
        missing_chunks=missing,
    )


def _print_table(results: dict[tuple[int, int], list[Trial]]) -> None:
    header = (
        "blocks conc refs selected_mib raw_mib "
        "median_s mean_s selected_mib_s raw_mib_s missing"
    )
    print(header)
    for (block_count, concurrency), trials in results.items():
        if not trials:
            continue
        elapsed = [trial.elapsed_s for trial in trials]
        selected_mib_s = [trial.selected_mib_s for trial in trials]
        raw_mib_s = [trial.estimated_raw_mib_s for trial in trials]
        selected_mib = trials[-1].selected_bytes / (1024.0 * 1024.0)
        raw_mib = trials[-1].estimated_raw_bytes / (1024.0 * 1024.0)
        refs = trials[-1].ref_count
        missing = max(trial.missing_chunks for trial in trials)
        print(
            f"{block_count:6d} "
            f"{concurrency:4d} "
            f"{refs:4d} "
            f"{selected_mib:12.1f} "
            f"{raw_mib:8.1f} "
            f"{_median(elapsed):8.3f} "
            f"{_mean(elapsed):7.3f} "
            f"{_median(selected_mib_s):14.1f} "
            f"{_median(raw_mib_s):9.1f} "
            f"{missing:7d}"
        )


def _json_payload(
    *,
    args: argparse.Namespace,
    plotfile: str,
    level: int,
    fields: list[str],
    components: list[int],
    results: dict[tuple[int, int], list[Trial]],
) -> dict[str, Any]:
    return {
        "plotfile": plotfile,
        "step": int(args.step),
        "level": level,
        "block_start": int(args.block_start),
        "fields": fields,
        "components": components,
        "field_order": "field-major" if args.field_major else "block-major",
        "path": "data-service" if args.via_data_service else "dataset-handle",
        "trials": [
            {
                "block_count": trial.block_count,
                "ref_count": trial.ref_count,
                "concurrency": trial.concurrency,
                "repeat": trial.repeat,
                "elapsed_s": trial.elapsed_s,
                "selected_bytes": trial.selected_bytes,
                "estimated_raw_bytes": trial.estimated_raw_bytes,
                "selected_mib_s": trial.selected_mib_s,
                "estimated_raw_mib_s": trial.estimated_raw_mib_s,
                "missing_chunks": trial.missing_chunks,
            }
            for trials in results.values()
            for trial in trials
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark AMReX plotfile chunk reads through Kangaroo dataset paths."
    )
    parser.add_argument("plotfile", help="AMReX plotfile path or Kangaroo dataset URI.")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--level", type=int, default=None, help="AMR level; defaults to finest.")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument(
        "--block-counts",
        type=_parse_int_csv,
        default=[1, 2, 4, 8],
        help="Comma-separated block batch sizes.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument(
        "--concurrency",
        type=_parse_int_csv,
        default=[1],
        help="Comma-separated numbers of simultaneous get_chunks calls.",
    )
    parser.add_argument(
        "--fields",
        type=_parse_csv,
        default=list(DEFAULT_FIELDS),
        help="Comma-separated field names.",
    )
    parser.add_argument(
        "--field-major",
        action="store_true",
        help="Issue refs field-major instead of block-major.",
    )
    parser.add_argument(
        "--via-data-service",
        action="store_true",
        help="Read through DataServiceLocal so queueing and HPX I/O pools are exercised.",
    )
    parser.add_argument("--json", action="store_true", help="Print detailed JSON.")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.block_start < 0:
        raise ValueError("--block-start must be non-negative")

    runtime = Runtime()
    ds = open_dataset(args.plotfile, runtime=runtime, step=args.step)
    meta = ds.metadata
    level = _resolve_level(meta, args.level)
    level_boxes = list(meta["level_boxes"][level])
    max_block_count = max(args.block_counts)
    if args.block_start + max_block_count > len(level_boxes):
        raise ValueError(
            f"requested blocks {args.block_start}..{args.block_start + max_block_count - 1}, "
            f"but level {level} has {len(level_boxes)} blocks"
        )

    fields = list(args.fields)
    field_ids, components = _resolve_fields(ds, meta, fields)

    print(
        f"plotfile={args.plotfile} level={level} block_start={args.block_start} "
        f"fields={','.join(fields)} components={components}",
        flush=True,
    )

    results: dict[tuple[int, int], list[Trial]] = {}
    for block_count in args.block_counts:
        for concurrency in args.concurrency:
            ref_batches = _build_ref_batches(
                step=args.step,
                level=level,
                field_ids=field_ids,
                version=args.version,
                block_start=args.block_start,
                block_count=block_count,
                concurrency=concurrency,
                field_major=bool(args.field_major),
            )
            for _ in range(args.warmups):
                _run_trial(
                    dataset_handle=ds._h,
                    ref_batches=ref_batches,
                    block_count=block_count,
                    concurrency=concurrency,
                    repeat=-1,
                    field_count=len(field_ids),
                    components=components,
                    via_data_service=bool(args.via_data_service),
                )
            trials = [
                _run_trial(
                    dataset_handle=ds._h,
                    ref_batches=ref_batches,
                    block_count=block_count,
                    concurrency=concurrency,
                    repeat=repeat,
                    field_count=len(field_ids),
                    components=components,
                    via_data_service=bool(args.via_data_service),
                )
                for repeat in range(args.repeats)
            ]
            results[(block_count, concurrency)] = trials

    payload = _json_payload(
        args=args,
        plotfile=args.plotfile,
        level=level,
        fields=fields,
        components=components,
        results=results,
    )
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
