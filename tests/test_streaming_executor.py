from __future__ import annotations

import json
import struct

import numpy as np

from analysis import Runtime
from analysis.dataset import open_dataset
from analysis.pipeline import Pipeline
from analysis.plan import Domain, FieldRef, Plan, Stage
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


def _runmeta(nblocks: int) -> RunMeta:
    boxes = [BlockBox((i, 0, 0), (i, 0, 0)) for i in range(nblocks)]
    return RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=boxes,
                    )
                ],
            )
        ]
    )


def _set_scalar(ds, *, field: int, block: int, value: float) -> None:
    arr = np.asarray([[[value]]], dtype=np.float64)
    ds._h.set_chunk_ref(0, 0, field, 0, block, arr.tobytes(order="C"))


def _max_base_task_concurrency(events: list[dict]) -> int:
    active: set[str] = set()
    max_active = 0
    for event in events:
        if event.get("type") != "task":
            continue
        name = str(event.get("name", ""))
        if "/" in name:
            continue
        task_id = str(event.get("id", ""))
        if event.get("status") == "start":
            active.add(task_id)
            max_active = max(max_active, len(active))
        elif event.get("status") in {"end", "error"}:
            active.discard(task_id)
    return max_active


def test_streaming_executor_bounds_active_stage_tasks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANGAROO_EXECUTOR_MODE", "streaming")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_TASKS_PER_STAGE", "2")

    nblocks = 12
    rt = Runtime()
    runmeta = _runmeta(nblocks)
    ds = open_dataset("memory://streaming-window", runmeta=runmeta, step=0, level=0, runtime=rt)
    left = 41001
    right = 41002
    ds.register_field("left", left)
    ds.register_field("right", right)
    for block in range(nblocks):
        _set_scalar(ds, field=left, block=block, value=float(block))
        _set_scalar(ds, field=right, block=block, value=100.0)

    log_path = tmp_path / "streaming-window.events.jsonl"
    rt.set_event_log_path(str(log_path))
    try:
        pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        summed = pipe.field_add(pipe.field(left), pipe.field(right), out="sum", bytes_per_value=8)
        pipe.run()
    finally:
        rt.set_event_log_path("")

    for block in range(nblocks):
        got = rt.get_task_chunk_array(
            step=0,
            level=0,
            field=summed.field,
            version=0,
            block=block,
            shape=(1, 1, 1),
            dtype=np.float64,
            dataset=ds,
        )
        assert got[0, 0, 0] == float(block) + 100.0

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    max_active = _max_base_task_concurrency(events)
    assert max_active > 0
    assert max_active <= 2


def test_streaming_executor_bounds_active_storage_units(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANGAROO_EXECUTOR_MODE", "streaming")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_TASKS_PER_LOCALITY", "16")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_STORAGE_UNITS_PER_LOCALITY", "2")

    nblocks = 12
    rt = Runtime()
    runmeta = _runmeta(nblocks)
    ds = open_dataset("memory://streaming-storage-units", runmeta=runmeta, step=0, level=0, runtime=rt)
    left = 42001
    right = 42002
    ds.register_field("left", left)
    ds.register_field("right", right)
    for block in range(nblocks):
        _set_scalar(ds, field=left, block=block, value=float(block))
        _set_scalar(ds, field=right, block=block, value=7.0)

    log_path = tmp_path / "streaming-storage-units.events.jsonl"
    rt.set_event_log_path(str(log_path))
    try:
        pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        summed = pipe.field_add(pipe.field(left), pipe.field(right), out="sum", bytes_per_value=8)
        pipe.run()
    finally:
        rt.set_event_log_path("")

    for block in range(nblocks):
        got = rt.get_task_chunk_array(
            step=0,
            level=0,
            field=summed.field,
            version=0,
            block=block,
            shape=(1, 1, 1),
            dtype=np.float64,
            dataset=ds,
        )
        assert got[0, 0, 0] == float(block) + 7.0

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    max_active = _max_base_task_concurrency(events)
    assert max_active > 0
    assert max_active <= 2


def test_streaming_executor_bounds_active_output_bytes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANGAROO_EXECUTOR_MODE", "streaming")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_TASKS_PER_LOCALITY", "16")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_STORAGE_UNITS_PER_LOCALITY", "16")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_OUTPUT_BYTES_PER_LOCALITY", "16")

    nblocks = 12
    rt = Runtime()
    runmeta = _runmeta(nblocks)
    ds = open_dataset("memory://streaming-output-bytes", runmeta=runmeta, step=0, level=0, runtime=rt)
    source = 43001
    output = 43002
    for block in range(nblocks):
        ds._h.set_chunk_ref(0, 0, source, 0, block, struct.pack("<q", block + 3))

    log_path = tmp_path / "streaming-output-bytes.events.jsonl"
    rt.set_event_log_path(str(log_path))
    try:
        stage = Stage(name="int64_copy")
        stage.map_blocks(
            name="copy_block_int64",
            kernel="particle_int64_sum_reduce",
            domain=Domain(step=0, level=0),
            inputs=[FieldRef(source)],
            outputs=[FieldRef(output)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={},
        )
        rt.run(Plan(stages=[stage]), runmeta=runmeta, dataset=ds)
    finally:
        rt.set_event_log_path("")

    for block in range(nblocks):
        raw = rt.get_task_chunk(
            step=0,
            level=0,
            field=output,
            version=0,
            block=block,
            dataset=ds,
        )
        assert struct.unpack("<q", raw)[0] == block + 3

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    max_active = _max_base_task_concurrency(events)
    assert max_active > 0
    assert max_active <= 2


def test_streaming_executor_bounds_active_input_bytes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANGAROO_EXECUTOR_MODE", "streaming")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_TASKS_PER_LOCALITY", "16")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_STORAGE_UNITS_PER_LOCALITY", "16")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_INPUT_BYTES_PER_LOCALITY", "32")

    nblocks = 12
    rt = Runtime()
    runmeta = _runmeta(nblocks)
    ds = open_dataset("memory://streaming-input-bytes", runmeta=runmeta, step=0, level=0, runtime=rt)
    left = 44001
    right = 44002
    ds.register_field("left", left)
    ds.register_field("right", right)
    for block in range(nblocks):
        _set_scalar(ds, field=left, block=block, value=float(block))
        _set_scalar(ds, field=right, block=block, value=5.0)

    log_path = tmp_path / "streaming-input-bytes.events.jsonl"
    rt.set_event_log_path(str(log_path))
    try:
        pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        summed = pipe.field_add(pipe.field(left), pipe.field(right), out="sum", bytes_per_value=8)
        pipe.run()
    finally:
        rt.set_event_log_path("")

    for block in range(nblocks):
        got = rt.get_task_chunk_array(
            step=0,
            level=0,
            field=summed.field,
            version=0,
            block=block,
            shape=(1, 1, 1),
            dtype=np.float64,
            dataset=ds,
        )
        assert got[0, 0, 0] == float(block) + 5.0

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    max_active = _max_base_task_concurrency(events)
    assert max_active > 0
    assert max_active <= 2


def test_streaming_executor_evicts_dataset_inputs_after_last_consumer(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANGAROO_EXECUTOR_MODE", "streaming")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_TASKS_PER_LOCALITY", "2")

    nblocks = 4
    rt = Runtime()
    runmeta = _runmeta(nblocks)
    ds = open_dataset("memory://streaming-input-eviction", runmeta=runmeta, step=0, level=0, runtime=rt)
    left = 45001
    right = 45002
    ds.register_field("left", left)
    ds.register_field("right", right)
    for block in range(nblocks):
        _set_scalar(ds, field=left, block=block, value=float(block))
        _set_scalar(ds, field=right, block=block, value=11.0)

    log_path = tmp_path / "streaming-input-eviction.events.jsonl"
    rt.set_event_log_path(str(log_path))
    try:
        pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        summed = pipe.field_add(pipe.field(left), pipe.field(right), out="sum", bytes_per_value=8)
        pipe.run()
    finally:
        rt.set_event_log_path("")

    for block in range(nblocks):
        got = rt.get_task_chunk_array(
            step=0,
            level=0,
            field=summed.field,
            version=0,
            block=block,
            shape=(1, 1, 1),
            dtype=np.float64,
            dataset=ds,
        )
        assert got[0, 0, 0] == float(block) + 11.0

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evictions = [event for event in events if event.get("op") == "chunk_evict"]
    evicted_refs = {(event["field"], event["block"]) for event in evictions}
    assert evicted_refs == {
        (field, block)
        for field in (left, right)
        for block in range(nblocks)
    }
    assert all(event["bytes"] == 8 for event in evictions)
    assert all(event["field"] != summed.field for event in evictions)
