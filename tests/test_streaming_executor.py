from __future__ import annotations

import json
import struct

from analysis.buffer import (
    BufferSpec,
    DType,
    DynamicShape,
    DynamicUpperBound,
    FixedShape,
    LikeInputShape,
)
from analysis.plan import OutputRef

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
    ds._h.set_chunk_ref(0, 0, field, 0, block, arr.tobytes(order="C"), "f64", [1, 1, 1])


def _write_single_fab_plotfile(path) -> None:
    level = path / "Level_0"
    level.mkdir(parents=True)
    (path / "Header").write_text(
        "\n".join(
            [
                "HyperCLaw-V1.1",
                "1",
                "density",
                "3",
                "0.0",
                "0",
                "0 0 0",
                "2 1 3",
                "((0,0,0) (1,0,2) (0,0,0))",
                "0",
                "1 1 1",
                "0",
                "0",
                "0 1 0.0",
                "0",
                "0 2",
                "0 1",
                "0 3",
                "Level_0/Cell",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (level / "Cell_H").write_text(
        "\n".join(
            [
                "1",
                "1",
                "1",
                "0",
                "(1 0",
                "((0,0,0) (1,0,2) (0,0,0))",
                ")",
                "1",
                "FabOnDisk: Cell_D_00000 0",
                "1,1",
                "0,",
                "1,1",
                "0,",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fab_header = (
        "FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))"
        "((0,0,0) (1,0,2) (0,0,0)) 1\n"
    ).encode("ascii")
    values = np.arange(6, dtype=np.float64)
    (level / "Cell_D_00000").write_bytes(fab_header + values.tobytes())


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
        summed = pipe.field_add(pipe.field(left), pipe.field(right), out="sum", dtype="f64")
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
        summed = pipe.field_add(pipe.field(left), pipe.field(right), out="sum", dtype="f64")
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
        ds._h.set_chunk_ref(0, 0, source, 0, block, struct.pack("<q", block + 3), "i64", [1])

    log_path = tmp_path / "streaming-output-bytes.events.jsonl"
    rt.set_event_log_path(str(log_path))
    try:
        stage = Stage(name="int64_copy")
        stage.map_blocks(
            name="copy_block_int64",
            kernel="particle_int64_sum_reduce",
            domain=Domain(step=0, level=0),
            inputs=[FieldRef(source)],
            outputs=[OutputRef(FieldRef(output), BufferSpec(DType.I64, FixedShape((1,))))],
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


def test_streaming_executor_accounts_like_input_output_bytes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANGAROO_EXECUTOR_MODE", "streaming")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_TASKS_PER_LOCALITY", "16")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_STORAGE_UNITS_PER_LOCALITY", "16")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_OUTPUT_BYTES_PER_LOCALITY", "16")

    nblocks = 12
    rt = Runtime()
    runmeta = _runmeta(nblocks)
    ds = open_dataset(
        "memory://streaming-like-input-output-bytes",
        runmeta=runmeta,
        step=0,
        level=0,
        runtime=rt,
    )
    source = 43501
    output = 43502
    values = np.arange(8, dtype=np.float64)
    for block in range(nblocks):
        ds._h.set_chunk_ref(0, 0, source, 0, block, values.tobytes(), "f64", [8])

    log_path = tmp_path / "streaming-like-input-output-bytes.events.jsonl"
    rt.set_event_log_path(str(log_path))
    try:
        stage = Stage(name="particle_mask")
        stage.map_blocks(
            name="particle_eq_mask",
            kernel="particle_eq_mask",
            domain=Domain(step=0, level=0),
            inputs=[FieldRef(source)],
            outputs=[
                OutputRef(
                    FieldRef(output),
                    BufferSpec(DType.U8, LikeInputShape(0)),
                )
            ],
            deps={"kind": "None"},
            params={"scalar": -1.0},
        )
        rt.run(Plan(stages=[stage]), runmeta=runmeta, dataset=ds)
    finally:
        rt.set_event_log_path("")

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    max_active = _max_base_task_concurrency(events)
    assert max_active > 0
    assert max_active <= 2


def test_streaming_executor_accounts_plotfile_like_input_output_bytes(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("KANGAROO_EXECUTOR_MODE", "streaming")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_OUTPUT_BYTES_PER_LOCALITY", "48")

    plotfile = tmp_path / "plt00000"
    _write_single_fab_plotfile(plotfile)
    rt = Runtime()
    ds = open_dataset(str(plotfile), runtime=rt, step=0, level=0)
    runmeta = ds.get_runmeta()
    source = ds.field_id("density")
    output = 43512

    stage = Stage(name="plotfile_expr")
    stage.map_blocks(
        name="plotfile_identity",
        kernel="field_expr",
        domain=Domain(step=0, level=0),
        inputs=[FieldRef(source)],
        outputs=[OutputRef(FieldRef(output), BufferSpec(DType.F64, LikeInputShape(0)))],
        deps={"kind": "None"},
        params={"expression": "density", "variables": ["density"]},
    )
    rt.run(Plan(stages=[stage]), runmeta=runmeta, dataset=ds)

    got = rt.get_task_chunk_array(
        step=0,
        level=0,
        field=output,
        version=0,
        block=0,
        shape=(2, 1, 3),
        dtype=np.float64,
        dataset=ds,
    )
    expected = np.arange(6, dtype=np.float64).reshape(3, 1, 2).transpose(2, 1, 0)
    np.testing.assert_array_equal(got, expected)


def test_streaming_executor_accounts_dynamic_like_input_capacity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANGAROO_EXECUTOR_MODE", "streaming")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_TASKS_PER_LOCALITY", "16")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_ACTIVE_STORAGE_UNITS_PER_LOCALITY", "16")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_OUTPUT_BYTES_PER_LOCALITY", "128")

    nblocks = 12
    rt = Runtime()
    runmeta = _runmeta(nblocks)
    ds = open_dataset(
        "memory://streaming-dynamic-output-bytes",
        runmeta=runmeta,
        step=0,
        level=0,
        runtime=rt,
    )
    values_field = 43601
    mask_field = 43602
    output = 43603
    values = np.arange(8, dtype=np.float64)
    mask = np.ones(8, dtype=np.uint8)
    for block in range(nblocks):
        ds._h.set_chunk_ref(
            0, 0, values_field, 0, block, values.tobytes(), "f64", [8]
        )
        ds._h.set_chunk_ref(0, 0, mask_field, 0, block, mask.tobytes(), "u8", [8])

    log_path = tmp_path / "streaming-dynamic-output-bytes.events.jsonl"
    rt.set_event_log_path(str(log_path))
    try:
        stage = Stage(name="particle_filter")
        stage.map_blocks(
            name="particle_filter",
            kernel="particle_filter",
            domain=Domain(step=0, level=0),
            inputs=[FieldRef(values_field), FieldRef(mask_field)],
            outputs=[
                OutputRef(
                    FieldRef(output),
                    BufferSpec(
                        DType.F64,
                        DynamicShape(DynamicUpperBound.like_input(0)),
                    ),
                )
            ],
            deps={"kind": "None"},
            params={},
        )
        rt.run(Plan(stages=[stage]), runmeta=runmeta, dataset=ds)
    finally:
        rt.set_event_log_path("")

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
        summed = pipe.field_add(pipe.field(left), pipe.field(right), out="sum", dtype="f64")
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
        summed = pipe.field_add(pipe.field(left), pipe.field(right), out="sum", dtype="f64")
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
