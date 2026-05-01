from __future__ import annotations

import json
import struct


def test_task_inputs_for_same_block_are_loaded_as_one_batch(tmp_path) -> None:
    from analysis import Runtime
    from analysis.dataset import open_dataset
    from analysis.pipeline import Pipeline
    from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta

    rt = Runtime()
    log_path = tmp_path / "coalesced-inputs.events.jsonl"
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    ds = open_dataset("memory://coalesced-inputs", runmeta=runmeta, step=0, level=0, runtime=rt)
    ds.register_field("a", 101)
    ds.register_field("b", 102)
    ds._h.set_chunk_ref(0, 0, 101, 0, 0, struct.pack("<d", 2.0))
    ds._h.set_chunk_ref(0, 0, 102, 0, 0, struct.pack("<d", 3.0))

    rt.set_event_log_path(str(log_path))
    try:
        pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        pipe.field_add(pipe.field(101), pipe.field(102), out="sum")
        pipe.run()
    finally:
        rt.set_event_log_path("")

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    starts = [
        event
        for event in events
        if event.get("type") == "dataflow" and event.get("op") == "dataset_load_start"
    ]
    reads = [
        event
        for event in events
        if event.get("type") == "dataflow" and event.get("op") == "dataset_load_read"
    ]
    unit_starts = [
        event
        for event in events
        if event.get("type") == "dataflow" and event.get("op") == "dataset_load_unit_start"
    ]
    unit_reads = [
        event
        for event in events
        if event.get("type") == "dataflow" and event.get("op") == "dataset_load_unit_read"
    ]

    assert [event["field"] for event in starts] == [101, 102]
    assert [event["field"] for event in reads] == [101, 102]
    assert len({event["start"] for event in starts}) == 1
    assert len({event["start"] for event in reads}) == 1
    assert {event["in_flight"] for event in starts} == {1}
    assert len(unit_starts) == 1
    assert len(unit_reads) == 1
    assert unit_starts[0]["comp_count"] == 2
    assert unit_reads[0]["comp_count"] == 2
    assert unit_reads[0]["bytes"] == 16


def test_dataset_load_byte_budget_limits_in_flight_storage_units(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KANGAROO_DATASET_LOAD_MAX_BYTES_IN_FLIGHT", "16")

    import numpy as np

    from analysis import Runtime
    from analysis.dataset import open_dataset
    from analysis.pipeline import Pipeline
    from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta

    nblocks = 8
    rt = Runtime()
    log_path = tmp_path / "byte-budget.events.jsonl"
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox((i, 0, 0), (i, 0, 0)) for i in range(nblocks)],
                    )
                ],
            )
        ]
    )
    ds = open_dataset("memory://byte-budget", runmeta=runmeta, step=0, level=0, runtime=rt)
    left = 201
    right = 202
    ds.register_field("left", left)
    ds.register_field("right", right)
    for block in range(nblocks):
        ds._h.set_chunk_ref(
            0,
            0,
            left,
            0,
            block,
            np.asarray([[[float(block)]]], dtype=np.float64).tobytes(order="C"),
        )
        ds._h.set_chunk_ref(
            0,
            0,
            right,
            0,
            block,
            np.asarray([[[10.0]]], dtype=np.float64).tobytes(order="C"),
        )

    rt.set_event_log_path(str(log_path))
    try:
        pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
        out = pipe.field_add(pipe.field(left), pipe.field(right), out="sum", bytes_per_value=8)
        pipe.run()
    finally:
        rt.set_event_log_path("")

    for block in range(nblocks):
        got = rt.get_task_chunk_array(
            step=0,
            level=0,
            field=out.field,
            version=0,
            block=block,
            shape=(1, 1, 1),
            dtype=np.float64,
            dataset=ds,
        )
        assert got[0, 0, 0] == float(block) + 10.0

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    unit_starts = [
        event
        for event in events
        if event.get("type") == "dataflow" and event.get("op") == "dataset_load_unit_start"
    ]

    assert len(unit_starts) == nblocks
    assert {event["estimated_bytes"] for event in unit_starts} == {16}
    assert {event["byte_limit"] for event in unit_starts} == {16}
    assert max(event["in_flight_bytes"] for event in unit_starts) <= 16


def _pack_f32_chunk(nx: int, ny: int, nz: int, base: float) -> bytes:
    buf = bytearray()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                value = base + 100.0 * float(i) + 10.0 * float(j) + float(k)
                buf.extend(struct.pack("<f", value))
    return bytes(buf)


def _unpack_f32(data: bytes) -> list[float]:
    if not data:
        return []
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


def test_get_host_returns_pending_future_before_put() -> None:
    from analysis import _core  # type: ignore

    ds = _core.DatasetHandle("memory://local", 0, 0)
    payload = b"\x01\x02\x03\x04"

    out = _core.test_data_service_pending_then_put(
        dataset=ds,
        step=0,
        level=0,
        field=11,
        version=0,
        block=0,
        payload=payload,
        timeout_ms=50,
    )

    assert out["returned_before_put"] is True
    assert out["future_ready_before_put"] is False
    assert out["data"] == payload


def test_put_host_before_get_returns_ready_value() -> None:
    from analysis import _core  # type: ignore

    ds = _core.DatasetHandle("memory://local", 0, 0)
    payload = b"ready-first"

    out = _core.test_data_service_put_then_get(
        dataset=ds,
        step=0,
        level=0,
        field=12,
        version=0,
        block=0,
        payload=payload,
    )

    assert out["ready_before_get"] is True
    assert out["data"] == payload


def test_data_service_event_log_records_structured_dataflow(tmp_path) -> None:
    from analysis import _core  # type: ignore

    ds = _core.DatasetHandle("memory://local", 0, 0)
    payload = b"dataflow-event"
    log_path = tmp_path / "events.jsonl"

    _core.set_event_log_path(str(log_path))
    try:
        _core.test_data_service_put_then_get(
            dataset=ds,
            step=0,
            level=0,
            field=12,
            version=0,
            block=0,
            payload=payload,
        )
    finally:
        _core.set_event_log_path("")

    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataflow = [event for event in events if event["type"] == "dataflow"]

    assert {event["op"] for event in dataflow} >= {"put_host", "get_host"}
    for event in dataflow:
        assert event["mode"] in {"local", "remote"}
        assert event["status"] == "end"
        assert event["target_locality"] >= 0
        assert event["bytes"] == len(payload)
        assert event["elapsed"] >= 0.0


def test_multiple_get_host_consumers_wait_on_one_pending_chunk() -> None:
    from analysis import _core  # type: ignore

    ds = _core.DatasetHandle("memory://local", 0, 0)
    payload = b"shared-payload"
    consumers = 16

    out = _core.test_data_service_many_pending_consumers_then_put(
        dataset=ds,
        step=0,
        level=0,
        field=13,
        version=0,
        block=0,
        payload=payload,
        consumers=consumers,
    )

    assert out["ready_before_put"] == [False] * consumers
    assert out["data"] == [payload] * consumers


def test_get_subbox_completes_when_backing_chunk_arrives_later() -> None:
    from analysis import _core  # type: ignore

    ds = _core.DatasetHandle("memory://local", 0, 0)

    out = _core.test_data_service_subbox_pending_then_put(
        dataset=ds,
        step=0,
        level=0,
        field=14,
        version=0,
        block=0,
        chunk_lo=(0, 0, 0),
        chunk_hi=(3, 1, 1),
        request_lo=(1, 0, 0),
        request_hi=(2, 1, 1),
        bytes_per_value=4,
        payload=_pack_f32_chunk(4, 2, 2, 0.0),
    )

    assert out["future_ready_before_put"] is False
    assert tuple(out["lo"]) == (1, 0, 0)
    assert tuple(out["hi"]) == (2, 1, 1)
    assert _unpack_f32(out["data"]) == [
        100.0,
        101.0,
        110.0,
        111.0,
        200.0,
        201.0,
        210.0,
        211.0,
    ]
