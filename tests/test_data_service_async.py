from __future__ import annotations

import json
import struct


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
