from __future__ import annotations

import struct


def _pack_f32_chunk(nx: int, ny: int, nz: int, base: float) -> bytes:
    buf = bytearray()
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = base + 100.0 * float(i) + 10.0 * float(j) + float(k)
                buf.extend(struct.pack("<f", v))
    return bytes(buf)


def _unpack_f32(data: bytes) -> list[float]:
    if not data:
        return []
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


def test_cross_level_subbox_fetch_memory_backend() -> None:
    from analysis import _core  # type: ignore

    ds = _core.DatasetHandle("memory://local", 0, 0)

    # Level 0 chunk: [0:3]x[0:1]x[0:1]
    ds.set_chunk_ref(0, 0, 7, 0, 0, _pack_f32_chunk(4, 2, 2, 0.0))
    # Level 1 chunk: [0:7]x[0:3]x[0:3]
    ds.set_chunk_ref(0, 1, 7, 0, 0, _pack_f32_chunk(8, 4, 4, 1000.0))

    # Fetch a level-0 subbox.
    out0 = _core.test_get_subbox(
        dataset=ds,
        step=0,
        level=0,
        field=7,
        version=0,
        block=0,
        chunk_lo=(0, 0, 0),
        chunk_hi=(3, 1, 1),
        request_lo=(1, 0, 0),
        request_hi=(2, 1, 1),
        bytes_per_value=4,
    )
    assert tuple(out0["lo"]) == (1, 0, 0)
    assert tuple(out0["hi"]) == (2, 1, 1)
    vals0 = _unpack_f32(out0["data"])
    assert vals0 == [100.0, 101.0, 110.0, 111.0, 200.0, 201.0, 210.0, 211.0]

    # Fetch a level-1 subbox (cross-level availability test).
    out1 = _core.test_get_subbox(
        dataset=ds,
        step=0,
        level=1,
        field=7,
        version=0,
        block=0,
        chunk_lo=(0, 0, 0),
        chunk_hi=(7, 3, 3),
        request_lo=(2, 1, 1),
        request_hi=(3, 2, 2),
        bytes_per_value=4,
    )
    assert tuple(out1["lo"]) == (2, 1, 1)
    assert tuple(out1["hi"]) == (3, 2, 2)
    vals1 = _unpack_f32(out1["data"])
    assert vals1 == [1211.0, 1212.0, 1221.0, 1222.0, 1311.0, 1312.0, 1321.0, 1322.0]

