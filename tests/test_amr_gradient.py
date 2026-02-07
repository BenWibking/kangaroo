from __future__ import annotations

import msgpack
import struct


def _pack_linear_field_double(box_lo, box_hi, geom, coeffs=(1.0, 2.0, 3.0), c0=0.0) -> bytes:
    nx = box_hi[0] - box_lo[0] + 1
    ny = box_hi[1] - box_lo[1] + 1
    nz = box_hi[2] - box_lo[2] + 1
    out = bytearray()
    for i in range(nx):
        gi = box_lo[0] + i
        x = geom["x0"][0] + (gi - geom["index_origin"][0] + 0.5) * geom["dx"][0]
        for j in range(ny):
            gj = box_lo[1] + j
            y = geom["x0"][1] + (gj - geom["index_origin"][1] + 0.5) * geom["dx"][1]
            for k in range(nz):
                gk = box_lo[2] + k
                z = geom["x0"][2] + (gk - geom["index_origin"][2] + 0.5) * geom["dx"][2]
                f = c0 + coeffs[0] * x + coeffs[1] * y + coeffs[2] * z
                out.extend(struct.pack("<d", f))
    return bytes(out)


def test_gradU_stencil_amr_remote_fetch_linear_field() -> None:
    from analysis import Runtime
    from analysis.dataset import open_dataset
    from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta

    rt = Runtime()
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=2),
                        boxes=[BlockBox((0, 0, 0), (3, 3, 3))],
                    ),
                    LevelMeta(
                        geom=LevelGeom(dx=(0.5, 0.5, 0.5), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox((2, 0, 0), (7, 7, 7))],
                    ),
                ],
            )
        ]
    )
    ds = open_dataset("memory://local", runmeta=runmeta, step=0, level=0, runtime=rt)

    coarse_geom = {"dx": (1.0, 1.0, 1.0), "x0": (0.0, 0.0, 0.0), "index_origin": (0, 0, 0)}
    fine_geom = {"dx": (0.5, 0.5, 0.5), "x0": (0.0, 0.0, 0.0), "index_origin": (0, 0, 0)}
    field = 11
    out_field = 20

    coarse_payload = _pack_linear_field_double((0, 0, 0), (3, 3, 3), coarse_geom)
    fine_payload = _pack_linear_field_double((2, 0, 0), (7, 7, 7), fine_geom)
    ds._h.set_chunk_ref(0, 0, field, 0, 0, coarse_payload)
    ds._h.set_chunk_ref(0, 1, field, 0, 0, fine_payload)

    nx = ny = nz = 4
    out_bytes = nx * ny * nz * 3 * 8
    fetch_field = 19
    plan_dict = {
        "stages": [
            {
                "name": "fetch",
                "plane": "chunk",
                "after": [],
                "templates": [
                    {
                        "name": "fetch_nbr",
                        "plane": "chunk",
                        "kernel": "amr_subbox_fetch_pack",
                        "domain": {"step": 0, "level": 0, "blocks": [0]},
                        "inputs": [],
                        "outputs": [{"field": fetch_field, "version": 0}],
                        "output_bytes": [0],
                        "deps": {"kind": "None"},
                        "params": {
                            "input_field": field,
                            "input_version": 0,
                            "input_step": 0,
                            "input_level": 0,
                            "bytes_per_value": 8,
                            "halo_cells": 1,
                        },
                    }
                ],
            },
            {
                "name": "grad",
                "plane": "chunk",
                "after": [0],
                "templates": [
                    {
                        "name": "gradU",
                        "plane": "chunk",
                        "kernel": "gradU_stencil",
                        "domain": {"step": 0, "level": 0, "blocks": [0]},
                        "inputs": [{"field": field, "version": 0}, {"field": fetch_field, "version": 0}],
                        "outputs": [{"field": out_field, "version": 0}],
                        "output_bytes": [out_bytes],
                        "deps": {"kind": "None"},
                        "params": {
                            "input_field": field,
                            "input_version": 0,
                            "input_step": 0,
                            "input_level": 0,
                            "bytes_per_value": 8,
                        },
                    }
                ],
            }
        ]
    }
    packed = msgpack.packb(plan_dict, use_bin_type=True)
    rt._rt.run_packed_plan(packed, runmeta._h, ds._h)

    out = rt.get_task_chunk(step=0, level=0, field=out_field, version=0, block=0)
    vals = struct.unpack(f"<{len(out) // 8}d", out)

    def grad_at(i: int, j: int, k: int) -> tuple[float, float, float]:
        idx = (i * ny + j) * nz + k
        return vals[3 * idx + 0], vals[3 * idx + 1], vals[3 * idx + 2]

    # Check across the coarse-fine interface neighborhood.
    for i in (0, 1, 2, 3):
        gx, gy, gz = grad_at(i, 2, 2)
        assert abs(gx - 1.0) < 1e-8
        assert abs(gy - 2.0) < 1e-8
        assert abs(gz - 3.0) < 1e-8
