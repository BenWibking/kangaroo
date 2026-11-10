from __future__ import annotations

import struct

from analysis.buffer import BlockShape, BufferSpec, DType, DynamicShape, DynamicUpperBound
from analysis.kernel_params import AmrSubboxPackParams, GradStencilParams
from analysis.plan import Domain, FieldRef, OutputRef, Plan, Stage, TaskTemplate
from analysis.plan_codec import encode_plan


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
    ds._h.set_chunk_ref(0, 0, field, 0, 0, coarse_payload, "f64", [4, 4, 4])
    ds._h.set_chunk_ref(0, 1, field, 0, 0, fine_payload, "f64", [6, 8, 8])

    nx = ny = nz = 4
    out_bytes = nx * ny * nz * 3 * 8
    fetch_field = 19
    fetch = Stage(
        name="fetch",
        templates=[TaskTemplate(
            name="fetch_nbr",
            plane="chunk",
            kernel="amr_subbox_fetch_pack",
            domain=Domain(0, 0, [0]),
            inputs=[],
            outputs=[OutputRef(FieldRef(fetch_field), BufferSpec(
                DType.OPAQUE,
                DynamicShape(DynamicUpperBound.amr_subbox_pack()),
            ))],
            params=AmrSubboxPackParams(field, 0, 0, 0, 1),
        )],
    )
    grad = Stage(
        name="grad",
        after=[fetch],
        templates=[TaskTemplate(
            name="gradU",
            plane="chunk",
            kernel="gradU_stencil",
            domain=Domain(0, 0, [0]),
            inputs=[FieldRef(field), FieldRef(fetch_field)],
            outputs=[OutputRef(
                FieldRef(out_field), BufferSpec(DType.F64, BlockShape(3))
            )],
            params=GradStencilParams(field, 0, 0, 0),
        )],
    )
    packed = encode_plan(Plan([fetch, grad]))
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


def _pack_quadratic_field_double(box_lo, box_hi, geom) -> bytes:
    nx = box_hi[0] - box_lo[0] + 1
    ny = box_hi[1] - box_lo[1] + 1
    nz = box_hi[2] - box_lo[2] + 1
    out = bytearray()
    for i in range(nx):
        x = geom["x0"][0] + (box_lo[0] + i + 0.5) * geom["dx"][0]
        for j in range(ny):
            y = geom["x0"][1] + (box_lo[1] + j + 0.5) * geom["dx"][1]
            for k in range(nz):
                z = geom["x0"][2] + (box_lo[2] + k + 0.5) * geom["dx"][2]
                out.extend(struct.pack("<d", x * x + 0.5 * y * y - z * z))
    return bytes(out)


def _gradient_of_block(rt, ds, runmeta, field, block, shape):
    fetch_field = 19 + 10 * block
    out_field = 20 + 10 * block
    fetch = Stage(
        name=f"fetch{block}",
        templates=[TaskTemplate(
            name="fetch_nbr",
            plane="chunk",
            kernel="amr_subbox_fetch_pack",
            domain=Domain(0, 0, [block]),
            inputs=[],
            outputs=[OutputRef(FieldRef(fetch_field), BufferSpec(
                DType.OPAQUE,
                DynamicShape(DynamicUpperBound.amr_subbox_pack()),
            ))],
            params=AmrSubboxPackParams(field, 0, 0, 0, 1),
        )],
    )
    grad = Stage(
        name=f"grad{block}",
        after=[fetch],
        templates=[TaskTemplate(
            name="gradU",
            plane="chunk",
            kernel="gradU_stencil",
            domain=Domain(0, 0, [block]),
            inputs=[FieldRef(field), FieldRef(fetch_field)],
            outputs=[OutputRef(
                FieldRef(out_field), BufferSpec(DType.F64, BlockShape(3))
            )],
            params=GradStencilParams(field, 0, 0, 0),
        )],
    )
    rt._rt.run_packed_plan(encode_plan(Plan([fetch, grad])), runmeta._h, ds._h)
    out = rt.get_task_chunk(step=0, level=0, field=out_field, version=0, block=block)
    vals = struct.unpack(f"<{len(out) // 8}d", out)
    nx, ny, nz = shape

    def grad_at(i: int, j: int, k: int) -> tuple[float, float, float]:
        idx = (i * ny + j) * nz + k
        return vals[3 * idx + 0], vals[3 * idx + 1], vals[3 * idx + 2]

    return grad_at


def test_gradU_stencil_quadratic_field_across_same_level_blocks() -> None:
    """A quadratic field's central-difference gradient is exact at interior
    cells; cells beside another block on the same level are interior too, so
    they must be exact as well rather than fall back to a one-sided fit."""
    from analysis import Runtime
    from analysis.dataset import open_dataset
    from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta

    rt = Runtime()
    geom = {"dx": (1.0, 1.0, 1.0), "x0": (0.0, 0.0, 0.0), "index_origin": (0, 0, 0)}
    boxes = [((0, 0, 0), (3, 7, 7)), ((4, 0, 0), (7, 7, 7))]
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox(lo, hi) for lo, hi in boxes],
                    ),
                ],
            )
        ]
    )
    ds = open_dataset("memory://local", runmeta=runmeta, step=0, level=0, runtime=rt)
    field = 11
    for block, (lo, hi) in enumerate(boxes):
        shape = [h - l + 1 for l, h in zip(lo, hi)]
        ds._h.set_chunk_ref(0, 0, field, 0, block,
                            _pack_quadratic_field_double(lo, hi, geom), "f64", shape)

    for block, (lo, hi) in enumerate(boxes):
        shape = tuple(h - l + 1 for l, h in zip(lo, hi))
        grad_at = _gradient_of_block(rt, ds, runmeta, field, block, shape)
        # Cells on the shared face x = 4, away from the domain walls.
        i = 3 if block == 0 else 0
        for j in (2, 3, 4, 5):
            for k in (2, 3, 4, 5):
                x = lo[0] + i + 0.5
                y = j + 0.5
                z = k + 0.5
                gx, gy, gz = grad_at(i, j, k)
                assert abs(gx - 2.0 * x) < 1e-9, (block, j, k, gx, 2 * x)
                assert abs(gy - y) < 1e-9
                assert abs(gz + 2.0 * z) < 1e-9
