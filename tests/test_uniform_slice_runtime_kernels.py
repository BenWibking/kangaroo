from __future__ import annotations

import numpy as np

from analysis import Domain, FieldRef, Plan, Runtime, Stage
from analysis.dataset import open_dataset
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


def _single_block_runmeta(step: int = 0) -> RunMeta:
    levels = [
        LevelMeta(
            geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
            boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
        )
    ]
    return RunMeta(steps=[StepMeta(step=i, levels=levels) for i in range(step + 1)])


def _set_block_double(ds, *, step: int, level: int, field: int, block: int, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float64)
    ds._h.set_chunk_ref(step, level, field, 0, block, arr.tobytes(order="C"))


def test_uniform_slice_finalize_zero_pixel_area_fills_nan() -> None:
    step = 0
    rt = Runtime()
    runmeta = _single_block_runmeta(step=step)
    ds = open_dataset("memory://uniform-slice-kernels", runmeta=runmeta, step=step, level=0, runtime=rt)

    sum_field = rt.alloc_field_id("uniform_slice_sum")
    area_field = rt.alloc_field_id("uniform_slice_area")
    out_field = rt.alloc_field_id("uniform_slice_out")

    _set_block_double(ds, step=step, level=0, field=sum_field, block=0, values=np.array([10.0, 20.0]))
    _set_block_double(ds, step=step, level=0, field=area_field, block=0, values=np.array([1.0, 0.0]))

    stage = Stage(name="uniform_slice_finalize", plane="graph")
    stage.map_blocks(
        name="uniform_slice_finalize",
        kernel="uniform_slice_finalize",
        domain=Domain(step=step, level=0, blocks=[0]),
        inputs=[FieldRef(sum_field), FieldRef(area_field)],
        outputs=[FieldRef(out_field)],
        output_bytes=[2 * 8],
        deps={"kind": "None"},
        params={
            "graph_kind": "reduce",
            "fan_in": 1,
            "num_inputs": 1,
            "input_base": 0,
            "output_base": 0,
            "bytes_per_value": 8,
            "pixel_area": 0.0,
        },
    )

    rt.run(Plan(stages=[stage]), runmeta=runmeta, dataset=ds)

    out = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=out_field,
        shape=(2,),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    assert np.isnan(out).all()
