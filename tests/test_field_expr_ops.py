from __future__ import annotations

import numpy as np

from analysis import Runtime
from analysis.dataset import open_dataset
from analysis.pipeline import Pipeline
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


def _set_block_double(ds, *, step: int, level: int, field: int, block: int, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float64)
    ds._h.set_chunk_ref(step, level, field, 0, block, arr.tobytes(order="C"))


def _runmeta(step: int) -> RunMeta:
    levels = [
        LevelMeta(
            geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
            boxes=[BlockBox((0, 0, 0), (1, 1, 1))],
        )
    ]
    return RunMeta(steps=[StepMeta(step=i, levels=levels) for i in range(step + 1)])


def test_field_expr_runtime_velocity_like_expression() -> None:
    rt = Runtime()
    step = 5
    runmeta = _runmeta(step)
    ds = open_dataset("memory://expr-local", runmeta=runmeta, step=step, level=0, runtime=rt)

    rho_f = 24001
    mx_f = 24002
    ds.register_field("density", rho_f)
    ds.register_field("xmom", mx_f)

    rho = np.array(
        [
            [[1.0, 2.0], [4.0, 8.0]],
            [[2.0, 4.0], [8.0, 16.0]],
        ],
        dtype=np.float64,
    )
    mx = np.array(
        [
            [[2.0, 2.0], [8.0, 24.0]],
            [[4.0, 8.0], [16.0, 32.0]],
        ],
        dtype=np.float64,
    )
    _set_block_double(ds, step=step, level=0, field=rho_f, block=0, values=rho)
    _set_block_double(ds, step=step, level=0, field=mx_f, block=0, values=mx)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    vel = pipe.field_expr("mx / rho", {"mx": pipe.field(mx_f), "rho": pipe.field(rho_f)}, out="velx")
    shifted = pipe.field_add(vel, pipe.field(rho_f), out="vel_plus_rho")
    pipe.run()

    vel_out = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=vel.field,
        shape=(2, 2, 2),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    shifted_out = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=shifted.field,
        shape=(2, 2, 2),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )

    expected_vel = mx / rho
    assert np.allclose(vel_out, expected_vel)
    assert np.allclose(shifted_out, expected_vel + rho)
