from __future__ import annotations

import numpy as np

from analysis import Runtime, cdf_from_histogram, cdf_from_samples
from analysis.dataset import open_dataset
from analysis.pipeline import Pipeline
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


def _set_block_double(ds, *, step: int, level: int, field: int, block: int, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float64)
    ds._h.set_chunk_ref(step, level, field, 0, block, arr.tobytes(order="C"))


def _runmeta_with_step_index(step: int, levels: list[LevelMeta]) -> RunMeta:
    steps: list[StepMeta] = []
    for idx in range(step + 1):
        steps.append(StepMeta(step=idx, levels=levels))
    return RunMeta(steps=steps)


def test_histogram1d_runtime_unweighted_and_weighted() -> None:
    rt = Runtime()
    step = 10
    levels = [
        LevelMeta(
            geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
            boxes=[BlockBox((0, 0, 0), (1, 1, 1))],
        )
    ]
    runmeta = _runmeta_with_step_index(
        step,
        levels,
    )
    ds = open_dataset("memory://local", runmeta=runmeta, step=step, level=0, runtime=rt)

    scalar = 21001
    weights = 21002
    ds.register_field("scalar", scalar)
    ds.register_field("weights", weights)
    vals = np.array(
        [
            [[0.1, 0.2], [0.6, 0.7]],
            [[0.9, 0.4], [0.3, 0.8]],
        ],
        dtype=np.float64,
    )
    wts = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float64,
    )
    _set_block_double(ds, step=step, level=0, field=scalar, block=0, values=vals)
    _set_block_double(ds, step=step, level=0, field=weights, block=0, values=wts)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    scalar_h = pipe.field(scalar)
    weights_h = pipe.field(weights)
    h_unweighted = pipe.histogram1d(scalar_h, hist_range=(0.0, 1.0), bins=5, out="hist1d_u")
    h_weighted = pipe.histogram1d(
        scalar_h,
        hist_range=(0.0, 1.0),
        bins=5,
        weights=weights_h,
        out="hist1d_w",
    )
    pipe.run()

    raw_u = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=h_unweighted.counts.field,
        shape=(5,),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    raw_w = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=h_weighted.counts.field,
        shape=(5,),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )

    assert np.allclose(raw_u, np.array([1.0, 2.0, 1.0, 2.0, 2.0]))
    assert np.allclose(raw_w, np.array([1.0, 9.0, 6.0, 7.0, 13.0]))


def test_histogram2d_runtime_unweighted() -> None:
    rt = Runtime()
    step = 20
    levels = [
        LevelMeta(
            geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
            boxes=[BlockBox((0, 0, 0), (1, 1, 1))],
        )
    ]
    runmeta = _runmeta_with_step_index(
        step,
        levels,
    )
    ds = open_dataset("memory://local", runmeta=runmeta, step=step, level=0, runtime=rt)

    x_field = 22001
    y_field = 22002
    ds.register_field("x", x_field)
    ds.register_field("y", y_field)
    x_vals = np.array(
        [
            [[0.1, 0.1], [0.9, 0.9]],
            [[1.1, 1.1], [1.9, 1.9]],
        ],
        dtype=np.float64,
    )
    y_vals = np.array(
        [
            [[0.1, 1.1], [0.1, 1.1]],
            [[0.1, 1.1], [0.1, 1.1]],
        ],
        dtype=np.float64,
    )
    _set_block_double(ds, step=step, level=0, field=x_field, block=0, values=x_vals)
    _set_block_double(ds, step=step, level=0, field=y_field, block=0, values=y_vals)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    x_h = pipe.field(x_field)
    y_h = pipe.field(y_field)
    hist = pipe.histogram2d(
        x_h,
        y_h,
        x_range=(0.0, 2.0),
        y_range=(0.0, 2.0),
        bins=(2, 2),
        out="hist2d",
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=hist.counts.field,
        shape=(2, 2),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    assert np.allclose(raw, np.array([[2.0, 2.0], [2.0, 2.0]]))


def test_histogram2d_runtime_cell_volume_weight_mode() -> None:
    rt = Runtime()
    step = 21
    levels = [
        LevelMeta(
            geom=LevelGeom(dx=(2.0, 2.0, 2.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
            boxes=[BlockBox((0, 0, 0), (1, 1, 1))],
        )
    ]
    runmeta = _runmeta_with_step_index(step, levels)
    ds = open_dataset("memory://local", runmeta=runmeta, step=step, level=0, runtime=rt)

    x_field = 22011
    y_field = 22012
    ds.register_field("x", x_field)
    ds.register_field("y", y_field)

    x_vals = np.full((2, 2, 2), 0.5, dtype=np.float64)
    y_vals = np.full((2, 2, 2), 0.5, dtype=np.float64)
    _set_block_double(ds, step=step, level=0, field=x_field, block=0, values=x_vals)
    _set_block_double(ds, step=step, level=0, field=y_field, block=0, values=y_vals)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    hist = pipe.histogram2d(
        pipe.field(x_field),
        pipe.field(y_field),
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        bins=(1, 1),
        weight_mode="cell_volume",
        out="hist2d_vol",
    )
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=hist.counts.field,
        shape=(1, 1),
        dtype=np.float64,
        dataset=ds,
        block=0,
    )
    # 8 cells each with cell volume 8.
    assert np.allclose(raw, np.array([[64.0]]))


def test_histogram1d_amr_covered_boxes_masking() -> None:
    rt = Runtime()
    step = 30
    levels = [
        LevelMeta(
            geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=2),
            boxes=[BlockBox((0, 0, 0), (3, 3, 3))],
        ),
        LevelMeta(
            geom=LevelGeom(dx=(0.5, 0.5, 0.5), x0=(0.0, 0.0, 0.0), ref_ratio=1),
            boxes=[BlockBox((0, 0, 0), (3, 7, 7))],
        ),
    ]
    runmeta = _runmeta_with_step_index(
        step,
        levels,
    )
    ds0 = open_dataset("memory://local", runmeta=runmeta, step=step, level=0, runtime=rt)
    ds1 = open_dataset("memory://local", runmeta=runmeta, step=step, level=1, runtime=rt)
    scalar = 23001
    ds0.register_field("scalar", scalar)

    coarse_vals = np.full((4, 4, 4), 0.5, dtype=np.float64)
    fine_vals = np.full((4, 8, 8), 0.5, dtype=np.float64)
    _set_block_double(ds0, step=step, level=0, field=scalar, block=0, values=coarse_vals)
    _set_block_double(ds1, step=step, level=1, field=scalar, block=0, values=fine_vals)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds0)
    scalar_h = pipe.field(scalar)
    hist = pipe.histogram1d(scalar_h, hist_range=(0.0, 1.0), bins=1, out="hist_amr")
    pipe.run()

    raw = rt.get_task_chunk_array(
        step=step,
        level=0,
        field=hist.counts.field,
        shape=(1,),
        dtype=np.float64,
        dataset=ds0,
        block=0,
    )
    assert np.allclose(raw, np.array([288.0]))


def test_histogram_cdf_helpers() -> None:
    cdf = cdf_from_histogram([1.0, 2.0, 3.0])
    assert np.allclose(cdf, np.array([1.0 / 6.0, 3.0 / 6.0, 1.0]))

    xs, ys = cdf_from_samples([3.0, 1.0, 2.0, 2.0])
    assert xs == [1.0, 2.0, 2.0, 3.0]
    assert np.allclose(ys, np.array([0.25, 0.5, 0.75, 1.0]))
