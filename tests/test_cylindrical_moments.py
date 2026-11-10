from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import Runtime
from analysis.buffer import BufferSpec, DType, FixedShape, InitPolicy
from analysis.dataset import open_dataset
from analysis.kernel_params import ToomreProfileParams
from analysis.plan import (
    DependencyRule,
    Domain,
    FieldRef,
    OutputRef,
    Plan,
    Stage,
    TaskTemplate,
)
from analysis.plan_codec import encode_plan
from analysis.pipeline import Pipeline
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta

NUM_MOMENTS = 2


def _single_level_runmeta() -> RunMeta:
    return RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(-5.0, -5.0, -2.0),
                            index_origin=(0, 0, 0),
                            ref_ratio=1,
                        ),
                        boxes=[BlockBox((0, 0, 0), (9, 9, 3))],
                    )
                ],
            )
        ]
    )


def _set_chunk(ds, field: int, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64)
    ds._h.set_chunk_ref(
        0, 0, field, 0, 0, array.tobytes(), "f64", list(array.shape)
    )


def test_cylindrical_moments_kernel_direct() -> None:
    runtime = Runtime()
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, 0.0, 0.0),
                            index_origin=(0, 0, 0),
                            ref_ratio=1,
                        ),
                        boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    ds = open_dataset("memory://cylmom-kernel", runmeta=runmeta, runtime=runtime)
    input_field = 401
    _set_chunk(ds, input_field, np.array([[[3.0]]]))
    output_field = 402
    packed = encode_plan(
        Plan(
            [
                Stage(
                    name="cylmom",
                    templates=[
                        TaskTemplate(
                            name="cylindrical_moments_accumulate",
                            plane="chunk",
                            kernel="cylindrical_moments_accumulate",
                            domain=Domain(step=0, level=0, blocks=[0]),
                            inputs=[FieldRef(input_field)],
                            outputs=[
                                OutputRef(
                                    FieldRef(output_field),
                                    BufferSpec(
                                        DType.F64,
                                        FixedShape((2 * 2, NUM_MOMENTS)),
                                        InitPolicy.ZERO,
                                    ),
                                )
                            ],
                            deps=DependencyRule(),
                            params=ToomreProfileParams(
                                radial_edges=(0.0, 1.0, 2.0),
                                z_bounds=(-2.0, 2.0),
                                center=(0.5, 0.5, 0.5),
                                z_edges=(0.0, 0.5, 1.5),
                            ),
                        )
                    ],
                )
            ]
        )
    )
    runtime._rt.run_packed_plan(packed, runmeta._h, ds._h)
    values = runtime.get_task_chunk_array(
        step=0,
        level=0,
        field=output_field,
        block=0,
        shape=(2 * 2, NUM_MOMENTS),
        dtype=np.float64,
        dataset=ds,
    )
    # Cell center is (0.5, 0.5, 0.5): radius sqrt(2)/2 ~ 0.707 -> bin 0,
    # height |0.5 - 0.5| = 0 -> bin 0.  Flattened row = 0 * 2 + 0.
    np.testing.assert_allclose(values[0], (3.0, 1.0))
    np.testing.assert_allclose(values[1:], 0.0)


def test_cylindrical_moments_runtime_accumulates_expected_moments() -> None:
    runtime = Runtime()
    runmeta = _single_level_runmeta()
    ds = open_dataset("memory://cylmom-runtime", runmeta=runmeta, runtime=runtime)
    shape = (10, 10, 4)
    x = -5.0 + (np.arange(shape[0]) + 0.5)
    y = -5.0 + (np.arange(shape[1]) + 0.5)
    z = -2.0 + (np.arange(shape[2]) + 0.5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    field = 301
    _set_chunk(ds, field, np.full(shape, 2.0))

    pipe = Pipeline(runtime=runtime, runmeta=runmeta, dataset=ds)
    radial_edges = (0.5, 1.6, 2.7, 3.5)
    z_edges = (0.0, 1.25, 2.5)
    handle = pipe.cylindrical_moments(
        field,
        radial_edges=radial_edges,
        z_edges=z_edges,
        out="cylmom",
    )
    assert handle.bins == (3, 2)
    assert handle.radial_range == (0.5, 3.5)
    assert handle.z_range == (0.0, 2.5)
    pipe.run()
    moments = runtime.get_task_chunk_array(
        step=0,
        level=0,
        field=handle.field,
        block=0,
        shape=(3 * 2, NUM_MOMENTS),
        dtype=np.float64,
        dataset=ds,
    ).reshape(3, 2, NUM_MOMENTS)

    radius = np.sqrt(xx**2 + yy**2)
    height = np.abs(zz)
    for radial_bin, (rlo, rhi) in enumerate(zip(radial_edges[:-1], radial_edges[1:])):
        for z_bin, (zlo, zhi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
            selected = (
                (radius >= rlo)
                & (radius <= rhi if radial_bin == 2 else radius < rhi)
                & (height >= zlo)
                & (height <= zhi if z_bin == 1 else height < zhi)
            )
            columns = int(np.count_nonzero(selected))
            np.testing.assert_allclose(moments[radial_bin, z_bin, 0], 2.0 * columns)
            np.testing.assert_allclose(moments[radial_bin, z_bin, 1], float(columns))


def test_cylindrical_moments_lowering_prunes_blocks_and_wires_reduction() -> None:
    runtime = Runtime()
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(-2.0, -2.0, -1.0),
                            index_origin=(0, 0, 0),
                            ref_ratio=2,
                        ),
                        boxes=[BlockBox((0, 0, 0), (3, 3, 1))],
                    ),
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(0.5, 0.5, 0.5),
                            x0=(-2.0, -2.0, -1.0),
                            index_origin=(0, 0, 0),
                            ref_ratio=1,
                        ),
                        boxes=[BlockBox((2, 2, 0), (5, 5, 3))],
                    ),
                ],
            )
        ]
    )
    ds = open_dataset("memory://cylmom-lowering", runmeta=runmeta, runtime=runtime)
    pipe = Pipeline(runtime=runtime, runmeta=runmeta, dataset=ds)
    handle = pipe.cylindrical_moments(
        501,
        radial_edges=(0.0, 1.0),
        z_edges=(0.0, 1.0),
        out="cylmom",
    )
    plan = pipe.plan()
    templates = [
        template
        for stage in plan.stages
        for template in stage.templates
    ]
    accumulate = [
        template
        for template in templates
        if template.kernel == "cylindrical_moments_accumulate"
    ]
    assert accumulate, "expected a cylindrical_moments_accumulate template"
    assert any(template.kernel == "uniform_slice_reduce" for template in templates)
    assert any(template.kernel == "uniform_slice_add" for template in templates)
    assert all(
        template.kernel != "gradU_stencil" for template in templates
    ), "cylindrical moments must not depend on potential gradients"
    assert handle.bins == (1, 1)


def test_cylindrical_moments_rejects_invalid_edges() -> None:
    runtime = Runtime()
    runmeta = _single_level_runmeta()
    ds = open_dataset("memory://cylmom-invalid", runmeta=runmeta, runtime=runtime)
    pipe = Pipeline(runtime=runtime, runmeta=runmeta, dataset=ds)
    for radial_edges, z_edges in (
        ((0.0, 1.0), ()),
        ((0.0, 1.0), (1.0,)),
        ((0.0, 1.0), (1.0, 0.5)),
        ((0.0, 1.0), (-1.0, 1.0)),
        ((1.0, 0.5), (0.0, 1.0)),
        ((0.0, math.inf), (0.0, 1.0)),
    ):
        with pytest.raises(ValueError):
            pipe.cylindrical_moments(
                601,
                radial_edges=radial_edges,
                z_edges=z_edges,
            )
