from __future__ import annotations

import numpy as np

from analysis import Runtime
from analysis.dataset import open_dataset
from analysis.pipeline import pipeline
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


def _single_block_runmeta() -> RunMeta:
    return RunMeta(
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


def test_particle_runtime_masks_filters_and_reductions_with_numpy_inputs() -> None:
    rt = Runtime()
    runmeta = _single_block_runmeta()
    ds = open_dataset("memory://particle-kernels", runmeta=runmeta, step=0, level=0, runtime=rt)
    pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    pipe._particle_max_chunks = 2

    values = np.array([1.0, 2.0, np.inf, -3.0, 2.0], dtype=np.float64)

    positive_finite = pipe.particle_and(
        pipe.particle_gt(values, 0.0),
        pipe.particle_isfinite(values),
    )
    equal_two = pipe.particle_equals(values, 2.0)
    selected = pipe.particle_filter(values, positive_finite)

    assert pipe.particle_count(positive_finite) == 3
    assert pipe.particle_count(equal_two) == 2
    assert pipe.particle_sum(selected) == 5.0
    assert pipe.particle_min(values, finite_only=True) == -3.0
    assert pipe.particle_max(values, finite_only=True) == 2.0
    assert [chunk.tolist() for chunk in selected.iter_chunks()] == [[1.0, 2.0], [2.0]]
