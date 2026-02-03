from __future__ import annotations

import pytest


def _make_amr_runmeta():
    from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta

    coarse_geom = LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), index_origin=(0, 0, 0), ref_ratio=2)
    fine_geom = LevelGeom(dx=(0.5, 0.5, 0.5), x0=(0.0, 0.0, 0.0), index_origin=(0, 0, 0), ref_ratio=1)

    coarse_level = LevelMeta(
        geom=coarse_geom,
        boxes=[BlockBox((0, 0, 0), (7, 7, 7))],
    )
    fine_level = LevelMeta(
        geom=fine_geom,
        boxes=[BlockBox((0, 0, 0), (7, 15, 15))],
    )

    return RunMeta(steps=[StepMeta(step=0, levels=[coarse_level, fine_level])])


def test_uniform_projection_amr_covered_boxes() -> None:
    import analysis._core  # type: ignore # noqa: F401

    from analysis import Plan, Runtime
    from analysis.ctx import LoweringContext
    from analysis.dataset import open_dataset
    from analysis.ops import UniformProjection

    try:
        rt = Runtime()
    except Exception:
        pytest.skip("Runtime init failed (likely missing built module)")

    runmeta = _make_amr_runmeta()
    ds = open_dataset("memory://projection-test", runmeta=runmeta, step=0, level=0, runtime=rt)
    field = ds.field_id("scalar")

    op = UniformProjection(
        field=field,
        axis="z",
        axis_bounds=(0.0, 8.0),
        rect=(0.0, 0.0, 8.0, 8.0),
        resolution=(8, 8),
    )

    ctx = LoweringContext(runtime=rt._rt, runmeta=runmeta, dataset=ds)
    plan = Plan(stages=op.lower(ctx))

    templates = [
        tmpl
        for stage in plan.stages
        for tmpl in stage.templates
        if tmpl.kernel == "uniform_projection_accumulate"
    ]
    assert templates

    level0 = [tmpl for tmpl in templates if tmpl.domain.level == 0]
    level1 = [tmpl for tmpl in templates if tmpl.domain.level == 1]
    assert level0
    assert level1

    expected_box = [[0, 0, 0], [3, 7, 7]]

    for tmpl in level0:
        assert tmpl.params["axis_bounds"] == [0.0, 8.0]
        assert expected_box in tmpl.params["covered_boxes"]

    for tmpl in level1:
        assert tmpl.params["axis_bounds"] == [0.0, 8.0]
        assert tmpl.params["covered_boxes"] == []
