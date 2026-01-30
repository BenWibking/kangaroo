from __future__ import annotations

import subprocess
import sys

import pytest


def test_smoke_demo_script() -> None:
    import analysis._core  # type: ignore # noqa: F401

    result = subprocess.run([sys.executable, "scripts/smoke_demo.py"], check=False)

    if result.returncode == 1:
        pytest.skip("Runtime init failed (likely missing built module)")
    if result.returncode == 2:
        pytest.xfail("Runtime ran but kernels are not registered yet")

    assert result.returncode == 0


def _make_runtime_handles():
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
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox((0, 0, 0), (7, 7, 7))],
                    )
                ],
            )
        ]
    )
    ds = open_dataset("memory://example", runmeta=runmeta, step=0, level=0, runtime=rt)
    return rt, runmeta, ds


def test_executor_detects_stage_cycle() -> None:
    import analysis._core  # type: ignore # noqa: F401

    import msgpack

    rt, runmeta, ds = _make_runtime_handles()

    plan_dict = {
        "stages": [
            {"name": "a", "plane": "chunk", "after": [1], "templates": []},
            {"name": "b", "plane": "chunk", "after": [0], "templates": []},
        ]
    }
    packed = msgpack.packb(plan_dict, use_bin_type=True)

    with pytest.raises(RuntimeError):
        rt._rt.run_packed_plan(packed, runmeta._h, ds._h)


def test_executor_rejects_bad_stage_dep_index() -> None:
    import analysis._core  # type: ignore # noqa: F401

    import msgpack

    rt, runmeta, ds = _make_runtime_handles()

    plan_dict = {
        "stages": [{"name": "s0", "plane": "chunk", "after": [2], "templates": []}]
    }

    packed = msgpack.packb(plan_dict, use_bin_type=True)
    with pytest.raises(RuntimeError):
        rt._rt.run_packed_plan(packed, runmeta._h, ds._h)


def test_neighbor_dep_faces_and_width() -> None:
    import analysis._core  # type: ignore # noqa: F401

    import msgpack

    rt, runmeta, ds = _make_runtime_handles()

    base_task = {
        "name": "noop",
        "plane": "chunk",
        "kernel": "gradU_stencil",
        "domain": {"step": 0, "level": 0, "blocks": None},
        "inputs": [{"field": 1, "version": 0}],
        "outputs": [{"field": 2, "version": 0}],
        "output_bytes": [0],
        "deps": {"kind": "FaceNeighbors", "width": 1, "faces": [1, 1, 1, 1, 1, 1]},
        "params": {},
    }

    # width=0 should be allowed (no neighbors)
    plan_dict = {"stages": [{"name": "s0", "plane": "chunk", "after": [], "templates": [
        {**base_task, "deps": {"kind": "FaceNeighbors", "width": 0, "faces": [1, 1, 1, 1, 1, 1]}}
    ]}]}
    packed = msgpack.packb(plan_dict, use_bin_type=True)
    rt._rt.run_packed_plan(packed, runmeta._h, ds._h)

    # all faces disabled should be allowed (no neighbors)
    plan_dict = {"stages": [{"name": "s0", "plane": "chunk", "after": [], "templates": [
        {**base_task, "deps": {"kind": "FaceNeighbors", "width": 1, "faces": [0, 0, 0, 0, 0, 0]}}
    ]}]}
    packed = msgpack.packb(plan_dict, use_bin_type=True)
    rt._rt.run_packed_plan(packed, runmeta._h, ds._h)


def test_neighbor_halo_inputs_indices() -> None:
    import analysis._core  # type: ignore # noqa: F401

    import msgpack

    rt, runmeta, ds = _make_runtime_handles()

    base_task = {
        "name": "noop",
        "plane": "chunk",
        "kernel": "gradU_stencil",
        "domain": {"step": 0, "level": 0, "blocks": None},
        "inputs": [{"field": 1, "version": 0}, {"field": 2, "version": 0}],
        "outputs": [{"field": 3, "version": 0}],
        "output_bytes": [0],
        "deps": {
            "kind": "FaceNeighbors",
            "width": 1,
            "faces": [1, 1, 1, 1, 1, 1],
            "halo_inputs": [1],
        },
        "params": {},
    }

    plan_dict = {"stages": [{"name": "s0", "plane": "chunk", "after": [], "templates": [base_task]}]}
    packed = msgpack.packb(plan_dict, use_bin_type=True)
    rt._rt.run_packed_plan(packed, runmeta._h, ds._h)

    bad_task = {**base_task, "deps": {**base_task["deps"], "halo_inputs": [2]}}
    plan_dict = {"stages": [{"name": "s0", "plane": "chunk", "after": [], "templates": [bad_task]}]}
    packed = msgpack.packb(plan_dict, use_bin_type=True)
    with pytest.raises(RuntimeError):
        rt._rt.run_packed_plan(packed, runmeta._h, ds._h)
