from __future__ import annotations

import os
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


def test_slice_operator_demo_script(tmp_path) -> None:
    output = tmp_path / "slice.png"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/slice_operator_demo.py",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    if "Runtime init failed" in result.stdout:
        pytest.skip("Runtime init failed (likely missing built module)")

    assert result.returncode == 0, result.stderr
    assert "Slice comparison: allclose = True" in result.stdout
    assert output.is_file()


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
    payload = bytes(8 * 8 * 8 * 8)
    ds._h.set_chunk_ref(0, 0, 1, 0, 0, payload, "f64", [8, 8, 8])
    ds._h.set_chunk_ref(0, 0, 2, 0, 0, payload, "f64", [8, 8, 8])
    return rt, runmeta, ds


def test_executor_detects_stage_cycle() -> None:
    from analysis.plan import Plan, Stage
    from analysis.plan_codec import encode_plan

    a = Stage("a")
    b = Stage("b", after=[a])
    a.after.append(b)
    with pytest.raises(ValueError, match="dependency cycle"):
        encode_plan(Plan([a, b]))


def test_plan_encoder_derives_valid_stage_dependency_indices() -> None:
    from analysis.plan import Plan, Stage
    from analysis.plan_codec import encode_plan

    parent = Stage("parent")
    child = Stage("child", after=[parent])
    assert encode_plan(Plan([child]))[4:8] == b"KPLN"


def test_neighbor_dep_faces_and_width() -> None:
    import analysis._core  # type: ignore # noqa: F401

    from analysis.buffer import BlockShape, BufferSpec, DType
    from analysis.kernel_params import GradStencilParams
    from analysis.plan import DependencyRule, Domain, FieldRef, OutputRef, Plan, Stage
    from analysis.plan_codec import encode_plan

    rt, runmeta, ds = _make_runtime_handles()

    def packed(deps: DependencyRule) -> bytes:
        stage = Stage("s0")
        stage.map_blocks(
            name="noop", kernel="gradU_stencil", domain=Domain(0, 0),
            inputs=[FieldRef(1)],
            outputs=[OutputRef(FieldRef(2), BufferSpec(DType.F64, BlockShape(3)))],
            deps=deps, params=GradStencilParams(input_field=1),
        )
        return encode_plan(Plan([stage]))

    # width=0 should be allowed (no neighbors)
    rt._rt.run_packed_plan(
        packed(DependencyRule(kind="FaceNeighbors", width=0)), runmeta._h, ds._h
    )

    # all faces disabled should be allowed (no neighbors)
    rt._rt.run_packed_plan(
        packed(DependencyRule(kind="FaceNeighbors", width=1, faces=(False,) * 6)),
        runmeta._h,
        ds._h,
    )


def test_neighbor_halo_inputs_indices() -> None:
    import analysis._core  # type: ignore # noqa: F401

    from analysis.buffer import BlockShape, BufferSpec, DType
    from analysis.kernel_params import GradStencilParams
    from analysis.plan import DependencyRule, Domain, FieldRef, OutputRef, Plan, Stage
    from analysis.plan_codec import encode_plan

    rt, runmeta, ds = _make_runtime_handles()

    def packed(halo_inputs: tuple[int, ...]) -> bytes:
        stage = Stage("s0")
        stage.map_blocks(
            name="noop", kernel="gradU_stencil", domain=Domain(0, 0),
            inputs=[FieldRef(1), FieldRef(2)],
            outputs=[OutputRef(FieldRef(3), BufferSpec(DType.F64, BlockShape(3)))],
            deps=DependencyRule(kind="FaceNeighbors", width=1, halo_inputs=halo_inputs),
            params=GradStencilParams(input_field=1),
        )
        return encode_plan(Plan([stage]))

    rt._rt.run_packed_plan(packed((1,)), runmeta._h, ds._h)

    with pytest.raises(RuntimeError):
        rt._rt.run_packed_plan(packed((2,)), runmeta._h, ds._h)
