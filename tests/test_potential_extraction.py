"""Native, leaf-cell extraction tests; run only in the dedicated pixi-hpx environment."""

from __future__ import annotations

import numpy as np
import pytest

from analysis import Runtime
from analysis.dataset import open_dataset
from analysis.pipeline import Pipeline
from analysis.potential_extract import extract_samples
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta
from potential_replay.fit import fit_samples, quadrature_points, solve_samples

UNITS = {"length_to_cm": 1, "time_to_s": 1, "potential_to_cm2_s2": 1}


def quadratic(p):
    x, y, z = np.moveaxis(p, -1, 0)
    return (
        7
        + x * x
        + 2 * y * y
        + 3 * z * z
        + 0.3 * x
        + 0.4 * y
        + 0.5 * z
        + 0.2 * x * y
        + 0.1 * x * z
    )


def dataset(
    runtime, *, n=16, split=False, refined=False, poison=False, function=quadratic
):
    # Nonzero index origins and block offsets exercise physical-coordinate mapping.
    origin = (3, 5, 7)
    geom = LevelGeom(dx=(4 / n,) * 3, x0=(-2,) * 3, ref_ratio=2, index_origin=origin)
    box = lambda low, high: BlockBox(
        tuple(origin[a] + low[a] for a in range(3)),
        tuple(origin[a] + high[a] for a in range(3)),
    )
    boxes = (
        [
            box((0, 0, 0), (n // 2 - 1, n - 1, n - 1)),
            box((n // 2, 0, 0), (n - 1, n - 1, n - 1)),
        ]
        if split
        else [box((0, 0, 0), (n - 1, n - 1, n - 1))]
    )
    levels = [LevelMeta(geom=geom, boxes=boxes)]
    if refined:
        levels.append(
            LevelMeta(
                geom=LevelGeom(
                    dx=(2 / n,) * 3, x0=(-2,) * 3, ref_ratio=1, index_origin=origin
                ),
                boxes=[box((n // 2,) * 3, (3 * n // 2 - 1,) * 3)],
            )
        )
    runmeta = RunMeta(steps=[StepMeta(step=0, levels=levels)])
    ds = open_dataset("memory://potential-test", runmeta=runmeta, runtime=runtime)
    field = runtime.alloc_field_id("synthetic-potential")
    for lev, level in enumerate(levels):
        for block, b in enumerate(level.boxes):
            coordinates = [
                level.geom.x0[a]
                + (np.arange(b.lo[a], b.hi[a] + 1) - origin[a] + 0.5) * level.geom.dx[a]
                for a in range(3)
            ]
            points = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1)
            values = np.array(function(points), dtype=np.float64)
            if poison and lev == 0:
                # All of these coarse cells are covered by level 1. Their NaNs must
                # not reach the RHS or the bad-value volume diagnostic.
                inside = np.all((points >= -1) & (points <= 1), axis=-1)
                values[inside] = np.nan
            ds._h.set_chunk_ref(
                0, lev, field, 0, block, values.tobytes(), "f64", list(values.shape)
            )
    return ds, runmeta, field


def stats_for(runtime, ds, runmeta, field, points, h, fan_in=None):
    pipe = Pipeline(runtime=runtime, runmeta=runmeta, dataset=ds)
    handle = pipe.potential_samples(
        field, points=points, half_width=h, reduce_fan_in=fan_in
    )
    pipe.run()
    return runtime.get_task_chunk_array(
        step=0, level=0, field=handle.field, block=0, dataset=ds
    )


def test_native_quadratic_layout_refinement_and_covered_coarse_nan():
    runtime = Runtime()
    points = np.array([[0, 0, 0], [0.8, 0.3, -0.2], [-1, 0.1, 0.2], [1, 0.2, -0.1]])
    results = []
    for split, refined in ((False, False), (True, False), (True, True)):
        ds, rm, field = dataset(runtime, split=split, refined=refined, poison=refined)
        stats = stats_for(runtime, ds, rm, field, points, 0.65, fan_in=2)
        samples, diagnostics = solve_samples(stats)
        results.append(samples)
        np.testing.assert_allclose(samples, quadratic(points), atol=5e-13)
        np.testing.assert_allclose(diagnostics["coverage"], 1, atol=2e-14)
        assert np.all(stats[:, 111] == 0)
    np.testing.assert_allclose(results, np.broadcast_to(results[0], (3, 4)), atol=5e-13)


def test_native_failures_for_coverage_nonfinite_and_small_support():
    runtime = Runtime()
    ds, rm, field = dataset(runtime)
    stats = stats_for(runtime, ds, rm, field, [[1.9, 0, 0]], 0.6)
    assert stats[0, 110] < 0.7
    with pytest.raises(ValueError, match="incomplete"):
        solve_samples(stats)
    stats = stats_for(runtime, ds, rm, field, [[0.1, 0.1, 0.1]], 0.01)
    with pytest.raises(ValueError, match="insufficient"):
        solve_samples(stats)
    ds, rm, field = dataset(runtime, function=lambda p: np.full(p.shape[:-1], np.nan))
    stats = stats_for(runtime, ds, rm, field, [[0, 0, 0]], 0.6)
    assert stats[0, 111] > 0.99
    with pytest.raises(ValueError, match="nonfinite"):
        solve_samples(stats)


def test_native_end_to_end_forces_and_batch_agreement():
    runtime = Runtime()
    ds, rm, field = dataset(runtime, split=True, refined=True, poison=True)
    radii, heights = np.linspace(0, 1, 7), np.linspace(-0.8, 0.8, 7)
    points = quadrature_points(radii, heights, 20)
    samples, _ = extract_samples(
        runtime, ds, rm, field, points.reshape(-1, 3), half_width=0.6, batch_size=128
    )
    model = fit_samples(
        radii, heights, samples.reshape(points.shape[:-1]), mmax=4, time=0, units=UNITS
    )
    heldout = np.array([[0, 0, 0], [0.7, -0.2, 0.3], [-0.3, 0.4, -0.5]])
    x, y, z = heldout.T
    force = -np.column_stack(
        (2 * x + 0.3 + 0.2 * y + 0.1 * z, 4 * y + 0.4 + 0.2 * x, 6 * z + 0.5 + 0.1 * x)
    )
    np.testing.assert_allclose(model.acceleration(heldout), force, atol=2e-10)
    # Batching must not change sample values or the shared axis sample.
    selected = points.reshape(-1, 3)[::71]
    one, _ = extract_samples(
        runtime, ds, rm, field, selected, half_width=0.6, batch_size=100
    )
    many, _ = extract_samples(
        runtime, ds, rm, field, selected, half_width=0.6, batch_size=3
    )
    np.testing.assert_allclose(one, many, atol=5e-13)


def test_native_spatial_convergence():
    runtime = Runtime()
    rng = np.random.default_rng(12)
    points = rng.uniform(-0.55, 0.55, (100, 3))
    r, z = np.linspace(0, 0.9, 13), np.linspace(-0.7, 0.7, 13)
    quadrature = quadrature_points(r, z, 20)

    def potential(p):
        return (
            -((1 + np.sum(p * p, axis=-1)) ** -0.5)
            + 0.1 * p[..., 0] * p[..., 1]
            + 0.08 * p[..., 2]
        )

    exact = -points * (1 + np.sum(points * points, axis=1))[:, None] ** -1.5
    exact -= np.column_stack(
        (0.1 * points[:, 1], 0.1 * points[:, 0], np.full(len(points), 0.08))
    )
    errors = []
    for n in (12, 24, 48):
        ds, rm, field = dataset(runtime, n=n, split=True, function=potential)
        samples, _ = extract_samples(
            runtime,
            ds,
            rm,
            field,
            quadrature.reshape(-1, 3),
            half_width=8 / n,
            batch_size=128,
        )
        model = fit_samples(
            r, z, samples.reshape(quadrature.shape[:-1]), mmax=4, time=0, units=UNITS
        )
        errors.append(
            float(
                np.sqrt(
                    np.mean(np.sum((model.acceleration(points) - exact) ** 2, axis=1))
                )
            )
        )
    print("native mesh convergence RMS force errors:", errors)
    assert errors[2] < errors[1] / 2 < errors[0] / 4
    assert errors[2] < 0.003


def write_plotfile(path, time, n=16):
    level = path / "Level_0"
    level.mkdir(parents=True)
    hi = n - 1
    box = f"((0,0,0) ({hi},{hi},{hi}) (0,0,0))"
    (path / "Header").write_text(
        "\n".join(
            [
                "HyperCLaw-V1.1",
                "1",
                "gpot",
                "3",
                str(time),
                "0",
                "-2 -2 -2",
                "2 2 2",
                box,
                "0",
                f"{4 / n} {4 / n} {4 / n}",
                "0",
                "0",
                f"0 1 {time}",
                "0",
                "-2 2",
                "-2 2",
                "-2 2",
                "Level_0/Cell",
                "",
            ]
        )
    )
    (level / "Cell_H").write_text(
        f"1\n1\n1\n0\n(1 0\n{box}\n)\n1\nFabOnDisk: Cell_D_00000 0\n1,1\n0,\n1,1\n0,\n"
    )
    header = (
        "FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))" + box + " 1\n"
    ).encode()
    coords = -2 + (np.arange(n) + 0.5) * 4 / n
    points = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    (level / "Cell_D_00000").write_bytes(
        header + quadratic(points).astype("float64").tobytes(order="F")
    )


def test_cli_actual_header_times_manifest_and_portable_outputs(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    from potential_replay import PotentialModel

    first, second = tmp_path / "plt99999", tmp_path / "plt00001"
    write_plotfile(first, 1.3)
    write_plotfile(second, 7.8, n=32)
    manifest = tmp_path / "inputs.csv"
    manifest.write_text(
        f"plotfile,phase,time_myr\n{second},Phase 2,999\n{first},Phase 1,999\n"
    )
    output = tmp_path / "models"
    script = Path(__file__).resolve().parents[1] / "scripts" / "plotfile_potential.py"
    command = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output),
        "--center",
        "0",
        "0",
        "0",
        "--rmax",
        "1",
        "--zmin=-.8",
        "--zmax=.8",
        "--nr",
        "5",
        "--nz",
        "5",
        "--nphi",
        "20",
        "--mmax",
        "4",
        "--support",
        ".6",
        "--length-to-cm",
        "1",
        "--time-to-s",
        "2",
        "--potential-to-cm2-s2",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    series = json.loads((output / "timeseries.json").read_text())
    assert [r["time"] for r in series["snapshots"]] == [1.3, 7.8]
    assert [r["phase"] for r in series["snapshots"]] == ["Phase 1", "Phase 2"]
    assert [r["time_seconds"] for r in series["snapshots"]] == [2.6, 15.6]
    assert series["snapshots"][0]["cell_size"] != series["snapshots"][1]["cell_size"]
    for entry in series["snapshots"]:
        model = PotentialModel.load(output / entry["model"])
        assert model.metadata["time"] == entry["time"]
        assert model.metadata["provenance"]["input_record"]["time_myr"] == "999"
        np.testing.assert_allclose(
            model.acceleration([0, 0, 0]), [-0.3, -0.4, -0.5], atol=1e-11
        )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "refusing to overwrite" in result.stderr


def test_independent_amr_reference_derivative_and_storage_order(tmp_path):
    from analysis import PlotfileReader
    from scripts.validate_potential import AMRReference

    write_plotfile(tmp_path / "plt", 3)
    ref = AMRReference(PlotfileReader(str(tmp_path / "plt")), "gpot")
    # For quadratic terms a cell-center central derivative is ambiguous for a
    # piecewise-linear interpolant; use midpoints between centers, where it is exact.
    point = np.array([0.5, -0.25, 0.75])
    _, force, level, interface = ref.evaluate(point)
    x, y, z = point
    exact = -np.array(
        [2 * x + 0.3 + 0.2 * y + 0.1 * z, 4 * y + 0.4 + 0.2 * x, 6 * z + 0.5 + 0.1 * x]
    )
    np.testing.assert_allclose(force, exact, atol=1e-13)
    assert level == 0 and not interface
