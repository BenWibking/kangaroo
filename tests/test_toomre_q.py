from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np

from analysis import Runtime
from analysis.dataset import open_dataset
from analysis.pipeline import Pipeline
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta
from scripts.plotfile_toomre_q import (
    NUM_MOMENTS,
    _find_plotfiles,
    _pick_field,
    _radial_edges,
    derive_toomre_profiles,
    plot_toomre_profiles,
    profile_rows,
    write_profile_csv,
)


def _analytic_moments(
    edges: np.ndarray,
    *,
    surface_density: float,
    sound_speed: float,
    radial_mean: float,
    radial_dispersion: float,
    alfven_speed: float,
    omega: float,
    gamma: float,
) -> np.ndarray:
    radius = 0.5 * (edges[:-1] + edges[1:])
    area = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    mass = surface_density * area
    moments = np.zeros((len(radius), NUM_MOMENTS), dtype=np.float64)
    moments[:, 0] = mass
    moments[:, 1] = mass * sound_speed**2 / (gamma * (gamma - 1.0))
    moments[:, 2] = mass * alfven_speed**2
    moments[:, 3] = mass * radial_mean
    moments[:, 4] = mass * (radial_mean**2 + radial_dispersion**2)
    moments[:, 5] = mass * omega**2 * radius
    moments[:, 6] = area
    return moments


def test_derive_toomre_profiles_solid_body_three_support_models() -> None:
    gamma = 5.0 / 3.0
    edges = np.linspace(1.0, 5.0, 9)
    sigma = 10.0
    cs = 2.0
    turbulent = 3.0
    alfven = 4.0
    omega = 5.0
    radial_mean = 7.0
    moments = _analytic_moments(
        edges,
        surface_density=sigma,
        sound_speed=cs,
        radial_mean=radial_mean,
        radial_dispersion=turbulent,
        alfven_speed=alfven,
        omega=omega,
        gamma=gamma,
    )

    result = derive_toomre_profiles(
        edges,
        moments,
        gamma=gamma,
        gravitational_constant=1.0,
    )

    np.testing.assert_allclose(result["radial_velocity_mean"], radial_mean)
    np.testing.assert_allclose(result["radial_dispersion"], turbulent)
    np.testing.assert_allclose(result["sound_speed"], cs)
    np.testing.assert_allclose(result["alfven_speed"], alfven)
    np.testing.assert_allclose(result["kappa"], 2.0 * omega)
    np.testing.assert_allclose(
        result["q_thermal_magnetic"],
        2.0 * omega * math.sqrt(cs**2 + alfven**2) / (math.pi * sigma),
    )
    np.testing.assert_allclose(
        result["q_thermal_turbulent"],
        2.0 * omega * math.sqrt(cs**2 + turbulent**2) / (math.pi * sigma),
    )
    np.testing.assert_allclose(
        result["q_thermal_turbulent_magnetic"],
        2.0
        * omega
        * math.sqrt(cs**2 + turbulent**2 + alfven**2)
        / (math.pi * sigma),
    )
    assert np.all(result["valid"])


def test_derive_toomre_profiles_flat_rotation_curve() -> None:
    gamma = 5.0 / 3.0
    edges = np.linspace(1.0, 11.0, 201)
    radius = 0.5 * (edges[:-1] + edges[1:])
    circular_speed = 3.0
    moments = _analytic_moments(
        edges,
        surface_density=2.0,
        sound_speed=1.0,
        radial_mean=0.0,
        radial_dispersion=0.0,
        alfven_speed=0.0,
        omega=1.0,
        gamma=gamma,
    )
    mass = moments[:, 0]
    moments[:, 5] = mass * circular_speed**2 / radius

    result = derive_toomre_profiles(
        edges,
        moments,
        gamma=gamma,
        gravitational_constant=1.0,
    )

    expected = math.sqrt(2.0) * circular_speed / radius
    np.testing.assert_allclose(result["kappa"][2:-2], expected[2:-2], rtol=2.0e-3)


def test_zero_support_terms_reduce_to_thermal_q() -> None:
    gamma = 5.0 / 3.0
    edges = np.linspace(1.0, 4.0, 7)
    moments = _analytic_moments(
        edges,
        surface_density=2.0,
        sound_speed=3.0,
        radial_mean=11.0,
        radial_dispersion=0.0,
        alfven_speed=0.0,
        omega=2.0,
        gamma=gamma,
    )
    result = derive_toomre_profiles(
        edges,
        moments,
        gamma=gamma,
        gravitational_constant=1.0,
    )
    np.testing.assert_allclose(
        result["q_thermal_magnetic"], result["q_thermal_turbulent"]
    )
    np.testing.assert_allclose(
        result["q_thermal_magnetic"], result["q_thermal_turbulent_magnetic"]
    )
    np.testing.assert_allclose(result["radial_dispersion"], 0.0, atol=2.0e-7)


def _two_level_runmeta() -> RunMeta:
    return RunMeta(
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


def test_toomre_q_profile_lowering_wires_gradient_amr_mask_and_reduction() -> None:
    runtime = Runtime()
    runmeta = _two_level_runmeta()
    ds = open_dataset("memory://toomre-lowering", runmeta=runmeta, runtime=runtime)
    pipe = Pipeline(runtime=runtime, runmeta=runmeta, dataset=ds)
    handle = pipe.toomre_q_profile(
        101,
        momentum=(102, 103),
        internal_energy=104,
        magnetic_field=(105, 106, 107),
        potential=108,
        radial_range=(0.25, 2.0),
        bins=7,
        z_bounds=(-0.75, 0.75),
        center=(0.0, 0.0, 0.0),
        bytes_per_value=8,
        out="toomre",
    )

    assert handle.components == (
        "mass",
        "internal_energy",
        "magnetic_b2_volume",
        "radial_momentum",
        "radial_velocity_second_moment",
        "radial_gravity_moment",
        "sampled_volume",
    )
    templates = [template for stage in pipe.plan().stages for template in stage.templates]
    accumulators = [
        template for template in templates if template.kernel == "toomre_profile_accumulate"
    ]
    fetch = [template for template in templates if template.kernel == "amr_subbox_fetch_pack"]
    gradients = [template for template in templates if template.kernel == "gradU_stencil"]
    assert len(accumulators) == 2
    assert len(fetch) == len(gradients) == len(accumulators)
    assert all(template.params["input_field"] == 108 for template in fetch)
    assert all(len(template.inputs) == 8 for template in accumulators)
    assert all(template.output_bytes == [7 * NUM_MOMENTS * 8] for template in accumulators)
    coarse = next(template for template in accumulators if template.domain.level == 0)
    assert [[[1, 1, 0], [2, 2, 1]]] == coarse.params["covered_boxes"]
    assert any(template.kernel == "uniform_slice_add" for template in templates)
    assert any(template.name == "toomre_q_profile_output" for template in templates)


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
    ds._h.set_chunk_ref(0, 0, field, 0, 0, np.asarray(values, dtype=np.float64).tobytes())


def test_toomre_profile_accumulator_kernel_direct() -> None:
    runtime = Runtime()
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(1.0, -0.5, -0.5),
                            index_origin=(0, 0, 0),
                            ref_ratio=1,
                        ),
                        boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    ds = open_dataset("memory://toomre-kernel", runmeta=runmeta, runtime=runtime)
    input_fields = list(range(301, 308))
    for field, value in zip(input_fields, (2.0, 0.0, 0.0, 3.0, 1.0, 2.0, 0.0)):
        _set_chunk(ds, field, np.array([[[value]]]))
    gradient_field = 308
    ds._h.set_chunk_ref(
        0,
        0,
        gradient_field,
        0,
        0,
        np.asarray([7.0, 0.0, 0.0], dtype=np.float64).tobytes(),
    )
    output_field = 309
    packed = msgpack.packb(
        {
            "stages": [
                {
                    "name": "toomre",
                    "plane": "chunk",
                    "after": [],
                    "templates": [
                        {
                            "name": "toomre_profile_accumulate",
                            "plane": "chunk",
                            "kernel": "toomre_profile_accumulate",
                            "domain": {"step": 0, "level": 0, "blocks": [0]},
                            "inputs": [
                                {"field": field, "version": 0}
                                for field in (*input_fields, gradient_field)
                            ],
                            "outputs": [{"field": output_field, "version": 0}],
                            "output_bytes": [NUM_MOMENTS * 8],
                            "deps": {"kind": "None"},
                            "params": {
                                "radial_range": [1.0, 2.0],
                                "bins": 1,
                                "z_bounds": [-0.25, 0.25],
                                "center": [0.0, 0.0, 0.0],
                                "bytes_per_value": 8,
                                "covered_boxes": [],
                            },
                        }
                    ],
                }
            ]
        },
        use_bin_type=True,
    )
    runtime._rt.run_packed_plan(packed, runmeta._h, ds._h)
    values = runtime.get_task_chunk_array(
        step=0,
        level=0,
        field=output_field,
        block=0,
        shape=(NUM_MOMENTS,),
        dtype=np.float64,
        dataset=ds,
    )
    np.testing.assert_allclose(values, [1.0, 1.5, 2.5, 0.0, 0.0, 7.0, 0.5])


def test_toomre_q_profile_runtime_accumulates_expected_moments() -> None:
    runtime = Runtime()
    runmeta = _single_level_runmeta()
    ds = open_dataset("memory://toomre-runtime", runmeta=runmeta, runtime=runtime)
    shape = (10, 10, 4)
    x = -5.0 + (np.arange(shape[0]) + 0.5)
    y = -5.0 + (np.arange(shape[1]) + 0.5)
    xx, yy, _ = np.meshgrid(x, y, np.arange(shape[2]), indexing="ij")
    rho = np.full(shape, 2.0)
    momx = np.zeros(shape)
    momy = np.zeros(shape)
    eint = np.full(shape, 3.0)
    bx = np.full(shape, 1.0)
    by = np.full(shape, 2.0)
    bz = np.zeros(shape)
    omega = 2.0
    potential = 0.5 * omega**2 * (xx * xx + yy * yy)
    fields = (201, 202, 203, 204, 205, 206, 207, 208)
    for field, values in zip(fields, (rho, momx, momy, eint, bx, by, bz, potential)):
        _set_chunk(ds, field, values)

    pipe = Pipeline(runtime=runtime, runmeta=runmeta, dataset=ds)
    handle = pipe.toomre_q_profile(
        fields[0],
        momentum=(fields[1], fields[2]),
        internal_energy=fields[3],
        magnetic_field=(fields[4], fields[5], fields[6]),
        potential=fields[7],
        radial_range=(0.5, 3.5),
        bins=3,
        z_bounds=(-1.5, 1.5),
        bytes_per_value=8,
        out="toomre",
    )
    pipe.run()
    moments = runtime.get_task_chunk_array(
        step=0,
        level=0,
        field=handle.field,
        block=0,
        shape=(3, NUM_MOMENTS),
        dtype=np.float64,
        dataset=ds,
    )

    radius = np.sqrt(xx[:, :, 0] ** 2 + yy[:, :, 0] ** 2)
    for radial_bin, (rlo, rhi) in enumerate(zip(handle.edges[:-1], handle.edges[1:])):
        selected = (radius >= rlo) & (radius < rhi if radial_bin < 2 else radius <= rhi)
        columns = int(np.count_nonzero(selected))
        volume = 3.0 * columns
        np.testing.assert_allclose(moments[radial_bin, 0], 2.0 * volume)
        np.testing.assert_allclose(moments[radial_bin, 1], 3.0 * volume)
        np.testing.assert_allclose(moments[radial_bin, 2], 5.0 * volume)
        np.testing.assert_allclose(moments[radial_bin, 3:5], 0.0, atol=1.0e-12)
        expected_gravity_moment = (
            np.sum(2.0 * volume / columns * omega**2 * radius[selected])
            if columns
            else 0.0
        )
        np.testing.assert_allclose(moments[radial_bin, 5], expected_gravity_moment)
        np.testing.assert_allclose(moments[radial_bin, 6], volume)


def test_plot_and_csv_contain_all_three_q_profiles(tmp_path: Path) -> None:
    gamma = 5.0 / 3.0
    edges = np.linspace(1.0, 5.0, 9)
    moments = _analytic_moments(
        edges,
        surface_density=10.0,
        sound_speed=2.0,
        radial_mean=0.0,
        radial_dispersion=3.0,
        alfven_speed=4.0,
        omega=5.0,
        gamma=gamma,
    )
    profile = derive_toomre_profiles(
        edges,
        moments,
        gamma=gamma,
        gravitational_constant=1.0,
    )
    rows = profile_rows(edges, profile)
    csv_path = tmp_path / "profile.csv"
    png_path = tmp_path / "profile.png"
    write_profile_csv(csv_path, rows)
    plot_toomre_profiles(
        png_path,
        profile,
        plotfile_name="plt00000",
        time_seconds=0.0,
    )

    assert png_path.stat().st_size > 0
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert set(
            (
                "q_thermal_magnetic",
                "q_thermal_turbulent",
                "q_thermal_turbulent_magnetic",
            )
        ).issubset(reader.fieldnames or [])
        assert len(list(reader)) == len(edges) - 1


def test_cli_discovers_only_immediate_main_plotfiles_and_default_fields(tmp_path: Path) -> None:
    first = tmp_path / "plt00000"
    second = tmp_path / "plt00100"
    projection = tmp_path / "proj" / "z" / "plt00100"
    for path in (first, second, projection):
        path.mkdir(parents=True)
        (path / "Header").write_text("fixture\n", encoding="utf-8")

    assert _find_plotfiles(tmp_path) == [first, second]
    available = [
        "gasDensity",
        "x-GasMomentum",
        "y-GasMomentum",
        "gasInternalEnergy",
        "x-BField",
        "y-BField",
        "z-BField",
        "gpot",
    ]
    assert _pick_field("density", None, available) == "gasDensity"
    assert _pick_field("potential", None, available) == "gpot"

    args = SimpleNamespace(r_min_kpc=0.5, r_max_kpc=16.0, bins=62, dr_kpc=None)
    edges = _radial_edges(args)
    np.testing.assert_allclose(np.diff(edges), 0.25 * 1.0e3 * 3.0856775814913673e18)
