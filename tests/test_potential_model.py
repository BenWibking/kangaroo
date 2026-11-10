from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from potential_replay import PotentialModel
from potential_replay.fit import fit_samples, quadrature_points, solve_samples

UNITS = {"length_to_cm": 1, "time_to_s": 1, "potential_to_cm2_s2": 1}


def analytic(points):
    x, y, z = np.moveaxis(points, -1, 0)
    w = x + 1j * y
    value = (
        -((1 + x * x + y * y + z * z) ** -0.5)
        + 0.08 * w.real
        + 0.12 * w.imag
        + 0.07 * z
    )
    gradient = (
        np.stack((x, y, z), axis=-1) * (1 + x * x + y * y + z * z)[..., None] ** -1.5
    )
    gradient += [0.08, 0.12, 0.07]
    for m in range(2, 5):
        value += 0.02 * (w**m).real + 0.03 * (w**m).imag
        d = m * w ** (m - 1)
        gradient[..., 0] += 0.02 * d.real + 0.03 * d.imag
        gradient[..., 1] += -0.02 * d.imag + 0.03 * d.real
    return value, -gradient


def make_model(n=17):
    r, z = np.linspace(0, 1, n), np.linspace(-1, 1, n)
    points = quadrature_points(r, z, 32)
    return fit_samples(r, z, analytic(points)[0], mmax=4, time=23.7, units=UNITS)


def test_force_and_potential_convergence():
    rng = np.random.default_rng(31)
    points = rng.uniform(-0.7, 0.7, (500, 3))
    exact, force = analytic(points)
    errors = []
    for n in (9, 17, 33):
        model = make_model(n)
        phi, acc = model.evaluate(points, restore_offset=True)
        errors.append(
            (np.max(np.abs(phi - exact)), np.max(np.linalg.norm(acc - force, axis=1)))
        )
    print(
        "spline convergence (n, max potential error, max force error):",
        list(zip((9, 17, 33), errors)),
    )
    assert errors[2][0] < errors[1][0] / 8 < errors[0][0] / 64
    assert errors[2][1] < errors[1][1] / 5 < errors[0][1] / 25
    assert errors[2][1] < 2e-5


def test_modes_signed_height_axis_and_rotated_frame():
    r, z = np.linspace(0, 2, 9), np.linspace(-1, 2, 8)
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    center = np.array([5, 6, 7])
    points = quadrature_points(r, z, 32)
    x, y, zz = np.moveaxis(points, -1, 0)
    w = x + 1j * y
    # Both parities, both phases, all retained modes, and a nonzero axis force.
    values = 3 + zz + zz**3 + x * x + y * y
    for m in range(1, 5):
        values += 0.01 * m * (w**m).real + 0.02 * m * (w**m).imag
    model = fit_samples(
        r, z, values, mmax=4, center=center, rotation=rotation, time=0, units=UNITS
    )
    local = np.array(
        [[0, 0, 0.2], [1e-9, -2e-9, 0.4], [0.3, 0.6, -0.7], [0.3, 0.6, 0.7]]
    )
    world = local @ rotation.T + center
    phi, acc = model.evaluate(world, restore_offset=True)
    xx, yy, zz = local.T
    exact = 3 + zz + zz**3 + xx * xx + yy * yy
    grad = np.column_stack((2 * xx, 2 * yy, 1 + 3 * zz**2))
    for m in range(1, 5):
        w = xx + 1j * yy
        exact += 0.01 * m * (w**m).real + 0.02 * m * (w**m).imag
        d = m * w ** (m - 1)
        grad[:, 0] += 0.01 * m * d.real + 0.02 * m * d.imag
        grad[:, 1] += -0.01 * m * d.imag + 0.02 * m * d.real
    np.testing.assert_allclose(phi, exact, atol=3e-12)
    np.testing.assert_allclose(acc, -grad @ rotation.T, atol=5e-11)
    np.testing.assert_allclose(
        acc[0], np.array([-0.01, -0.02, -1.12]) @ rotation.T, atol=5e-11
    )


def test_analytic_derivative_matches_finite_difference_and_spline_joins():
    model = make_model()
    points = np.array([[0.37, 0.2, 0.21], [0, 0, 0.3], [0.5, 0, 0.25], [0.75, 0, -0.5]])
    epsilon = 1e-6
    numerical = np.column_stack(
        [
            -(
                model.potential(points + np.eye(3)[i] * epsilon)
                - model.potential(points - np.eye(3)[i] * epsilon)
            )
            / (2 * epsilon)
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(model.acceleration(points), numerical, atol=1e-9)


def test_gauge_and_serialization_roundtrip(tmp_path):
    model = make_model()
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = PotentialModel.load(path)
    points = np.array([[0.1, -0.4, 0.2], [0, 0, -1]])
    for a, b in zip(model.evaluate(points), loaded.evaluate(points)):
        np.testing.assert_allclose(a, b, atol=1e-14, rtol=1e-14)
    assert loaded.metadata == model.metadata
    assert abs(loaded.potential(points)[1]) < 1e-14
    with pytest.raises(FileExistsError):
        model.save(path)
    r, z = np.linspace(0, 1, 17), np.linspace(-1, 1, 17)
    shifted = fit_samples(
        r, z, analytic(quadrature_points(r, z, 32))[0] + 100, time=23.7, units=UNITS
    )
    np.testing.assert_allclose(
        shifted.acceleration(points), model.acceleration(points), atol=1e-10
    )


def test_standalone_reader_has_no_runtime_dependencies(tmp_path):
    model = make_model()
    model.save(tmp_path / "model.npz")
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "potential_replay",
        tmp_path / "potential_replay",
    )
    script = """
import sys
class NoRuntime:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'analysis', 'kangaroo', 'hpx', 'amrex'}:
            raise AssertionError('forbidden runtime import: '+fullname)
sys.meta_path.insert(0, NoRuntime())
from potential_replay import PotentialModel
m = PotentialModel.load('model.npz')
assert m.acceleration([0,0,0]).shape == (3,)
assert 'analysis._core' not in sys.modules
"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, check=True)


def test_outside_nonfinite_and_invalid_schema():
    model = make_model()
    for point in ([1.01, 0, 0], [0, 0, 1.01], [0, 0, -1.01], [np.nan, 0, 0]):
        with pytest.raises(ValueError):
            model.evaluate(point)
    model.evaluate([[1, 0, 1], [0, 0, -1]])  # closed domain
    metadata = dict(model.metadata, version=999)
    with pytest.raises(ValueError, match="version"):
        PotentialModel(model.s_knots, model.z_knots, model.coefficients, metadata)
    with pytest.raises(ValueError, match="orientation"):
        quadrature_points(
            np.linspace(0, 1, 5),
            np.linspace(-1, 1, 4),
            20,
            rotation=np.diag([1, 1, -1]),
        )
    with pytest.raises(ValueError, match="nphi"):
        fit_samples(
            np.linspace(0, 1, 5),
            np.linspace(-1, 1, 4),
            np.zeros((5, 4, 8)),
            time=0,
            units=UNITS,
        )


def test_invalid_statistics():
    stats = np.zeros((1, 114))
    for index in (110, 111, 112):
        invalid = stats.copy()
        invalid[0, 110] = 1
        invalid[0, 112] = 20
        invalid[0, index] = {110: 0.9, 111: 0.001, 112: 9}[index]
        with pytest.raises(ValueError, match="samples"):
            solve_samples(invalid)
    stats[0, 110], stats[0, 112] = 1, 20
    stats[0, :100] = np.ones((10, 10)).ravel()
    with pytest.raises(ValueError, match="ill-conditioned"):
        solve_samples(stats)


def test_explicit_manifest_paths_and_input_times_are_not_authoritative(tmp_path):
    from scripts.plotfile_potential import read_inputs

    path = tmp_path / "plt999"
    path.mkdir()
    (path / "Header").write_text("synthetic")
    manifest = tmp_path / "inputs.csv"
    manifest.write_text("plotfile,phase,time_myr\nplt999,Phase 2,9000\n")
    [(actual, provenance)] = read_inputs([], manifest)
    assert actual == path
    assert provenance["phase"] == "Phase 2"
    assert provenance["time_myr"] == "9000"
    with pytest.raises(ValueError):
        read_inputs([str(path)], manifest)
    with pytest.raises(FileNotFoundError):
        read_inputs([str(tmp_path / "missing")], None)
