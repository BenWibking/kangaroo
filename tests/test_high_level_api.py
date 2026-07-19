from __future__ import annotations

import os

import numpy as np
import pytest

import kangaroo as kr


PLOTFILE = os.getenv(
    "KANGAROO_TEST_PLOTFILE",
    "/Users/benwibking/amrex_codes/amrex-visit-plugin/example_data/DiskGalaxy/plt0000020",
)


def _plotfile() -> str:
    if not os.path.isdir(PLOTFILE):
        pytest.skip("DiskGalaxy example plotfile not found")
    return PLOTFILE


def test_four_line_slice_workflow_returns_numpy_array() -> None:
    ds = kr.open_dataset(_plotfile())
    image = ds["gasDensity"].slice(axis="z", resolution=(8, 8))
    result = image.compute()

    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 8)
    assert image.is_materialized


def test_arithmetic_after_bounded_slice_preserves_domain_and_shape() -> None:
    ds = kr.open_dataset(_plotfile())
    sliced = ds["gasDensity"].slice(axis="z", resolution=(8, 8))

    result = (sliced * 2.0).compute()

    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 8)
    np.testing.assert_allclose(result, sliced.compute() * 2.0)


def test_multi_output_compute_returns_typed_results() -> None:
    ds = kr.open_dataset(_plotfile())
    density = ds["gasDensity"]
    image = (density * 2.0).slice(axis="z", resolution=(4, 4))
    histogram = density.histogram(bins=4, range=(0.0, 1.0e-20))

    image_result, histogram_result = kr.compute(
        image, histogram, gather=True, max_bytes=1024
    )

    assert image_result.shape == (4, 4)
    assert isinstance(histogram_result, kr.HistogramResult)
    assert histogram_result.counts.shape == (4,)
    assert histogram_result.edges.shape == (5,)


def test_particle_reductions_are_lazy_and_gather_is_explicit() -> None:
    ds = kr.open_dataset(_plotfile())
    mass = ds.particles["StochasticStellarPop_particles"]["mass"]
    selected = mass[mass.isfinite() & (mass > 0.0)]
    total = selected.sum()

    assert isinstance(total, kr.Scalar)
    assert not total.is_materialized
    with pytest.raises(TypeError, match="compute"):
        float(total)
    with pytest.raises(TypeError, match="unexpected compute options: gather"):
        total.compute(gather=True)

    chunks = mass.compute()
    assert isinstance(chunks, kr.ChunkedArray)
    gathered = mass.compute(gather=True, max_bytes=chunks.nbytes)
    assert isinstance(gathered, np.ndarray)


def test_particle_operator_syntax_and_lazy_histogram() -> None:
    ds = kr.open_dataset(_plotfile())
    mass = ds.particles["StochasticStellarPop_particles"]["mass"]
    transformed = ((mass + 2.0) * 3.0 - 1.0) / 2.0
    selected = transformed[(transformed >= 0.0) & (transformed != 4.0)]
    histogram = selected.histogram(bins=4, range=(0.0, 1.0e40))

    assert not histogram.is_materialized
    result = histogram.compute()
    assert isinstance(result, kr.HistogramResult)
    assert result.counts.shape == (4,)


@pytest.mark.parametrize("operation", ["arithmetic", "filter", "mask", "weights"])
def test_cross_species_particle_position_operations_fail(operation: str) -> None:
    ds = kr.open_dataset(_plotfile())
    cic_mass = ds.particles["CIC_particles"]["mass"]
    stellar_mass = ds.particles["StochasticStellarPop_particles"]["mass"]

    with pytest.raises(ValueError, match="different particle species"):
        if operation == "arithmetic":
            _ = cic_mass + stellar_mass
        elif operation == "filter":
            _ = cic_mass[stellar_mass > 0.0]
        elif operation == "mask":
            _ = (cic_mass > 0.0) & (stellar_mass > 0.0)
        else:
            _ = cic_mass.histogram(
                bins=4, range=(0.0, 1.0e40), weights=stellar_mass
            )


def test_particle_topk_uses_backend_provenance_not_display_name() -> None:
    ds = kr.open_dataset(_plotfile())
    raw = ds.particles["StochasticStellarPop_particles"]["evolution_stage"]

    renamed_result = raw.rename("stage").topk(3).compute()
    raw_result = raw.topk(3).compute()

    np.testing.assert_array_equal(renamed_result.values, raw_result.values)
    np.testing.assert_array_equal(renamed_result.counts, raw_result.counts)

    disguised_expression = (raw + 1.0).rename("CIC_particles/mass")
    with pytest.raises(ValueError, match="backend particle field"):
        disguised_expression.topk(3)


def test_cross_dataset_expressions_fail_at_construction() -> None:
    left = kr.open_dataset(_plotfile())
    right = kr.open_dataset(_plotfile())
    with pytest.raises(ValueError, match="different dataset contexts"):
        _ = left["gasDensity"] + right["gasDensity"]


def test_diagnostics_do_not_materialize() -> None:
    ds = kr.open_dataset(_plotfile())
    value = ds["gasDensity"] * 2.0

    explanation = value.explain()
    dot = value.visualize()

    assert "field_expr" in explanation
    assert dot.startswith("digraph kangaroo")
    assert not value.is_materialized


def test_persisted_field_is_reused_by_later_graphs() -> None:
    ds = kr.open_dataset(_plotfile())
    prepared = (ds["gasDensity"] * 2.0).persist()
    assert prepared.is_materialized
    assert not ds._pipeline._field_producers.get(prepared._field_handle.field)

    histogram = prepared.histogram(bins=4, range=(0.0, 1.0e-20)).compute()
    image = prepared.slice(axis="z", resolution=(4, 4)).compute()

    assert np.sum(histogram.counts) > 0.0
    assert image.shape == (4, 4)


def test_compatibility_mesh_handle_owns_materialization() -> None:
    ds = kr.open_dataset(_plotfile())
    geometry = ds.geometry.plane(axis="z", resolution=(4, 4))
    handle = ds._pipeline.uniform_slice(
        ds._pipeline.field("gasDensity"),
        axis="z",
        coord=geometry.coord,
        rect=geometry.rect,
        resolution=geometry.resolution,
    )

    result = handle.compute()

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 4)
