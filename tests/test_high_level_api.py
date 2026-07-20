from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import kangaroo as kr
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta
from kangaroo.dataset import ParticleSpecies


PLOTFILE = os.getenv(
    "KANGAROO_TEST_PLOTFILE",
    "/Users/benwibking/amrex_codes/amrex-visit-plugin/example_data/DiskGalaxy/plt0000020",
)


def _plotfile() -> str:
    if not os.path.isdir(PLOTFILE):
        pytest.skip("DiskGalaxy example plotfile not found")
    return PLOTFILE


def _memory_mesh_array() -> kr.Array:
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, 0.0, 0.0),
                            ref_ratio=1,
                        ),
                        boxes=[BlockBox((0, 0, 0), (1, 1, 1))],
                    )
                ],
            )
        ]
    )
    ds = kr.open_dataset(
        "memory://high-level-mesh-dtype",
        runmeta=runmeta,
    )
    field = 61001
    ds._backend.register_field("image", field)
    ds._backend.set_chunk(
        field=field,
        block=0,
        data=np.arange(8, dtype=np.float64).reshape(2, 2, 2),
    )
    return kr.Array._from_handle(
        ds,
        ds._pipeline.field(field),
        name="image",
        shape=(2, 2, 2),
    )


def _memory_amr_array() -> kr.Array:
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, 0.0, 0.0),
                            ref_ratio=1,
                        ),
                        boxes=[
                            BlockBox((0, 0, 0), (1, 1, 1)),
                            BlockBox((2, 0, 0), (3, 1, 1)),
                        ],
                    )
                ],
            )
        ]
    )
    ds = kr.open_dataset("memory://high-level-amr", runmeta=runmeta)
    ds.geometry = SimpleNamespace(
        plane=lambda **kwargs: SimpleNamespace(
            coord=0.5 if kwargs.get("coord") is None else kwargs["coord"],
            rect=(0.0, 0.0, 4.0, 2.0),
            resolution=kwargs["resolution"],
            axis_bounds=(0.0, 2.0),
            axis_index={"x": 0, "y": 1, "z": 2}.get(
                kwargs["axis"], kwargs["axis"]
            ),
        )
    )
    field = 61002
    ds._backend.register_field("image", field)
    ds._backend.set_chunk(
        field=field,
        block=0,
        data=np.arange(8, dtype=np.float64).reshape(2, 2, 2),
    )
    ds._backend.set_chunk(
        field=field,
        block=1,
        data=np.arange(8, 16, dtype=np.float64).reshape(2, 2, 2),
    )
    return kr.Array._from_handle(ds, ds._pipeline.field(field), name="image")


def _memory_amr_array_at_level_one() -> kr.Array:
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, 0.0, 0.0),
                            ref_ratio=2,
                        ),
                        boxes=[BlockBox((0, 0, 0), (1, 1, 1))],
                    ),
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(0.5, 0.5, 0.5),
                            x0=(0.0, 0.0, 0.0),
                            ref_ratio=1,
                        ),
                        boxes=[BlockBox((0, 0, 0), (3, 3, 3))],
                    ),
                ],
            )
        ]
    )
    ds = kr.open_dataset(
        "memory://high-level-amr-level-one",
        runmeta=runmeta,
        level=1,
    )
    ds.geometry = SimpleNamespace(
        plane=lambda **kwargs: SimpleNamespace(
            coord=0.5,
            rect=(0.0, 0.0, 2.0, 2.0),
            resolution=kwargs["resolution"],
            axis_bounds=(0.0, 2.0),
            axis_index={"x": 0, "y": 1, "z": 2}.get(
                kwargs["axis"], kwargs["axis"]
            ),
        )
    )
    field = 61003
    ds._backend.register_field("image", field)
    for level, values in (
        (0, np.arange(8, dtype=np.float64).reshape(2, 2, 2)),
        (1, np.arange(64, dtype=np.float64).reshape(4, 4, 4)),
    ):
        array = np.ascontiguousarray(values)
        ds._backend._h.set_chunk_ref(
            0,
            level,
            field,
            0,
            0,
            array.tobytes(order="C"),
            "f64",
            list(array.shape),
        )
    return kr.Array._from_handle(ds, ds._pipeline.field(field), name="image")


class _ParticleLineagePipeline:
    def __init__(self) -> None:
        self._next_field = 1

    def _handle(self, *, mask: bool = False) -> SimpleNamespace:
        handle = SimpleNamespace(
            field=self._next_field,
            chunk_count=1,
            dtype="mask_u8" if mask else "float64",
        )
        self._next_field += 1
        return handle

    def particle_compare_scalar(self, *_args, **_kwargs) -> SimpleNamespace:
        return self._handle(mask=True)

    def particle_filter(self, *_args, **_kwargs) -> SimpleNamespace:
        return self._handle()

    def particle_binary(self, *_args, **_kwargs) -> SimpleNamespace:
        return self._handle()

    def particle_and(self, *_args, **_kwargs) -> SimpleNamespace:
        return self._handle(mask=True)

    def particle_histogram1d_lazy(self, *_args, **_kwargs) -> SimpleNamespace:
        return self._handle()


def _particle_lineage_arrays() -> tuple[kr.ParticleArray, kr.ParticleArray]:
    pipeline = _ParticleLineagePipeline()
    dataset = SimpleNamespace(_pipeline=pipeline)
    left = kr.ParticleArray._from_handle(
        dataset, pipeline._handle(), name="stars/x", dtype="float64", species="stars"
    )
    right = kr.ParticleArray._from_handle(
        dataset, pipeline._handle(), name="stars/y", dtype="float64", species="stars"
    )
    return left, right


def test_four_line_slice_workflow_returns_numpy_array() -> None:
    ds = kr.open_dataset(_plotfile())
    image = ds["gasDensity"].slice(axis="z", resolution=(8, 8))
    result = image.compute()

    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 8)
    assert image.is_materialized


@pytest.mark.parametrize("operation", ["slice", "project"])
def test_non_square_bounded_mesh_shape_uses_descriptor_order(operation: str) -> None:
    image = _memory_amr_array()

    bounded = getattr(image, operation)(axis="z", resolution=(6, 4))

    assert bounded.shape == (4, 6)
    assert bounded.compute().shape == (4, 6)


@pytest.mark.parametrize("bounded_operation", ["slice", "project"])
@pytest.mark.parametrize("followup_operation", ["slice", "project"])
def test_amr_only_operations_reject_bounded_arrays_before_plan_lowering(
    bounded_operation: str,
    followup_operation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _memory_amr_array()
    bounded = getattr(image, bounded_operation)(axis="z", resolution=(4, 4))

    def unexpected_lowering(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("bounded operation reached AMR plan lowering")

    monkeypatch.setattr(
        image.dataset._pipeline, "uniform_slice", unexpected_lowering
    )
    monkeypatch.setattr(
        image.dataset._pipeline, "uniform_projection", unexpected_lowering
    )

    with pytest.raises(ValueError, match="only defined for unbounded AMR arrays"):
        getattr(bounded, followup_operation)(axis="z", resolution=(4, 4))


@pytest.mark.parametrize("bounded_operation", ["slice", "project"])
@pytest.mark.parametrize(
    ("operation", "pipeline_method"),
    [
        ("vorticity", "vorticity_mag"),
        ("flux_surface_integral", "flux_surface_integral"),
        ("cylindrical_flux_surface_integral", "cylindrical_flux_surface_integral"),
        ("toomre_q_profile", "toomre_q_profile"),
    ],
)
def test_amr_only_reductions_reject_bounded_arrays_before_plan_lowering(
    bounded_operation: str,
    operation: str,
    pipeline_method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _memory_amr_array()
    bounded = getattr(image, bounded_operation)(axis="z", resolution=(4, 4))

    def unexpected_lowering(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("bounded operation reached AMR plan lowering")

    monkeypatch.setattr(image.dataset._pipeline, pipeline_method, unexpected_lowering)

    with pytest.raises(ValueError, match="only defined for unbounded AMR arrays"):
        if operation == "vorticity":
            bounded.vorticity(bounded, bounded)
        elif operation == "flux_surface_integral":
            bounded.flux_surface_integral(
                momentum=(bounded, bounded, bounded),
                energy=bounded,
                passive_scalar=bounded,
                magnetic_field=(bounded, bounded, bounded),
                radius=1.0,
            )
        elif operation == "cylindrical_flux_surface_integral":
            bounded.cylindrical_flux_surface_integral(
                momentum=(bounded, bounded, bounded),
                energy=bounded,
                passive_scalar=bounded,
                magnetic_field=(bounded, bounded, bounded),
                radius=1.0,
                height=1.0,
            )
        else:
            bounded.toomre_q_profile(
                momentum=(bounded, bounded),
                internal_energy=bounded,
                magnetic_field=(bounded, bounded, bounded),
                potential=bounded,
                z_bounds=(-1.0, 1.0),
                radial_range=(0.0, 1.0),
                bins=1,
            )


def test_non_square_particle_projection_shape_uses_descriptor_order() -> None:
    image = _memory_amr_array()

    projected = ParticleSpecies(image.dataset, "stars").project(
        axis="z", resolution=(6, 4)
    )

    assert projected.shape == (4, 6)


def test_unbounded_amr_gather_is_rejected_before_execution(monkeypatch) -> None:
    image = _memory_amr_array()
    executed = False

    def run_for(**_kwargs) -> None:
        nonlocal executed
        executed = True

    monkeypatch.setattr(image.dataset._pipeline, "run_for", run_for)

    with pytest.raises(ValueError, match="not defined for an unbounded AMR"):
        image.compute(gather=True)

    assert not executed
    assert not image.is_materialized


def test_unbounded_amr_result_retains_geometry_and_refuses_gather() -> None:
    image = _memory_amr_array()

    result = image.compute()

    assert isinstance(result, kr.AMRChunkedArray)
    assert len(result) == 2
    assert result.nbytes == 16 * np.dtype(np.float64).itemsize
    first = result.chunks[0]
    assert first.step == 0
    assert first.level == 0
    assert first.block == 0
    assert first.box == BlockBox((0, 0, 0), (1, 1, 1))
    assert first.geometry == image.dataset._pipeline.runmeta.steps[0].levels[0].geom
    np.testing.assert_array_equal(
        first.values, np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    )
    with pytest.raises(ValueError, match="not defined for an AMR hierarchy"):
        result.gather()


def test_unbounded_amr_iter_chunks_still_yields_arrays() -> None:
    image = _memory_amr_array()

    chunks = tuple(image.iter_chunks())

    assert len(chunks) == 2
    assert all(isinstance(chunk, np.ndarray) for chunk in chunks)


@pytest.mark.parametrize("operation", ["slice", "project"])
def test_bounded_array_compute_uses_selected_level(operation: str) -> None:
    image = _memory_amr_array_at_level_one()

    result = getattr(image, operation)(axis="z", resolution=(4, 4)).compute()

    assert result.shape == (4, 4)


@pytest.mark.parametrize("operation", ["arithmetic", "histogram"])
def test_bounded_followup_uses_selected_level(operation: str) -> None:
    image = _memory_amr_array_at_level_one()
    bounded = image.slice(axis="z", resolution=(4, 4))

    if operation == "arithmetic":
        result = (bounded * 2.0).compute()
        assert result.shape == (4, 4)
    else:
        result = bounded.histogram(bins=4, range=(0.0, 64.0)).compute()
        assert result.counts.sum() == 16.0


@pytest.mark.parametrize(
    "result_type",
    ("spherical-flux", "cylindrical-flux", "toomre-q"),
)
def test_reduced_result_materialization_uses_selected_level(result_type: str) -> None:
    class _SelectedLevelRuntime:
        def get_task_chunk_array(self, **kwargs):
            assert kwargs["level"] == 1
            return np.ones(1, dtype=np.float64)

    dataset = SimpleNamespace(
        step=0,
        level=1,
        client=SimpleNamespace(runtime=_SelectedLevelRuntime(), progress=False),
        _backend=object(),
        _pipeline=SimpleNamespace(run_for=lambda **_kwargs: None),
    )
    if result_type == "spherical-flux":
        value = kr.FluxSurfaceIntegral(
            dataset,
            SimpleNamespace(
                name="flux",
                field=62001,
                radii=(1.0,),
                components=("mass_flux",),
                temperature_bins=None,
            ),
        )
    elif result_type == "cylindrical-flux":
        value = kr.CylindricalFluxSurfaceIntegral(
            dataset,
            SimpleNamespace(
                name="cylindrical_flux",
                field=62002,
                radius=1.0,
                heights=(1.0,),
                geometric_sections=("walls",),
                components=("mass_flux",),
                temperature_bins=None,
            ),
        )
    else:
        value = kr.ToomreQProfile(
            dataset,
            SimpleNamespace(
                name="toomre_q",
                field=62003,
                edges=np.array([0.0, 1.0]),
                components=("mass",),
                z_bounds=(-1.0, 1.0),
                center=(0.0, 0.0, 0.0),
                gamma=5.0 / 3.0,
            ),
        )

    value.compute()


def test_arithmetic_after_bounded_slice_preserves_domain_and_shape() -> None:
    ds = kr.open_dataset(_plotfile())
    sliced = ds["gasDensity"].slice(axis="z", resolution=(8, 8))

    result = (sliced * 2.0).compute()

    assert isinstance(result, np.ndarray)
    assert result.shape == (8, 8)
    np.testing.assert_allclose(result, sliced.compute() * 2.0)


def test_matching_physical_domains_can_be_combined() -> None:
    image = _memory_amr_array()
    left = image.slice(axis="z", coord=0.5, resolution=(4, 4))
    right = image.slice(axis=2, coord=0.5, resolution=(4, 4))

    result = (left + right).compute()

    np.testing.assert_allclose(result, left.compute() + right.compute())


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            lambda image: image.slice(axis="z", coord=0.25, resolution=(4, 4)),
            lambda image: image.slice(axis="z", coord=0.75, resolution=(4, 4)),
        ),
        (
            lambda image: image.slice(axis="x", coord=0.5, resolution=(4, 4)),
            lambda image: image.slice(axis="z", coord=0.5, resolution=(4, 4)),
        ),
        (
            lambda image: image.slice(
                axis="z", coord=0.5, resolution=(4, 4), rect=(0.0, 0.0, 2.0, 2.0)
            ),
            lambda image: image.slice(
                axis="z", coord=0.5, resolution=(4, 4), rect=(0.0, 0.0, 4.0, 2.0)
            ),
        ),
        (
            lambda image: image.project(
                axis="z", bounds=(0.0, 1.0), resolution=(4, 4)
            ),
            lambda image: image.project(
                axis="z", bounds=(0.0, 2.0), resolution=(4, 4)
            ),
        ),
        (
            lambda image: image.slice(axis="z", coord=0.5, resolution=(4, 4)),
            lambda image: image.project(
                axis="z", bounds=(0.0, 2.0), resolution=(4, 4)
            ),
        ),
    ],
)
def test_matching_shapes_with_different_physical_domains_are_rejected(
    left: Any, right: Any
) -> None:
    image = _memory_amr_array()

    with pytest.raises(ValueError, match="different domains or shapes"):
        left(image) + right(image)


def test_mesh_float32_arithmetic_preserves_runtime_dtype() -> None:
    image = _memory_mesh_array()

    result = (image.astype("float32") * 2).compute()

    assert result.dtype == np.float32


def test_float32_slice_preserves_runtime_dtype() -> None:
    image = _memory_amr_array()

    sliced = image.astype("float32").slice(axis="z", resolution=(4, 4))
    result = sliced.compute()

    assert sliced.dtype == "float32"
    assert result.dtype == np.float32


@pytest.mark.parametrize("operation", ["slice", "project"])
def test_histogram_of_bounded_array_uses_bounded_domain(operation: str) -> None:
    image = _memory_amr_array()
    bounded = (
        image.slice(axis="z", resolution=(4, 4))
        if operation == "slice"
        else image.project(axis="z", resolution=(4, 4))
    )

    result = bounded.histogram(bins=4, range=(0.0, 32.0)).compute()

    assert result.counts.sum() == 16.0


@pytest.mark.parametrize("symbol", ["eq", "ne"])
def test_mesh_equality_builds_lazy_expression(symbol: str) -> None:
    image = _memory_mesh_array()

    comparison = image == 0 if symbol == "eq" else image != 0

    assert isinstance(comparison, kr.Array)
    assert isinstance(comparison, kr.MeshMask)
    assert comparison.dtype == "bool"
    assert not comparison.is_materialized
    expected = np.equal(np.arange(8).reshape(2, 2, 2), 0)
    if symbol == "ne":
        expected = np.logical_not(expected)
    result = comparison.compute()
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)


def test_mesh_comparison_materializes_a_boolean_index_mask() -> None:
    image = _memory_mesh_array()

    mask = (image >= 2) & (image < 6)
    result = mask.compute()

    assert isinstance(mask, kr.MeshMask)
    assert mask.dtype == "bool"
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(
        image.compute()[result], np.array([2.0, 3.0, 4.0, 5.0])
    )


def test_amr_mesh_comparison_materializes_boolean_chunks() -> None:
    image = _memory_amr_array()

    result = (image >= 8).compute()

    assert isinstance(result, kr.AMRChunkedArray)
    assert all(chunk.values.dtype == np.bool_ for chunk in result.chunks)
    np.testing.assert_array_equal(result.chunks[0].values, False)
    np.testing.assert_array_equal(result.chunks[1].values, True)


def test_mesh_mask_and_rejects_different_physical_domains() -> None:
    image = _memory_amr_array()
    left = image.slice(axis="z", coord=0.25, resolution=(4, 4)) > 0
    right = image.slice(axis="z", coord=0.75, resolution=(4, 4)) > 0

    with pytest.raises(ValueError, match="different domains or shapes"):
        left & right


def test_bitwise_and_rejects_numeric_mesh_arrays() -> None:
    image = _memory_mesh_array()

    with pytest.raises(TypeError, match="only supported between boolean mesh masks"):
        image & image


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


@pytest.mark.parametrize("operation", ["arithmetic", "filter", "mask", "weights"])
def test_differently_filtered_particle_position_operations_fail(operation: str) -> None:
    left, right = _particle_lineage_arrays()
    left_filtered = left[left > 0.0]
    right_filtered = right[right < 1.0]

    with pytest.raises(ValueError, match="different filtered particle domains"):
        if operation == "arithmetic":
            _ = left_filtered + right_filtered
        elif operation == "filter":
            _ = left_filtered[right_filtered > 0.0]
        elif operation == "mask":
            _ = (left_filtered > 0.0) & (right_filtered > 0.0)
        else:
            _ = left_filtered.histogram(
                bins=4, range=(0.0, 1.0), weights=right_filtered
            )


def test_particle_fields_filtered_by_the_same_mask_remain_aligned() -> None:
    left, right = _particle_lineage_arrays()
    common_mask = left > 0.0

    combined = left[common_mask] + right[common_mask]
    weighted = left[common_mask].histogram(
        bins=4, range=(0.0, 1.0), weights=right[common_mask]
    )

    assert isinstance(combined, kr.ParticleArray)
    assert isinstance(weighted, kr.Histogram)


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
