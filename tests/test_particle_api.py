from __future__ import annotations

import os

import numpy as np
import pytest

from analysis.dataset import open_dataset
from analysis.pipeline import Pipeline, pipeline
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta
from analysis.runtime import Runtime


PLOTFILE = "/Users/benwibking/amrex_codes/amrex-visit-plugin/example_data/DiskGalaxy/plt0000020"


def _memory_particle_pipeline(name: str) -> Pipeline:
    rt = Runtime()
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
                        boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
                    )
                ],
            )
        ]
    )
    ds = open_dataset(f"memory://{name}", runmeta=runmeta, runtime=rt)
    return Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)


def test_particle_elementwise_ops_truncate_unequal_chunk_lengths() -> None:
    p = _memory_particle_pipeline("particle-unequal-lengths")

    difference = p.particle_subtract(
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([1.0, 2.0, 3.0]),
    )
    intersection = p.particle_and(
        np.array([True, False, True, True]),
        np.array([True, True]),
    )
    distance = p.particle_distance3(
        np.array([0.0, 3.0, 100.0, 200.0]),
        np.array([0.0, 4.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0]),
    )

    difference_chunk = difference.iter_chunks()[0]
    intersection_chunk = intersection.iter_chunks()[0]
    distance_chunk = distance.iter_chunks()[0]

    assert [difference_chunk.shape, intersection_chunk.shape, distance_chunk.shape] == [
        (3,),
        (2,),
        (2,),
    ]
    np.testing.assert_array_equal(difference_chunk, [9.0, 18.0, 27.0])
    np.testing.assert_array_equal(intersection_chunk, [True, False])
    np.testing.assert_array_equal(distance_chunk, [0.0, 5.0])


@pytest.mark.parametrize(
    ("values", "weights"),
    [
        ([0.1, 0.2, 0.3], [10.0]),
        ([0.1], [10.0, 20.0, 30.0]),
    ],
)
def test_weighted_particle_histogram_rejects_unequal_chunk_lengths(
    values: list[float], weights: list[float]
) -> None:
    p = _memory_particle_pipeline("particle-histogram-unequal-lengths")

    with pytest.raises(ValueError, match="values and weights must have matching shapes"):
        p.particle_histogram1d(
            np.asarray(values),
            bins=1,
            hist_range=(0.0, 1.0),
            weights=np.asarray(weights),
        )
    assert not p._particle_stages


def test_weighted_particle_histogram_rejects_unequal_dynamic_handle_extents() -> None:
    p = _memory_particle_pipeline("particle-histogram-dynamic-unequal-lengths")
    values = p.particle_filter(
        np.array([0.1, 0.2, 0.3]),
        np.array([True, True, True]),
    )
    weights = p.particle_filter(
        np.array([10.0, 20.0, 30.0]),
        np.array([True, False, False]),
    )

    with pytest.raises(RuntimeError, match="values and weights must have matching extents"):
        p.particle_histogram1d(
            values,
            bins=1,
            hist_range=(0.0, 1.0),
            weights=weights,
        )


def test_particle_histogram_weighted_and_unweighted_kernel_contracts() -> None:
    weighted = _memory_particle_pipeline("particle-histogram-weighted")
    weighted_counts, weighted_edges = weighted.particle_histogram1d(
        np.array([0.1, 0.2, 0.8]),
        bins=2,
        hist_range=(0.0, 1.0),
        weights=np.array([2.0, 3.0, 5.0]),
    )
    weighted_kernels = [
        tmpl.kernel for stage in weighted._particle_stages for tmpl in stage.templates
    ]

    unweighted = _memory_particle_pipeline("particle-histogram-unweighted")
    unweighted_counts, unweighted_edges = unweighted.particle_histogram1d(
        np.array([0.1, 0.2, 0.8]),
        bins=2,
        hist_range=(0.0, 1.0),
    )
    unweighted_kernels = [
        tmpl.kernel for stage in unweighted._particle_stages for tmpl in stage.templates
    ]

    np.testing.assert_array_equal(weighted_counts, [5.0, 5.0])
    np.testing.assert_array_equal(weighted_edges, [0.0, 0.5, 1.0])
    assert "particle_histogram1d_weighted" in weighted_kernels
    np.testing.assert_array_equal(unweighted_counts, [2.0, 1.0])
    np.testing.assert_array_equal(unweighted_edges, [0.0, 0.5, 1.0])
    assert "particle_histogram1d" in unweighted_kernels
    assert "particle_histogram1d_weighted" not in unweighted_kernels


def test_dataset_particle_api_diskgalaxy() -> None:
    if not os.path.isdir(PLOTFILE):
        pytest.skip("DiskGalaxy example plotfile not found")

    rt = Runtime()
    ds = open_dataset(PLOTFILE, runtime=rt, step=0, level=0)

    ptypes = ds.list_particle_types()
    assert "CIC_particles" in ptypes

    pfields = ds.list_particle_fields("StochasticStellarPop_particles")
    assert "evolution_stage" in pfields
    assert "x" in pfields

    nchunks = ds.get_particle_chunk_count("StochasticStellarPop_particles")
    assert nchunks >= 1

    with pytest.raises(RuntimeError, match="disabled"):
        ds.get_particle_array("StochasticStellarPop_particles", "mass")

    x_parts = [
        ds.get_particle_chunk_array("StochasticStellarPop_particles", "x", i)
        for i in range(nchunks)
    ]
    stage_parts = [
        ds.get_particle_chunk_array("StochasticStellarPop_particles", "evolution_stage", i)
        for i in range(nchunks)
    ]
    mass_parts = [
        ds.get_particle_chunk_array("StochasticStellarPop_particles", "mass", i)
        for i in range(nchunks)
    ]
    assert len(x_parts) == nchunks
    assert len(stage_parts) == nchunks
    assert len(mass_parts) == nchunks
    assert all(isinstance(chunk, np.ndarray) and chunk.ndim == 1 for chunk in x_parts)
    assert all(isinstance(chunk, np.ndarray) and chunk.ndim == 1 for chunk in stage_parts)
    assert all(isinstance(chunk, np.ndarray) and chunk.ndim == 1 for chunk in mass_parts)
    assert all(chunk.dtype in (np.float32, np.float64) for chunk in x_parts)
    assert all(chunk.dtype == np.int64 for chunk in stage_parts)
    assert all(chunk.dtype in (np.float32, np.float64) for chunk in mass_parts)
    assert all(
        x_parts[i].shape == stage_parts[i].shape == mass_parts[i].shape for i in range(nchunks)
    )


def test_pipeline_particle_field_uses_runtime_diskgalaxy() -> None:
    if not os.path.isdir(PLOTFILE):
        pytest.skip("DiskGalaxy example plotfile not found")

    rt = Runtime()
    ds = open_dataset(PLOTFILE, runtime=rt, step=0, level=0)
    p = pipeline(runtime=rt, runmeta=ds.get_runmeta(), dataset=ds)

    def _forbidden(_ptype: str, _field: str) -> np.ndarray:
        raise RuntimeError("get_particle_array should not be used by Pipeline.particle_field")

    ds.get_particle_array = _forbidden  # type: ignore[assignment]
    masses = p.particle_field("StochasticStellarPop_particles", "mass")
    with pytest.raises(RuntimeError, match="disabled"):
        _ = masses.values
    chunks = masses.iter_chunks()
    assert isinstance(chunks, list)
    assert chunks
    assert all(isinstance(chunk, np.ndarray) and chunk.ndim == 1 for chunk in chunks)
