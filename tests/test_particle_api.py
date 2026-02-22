from __future__ import annotations

import os

import numpy as np
import pytest

from analysis.dataset import open_dataset
from analysis.pipeline import pipeline
from analysis.runtime import Runtime


PLOTFILE = "/Users/benwibking/amrex_codes/amrex-visit-plugin/example_data/DiskGalaxy/plt0000020"


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
