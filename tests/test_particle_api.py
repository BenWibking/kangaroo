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

    x = ds.get_particle_array("StochasticStellarPop_particles", "x")
    stage = ds.get_particle_array("StochasticStellarPop_particles", "evolution_stage")
    mass = ds.get_particle_array("StochasticStellarPop_particles", "mass")
    nchunks = ds.get_particle_chunk_count("StochasticStellarPop_particles")

    assert isinstance(x, np.ndarray)
    assert isinstance(stage, np.ndarray)
    assert isinstance(mass, np.ndarray)
    assert x.ndim == 1
    assert x.shape == stage.shape
    assert mass.shape == stage.shape
    assert stage.dtype == np.int64
    assert x.dtype in (np.float32, np.float64)
    assert nchunks >= 1

    parts = [
        ds.get_particle_chunk_array("StochasticStellarPop_particles", "mass", i)
        for i in range(nchunks)
    ]
    mass_chunked = np.concatenate(parts) if parts else np.zeros(0, dtype=mass.dtype)
    np.testing.assert_allclose(mass_chunked, mass)


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
    assert isinstance(masses.values, np.ndarray)
    assert masses.values.ndim == 1
