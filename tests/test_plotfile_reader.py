from __future__ import annotations

import math
import os
import struct
import subprocess
import sys

import pytest


PLOTFILE = "/Users/benwibking/amrex_codes/amrex-visit-plugin/example_data/DiskGalaxy/plt0000020"


def test_plotfile_reader_diskgalaxy() -> None:
    from analysis import PlotfileReader

    if not os.path.isdir(PLOTFILE):
        pytest.skip("DiskGalaxy example plotfile not found")

    reader = PlotfileReader(PLOTFILE)
    hdr = reader.header()

    assert hdr["ncomp"] == 17
    assert hdr["spacedim"] == 3
    assert hdr["finest_level"] == 5
    assert hdr["var_names"][0] == "gasDensity"

    assert reader.num_levels() == 6
    assert reader.num_fabs(0) == 8

    fab = reader.read_fab(0, 0, 0, 1)
    shape = fab["shape"]
    assert shape == (1, 64, 64, 64)

    dtype = fab["dtype"]
    data = fab["data"]
    bytes_per = 4 if dtype == "float32" else 8
    expected_bytes = 1 * 64 * 64 * 64 * bytes_per
    assert len(data) == expected_bytes

    fmt = "=f" if dtype == "float32" else "=d"
    first = struct.unpack(fmt, data[:bytes_per])[0]
    assert math.isfinite(first)

    script = f"""
import numpy as np
from analysis import PlotfileReader

reader = PlotfileReader(r\"{PLOTFILE}\")
fab_nd = reader.read_fab(0, 0, 0, 1, return_ndarray=True)
arr = fab_nd[\"data\"]
assert isinstance(arr, np.ndarray)
assert arr.shape == (1, 64, 64, 64)
expected = np.float32 if fab_nd[\"dtype\"] == \"float32\" else np.float64
assert arr.dtype == expected
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"ndarray subprocess failed: {result.returncode}\n{result.stdout}\n{result.stderr}"
        )


def test_plotfile_reader_particles_diskgalaxy() -> None:
    from analysis import PlotfileReader

    if not os.path.isdir(PLOTFILE):
        pytest.skip("DiskGalaxy example plotfile not found")

    reader = PlotfileReader(PLOTFILE)
    particle_types = reader.particle_types()
    assert "CIC_particles" in particle_types
    assert "StochasticStellarPop_particles" in particle_types

    cic_fields = reader.particle_fields("CIC_particles")
    assert "x" in cic_fields
    assert "mass" in cic_fields
    assert "vx" in cic_fields

    stars_fields = reader.particle_fields("StochasticStellarPop_particles")
    assert "evolution_stage" in stars_fields
    assert "birth_time" in stars_fields

    if not hasattr(reader._reader, "particle_chunk_count") or not hasattr(
        reader._reader, "read_particle_field_chunk"
    ):
        pytest.skip("plotfile extension missing particle chunk APIs; rebuild extensions")

    nchunks = reader.particle_chunk_count("CIC_particles")
    assert nchunks > 0

    mass_chunk = reader.read_particle_field_chunk(
        "CIC_particles", "mass", 0, return_ndarray=True
    )
    assert mass_chunk["dtype"] in {"float32", "float64"}
    assert int(mass_chunk["count"]) == mass_chunk["data"].shape[0]
    assert mass_chunk["data"].shape[0] > 0

    stage_chunk = reader.read_particle_field_chunk(
        "StochasticStellarPop_particles", "evolution_stage", 0, return_ndarray=True
    )
    assert stage_chunk["dtype"] == "int64"
    assert int(stage_chunk["count"]) == stage_chunk["data"].shape[0]
