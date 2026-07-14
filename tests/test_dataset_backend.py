from __future__ import annotations

import os

import pytest


PLOTFILE = "/Users/benwibking/amrex_codes/amrex-visit-plugin/example_data/DiskGalaxy/plt0000020"


def test_memory_dataset_uses_backend_capabilities() -> None:
    from analysis import Runtime, _core
    from analysis.dataset import open_dataset

    runtime = Runtime()
    dataset = open_dataset("memory://backend-capabilities", runtime=runtime)

    assert dataset.kind == "memory"
    assert dataset.metadata == {}
    assert dataset.list_meshes() == []
    assert dataset.list_particle_types() == []

    dataset.register_field("density", 101)
    assert dataset.field_id("density") == 101

    with pytest.raises(RuntimeError, match="does not support mesh selection"):
        dataset.select_mesh("mesh")
    with pytest.raises(RuntimeError, match="unsupported dataset URI scheme"):
        _core.DatasetHandle("unknown://dataset", 0, 0)


def test_plotfile_dataset_metadata_and_particles_cross_backend_seam() -> None:
    if not os.path.isdir(PLOTFILE):
        pytest.skip("DiskGalaxy example plotfile not found")

    from analysis import Runtime
    from analysis.dataset import open_dataset

    runtime = Runtime()
    dataset = open_dataset(f"amrex://{PLOTFILE}", runtime=runtime)

    assert dataset.kind == "amrex"
    assert dataset.metadata["var_names"][0] == "gasDensity"
    assert dataset.field_id("gasDensity") >= 0
    assert "CIC_particles" in dataset.list_particle_types()
    assert "mass" in dataset.list_particle_fields("CIC_particles")
    assert dataset.get_particle_chunk_count("CIC_particles") > 0
