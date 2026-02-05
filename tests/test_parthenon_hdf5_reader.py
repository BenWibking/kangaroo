from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _maybe_h5py():
    return pytest.importorskip("h5py")


def _write_minimal_parthenon_file(path: Path) -> None:
    h5py = _maybe_h5py()
    with h5py.File(path, "w") as f:
        info = f.create_group("Info")
        info.attrs["OutputDatasetNames"] = np.array([b"density", b"velocity"])
        info.attrs["NumComponents"] = np.array([1, 3], dtype=np.int64)
        info.attrs["ComponentNames"] = np.array([b"density", b"vel1", b"vel2", b"vel3"])
        info.attrs["NumMeshBlocks"] = np.int64(2)
        info.attrs["MeshBlockSize"] = np.array([4, 3, 2], dtype=np.int64)
        info.attrs["RootGridSize"] = np.array([8, 6, 4], dtype=np.int64)
        info.attrs["RootGridDomain"] = np.array(
            [0.0, 1.0, 1.0, -1.0, 1.0, 1.0, 10.0, 14.0, 1.0],
            dtype=np.float64,
        )
        info.attrs["Time"] = np.float64(2.5)

        f.create_dataset("Levels", data=np.array([0, 1], dtype=np.int64))
        f.create_dataset("LogicalLocations", data=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64))

        density = np.arange(2 * 2 * 3 * 4, dtype=np.float32).reshape(2, 2, 3, 4)
        velocity = np.arange(2 * 3 * 2 * 3 * 4, dtype=np.float64).reshape(2, 3, 2, 3, 4)
        f.create_dataset("density", data=density)
        f.create_dataset("velocity", data=velocity)


def test_parthenon_hdf5_reader(tmp_path: Path) -> None:
    from analysis.parthenon_hdf5 import ParthenonHDF5Reader

    p = tmp_path / "sample.phdf"
    _write_minimal_parthenon_file(p)

    reader = ParthenonHDF5Reader(str(p))
    hdr = reader.header()
    meta = reader.metadata()

    assert reader.num_levels() == 2
    assert reader.num_fabs(0) == 1
    assert reader.num_fabs(1) == 1

    assert hdr["var_names"] == ["density", "velocity"]
    assert meta["finest_level"] == 1
    assert meta["prob_lo"] == [0.0, -1.0, 10.0]
    assert meta["prob_hi"] == [1.0, 1.0, 14.0]

    b0 = reader.read_block("density", level=0, fab=0, comp_start=0, comp_count=1)
    assert b0["shape"] == (1, 2, 3, 4)
    assert b0["dtype"] == "float32"
    assert len(b0["data"]) == 1 * 2 * 3 * 4 * 4

    b1 = reader.read_block(
        "velocity",
        level=1,
        fab=0,
        comp_start=1,
        comp_count=2,
        return_ndarray=True,
    )
    arr = b1["data"]
    assert arr.shape == (2, 2, 3, 4)
    assert arr.dtype == np.float64


def test_open_dataset_parthenon_uri_loads_chunks(tmp_path: Path) -> None:
    pytest.importorskip("analysis._core")
    from analysis.dataset import open_dataset
    from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta
    from analysis.runtime import Runtime

    p = tmp_path / "sample_runtime.phdf"
    _write_minimal_parthenon_file(p)

    rt = Runtime()
    ds = open_dataset(f"parthenon://{p}", runtime=rt, step=0, level=0)
    fid = ds.field_id("density")

    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox((0, 0, 0), (3, 2, 1))],
                    )
                ],
            )
        ]
    )
    rt.preload(runmeta=runmeta, dataset=ds, fields=[fid])
    chunk = rt.get_task_chunk(step=0, level=0, field=fid, version=0, block=0)
    assert len(chunk) == 1 * 2 * 3 * 4 * 4
