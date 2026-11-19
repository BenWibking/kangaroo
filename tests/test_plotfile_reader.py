from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

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


def _write_grown_plotfile(root, ngrow: int, *, single_precision: bool = False) -> list[float]:
    """A one-level 4^3 plotfile with one 2x4x4 grid pair whose FABs are stored
    over their grown boxes, ghost cells holding a sentinel. Returns grid 1's
    valid values in x-fastest order."""
    os.makedirs(root / "Level_0")
    (root / "Header").write_text(
        "HyperCLaw-V1.1\n1\nrho\n3\n0.0\n0\n0 0 0\n4 4 4\n\n"
        "((0,0,0) (3,3,3) (0,0,0))\n0\n1 1 1\n0\n0\n"
        "0 2 0.0\n0\n0 2\n0 4\n0 4\n2 4\n0 4\n0 4\nLevel_0/Cell\n"
    )
    if single_precision:
        desc = "((8, (32 8 23 0 1 9 0 127)),(4, (4 3 2 1)))"
        scalar_format = "<f"
    else:
        desc = "((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))"
        scalar_format = "<d"
    boxes = [((0, 0, 0), (1, 3, 3)), ((2, 0, 0), (3, 3, 3))]
    data = bytearray()
    offsets = []
    expected = []
    for grid, (lo, hi) in enumerate(boxes):
        glo = [v - ngrow for v in lo]
        ghi = [v + ngrow for v in hi]
        offsets.append(len(data))
        box = f"(({glo[0]},{glo[1]},{glo[2]}) ({ghi[0]},{ghi[1]},{ghi[2]}) (0,0,0))"
        data += f"FAB {desc}{box} 1\n".encode()
        for k in range(glo[2], ghi[2] + 1):
            for j in range(glo[1], ghi[1] + 1):
                for i in range(glo[0], ghi[0] + 1):
                    valid = all(lo[a] <= c <= hi[a] for a, c in enumerate((i, j, k)))
                    value = 1.0 + i + 10.0 * j + 100.0 * k if valid else -1.0e30
                    data += struct.pack(scalar_format, value)
                    if valid and grid == 1:
                        expected.append(value)
    (root / "Level_0" / "Cell_D_00000").write_bytes(bytes(data))
    lines = ["1", "1", "1", str(ngrow), "(2 0"]
    lines += [f"(({lo[0]},{lo[1]},{lo[2]}) ({hi[0]},{hi[1]},{hi[2]}) (0,0,0))" for lo, hi in boxes]
    lines += [")", "2"] + [f"FabOnDisk: Cell_D_00000 {o}" for o in offsets]
    lines += ["", "2,1", "-1e30,", "-1e30,", "", "2,1", "1e30,", "1e30,", ""]
    (root / "Level_0" / "Cell_H").write_text("\n".join(lines) + "\n")
    return expected


def test_read_fab_crops_stored_ghost_cells(tmp_path) -> None:
    from analysis import PlotfileReader

    expected = _write_grown_plotfile(tmp_path / "plt", ngrow=1)
    reader = PlotfileReader(str(tmp_path / "plt"))
    fab = reader.read_fab(0, 1, 0, 1)
    assert fab["shape"] == (1, 4, 4, 2)
    values = list(struct.unpack(f"<{len(fab['data']) // 8}d", fab["data"]))
    assert values == expected
    assert -1.0e30 not in values


def test_temperature_density_cli_writes_mass_bins_and_plot(tmp_path) -> None:
    import numpy as np

    from scripts.plotfile_flux_surface import MSUN_G

    repo = Path(__file__).resolve().parents[1]
    plotfile = tmp_path / "plt"
    values = _write_grown_plotfile(plotfile, ngrow=1)
    # The two blocks are offset by two cells along x. Use rho for both axes
    # in this single-field plotfile to check loading, weighting and output units.
    density = np.r_[values, np.asarray(values) - 2.0]
    directory = tmp_path / "results/demo/plt/temperature_density"
    directory.mkdir(parents=True)
    output = directory / "temperature_density_histogram.json"
    env = dict(os.environ, KANGAROO_EXECUTOR_MODE="streaming", KANGAROO_EVENT_LOG="",
               KANGAROO_REPO_ROOT=str(repo), MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/plotfile_temperature_density.py"), str(plotfile),
         "--density-range", "1", "1000", "--temperature-range", "1", "1000",
         "--temperature", "rho", "--bins", "3", "4", "--output-json", str(output)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    expected, _, _ = np.histogram2d(
        density, density, bins=(payload["density_edges"], payload["temperature_edges"]),
        weights=density / MSUN_G,
    )
    np.testing.assert_allclose(payload["mass_msun"], expected, atol=0.0)
    assert payload["total_mass_msun"] == pytest.approx(density.sum() / MSUN_G, abs=0.0)

    result = subprocess.run(
        [str(repo / "scripts/kangaroo"), "plot-temperature-density", "plt", "--campaign", "demo"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (directory / "temperature_density.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("save_trace", [True, False])
def test_cylindrical_flux_cli_traces_startup_phases(tmp_path, save_trace) -> None:
    plotfile = tmp_path / "plt"
    _write_grown_plotfile(plotfile, ngrow=1)
    trace = tmp_path / "events.jsonl"
    output = tmp_path / "flux.json"
    script = Path(__file__).resolve().parents[1] / "scripts/plotfile_cylindrical_flux_surface.py"
    cmd = [sys.executable, str(script), str(plotfile), "--radius", "1.5", "--height", "1.5",
           "--output-json", str(output), "--progress"]
    for field in ("density", "momx", "momy", "momz", "energy", "scalar", "bx", "by", "bz"):
        cmd.extend([f"--{field}", "rho"])
    env = dict(os.environ, KANGAROO_EXECUTOR_MODE="streaming",
               KANGAROO_EVENT_LOG=str(trace) if save_trace else "")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    if not save_trace:
        assert not trace.exists()
        return

    events = [json.loads(line) for line in trace.read_text().splitlines()]
    phases = {(e["name"], e["status"]): e for e in events if e["type"] == "phase"}
    previous_end = 0.0
    for name in ("python_open_plotfile", "python_load_plotfile_metadata",
                 "python_build_cylindrical_flux_plan", "python_encode_plan_flatbuffer"):
        start = phases[name, "start"]
        end = phases[name, "end"]
        assert start["category"] == end["category"] == "kangaroo.python.setup"
        assert previous_end <= start["ts"] == end["start"] <= end["ts"]
        previous_end = end["ts"]
    assert previous_end <= phases["runtime_execute_plan", "start"]["ts"]


@pytest.mark.parametrize("like_input", [False, True], ids=["fixed-shape", "like-input"])
@pytest.mark.parametrize("single_precision", [False, True], ids=["f64", "f32"])
def test_streaming_plotfile_estimates_use_valid_fab_shape(
    tmp_path, monkeypatch, like_input, single_precision
) -> None:
    import numpy as np

    from analysis import Runtime
    from analysis.buffer import BufferSpec, DType, FixedShape, LikeInputShape
    from analysis.dataset import open_dataset
    from analysis.kernel_params import FieldExprParams
    from analysis.plan import Domain, FieldRef, OutputRef, Plan, Stage

    monkeypatch.setenv("KANGAROO_EXECUTOR_MODE", "streaming")
    monkeypatch.setenv("KANGAROO_EXECUTOR_MAX_OUTPUT_BYTES_PER_LOCALITY", "256")
    plotfile = tmp_path / "plt"
    expected = _write_grown_plotfile(plotfile, ngrow=1, single_precision=single_precision)
    shape = (2, 4, 4)
    output_shape = LikeInputShape(0) if like_input else FixedShape(shape)
    rt = Runtime()
    ds = open_dataset(str(plotfile), runtime=rt, step=0, level=0)
    source = ds.field_id("rho")
    output = rt.alloc_field_id("identity")
    stage = Stage("identity")
    stage.map_blocks(
        name="identity",
        kernel="field_expr",
        domain=Domain(step=0, level=0),
        inputs=[FieldRef(source)],
        outputs=[OutputRef(FieldRef(output), BufferSpec(DType.F64, output_shape))],
        params=FieldExprParams("value", ("value",)),
    )
    events_path = tmp_path / "setup.jsonl"
    rt.set_event_log_path(str(events_path))
    try:
        rt.run(Plan([stage]), runmeta=ds.get_runmeta(), dataset=ds)
    finally:
        rt.set_event_log_path("")

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    phases = {
        (e["name"], e["status"]): e["ts"]
        for e in events if e["type"] == "phase" and e["category"] == "kangaroo.executor.setup"
    }
    build = "executor_streaming_build_output_estimates"
    register = "executor_streaming_register_input_consumers"
    count = "executor_streaming_count_input_consumers"
    expand = "executor_streaming_expand_stage/0"
    partition = "executor_streaming_partition_stage/0"
    order = [(build, "start"), (build, "end"), (register, "start"), (count, "start"),
             (count, "end"), (register, "end"), (expand, "start"), (expand, "end"),
             (partition, "start"), (partition, "end")]
    timestamps = [phases[key] for key in order]
    assert timestamps == sorted(timestamps)
    reads = [e for e in events if e.get("op") == "plotfile_read_fab_header"]
    assert sorted(e["block"] for e in reads) == ([0, 1] if like_input else [])
    for read in reads:
        assert read["status"] == "end"
        assert phases[build, "start"] <= read["start"] <= read["end"] <= phases[build, "end"]

    queues = [e for e in events if e.get("op") == "dataset_load_queue"]
    assert sorted(e["block"] for e in queues) == [0, 1]
    assert all(e["estimated_bytes"] == 2 * 4 * 4 * 8 for e in queues)
    payloads = [e for e in events if e.get("op") == "plotfile_read_fab" and e["status"] == "end"]
    assert sorted(e["block"] for e in payloads) == [0, 1]
    assert all(e["bytes"] == 2 * 4 * 4 * (4 if single_precision else 8) for e in payloads)
    assert all(e["start"] >= phases[partition, "end"] for e in payloads)

    got = rt.get_task_chunk_array(
        step=0, level=0, field=output, version=0, block=1,
        shape=shape, dtype=np.float64, dataset=ds,
    )
    np.testing.assert_array_equal(got.ravel(order="F"), expected)
