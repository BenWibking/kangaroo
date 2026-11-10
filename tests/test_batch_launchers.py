import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPTS = sorted((REPO_ROOT / "batch").glob("make_*.submit"))


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    # Spaces expose accidental word splitting in launchers and analysis paths.
    root = tmp_path / "kangaroo checkout"
    shutil.copytree(REPO_ROOT / "batch", root / "batch")
    utils = root / "scripts" / "utils"
    utils.mkdir(parents=True)
    for name in ("hpxrun_pixi.sh", "hpxrun_spack_hpx.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / "utils" / name, utils / name)
    for backend in ("pixi-hpx", "spack-hpx"):
        bin_dir = root / ".pixi" / "envs" / backend / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").symlink_to(sys.executable)
        (bin_dir / "hpxrun.py").write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['CAPTURE']).write_text(json.dumps({"
            "'args': sys.argv[1:], 'prefix': os.environ['CONDA_PREFIX'], "
            "'hpx_args': os.environ.get('HPXRUN_ARGS', '')}))\n"
        )
    return root


def run_batch(root: Path, script: str, **overrides: str) -> subprocess.CompletedProcess:
    env = {name: os.environ[name] for name in ("PATH", "HOME") if name in os.environ}
    env.update(
        KANGAROO_REPO_ROOT=str(root),
        CAPTURE=str(root / "capture.json"),
        SLURM_NNODES="1",
        SLURM_NTASKS="1",
        SLURM_CPUS_PER_TASK="32",
        SLURM_TRES_PER_TASK="cpu=24",
        CONDA_PREFIX="/wrong/activated/environment",
    )
    env.update(overrides)
    return subprocess.run(
        ["bash", str(root / "batch" / script), "/data/plot file"],
        cwd=root.parent,
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("script", [p.name for p in BATCH_SCRIPTS])
def test_batch_uses_vendored_hpx(checkout: Path, script: str) -> None:
    result = run_batch(checkout, script)
    assert result.returncode == 0, result.stderr
    captured = json.loads((checkout / "capture.json").read_text())
    args = captured["args"]
    assert args[:10] == [
        "-p", "none", "-r", "srun", "-p", "none", "-l", "1", "-t", "24"
    ]
    assert args[10] == str(checkout / ".pixi/envs/pixi-hpx/bin/python")
    assert args[11].startswith(str(checkout / "scripts"))
    assert args[12:14] == ["--", "/data/plot file"]
    assert captured["prefix"] == str(checkout / ".pixi/envs/pixi-hpx")
    assert "hpx.stacks.use_guard_pages=0" in captured["hpx_args"]


@pytest.mark.parametrize("overrides", [{"SLURM_NNODES": "2"}, {"SLURM_NTASKS": "2"}])
def test_pixi_batch_rejects_distributed_allocation(checkout: Path, overrides: dict) -> None:
    result = run_batch(checkout, "make_projection.submit", **overrides)
    assert result.returncode == 2
    assert "networking disabled" in result.stderr
    assert not (checkout / "capture.json").exists()


def test_batch_rejects_unknown_backend(checkout: Path) -> None:
    result = run_batch(checkout, "make_projection.submit", KANGAROO_HPX_BACKEND="typo")
    assert result.returncode == 2
    assert "Unknown KANGAROO_HPX_BACKEND" in result.stderr


def test_batch_reports_missing_environment(checkout: Path) -> None:
    (checkout / ".pixi/envs/pixi-hpx/bin/python").unlink()
    result = run_batch(checkout, "make_projection.submit")
    assert result.returncode == 2
    assert "pixi install -e pixi-hpx" in result.stderr


def test_spack_batch_keeps_mpi_runtime(checkout: Path) -> None:
    # Supply a module function without depending on the cluster module system.
    shell_env = checkout / "shell-env"
    shell_env.write_text("module() { :; }\n")
    prefix = checkout / ".pixi/envs/spack-hpx"
    hpx_dir = prefix / "lib/cmake/HPX"
    hpx_dir.mkdir(parents=True)
    (hpx_dir / "HPXCacheVariables.cmake").write_text(
        "set(HPX_WITH_PARCELPORT_MPI ON)\n"
    )
    result = run_batch(
        checkout, "make_projection.submit",
        KANGAROO_HPX_BACKEND="spack-hpx", SLURM_NNODES="4", SLURM_NTASKS="4",
        SPACK_HPX_PREFIX=str(prefix), SPACK_MPI_PREFIX=str(prefix),
        HPX_DIR=str(hpx_dir), BASH_ENV=str(shell_env),
    )
    assert result.returncode == 0, result.stderr
    captured = json.loads((checkout / "capture.json").read_text())
    assert captured["args"][:8] == ["-r", "srun", "-p", "mpi", "-l", "4", "-t", "24"]
    assert captured["args"][8] == str(prefix / "bin/python")


def test_plot_cli_uses_pixi_hpx_from_callers_results(checkout: Path) -> None:
    launcher = checkout / "scripts/kangaroo"
    shutil.copy2(REPO_ROOT / "scripts/kangaroo", launcher)
    output = checkout.parent / "results/test-campaign/plt000100/spherical_flux"
    output.mkdir(parents=True)
    (output / "flux_surface.json").write_text("{}")
    for kind in ("mass", "energy", "passive_scalar"):
        (checkout / f"scripts/plot_flux_surface_{kind}_flux.py").write_text(
            "from pathlib import Path\n"
            f"Path('{kind}.called').write_text('ok')\n"
        )
    result = subprocess.run(
        [str(launcher), "plot-spherical-flux", "plt000100", "--campaign", "test-campaign"],
        cwd=checkout.parent, capture_output=True, text=True,
        env={"PATH": os.environ["PATH"], "KANGAROO_REPO_ROOT": str(checkout)},
    )
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in output.glob("*.called")) == [
        "energy.called", "mass.called", "passive_scalar.called"
    ]
