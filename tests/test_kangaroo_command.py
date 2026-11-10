from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
KANGAROO = REPO_ROOT / "scripts" / "kangaroo"


def run_kangaroo(
    *args: str,
    campaign: str = "test-campaign",
    executable: Path = KANGAROO,
    cwd: Path = REPO_ROOT,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {"KANGAROO_CAMPAIGN": campaign}
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        [str(executable), *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_lists_plot_commands() -> None:
    result = run_kangaroo("--help")

    assert result.returncode == 0
    assert "plot-cylindrical-flux" in result.stdout
    assert "plot-cylindrical-radial-flux" in result.stdout
    assert "plot-spherical-flux" in result.stdout
    assert "plot-radial-flux-to-sfr" in result.stdout
    assert "plot-wind-disk-angular-momentum-flux" in result.stdout


def test_plot_command_reports_missing_result_directory() -> None:
    result = run_kangaroo("plot-spherical-flux", "plt-does-not-exist")

    assert result.returncode == 2
    assert "result directory does not exist" in result.stderr


def test_symlinked_command_resolves_repository_root(tmp_path: Path) -> None:
    symlink = tmp_path / "kangaroo"
    symlink.symlink_to(KANGAROO)

    result = run_kangaroo(
        "plot-spherical-flux", "plt-does-not-exist", executable=symlink
    )

    assert result.returncode == 2
    assert str(REPO_ROOT / "results") in result.stderr
    assert str(tmp_path / "results") not in result.stderr


def test_copied_command_resolves_current_worktree(tmp_path: Path) -> None:
    copied_launcher = tmp_path / "kangaroo"
    shutil.copy2(KANGAROO, copied_launcher)

    result = run_kangaroo(
        "plot-spherical-flux", "plt-does-not-exist", executable=copied_launcher
    )

    assert result.returncode == 2
    assert str(REPO_ROOT / "results") in result.stderr
    assert str(tmp_path / "results") not in result.stderr


def test_plot_command_rejects_nonexistent_plotfile_path() -> None:
    result = run_kangaroo("plot-spherical-flux", "missing/plt000100")

    assert result.returncode == 2
    assert "plotfile directory does not exist" in result.stderr


def test_plot_command_looks_for_results_in_current_directory(tmp_path: Path) -> None:
    result = run_kangaroo(
        "plot-spherical-flux",
        "plt-does-not-exist",
        cwd=tmp_path,
        extra_env={"KANGAROO_REPO_ROOT": str(REPO_ROOT)},
    )

    assert result.returncode == 2
    assert str(tmp_path / "results") in result.stderr


@pytest.mark.parametrize("sbatch_options", [[], ["--", "--time=01:00:00"]])
def test_submission_dry_run(tmp_path: Path, sbatch_options: list[str]) -> None:
    plotfile = tmp_path / "plt000100"
    plotfile.mkdir()

    result = run_kangaroo("toomre-q", str(plotfile), "--dry-run", *sbatch_options)

    assert result.returncode == 0, result.stderr
    assert "sbatch " in result.stdout
    assert "make_toomre_q.submit" in result.stdout
    assert str(plotfile) in result.stdout
    if sbatch_options:
        assert "--time=01:00:00" in result.stdout
    assert not (REPO_ROOT / "results" / "test-campaign" / plotfile.name).exists()
