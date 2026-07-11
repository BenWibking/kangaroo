from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR_MIGRATED_CLIS = (
    "scripts/plotfile_projection.py",
    "scripts/plotfile_flux_surface.py",
    "scripts/plotfile_cylindrical_flux_surface.py",
    "scripts/plotfile_projection_cic_stellar.py",
)


@pytest.mark.parametrize("relative_path", DESCRIPTOR_MIGRATED_CLIS)
def test_plotting_cli_uses_descriptor_api(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "infer_bytes_per_value" not in source
    assert "bytes_per_value" not in source
    assert "--bytes-per-value" not in source
