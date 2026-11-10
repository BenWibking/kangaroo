from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR_MIGRATED_CLIS = (
    "scripts/plotfile_slice.py",
    "scripts/plotfile_projection.py",
    "scripts/plotfile_flux_surface.py",
    "scripts/plotfile_cylindrical_flux_surface.py",
    "scripts/plotfile_projection_cic_stellar.py",
    "scripts/plotfile_toomre_q.py",
)

RESHAPED_DESCRIPTOR_CLIS = (
    "scripts/plotfile_slice.py",
    "scripts/plotfile_projection.py",
    "scripts/plotfile_flux_surface.py",
    "scripts/plotfile_cylindrical_flux_surface.py",
    "scripts/plotfile_projection_cic_stellar.py",
    "scripts/plotfile_toomre_q.py",
)

RAW_CHUNK_DEMOS = ("scripts/slice_operator_demo.py",)

FLUX_SURFACE_CLIS = (
    "scripts/plotfile_flux_surface.py",
    "scripts/plotfile_cylindrical_flux_surface.py",
)


@pytest.mark.parametrize("relative_path", DESCRIPTOR_MIGRATED_CLIS)
def test_plotting_cli_uses_descriptor_api(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "infer_bytes_per_value" not in source
    assert "bytes_per_value" not in source
    assert "--bytes-per-value" not in source


@pytest.mark.parametrize("relative_path", RESHAPED_DESCRIPTOR_CLIS)
def test_cli_uses_resolved_output_shape(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)

    chunk_reads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_task_chunk_array"
    ]
    if chunk_reads:
        assert all(
            keyword.arg != "shape" for call in chunk_reads for keyword in call.keywords
        )
    else:
        compute_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compute"
        ]
        assert "import kangaroo as kr" in source
        assert compute_calls


@pytest.mark.parametrize("relative_path", FLUX_SURFACE_CLIS)
def test_flux_surface_cli_reshapes_descriptor_array(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert "values = values.reshape(" in source


@pytest.mark.parametrize("relative_path", RAW_CHUNK_DEMOS)
def test_demo_raw_chunk_writes_include_descriptors(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)

    chunk_writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_chunk"
    ]
    assert chunk_writes
    assert all(
        {"dtype", "shape"} <= {keyword.arg for keyword in call.keywords}
        for call in chunk_writes
    )
