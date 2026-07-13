from __future__ import annotations

from pathlib import Path

import pytest

from analysis import _core


@pytest.mark.skipif(
    not hasattr(_core, "test_parthenon_component_storage"),
    reason="Parthenon HDF5 backend is disabled",
)
def test_selected_component_owns_only_selected_storage(tmp_path: Path) -> None:
    assert _core.test_parthenon_component_storage(str(tmp_path / "fixture.phdf")) == (
        [20.0, 21.0],
        16,
        16,
        16,
    )
