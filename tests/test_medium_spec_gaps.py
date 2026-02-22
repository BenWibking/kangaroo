from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from analysis import histogram_edges_1d, histogram_edges_2d
from analysis.dataset import Dataset
from analysis.dashboard import DashboardApp, DashboardConfig
from analysis.ops import histogram_edges_1d as histogram_edges_1d_ops
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


def test_histogram_edges_helpers_exported_from_package() -> None:
    xs = histogram_edges_1d((0.0, 1.0), 4)
    x2, y2 = histogram_edges_2d((0.0, 1.0), (0.0, 2.0), (2, 4))
    assert xs == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert x2 == [0.0, 0.5, 1.0]
    assert y2 == [0.0, 0.5, 1.0, 1.5, 2.0]


@pytest.mark.parametrize(
    "hist_range",
    [
        (math.nan, 1.0),
        (0.0, math.nan),
        (math.inf, 1.0),
        (0.0, math.inf),
        (-math.inf, 1.0),
    ],
)
def test_histogram_edges_reject_non_finite_bounds(hist_range: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        histogram_edges_1d_ops(hist_range, 8)


def test_runmeta_forwards_particle_species_to_core_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeRunMetaHandle:
        def __init__(self, payload) -> None:
            captured["payload"] = payload

    monkeypatch.setattr("analysis.runmeta._core.RunMetaHandle", _FakeRunMetaHandle)

    rm = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox((0, 0, 0), (1, 1, 1))],
                    )
                ],
            )
        ],
        particle_species={"dm": 3, "stars": 7},
    )
    assert rm._h is not None
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["particle_species"] == {"dm": 3, "stars": 7}
    assert isinstance(payload["steps"], list)


def test_dashboard_raises_explicit_error_for_malformed_plan_payload(tmp_path: Path) -> None:
    bad_plan = tmp_path / "bad_plan.json"
    bad_plan.write_text(json.dumps({"stages": "not-a-list"}), encoding="utf-8")

    app = DashboardApp(DashboardConfig(plan_path=bad_plan))
    with pytest.raises(RuntimeError, match="malformed plan payload"):
        app._ensure_plan_loaded()


def test_dataset_classifies_any_memory_uri_as_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeHandle:
        def __init__(self, uri: str, step: int, level: int) -> None:
            self.uri = uri
            self.step = step
            self.level = level

    monkeypatch.setattr("analysis.dataset._core.DatasetHandle", _FakeHandle)
    ds = Dataset(uri="memory://session-abc", runtime=None)
    assert ds.kind == "memory"
