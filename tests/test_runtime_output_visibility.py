from __future__ import annotations

import json
import threading
import time

import pytest

from analysis.plan import Plan
from analysis import runtime as runtime_mod
from analysis.runtime import Runtime


class _FakeCoreRuntime:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self._started = started
        self._release = release

    def run_packed_plan(self, packed, runmeta_h, dataset_h) -> None:
        self._started.set()
        self._release.wait(timeout=2.0)

    def get_task_chunk(self, step, level, field, version, block) -> bytes:
        return b""


class _FakeHandle:
    pass


class _FakeRunMeta:
    _h = _FakeHandle()


class _FakeDataset:
    _h = _FakeHandle()


class _ImmediateCoreRuntime:
    def run_packed_plan(self, packed, runmeta_h, dataset_h) -> None:
        return None


def test_runtime_rejects_get_task_chunk_during_inflight_run() -> None:
    started = threading.Event()
    release = threading.Event()
    rt = Runtime(core_runtime=_FakeCoreRuntime(started, release))
    rt._bind_dataset_handle = lambda dataset: None  # type: ignore[method-assign]
    plan = Plan(stages=[])
    runmeta = _FakeRunMeta()
    dataset = _FakeDataset()

    err: list[BaseException] = []

    def _run() -> None:
        try:
            rt.run(plan, runmeta=runmeta, dataset=dataset)
        except BaseException as exc:  # noqa: BLE001
            err.append(exc)

    t = threading.Thread(target=_run)
    t.start()
    assert started.wait(timeout=1.0)
    time.sleep(0.01)

    with pytest.raises(RuntimeError, match="in progress"):
        rt.get_task_chunk(step=0, level=0, field=1, block=0, dataset=dataset)

    release.set()
    t.join(timeout=2.0)
    assert not err


def test_runtime_logs_python_prepare_phases(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def _record_phase(name, status, start, end, category, worker_label, locality) -> None:
        seen.append((name, status))

    monkeypatch.setattr(runtime_mod._core, "log_phase_event", _record_phase)
    rt = Runtime(core_runtime=_ImmediateCoreRuntime())
    rt._bind_dataset_handle = lambda dataset: None  # type: ignore[method-assign]

    rt.run(Plan(stages=[]), runmeta=_FakeRunMeta(), dataset=_FakeDataset())

    assert ("python_plan_to_dict", "start") in seen
    assert ("python_plan_to_dict", "end") in seen
    assert ("python_pack_plan_msgpack", "start") in seen
    assert ("python_pack_plan_msgpack", "end") in seen
    assert ("python_prepare_plan", "start") in seen
    assert ("python_prepare_plan", "end") in seen


def test_runtime_writes_dashboard_plan_only_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    rt = Runtime(core_runtime=_ImmediateCoreRuntime())
    rt._bind_dataset_handle = lambda dataset: None  # type: ignore[method-assign]
    plan = Plan(stages=[])
    plan_path = tmp_path / "dashboard" / "plan.json"

    monkeypatch.delenv("KANGAROO_DASHBOARD_PLAN", raising=False)
    rt.run(plan, runmeta=_FakeRunMeta(), dataset=_FakeDataset())
    assert not plan_path.exists()

    monkeypatch.setenv("KANGAROO_DASHBOARD_PLAN", str(plan_path))
    rt.run(plan, runmeta=_FakeRunMeta(), dataset=_FakeDataset())
    assert plan_path.exists()
    assert json.loads(plan_path.read_text(encoding="utf-8")) == {"stages": []}
