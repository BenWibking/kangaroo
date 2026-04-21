from __future__ import annotations

import json
import threading
import time

import pytest

from analysis.plan import Domain, FieldRef, Plan, Stage
from analysis import runtime as runtime_mod
from analysis.runtime import Runtime, plan_to_dict, run_console_main


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


class _FakeLocalityCoreRuntime:
    def __init__(self, locality: int) -> None:
        self._locality = locality
        self.wait_called = False
        self.release_called = False

    def locality_id(self) -> int:
        return self._locality

    def num_localities(self) -> int:
        return 2

    def wait_for_console_release(self) -> None:
        self.wait_called = True

    def release_console_workers(self) -> None:
        self.release_called = True


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


def test_plan_to_dict_hoists_shared_covered_boxes() -> None:
    shared_boxes = [[[0, 0, 0], [3, 3, 3]], [[4, 0, 0], [7, 3, 3]]]
    stage = Stage(name="projection")
    domain = Domain(step=0, level=0, blocks=[0])
    for block in (0, 1):
        stage.map_blocks(
            name=f"uniform_projection_b{block}",
            kernel="uniform_projection_accumulate",
            domain=Domain(step=0, level=0, blocks=[block]),
            inputs=[FieldRef(1, domain=domain)],
            outputs=[FieldRef(2)],
            output_bytes=[128],
            deps={"kind": "None"},
            params={
                "axis": 2,
                "resolution": [8, 8],
                "covered_boxes": shared_boxes,
            },
        )

    plan_dict = plan_to_dict(Plan(stages=[stage]))

    assert plan_dict["shared_covered_boxes"] == [shared_boxes]
    params = [tmpl["params"] for tmpl in plan_dict["stages"][0]["templates"]]
    assert all("covered_boxes" not in p for p in params)
    assert all(p["covered_boxes_ref"] == 0 for p in params)


def test_run_console_main_executes_only_on_console_locality() -> None:
    core = _FakeLocalityCoreRuntime(locality=0)
    rt = Runtime(core_runtime=core)
    seen: list[str] = []

    result = run_console_main(rt, lambda: seen.append("ran") or 7)

    assert result == 7
    assert seen == ["ran"]
    assert core.release_called
    assert not core.wait_called


def test_run_console_main_waits_on_worker_locality() -> None:
    core = _FakeLocalityCoreRuntime(locality=1)
    rt = Runtime(core_runtime=core)
    seen: list[str] = []

    result = run_console_main(rt, lambda: seen.append("ran") or 7)

    assert result == 0
    assert seen == []
    assert core.wait_called
    assert not core.release_called
