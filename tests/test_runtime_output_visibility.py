from __future__ import annotations

import threading
import time

import pytest

from analysis.plan import Plan
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
