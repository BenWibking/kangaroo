from __future__ import annotations

from typing import Any

import importlib.util
from pathlib import Path
import sys

import msgpack

from .plan import Plan

try:
    from . import _core  # type: ignore
except ImportError as exc:
    candidates: list[Path] = []
    for base in sys.path:
        try:
            base_path = Path(base)
        except OSError:
            continue
        if not base_path.is_dir():
            continue
        candidates.extend(base_path.glob("analysis/_core*.so"))
    if not candidates:
        raise ImportError(
            "analysis._core extension not found. Run `pixi run install` to install the C++ bindings."
        ) from exc
    ext_path = candidates[0]
    spec = importlib.util.spec_from_file_location("analysis._core", ext_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load analysis._core from {ext_path}") from exc
    module = importlib.util.module_from_spec(spec)
    sys.modules["analysis._core"] = module
    spec.loader.exec_module(module)
    _core = module  # type: ignore


def hpx_configuration_string() -> str:
    return _core.hpx_configuration_string()


class Runtime:
    def __init__(
        self,
        core_runtime: Any | None = None,
        *,
        hpx_config: list[str] | None = None,
        hpx_args: list[str] | None = None,
    ) -> None:
        if core_runtime is not None:
            self._rt = core_runtime
        elif hpx_config is not None or hpx_args is not None:
            self._rt = _core.Runtime(hpx_config or [], hpx_args or [])
        else:
            self._rt = _core.Runtime()

    @property
    def kernels(self):
        return self._rt.kernels()

    def alloc_field_id(self, name: str) -> int:
        return self._rt.alloc_field_id(name)

    def mark_field_persistent(self, fid: int, name: str) -> None:
        self._rt.mark_field_persistent(fid, name)

    def run(self, plan: Plan, *, runmeta, dataset) -> None:
        packed = msgpack.packb(plan_to_dict(plan), use_bin_type=True)
        self._rt.run_packed_plan(packed, runmeta._h, dataset._h)

    def set_event_log_path(self, path: str) -> None:
        self._rt.set_event_log_path(path)

    def preload(self, *, runmeta, dataset, fields: list[int]) -> None:
        self._rt.preload_dataset(runmeta._h, dataset._h, list(fields))

    def get_task_chunk(
        self,
        *,
        step: int,
        level: int,
        field: int,
        version: int = 0,
        block: int,
    ) -> bytes:
        return self._rt.get_task_chunk(step, level, field, version, block)


def plan_to_dict(plan: Plan) -> dict:
    stages = []
    topo = plan.topo_stages()
    stage_ids = {id(s): i for i, s in enumerate(topo)}
    for stage in topo:
        stages.append(
            {
                "name": stage.name,
                "plane": stage.plane,
                "after": [stage_ids[id(parent)] for parent in stage.after],
                "templates": [
                    {
                        "name": tmpl.name,
                        "plane": tmpl.plane,
                        "kernel": tmpl.kernel,
                        "domain": {
                            "step": tmpl.domain.step,
                            "level": tmpl.domain.level,
                            "blocks": list(tmpl.domain.blocks)
                            if tmpl.domain.blocks is not None
                            else None,
                        },
                    "inputs": [
                        {"field": ref.field, "version": ref.version}
                        for ref in tmpl.inputs
                    ],
                    "outputs": [
                        {"field": ref.field, "version": ref.version}
                        for ref in tmpl.outputs
                    ],
                    "output_bytes": list(tmpl.output_bytes),
                    "deps": tmpl.deps,
                    "params": tmpl.params,
                }
                    for tmpl in stage.templates
                ],
            }
        )
    return {"stages": stages}


def log_task_event(
    name: str,
    status: str,
    *,
    start: float | None = None,
    end: float | None = None,
    event_id: str | None = None,
    worker_label: str | None = None,
) -> None:
    _core.log_task_event(name, status, start, end, event_id, worker_label)


def set_event_log_path(path: str) -> None:
    _core.set_event_log_path(path)
