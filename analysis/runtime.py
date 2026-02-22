from __future__ import annotations

import sys
from typing import Any

import importlib.util
from pathlib import Path

import msgpack
import numpy as np

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
        self._run_in_progress = False

    @classmethod
    def from_parsed_args(
        cls,
        parsed_args: Any,
        *,
        unknown_args: list[str] | None = None,
        argv0: str | None = None,
    ) -> "Runtime":
        hpx_config = getattr(parsed_args, "hpx_config", None)
        hpx_args = list(getattr(parsed_args, "hpx_arg", None) or [])
        unknown = list(unknown_args or [])
        if unknown:
            hpx_args.extend([argv0 or sys.argv[0], *unknown])
        if hpx_config or hpx_args:
            return cls(hpx_config=hpx_config, hpx_args=hpx_args)
        return cls()

    @staticmethod
    def _bind_dataset_handle(dataset: Any | None) -> None:
        if dataset is None or not hasattr(dataset, "_h"):
            return
        if hasattr(_core, "set_global_dataset"):
            _core.set_global_dataset(dataset._h)

    @property
    def kernels(self):
        return self._rt.kernels()

    def alloc_field_id(self, name: str) -> int:
        return self._rt.alloc_field_id(name)

    def mark_field_persistent(self, fid: int, name: str) -> None:
        self._rt.mark_field_persistent(fid, name)

    def run(self, plan: Plan, *, runmeta, dataset) -> None:
        if self._run_in_progress:
            raise RuntimeError("runtime run is already in progress")
        self._bind_dataset_handle(dataset)
        packed = msgpack.packb(plan_to_dict(plan), use_bin_type=True)
        self._run_in_progress = True
        try:
            self._rt.run_packed_plan(packed, runmeta._h, dataset._h)
        finally:
            self._run_in_progress = False

    def set_event_log_path(self, path: str) -> None:
        self._rt.set_event_log_path(path)

    def preload(self, *, runmeta, dataset, fields: list[int]) -> None:
        self._bind_dataset_handle(dataset)
        self._rt.preload_dataset(runmeta._h, dataset._h, list(fields))

    def get_task_chunk(
        self,
        *,
        step: int,
        level: int,
        field: int,
        version: int = 0,
        block: int,
        dataset: Any | None = None,
    ) -> bytes:
        if self._run_in_progress:
            raise RuntimeError("output retrieval is not allowed while a plan run is in progress")
        self._bind_dataset_handle(dataset)
        return self._rt.get_task_chunk(step, level, field, version, block)

    def get_task_chunk_array(
        self,
        *,
        step: int,
        level: int,
        field: int,
        shape: tuple[int, ...],
        version: int = 0,
        block: int,
        dtype: Any | None = None,
        bytes_per_value: int | None = None,
        dataset: Any | None = None,
    ) -> np.ndarray:
        count = int(np.prod(shape))
        if count <= 0:
            raise ValueError("shape must have a positive number of elements")

        raw = self.get_task_chunk(
            step=step,
            level=level,
            field=field,
            version=version,
            block=block,
            dataset=dataset,
        )
        if dtype is None:
            if bytes_per_value is None:
                bytes_per_value = len(raw) // count
            if bytes_per_value == 8:
                dtype = np.float64
            elif bytes_per_value == 4:
                dtype = np.float32
            else:
                raise ValueError(f"unsupported bytes_per_value: {bytes_per_value}")
        return np.frombuffer(raw, dtype=dtype, count=count).reshape(shape)


def plan_to_dict(plan: Plan) -> dict:
    stages = []
    topo = plan.topo_stages()
    stage_ids = {id(s): i for i, s in enumerate(topo)}

    def domain_to_dict(domain) -> dict:
        return {
            "step": domain.step,
            "level": domain.level,
            "blocks": list(domain.blocks) if domain.blocks is not None else None,
        }

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
                        "domain": domain_to_dict(tmpl.domain),
                        "inputs": [
                            {
                                "field": ref.field,
                                "version": ref.version,
                                "domain": domain_to_dict(
                                    ref.domain if ref.domain is not None else tmpl.domain
                                ),
                            }
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
