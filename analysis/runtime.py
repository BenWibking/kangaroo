from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
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


def run_console_main(runtime: "Runtime", fn: Any) -> Any:
    locality = runtime.locality_id()
    if locality != 0:
        runtime.wait_for_console_release()
        return 0
    try:
        return fn()
    finally:
        runtime.release_console_workers()


def _log_phase_span(
    name: str,
    start: float,
    end: float,
    *,
    category: str = "kangaroo.python.setup",
    worker_label: str = "python-main",
    locality: int = 0,
) -> None:
    if not hasattr(_core, "log_phase_event"):
        return
    _core.log_phase_event(
        name,
        "start",
        start,
        start,
        category,
        worker_label,
        locality,
    )
    _core.log_phase_event(
        name,
        "end",
        start,
        end,
        category,
        worker_label,
        locality,
    )


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
        return

    @property
    def kernels(self):
        return self._rt.kernels()

    def alloc_field_id(self, name: str) -> int:
        return self._rt.alloc_field_id(name)

    def mark_field_persistent(self, fid: int, name: str) -> None:
        self._rt.mark_field_persistent(fid, name)

    def run(self, plan: Plan, *, runmeta, dataset, progress_bar: bool = False) -> None:
        if self._run_in_progress:
            raise RuntimeError("runtime run is already in progress")
        prepare_start = time.time()
        phase_start = prepare_start
        plan_ir = plan_to_dict(plan)
        phase_end = time.time()
        _log_phase_span("python_plan_to_dict", phase_start, phase_end)
        configured_plan_path = os.environ.get("KANGAROO_DASHBOARD_PLAN", "").strip()
        if configured_plan_path:
            phase_start = time.time()
            plan_path = Path(configured_plan_path)
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(plan_ir, indent=2), encoding="utf-8")
            phase_end = time.time()
            _log_phase_span("python_write_dashboard_plan", phase_start, phase_end)
        phase_start = time.time()
        packed = msgpack.packb(plan_ir, use_bin_type=True)
        phase_end = time.time()
        _log_phase_span("python_pack_plan_msgpack", phase_start, phase_end)
        _log_phase_span("python_prepare_plan", prepare_start, phase_end)
        self._run_in_progress = True
        try:
            if progress_bar:
                self._run_packed_plan_with_progress(plan, packed, runmeta=runmeta, dataset=dataset)
            else:
                self._rt.run_packed_plan(packed, runmeta._h, dataset._h)
        finally:
            self._run_in_progress = False

    def _run_packed_plan_with_progress(self, plan: Plan, packed: bytes, *, runmeta, dataset) -> None:
        total_tasks = _count_plan_tasks(plan, runmeta=runmeta)
        stop_progress = threading.Event()
        progress_thread: threading.Thread | None = None
        configured_event_log = os.environ.get("KANGAROO_EVENT_LOG", "").strip()
        if configured_event_log:
            event_log = Path(configured_event_log)
            event_log.parent.mkdir(parents=True, exist_ok=True)
            self.set_event_log_path(str(event_log))
            progress_thread = threading.Thread(
                target=_task_progress_monitor,
                args=(event_log, total_tasks, stop_progress),
                daemon=True,
            )
            progress_thread.start()
            try:
                self._rt.run_packed_plan(packed, runmeta._h, dataset._h)
            finally:
                stop_progress.set()
                if progress_thread is not None:
                    progress_thread.join(timeout=2.0)
                try:
                    self.set_event_log_path("")
                except Exception:
                    pass
            return

        with tempfile.TemporaryDirectory(prefix="kangaroo-events-") as tmpdir:
            event_log = Path(tmpdir) / "events.jsonl"
            self.set_event_log_path(str(event_log))
            progress_thread = threading.Thread(
                target=_task_progress_monitor,
                args=(event_log, total_tasks, stop_progress),
                daemon=True,
            )
            progress_thread.start()
            try:
                self._rt.run_packed_plan(packed, runmeta._h, dataset._h)
            finally:
                stop_progress.set()
                if progress_thread is not None:
                    progress_thread.join(timeout=2.0)
                # Clear the event log path so later runs do not target a deleted temp file.
                try:
                    self.set_event_log_path("")
                except Exception:
                    pass

    def set_event_log_path(self, path: str) -> None:
        self._rt.set_event_log_path(path)

    def set_perfetto_trace_path(self, path: str) -> None:
        self._rt.set_perfetto_trace_path(path)

    def locality_id(self) -> int:
        return int(self._rt.locality_id())

    def num_localities(self) -> int:
        return int(self._rt.num_localities())

    def chunk_home_rank(self, *, step: int, level: int, block: int) -> int:
        return int(self._rt.chunk_home_rank(int(step), int(level), int(block)))

    def is_console_locality(self) -> bool:
        return self.locality_id() == 0

    def wait_for_console_release(self) -> None:
        self._rt.wait_for_console_release()

    def release_console_workers(self) -> None:
        self._rt.release_console_workers()

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
        dataset: Any | None = None,
    ) -> bytes:
        if self._run_in_progress:
            raise RuntimeError("output retrieval is not allowed while a plan run is in progress")
        dataset_h = dataset._h if dataset is not None and hasattr(dataset, "_h") else None
        return self._rt.get_task_chunk(step, level, field, version, block, dataset_h)

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
    shared_covered_boxes: list[Any] = []
    shared_covered_boxes_by_objid: dict[int, int] = {}
    shared_covered_boxes_by_key: dict[str, int] = {}

    def domain_to_dict(domain) -> dict:
        return {
            "step": domain.step,
            "level": domain.level,
            "blocks": list(domain.blocks) if domain.blocks is not None else None,
        }

    def params_to_dict(params: dict[str, Any]) -> dict[str, Any]:
        if "covered_boxes" not in params:
            return params
        covered_boxes = params["covered_boxes"]
        shared_idx = shared_covered_boxes_by_objid.get(id(covered_boxes))
        if shared_idx is None:
            key = json.dumps(covered_boxes, separators=(",", ":"))
            shared_idx = shared_covered_boxes_by_key.get(key)
            if shared_idx is None:
                shared_idx = len(shared_covered_boxes)
                shared_covered_boxes.append(covered_boxes)
                shared_covered_boxes_by_key[key] = shared_idx
            shared_covered_boxes_by_objid[id(covered_boxes)] = shared_idx
        out = dict(params)
        out.pop("covered_boxes", None)
        out["covered_boxes_ref"] = shared_idx
        return out

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
                        "params": params_to_dict(tmpl.params),
                    }
                    for tmpl in stage.templates
                ],
            }
        )
    plan_dict = {"stages": stages}
    if shared_covered_boxes:
        plan_dict["shared_covered_boxes"] = shared_covered_boxes
    return plan_dict


def _count_plan_tasks(plan: Plan, *, runmeta: Any) -> int:
    def _blocks_for_domain(domain: Any) -> int:
        blocks = getattr(domain, "blocks", None)
        if blocks is None:
            try:
                step = int(getattr(domain, "step", 0))
                level = int(getattr(domain, "level", 0))
                levels = runmeta.steps[step].levels
                return len(levels[level].boxes)
            except Exception:
                return 1
        try:
            return len(blocks)
        except Exception:
            return 1

    def _graph_groups_for_template(tmpl: Any) -> int:
        params = getattr(tmpl, "params", None)
        if not isinstance(params, dict):
            return 1
        if str(params.get("graph_kind", "")) != "reduce":
            return 1
        try:
            fan_in = max(1, int(params.get("fan_in", 1)))
        except Exception:
            fan_in = 1
        try:
            num_inputs = int(params.get("num_inputs", 0))
        except Exception:
            num_inputs = 0
        if num_inputs <= 0:
            input_blocks = params.get("input_blocks")
            if input_blocks is not None:
                try:
                    num_inputs = len(input_blocks)
                except Exception:
                    num_inputs = 0
        if num_inputs <= 0:
            return 1
        return max(1, (num_inputs + fan_in - 1) // fan_in)

    total = 0
    for stage in plan.topo_stages():
        for tmpl in stage.templates:
            if str(getattr(tmpl, "plane", getattr(stage, "plane", ""))) == "graph":
                total += _graph_groups_for_template(tmpl)
            else:
                total += _blocks_for_domain(tmpl.domain)
    return total


def _fallback_task_id(event: dict) -> str:
    return (
        f"{event.get('stage','?')}:{event.get('template','?')}:"
        f"{event.get('block','?')}:{event.get('start',0.0)}"
    )


def _event_log_paths(log_path: Path) -> list[Path]:
    paths = [log_path]
    suffix = log_path.suffix
    if suffix:
        pattern = f"{log_path.stem}.locality*{suffix}"
    else:
        pattern = f"{log_path.name}.locality*"
    paths.extend(sorted(p for p in log_path.parent.glob(pattern) if p != log_path))
    return paths


def _is_main_task_completion(event: dict) -> bool:
    if not isinstance(event, dict) or event.get("type") != "task":
        return False
    status = str(event.get("status", ""))
    if status not in {"end", "error"}:
        return False
    task_id = str(event.get("id") or _fallback_task_id(event))
    return task_id.count(":") == 3


def _task_progress_monitor(log_path: Path, total_tasks: int, stop_event: threading.Event) -> None:
    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    seen_done: set[str] = set()
    offsets: dict[Path, int] = {}
    bar = None
    if tqdm is not None:
        bar = tqdm(
            total=total_tasks or None,
            desc="pipeline tasks",
            unit="task",
            mininterval=1.0,
            miniters=1,
            smoothing=0.0,
        )
        bar.refresh()

    def _update_bar() -> None:
        if bar is None:
            return
        delta = len(seen_done) - bar.n
        if delta > 0:
            bar.update(delta)

    try:
        while True:
            progressed = False
            for path in _event_log_paths(log_path):
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        handle.seek(offsets.get(path, 0))
                        while True:
                            pos = handle.tell()
                            line = handle.readline()
                            if not line:
                                offsets[path] = pos
                                break
                            offsets[path] = handle.tell()
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if not _is_main_task_completion(event):
                                continue
                            task_id = str(event.get("id") or _fallback_task_id(event))
                            if task_id in seen_done:
                                continue
                            seen_done.add(task_id)
                            progressed = True
                except FileNotFoundError:
                    pass

            if progressed:
                _update_bar()
            elif bar is not None:
                bar.refresh()

            if stop_event.is_set():
                time.sleep(0.1)
                for path in _event_log_paths(log_path):
                    try:
                        with path.open("r", encoding="utf-8") as handle:
                            handle.seek(offsets.get(path, 0))
                            for line in handle:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    event = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                if not _is_main_task_completion(event):
                                    continue
                                task_id = str(event.get("id") or _fallback_task_id(event))
                                seen_done.add(task_id)
                    except FileNotFoundError:
                        pass
                _update_bar()
                break

            time.sleep(1.0)
    finally:
        if bar is not None:
            bar.close()


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


def log_phase_event(
    name: str,
    status: str,
    *,
    start: float | None = None,
    end: float | None = None,
    category: str | None = None,
    worker_label: str | None = None,
    locality: int | None = None,
) -> None:
    if not hasattr(_core, "log_phase_event"):
        return
    _core.log_phase_event(name, status, start, end, category, worker_label, locality)


def set_event_log_path(path: str) -> None:
    _core.set_event_log_path(path)


def set_perfetto_trace_path(path: str) -> None:
    _core.set_perfetto_trace_path(path)
