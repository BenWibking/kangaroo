from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import psutil

if sys.version_info >= (3, 14):
    raise SystemExit(
        "Kangaroo dashboard requires Python < 3.14 (NumPy/Bokeh crash on 3.14). "
        "Please use Python 3.12/3.13."
    )

from bokeh.layouts import column, gridplot, row
from bokeh.models import Button, ColumnDataSource, Div, HoverTool, Range1d, Spinner
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure
from bokeh.server.server import Server

ColorPalette = List[str]


@dataclass
class DashboardConfig:
    metrics_path: Optional[Path] = None
    plan_path: Optional[Path] = None
    run_command: Optional[List[str]] = None
    threads_per_locality: Optional[int] = None
    update_interval_ms: int = 500
    history_seconds: int = 300
    max_tasks: int = 2000
    port: int = 5006
    title: str = "Kangaroo Dashboard"
    allow_websocket_origin: Optional[List[str]] = None


class EventLogReader:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._inode = None

    def _refresh_handle(self) -> Optional[Any]:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return None
        if self._inode != stat.st_ino:
            self._inode = stat.st_ino
            self._offset = 0
        return True

    def read_events(self) -> Iterable[Dict[str, Any]]:
        if not self._refresh_handle():
            return []
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                while True:
                    pos = handle.tell()
                    line = handle.readline()
                    if not line:
                        self._offset = pos
                        break
                    self._offset = handle.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield payload
        except FileNotFoundError:
            return []


class MetricSampler:
    def __init__(self) -> None:
        self._last_io = psutil.disk_io_counters()
        self._last_ts = time.monotonic()

    def sample(self) -> Dict[str, float]:
        now = time.monotonic()
        elapsed = max(now - self._last_ts, 1e-6)
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        io_now = psutil.disk_io_counters()
        read_rate = (io_now.read_bytes - self._last_io.read_bytes) / elapsed
        write_rate = (io_now.write_bytes - self._last_io.write_bytes) / elapsed
        self._last_io = io_now
        self._last_ts = now
        return {
            "mem_used_gb": vm.used / 1e9,
            "mem_total_gb": vm.total / 1e9,
            "cpu_percent": cpu,
            "io_read_mbps": read_rate / 1e6,
            "io_write_mbps": write_rate / 1e6,
        }


class DashboardApp:
    def __init__(self, config: DashboardConfig) -> None:
        self._config = config
        self._start_ts = time.monotonic()
        self._epoch_offset = time.time() - self._start_ts
        self._metrics = MetricSampler()
        self._log_reader = EventLogReader(config.metrics_path) if config.metrics_path else None
        self._task_by_id: Dict[str, Dict[str, Any]] = {}
        self._task_order: Deque[str] = deque(maxlen=config.max_tasks)
        self._task_end_times: Dict[str, float] = {}
        self._task_palette = self._select_palette()
        self._task_colors: Dict[str, str] = {}
        self._workers: Dict[str, int] = {}
        self._saw_metrics_event = False
        self._proc: Optional[subprocess.Popen[str]] = None
        self._proc_status = "idle"
        self._workflow_start_ts: Optional[float] = None
        self._workflow_end_ts: Optional[float] = None
        self._task_stream_zero: Optional[float] = None
        self._status_div: Optional[Div] = None

        self.metrics_source = ColumnDataSource(
            data={
                "time_s": [],
                "mem_used_gb": [],
                "mem_total_gb": [],
                "cpu_percent": [],
                "io_read_mbps": [],
                "io_write_mbps": [],
                "runtime_s": [],
            }
        )
        self.task_stream_source = ColumnDataSource(
            data={
                "start": [],
                "duration_s": [],
                "duration_text": [],
                "worker": [],
                "y": [],
                "color": [],
                "name": [],
                "duration": [],
            }
        )
        self.flame_source = ColumnDataSource(
            data={"left": [], "right": [], "label": [], "color": [], "duration": []}
        )
        self._task_stream_plot_ref = None

    def _select_palette(self) -> ColorPalette:
        return list(Category20[20]) + list(Category10[10])

    def _color_for_task(self, name: str) -> str:
        if name not in self._task_colors:
            color = self._task_palette[len(self._task_colors) % len(self._task_palette)]
            self._task_colors[name] = color
        return self._task_colors[name]

    def _update_metrics(self) -> None:
        self._saw_metrics_event = False
        if self._log_reader:
            for event in self._log_reader.read_events():
                self._handle_event(event)
        if self._proc is not None and self._proc_status == "running":
            if self._proc.poll() is not None:
                self._proc_status = "finished"
                if self._workflow_start_ts is not None and self._workflow_end_ts is None:
                    self._workflow_end_ts = time.monotonic()
                    self._sample_metrics_once()
                if self._status_div is not None:
                    self._status_div.text = f"<b>Workflow:</b> {self._proc_status}"
        if not self._saw_metrics_event:
            if self._proc is not None and self._proc_status == "running":
                self._sample_metrics_once()
        self._refresh_task_stream()
        self._refresh_flamegraph()

    def _handle_event(self, event: Dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "task":
            self._handle_task_event(event)
        elif etype == "metrics":
            self._ingest_metrics(event)
            self._saw_metrics_event = True

    def _handle_task_event(self, event: Dict[str, Any]) -> None:
        normalized = self._normalize_task_event(event)
        task_id = str(normalized.get("id") or self._fallback_task_id(normalized))
        status = str(normalized.get("status", "complete"))
        normalized["id"] = task_id
        normalized["status"] = status
        existing = self._task_by_id.get(task_id)
        if existing is not None:
            existing_status = str(existing.get("status", "complete"))
            if existing_status in {"end", "error"} and status == "start":
                return
        if status == "start":
            if task_id not in self._task_by_id:
                self._task_order.append(task_id)
            self._task_by_id[task_id] = normalized
        elif status in {"end", "error"}:
            end_ts = float(normalized.get("end", normalized.get("ts", time.time())))
            self._task_end_times[task_id] = end_ts
            if task_id not in self._task_by_id:
                self._task_order.append(task_id)
                self._task_by_id[task_id] = normalized
            else:
                self._task_by_id[task_id].update(normalized)
        else:
            if task_id not in self._task_by_id:
                self._task_order.append(task_id)
            self._task_by_id[task_id] = normalized
        while len(self._task_order) > self._config.max_tasks:
            old_id = self._task_order.popleft()
            self._task_by_id.pop(old_id, None)

    def _normalize_task_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        start = float(event.get("start", event.get("ts", 0.0)))
        end = float(event.get("end", start))
        event = dict(event)
        event["start"] = self._normalize_time(start)
        event["end"] = self._normalize_time(end)
        return event

    def _fallback_task_id(self, event: Dict[str, Any]) -> str:
        return f"{event.get('stage','?')}:{event.get('template','?')}:{event.get('block','?')}:{event.get('start',0.0)}"

    def _normalize_time(self, ts: float) -> float:
        if ts < 1.0e9:
            return self._epoch_offset + ts
        return ts

    def _ingest_metrics(self, event: Dict[str, Any]) -> None:
        if self._proc is None or self._proc_status != "running":
            return
        runtime_s = 0.0
        if self._workflow_start_ts is not None:
            if self._workflow_end_ts is not None:
                runtime_s = self._workflow_end_ts - self._workflow_start_ts
            else:
                runtime_s = time.monotonic() - self._workflow_start_ts
        payload = {
            "time_s": [runtime_s],
            "mem_used_gb": [float(event.get("mem_used_gb", 0.0))],
            "mem_total_gb": [float(event.get("mem_total_gb", 0.0))],
            "cpu_percent": [float(event.get("cpu_percent", 0.0))],
            "io_read_mbps": [float(event.get("io_read_mbps", 0.0))],
            "io_write_mbps": [float(event.get("io_write_mbps", 0.0))],
            "runtime_s": [float(event.get("runtime_s", runtime_s))],
        }
        max_points = max(int(self._config.history_seconds * 1000 / self._config.update_interval_ms), 1)
        self.metrics_source.stream(payload, rollover=max_points)

    def _sample_metrics_once(self) -> None:
        metrics = self._metrics.sample()
        runtime_s = 0.0
        if self._workflow_start_ts is not None:
            if self._workflow_end_ts is not None:
                runtime_s = self._workflow_end_ts - self._workflow_start_ts
            else:
                runtime_s = time.monotonic() - self._workflow_start_ts
        new_data = {
            "time_s": [runtime_s],
            "mem_used_gb": [metrics["mem_used_gb"]],
            "mem_total_gb": [metrics["mem_total_gb"]],
            "cpu_percent": [metrics["cpu_percent"]],
            "io_read_mbps": [metrics["io_read_mbps"]],
            "io_write_mbps": [metrics["io_write_mbps"]],
            "runtime_s": [runtime_s],
        }
        max_points = max(int(self._config.history_seconds * 1000 / self._config.update_interval_ms), 1)
        self.metrics_source.stream(new_data, rollover=max_points)

    def _refresh_task_stream(self) -> None:
        window = time.time() - self._config.history_seconds
        now_epoch = time.time()
        centers: List[float] = []
        duration_s: List[float] = []
        duration_text: List[str] = []
        workers: List[str] = []
        ys: List[int] = []
        colors: List[str] = []
        names: List[str] = []
        durations: List[float] = []
        stale: List[str] = []
        zero = self._task_stream_zero or time.time()
        for task_id in list(self._task_order):
            event = self._task_by_id.get(task_id)
            if event is None:
                continue
            start = float(event.get("start", event.get("ts", 0.0)))
            end = float(event.get("end", start))
            cached_end = self._task_end_times.get(task_id)
            if cached_end is not None and cached_end > start:
                end = cached_end
            if str(event.get("status")) == "start" and end <= start:
                end = now_epoch
            if end < window:
                stale.append(task_id)
                continue
            worker = self._lane_key(event)
            if worker not in self._workers:
                self._workers[worker] = len(self._workers)
            y = self._workers[worker]
            name = str(event.get("name", "task"))
            color = self._color_for_task(name)
            duration = max(end - start, 0.0)
            start_rel = start - zero
            centers.append(start_rel + duration / 2.0)
            duration_s.append(duration)
            workers.append(worker)
            ys.append(y)
            colors.append(color)
            names.append(name)
            durations.append(duration)
            duration_text.append(f"{duration * 1000.0:0.1f} ms")
        for task_id in stale:
            self._task_by_id.pop(task_id, None)
            if task_id in self._task_order:
                self._task_order.remove(task_id)
        self.task_stream_source.data = {
            "start": centers,
            "duration_s": duration_s,
            "duration_text": duration_text,
            "worker": workers,
            "y": ys,
            "color": colors,
            "name": names,
            "duration": durations,
        }
        if self._task_stream_plot_ref is not None:
            upper = max(5, len(self._workers) + 1)
            self._task_stream_plot_ref.y_range.end = upper

    def _refresh_flamegraph(self) -> None:
        totals: Dict[str, float] = defaultdict(float)
        now_epoch = time.time()
        for task_id in list(self._task_order):
            event = self._task_by_id.get(task_id)
            if event is None:
                continue
            start = float(event.get("start", event.get("ts", 0.0)))
            end = float(event.get("end", start))
            if str(event.get("status")) == "start" and end <= start:
                end = now_epoch
            name = str(event.get("name", "task"))
            totals[name] += max(end - start, 0.0)
        lefts: List[float] = []
        rights: List[float] = []
        labels: List[str] = []
        colors: List[str] = []
        durations: List[float] = []
        cursor = 0.0
        for name, duration in sorted(totals.items(), key=lambda item: item[1], reverse=True):
            lefts.append(cursor)
            cursor += duration
            rights.append(cursor)
            labels.append(name)
            colors.append(self._color_for_task(name))
            durations.append(duration)
        self.flame_source.data = {
            "left": lefts,
            "right": rights,
            "label": labels,
            "color": colors,
            "duration": durations,
        }

    def make_document(self, doc) -> None:
        doc.title = self._config.title
        header = Div(text=f"<h2>{self._config.title}</h2>")
        kpi = Div(text="Runtime: 0.0s")
        controls = self._controls_panel()
        mem_plot = self._time_series_plot(
            "Memory (GB)",
            "mem_used_gb",
            "mem_total_gb",
            "mem_used_gb",
            "mem_total_gb",
        )
        cpu_plot = self._time_series_plot("CPU Usage (%)", "cpu_percent", None, "cpu_percent", None)
        io_plot = self._time_series_plot(
            "I/O (MB/s)",
            "io_read_mbps",
            "io_write_mbps",
            "io_read_mbps",
            "io_write_mbps",
        )
        task_stream_plot = self._task_stream_plot()
        flame_plot = self._flamegraph_plot()

        grid = gridplot(
            [
                [mem_plot, cpu_plot],
                [io_plot, None],
            ],
            sizing_mode="stretch_width",
        )

        doc.add_root(
            column(
                header,
                controls,
                kpi,
                grid,
                row(task_stream_plot, flame_plot, sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
            )
        )

        def _update_kpi() -> None:
            if self.metrics_source.data["runtime_s"]:
                runtime = self.metrics_source.data["runtime_s"][-1]
                kpi.text = f"<b>Runtime:</b> {runtime:0.1f}s"

        doc.add_periodic_callback(self._update_metrics, self._config.update_interval_ms)
        doc.add_periodic_callback(_update_kpi, self._config.update_interval_ms)

    def _controls_panel(self):
        if not self._config.run_command:
            return Div(text="")

        status = Div(text=f"<b>Workflow:</b> {self._proc_status}")
        self._status_div = status
        button = Button(label="Start workflow", button_type="success")
        threads_spinner = Spinner(
            title="Threads per locality",
            low=1,
            step=1,
            value=self._config.threads_per_locality or 1,
            width=160,
        )

        def _start() -> None:
            if self._proc is not None and self._proc.poll() is None:
                return
            self._proc = self._launch_workflow(int(threads_spinner.value))
            self._proc_status = "running" if self._proc is not None else "failed"
            if self._proc is not None:
                self._task_by_id.clear()
                self._task_order.clear()
                self._task_end_times.clear()
                self._workers.clear()
                self._workflow_start_ts = time.monotonic()
                self._workflow_end_ts = None
                self._task_stream_zero = self._workflow_start_ts + self._epoch_offset
                self._sample_metrics_once()
            status.text = f"<b>Workflow:</b> {self._proc_status}"

        button.on_click(_start)
        return row(button, threads_spinner, status, sizing_mode="stretch_width")

    def _launch_workflow(self, threads_per_locality: Optional[int] = None) -> Optional[subprocess.Popen[str]]:
        if not self._config.run_command:
            return None
        if threads_per_locality is not None and threads_per_locality < 1:
            raise ValueError("threads_per_locality must be >= 1")
        run_command = list(self._config.run_command)
        if threads_per_locality:
            if not self._threads_config_present(run_command):
                run_command = self._inject_threads(run_command, threads_per_locality)
        env = os.environ.copy()
        if self._config.metrics_path:
            env["KANGAROO_EVENT_LOG"] = str(self._config.metrics_path)
        if self._config.plan_path:
            env["KANGAROO_DASHBOARD_PLAN"] = str(self._config.plan_path)
        proc = subprocess.Popen(
            run_command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def _drain_output() -> None:
            if proc.stdout is None:
                return
            for line in proc.stdout:
                sys.stderr.write(f"[workflow] {line}")

        threading.Thread(target=_drain_output, daemon=True).start()
        return proc

    @staticmethod
    def _threads_config_present(argv: List[str]) -> bool:
        for idx, arg in enumerate(argv):
            if "hpx.os_threads" in arg:
                return True
            if arg == "--hpx-arg" and idx + 1 < len(argv):
                if "--hpx:threads" in argv[idx + 1]:
                    return True
        return False

    @staticmethod
    def _inject_threads(argv: List[str], threads: int) -> List[str]:
        insert_at = len(argv)
        if "--" in argv:
            insert_at = argv.index("--") + 1
        return [*argv[:insert_at], "--hpx-config", f"hpx.os_threads={threads}", *argv[insert_at:]]

    def _time_series_plot(
        self,
        title: str,
        field_a: str,
        field_b: Optional[str],
        legend_a: str,
        legend_b: Optional[str],
    ):
        p = figure(
            title=title,
            x_axis_label="Time since workflow start (s)",
            height=200,
            sizing_mode="stretch_width",
        )
        p.line("time_s", field_a, source=self.metrics_source, legend_label=legend_a, line_width=2)
        if field_b:
            p.line(
                "time_s",
                field_b,
                source=self.metrics_source,
                legend_label=legend_b or field_b,
                line_width=2,
                line_dash="dashed",
                color="orange",
            )
        p.legend.location = "top_left"
        p.add_tools(HoverTool(tooltips=[("value", f"@{field_a}")]))
        return p

    def _task_stream_plot(self):
        p = figure(
            title="Task Stream",
            x_axis_label="Time since workflow start (s)",
            height=280,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset",
        )
        p.y_range = Range1d(-1, 10)
        p.rect(
            x="start",
            y="y",
            width="duration_s",
            height=0.8,
            source=self.task_stream_source,
            color="color",
            line_color="color",
        )
        p.add_tools(
            HoverTool(
                tooltips=[
                    ("task", "@name"),
                    ("worker", "@worker"),
                    ("duration", "@duration_text"),
                ]
            )
        )
        self._task_stream_plot_ref = p
        return p

    def _flamegraph_plot(self):
        p = figure(
            title="Flamegraph (total durations)",
            height=280,
            sizing_mode="stretch_width",
            tools="pan,wheel_zoom,box_zoom,reset",
            x_axis_label="Total task time (s)",
        )
        p.y_range = Range1d(0, 1)
        p.quad(
            left="left",
            right="right",
            bottom=0,
            top=1,
            source=self.flame_source,
            color="color",
            line_color="white",
        )
        p.add_tools(HoverTool(tooltips=[("task", "@label"), ("duration", "@duration")]))
        p.yaxis.visible = False
        return p

    @staticmethod
    def _lane_key(event: Dict[str, Any]) -> str:
        worker = event.get("worker")
        locality = event.get("locality")
        if worker is None and locality is None:
            return "worker-0"
        if worker is None:
            return f"loc-{locality}"
        if locality is None:
            return str(worker)
        return f"loc-{locality}/{worker}"


def serve_dashboard(config: DashboardConfig) -> None:
    if config.run_command:
        temp_dir = Path(tempfile.mkdtemp(prefix="kangaroo-dashboard-"))
        if config.metrics_path is None:
            config.metrics_path = temp_dir / "events.jsonl"
        if config.plan_path is None:
            config.plan_path = temp_dir / "plan.json"
        print(f"[dashboard] workflow log: {config.metrics_path}")
        print(f"[dashboard] workflow plan: {config.plan_path}")
        print(f"[dashboard] ready to run: {' '.join(config.run_command)}")

    def _bkapp(doc) -> None:
        app = DashboardApp(config)
        app.make_document(doc)

    origins = config.allow_websocket_origin
    if origins is None:
        origins = [f"localhost:{config.port}", f"127.0.0.1:{config.port}"]
    server = Server({"/": _bkapp}, port=config.port, allow_websocket_origin=origins)
    server.start()
    print(f"[dashboard] http://localhost:{config.port}/")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


def parse_args(argv: Optional[List[str]] = None) -> DashboardConfig:
    parser = argparse.ArgumentParser(description="Run the Kangaroo Bokeh dashboard.")
    parser.add_argument("--metrics", type=Path, help="Path to JSONL metrics/event log")
    parser.add_argument("--plan", type=Path, help="Path to plan JSON for the DAG")
    parser.add_argument("--run", type=str, help="Python script to run alongside the dashboard")
    parser.add_argument("--port", type=int, default=5006, help="Port for the Bokeh server")
    parser.add_argument("--interval-ms", type=int, default=500, help="Update interval in milliseconds")
    parser.add_argument("--history-seconds", type=int, default=300, help="History window for charts")
    parser.add_argument("--title", type=str, default="Kangaroo Dashboard", help="Dashboard title")
    args, remainder = parser.parse_known_args(argv)
    run_cmd = None
    if args.run:
        if remainder and remainder[0] == "--":
            remainder = remainder[1:]
        run_cmd = [sys.executable, args.run, *remainder]
    return DashboardConfig(
        metrics_path=args.metrics,
        plan_path=args.plan,
        run_command=run_cmd,
        update_interval_ms=args.interval_ms,
        history_seconds=args.history_seconds,
        port=args.port,
        title=args.title,
    )


def main(argv: Optional[List[str]] = None) -> None:
    config = parse_args(argv)
    serve_dashboard(config)


if __name__ == "__main__":
    main()
