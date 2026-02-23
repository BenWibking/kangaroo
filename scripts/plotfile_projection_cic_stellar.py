#!/usr/bin/env python3
"""Project stellar particle density with native-grid CIC deposition."""

from __future__ import annotations

import argparse
import json
import numpy as np
import tempfile
import threading
import time
from pathlib import Path

from analysis import Runtime  # noqa: E402
from analysis.dataset import open_dataset  # noqa: E402
from analysis.pipeline import pipeline  # noqa: E402
from analysis.runtime import plan_to_dict  # noqa: E402


def _parse_bounds(bounds: str) -> tuple[float, float]:
    try:
        b0, b1 = (float(v.strip()) for v in bounds.split(","))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("--axis-bounds must be in min,max format") from exc
    if b0 == b1:
        raise ValueError("--axis-bounds must specify two distinct values")
    return b0, b1


def _count_plan_tasks(plan_dict: dict) -> int:
    total = 0
    for stage in plan_dict.get("stages", []):
        for tmpl in stage.get("templates", []):
            domain = tmpl.get("domain") or {}
            blocks = domain.get("blocks")
            if isinstance(blocks, list) and blocks:
                total += len(blocks)
            else:
                total += 1
    return total


def _fallback_task_id(event: dict) -> str:
    return (
        f"{event.get('stage','?')}:{event.get('template','?')}:"
        f"{event.get('block','?')}:{event.get('start',0.0)}"
    )


def _task_progress_monitor(log_path: Path, total_tasks: int, stop_event: threading.Event) -> None:
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None

    seen_done: set[str] = set()
    offset = 0
    bar = None
    if tqdm is not None:
        bar = tqdm(total=total_tasks or None, desc="pipeline tasks", unit="task")

    def _update_bar() -> None:
        if bar is None:
            return
        delta = len(seen_done) - bar.n
        if delta > 0:
            bar.update(delta)

    try:
        while True:
            progressed = False
            try:
                with log_path.open("r", encoding="utf-8") as handle:
                    handle.seek(offset)
                    while True:
                        pos = handle.tell()
                        line = handle.readline()
                        if not line:
                            offset = pos
                            break
                        offset = handle.tell()
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        if event.get("type") != "task":
                            continue
                        status = str(event.get("status", ""))
                        if status not in {"end", "error"}:
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
                # Drain any final buffered events written just before/after run completion.
                time.sleep(0.05)
                try:
                    with log_path.open("r", encoding="utf-8") as handle:
                        handle.seek(offset)
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(event, dict) or event.get("type") != "task":
                                continue
                            status = str(event.get("status", ""))
                            if status not in {"end", "error"}:
                                continue
                            task_id = str(event.get("id") or _fallback_task_id(event))
                            seen_done.add(task_id)
                except FileNotFoundError:
                    pass
                _update_bar()
                break

            time.sleep(0.1)
    finally:
        if bar is not None:
            bar.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Deposit stellar particles with CIC on native 2D AMR grids, then project."
    )
    p.add_argument("plotfile")
    p.add_argument("--axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--axis-bounds")
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--output")
    p.add_argument("--resolution")
    p.add_argument(
        "--mass-max",
        type=float,
        help="Only accumulate particles with mass <= this threshold [Msun].",
    )
    p.add_argument(
        "--particle-type",
        "--particles",
        dest="particle_type",
        choices=("CIC_particles", "StochasticStellarPop_particles"),
        default="StochasticStellarPop_particles",
        help="Particle species to deposit.",
    )
    a = p.parse_args()

    rt = Runtime.from_parsed_args(a, unknown_args=[])

    base_level = 0
    ds = open_dataset(a.plotfile, level=base_level, runtime=rt)
    runmeta = ds.metadata_bundle().runmeta

    view = ds.plane_geometry(
        axis=a.axis,
        level=base_level,
        zoom=a.zoom,
        resolution=a.resolution,
    )
    rect = view["rect"]
    res = view["resolution"]
    labels = view["labels"]
    plane = view["plane"]
    axis_bounds = _parse_bounds(a.axis_bounds) if a.axis_bounds else view["axis_bounds"]

    particle_types = ds.list_particle_types()
    if a.particle_type not in particle_types:
        available = ", ".join(particle_types) if particle_types else "<none>"
        raise RuntimeError(
            f"particle type '{a.particle_type}' not found in plotfile; available: {available}"
        )

    mass_max = None
    if a.mass_max is not None:
        if a.mass_max <= 0.0:
            raise ValueError("--mass-max must be positive when specified")
        msun_g = 1.98847e33
        mass_max = a.mass_max * msun_g

    pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    out = pipe.particle_cic_projection(
        particle_type=a.particle_type,
        axis=a.axis,
        axis_bounds=axis_bounds,
        rect=rect,
        resolution=res,
        mass_max=mass_max,
        out="stellar_projection",
    )
    plan = pipe.plan()

    # Stream task events to a temporary JSONL file and drive a tqdm-like progress bar
    # from terminal task completion events while the runtime executes.
    plan_task_total = _count_plan_tasks(plan_to_dict(plan))
    stop_progress = threading.Event()
    progress_thread = None
    with tempfile.TemporaryDirectory(prefix="kangaroo-events-") as tmpdir:
        event_log = Path(tmpdir) / "events.jsonl"
        print(f"event log: {event_log}", flush=True)
        rt.set_event_log_path(str(event_log))
        progress_thread = threading.Thread(
            target=_task_progress_monitor,
            args=(event_log, plan_task_total, stop_progress),
            daemon=True,
        )
        progress_thread.start()
        try:
            rt.run(plan, runmeta=runmeta, dataset=ds)
        finally:
            stop_progress.set()
            progress_thread.join(timeout=2.0)

    arr = rt.get_task_chunk_array(
        step=0,
        level=base_level,
        field=out.field,
        version=0,
        block=0,
        shape=res,
        bytes_per_value=8,
        dataset=ds,
    )

    nx, ny = res
    pixel_area = abs((rect[2] - rect[0]) / nx) * abs((rect[3] - rect[1]) / ny)
    arr = arr / pixel_area

    msun_g = 1.98847e33
    pc_cm = 3.0856775814913673e18
    cgs_surface_density_to_msun_pc2 = (pc_cm**2) / msun_g
    arr = arr * cgs_surface_density_to_msun_pc2

    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    fig, axp = plt.subplots(figsize=(6, 5))
    kpc = 1.0e3 * pc_cm
    extent = (rect[0] / kpc, rect[2] / kpc, rect[1] / kpc, rect[3] / kpc)
    masked = np.ma.masked_invalid(arr)
    vmax = np.nanmax(arr)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    norm = SymLogNorm(linthresh=1.0e-2, vmin=0.0, vmax=vmax)
    im = axp.imshow(masked, origin="lower", cmap="viridis", extent=extent, norm=norm)
    axp.set_title(
        f"CIC stellar projection [{a.particle_type}] ({plane} plane)"
    )
    axp.set_xlabel(f"{labels[0]} [kpc]")
    axp.set_ylabel(f"{labels[1]} [kpc]")
    fig.colorbar(im, ax=axp, label=f"{a.particle_type} [Msun pc^-2] (symlog)")
    fig.tight_layout()
    fig.savefig(a.output, dpi=150) if a.output else plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
