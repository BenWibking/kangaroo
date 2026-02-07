Kangaroo prototype runtime

This repo contains a prototype implementation of the distributed task runtime described in `PLAN.md`.
The Python side provides a DSL to author operators and plans, and the C++ side (HPX-based) provides
execution, data services, adjacency, and kernel registry scaffolding. Output buffers are allocated
by the DataService, with optional per-output sizes provided in the plan.

Status
- Python DSL: implemented in `analysis/`
- C++ runtime: scaffolded in `cpp/` (HPX target with local adjacency/data services)
- Msgpack decoding: wired via msgpack-c (required)

Python quickstart (prototype)
```
from analysis import Runtime, Plan
from analysis.ctx import LoweringContext
from analysis.ops import VorticityMag
from analysis.runmeta import RunMeta, StepMeta, LevelMeta, LevelGeom, BlockBox
from analysis.dataset import open_dataset

runmeta = RunMeta(steps=[
    StepMeta(step=0, levels=[
        LevelMeta(
            geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=1),
            boxes=[BlockBox((0, 0, 0), (7, 7, 7)), BlockBox((8, 0, 0), (15, 7, 7))],
        )
    ])
])

rt = Runtime()

ds = open_dataset("memory://example", runmeta=runmeta, step=0, level=0, runtime=rt)
vel = ds.field_id("vel")

op = VorticityMag(vel_field=vel)
ctx = LoweringContext(runtime=rt._rt, runmeta=runmeta._h, dataset=ds)
plan = Plan(stages=op.lower(ctx))

rt.run(plan, runmeta=runmeta, dataset=ds)
```

Output allocation
- Outputs are allocated by the DataService.
- In Python, supply `output_bytes` in `map_blocks(...)` to request sizes per output.
- If omitted, outputs are allocated with size 0 (kernels may resize if desired).

Example:
```
s1.map_blocks(
    name="gradU",
    kernel="gradU_stencil",
    domain=dom,
    inputs=[FieldRef(self.vel_field)],
    outputs=[gradU],
    output_bytes=[1024],
    deps={"kind": "FaceNeighbors", "width": 1, "faces": [1, 1, 1, 1, 1, 1]},
    params={"order": 2},
)
```

C++ build (HPX required)
- Configure and build in `cpp/` using CMake. The runtime builds as a static library and installs the `_core` module plus `analysis/` into site-packages.
- msgpack-c is required and fetched automatically if not found.
- Python bindings are required and build via nanobind.
- Pixi environment (recommended for HPX):
  - Minimal (Python + extension):
    - `pixi install`
    - `pixi run install`
    - `pixi run test`
  - C++ dev loop (optional):
    - `pixi run configure`
    - `pixi run build`
  - `scripts/utils/detect_hpx.sh` will find HPX inside the Pixi/conda prefix automatically.
  - Build deps (`scikit-build-core`, `nanobind`) are provided by Pixi; `install` uses `--no-build-isolation`.
  - The C++ msgpack headers come from `msgpack-cxx` in the Pixi env.
  - Run Python with `pixi run python ...` so it picks up the Pixi HPX + extension module.
  - Example (set HPX worker threads): `pixi run python scripts/plotfile_slice.py /path/to/plotfile --var density --hpx:threads=8`
  - Note: the conda-forge HPX package used by Pixi may be built without networking enabled, which prevents multi-locality runs. For multi-rank execution, build HPX with `-DHPX_WITH_NETWORKING=On` and point `scripts/utils/detect_hpx.sh` at that install.
HPX autodetect helper
- Use `scripts/utils/detect_hpx.sh` to verify HPX resolution inside the Pixi env:
  - `HPX_DIR=$(scripts/utils/detect_hpx.sh)`

Kangaroo Dashboard (Bokeh)
- Run the dashboard (local system metrics only):
  - `python scripts/kangaroo_dashboard.py`
- Note: the dashboard requires Python < 3.14 (NumPy/Bokeh crash on 3.14 in current env).
- Launch a workflow and monitor it (event log + DAG auto-wired):
  - `python scripts/kangaroo_dashboard.py --run scripts/plotfile_slice_operator_demo.py -- /path/to/plotfile`
- Provide a JSONL event log for task stream/flamegraph/metrics:
  - `python scripts/kangaroo_dashboard.py --metrics path/to/events.jsonl`
- Provide a plan JSON to render a DAG (e.g. `analysis.runtime.plan_to_dict` output):
  - `python scripts/kangaroo_dashboard.py --plan path/to/plan.json`

Event log schema (one JSON object per line):
```
{"type": "metrics", "mem_used_gb": 12.3, "mem_total_gb": 64.0, "cpu_percent": 80.1,
 "io_read_mbps": 120.0, "io_write_mbps": 55.0, "runtime_s": 42.5}
{"type": "task", "id": "1:0:0:3", "status": "start", "name": "plotfile_load_b3",
 "worker": "rank-0", "start": 1700000000.0, "end": 1700000000.0}
{"type": "task", "id": "1:0:0:3", "status": "end", "name": "plotfile_load_b3",
 "worker": "rank-0", "start": 1700000000.0, "end": 1700000001.2}
{"type": "dag", "nodes": [{"id": 0, "name": "stage-a"}], "edges": [[0, 1]]}
```
