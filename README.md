# Kangaroo [experimental]

![image](logo.png)

Kangaroo is a prototype distributed analysis runtime with:
- a Python API/DSL in `analysis/`
- an HPX/C++ runtime in `cpp/`
- plotfile-focused workflows in `scripts/plotfile_*.py`

The current recommended usage pattern is:
1. create a `Runtime` (typically via `Runtime.from_parsed_args(...)`),
2. open a dataset with `open_dataset(...)`,
3. derive geometry with dataset helpers (`metadata_bundle`, `resolve_field`, `plane_geometry`),
4. build operations with the imperative pipeline API (`pipeline(...).uniform_slice(...)`, `uniform_projection(...)`),
5. run and fetch output bytes/arrays via `Runtime`.

**Please note: pull requests are not accepted for this repository, and will be automatically closed.**

## Requirements

- Python 3.10-3.13
- CMake + C++ toolchain
- HPX (Pixi-managed by default)

## Setup and Build

Using [Pixi](https://prefix.dev/) (recommended):

```bash
pixi install
pixi run configure
pixi run build
pixi run install
```

Run tests:

```bash
pixi run test
```

If `analysis._core` fails to import, install the extension module with `pixi run install`.

## Plotfile Scripts (Current Workflows)

All examples below assume:

```bash
pixi run python ...
```

### 1) Uniform slice with Kangaroo

```bash
pixi run python scripts/plotfile_slice.py /path/to/plotfile \
  --var density \
  --axis z \
  --coord 0.0 \
  --zoom 1.0 \
  --resolution 1024,1024 \
  --output slice.png
```

Notes:
- `--var` defaults to the first dataset variable if omitted.
- `--axis` is one of `x|y|z`.
- `--resolution` is `Nx,Ny`.
- unknown CLI args are forwarded to HPX through `Runtime.from_parsed_args(...)`, so flags like `--hpx:threads=8` can be passed directly.

### 2) Uniform projection with Kangaroo

```bash
pixi run python scripts/plotfile_projection.py /path/to/plotfile \
  --var density \
  --axis z \
  --axis-bounds 0.0,3.0e22 \
  --zoom 1.0 \
  --resolution 1024,1024 \
  --output projection.png
```

Notes:
- projection currently requires AMR cell-average semantics (`--amr-cell-average`, enabled by default).
- byte width is inferred from dataset chunks via `Dataset.infer_bytes_per_value(...)`.

### 3) Per-FAB min/max inspection

```bash
pixi run python scripts/plotfile_fab_minmax.py /path/to/plotfile --level 0 --fab 0
```

This uses `analysis.PlotfileReader` directly to inspect component-wise min/max values per FAB.

### 4) yt reference/benchmark slice path

```bash
pixi run python scripts/plotfile_slice_yt.py /path/to/plotfile \
  --var density \
  --axis z \
  --resolution 1024,1024
```

This is useful for comparison/benchmarking against Kangaroo slice behavior.

## Pipeline API Example (Recommended)

```python
from analysis import Runtime
from analysis.dataset import open_dataset
from analysis.pipeline import pipeline

rt = Runtime()
ds = open_dataset("/path/to/plotfile", runtime=rt)

metadata = ds.metadata_bundle()
runmeta = metadata.runmeta
comp, field_id, _ = ds.resolve_field("density")

view = ds.plane_geometry(axis="z", level=0, coord=None, zoom=1.0, resolution="512,512")

pipe = pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
out = pipe.uniform_slice(
    field=pipe.field(field_id),
    axis="z",
    coord=view["coord"],
    rect=view["rect"],
    resolution=view["resolution"],
    out="slice",
)

rt.run(pipe.plan(), runmeta=runmeta, dataset=ds)
arr = rt.get_task_chunk_array(
    step=0,
    level=0,
    field=out.field,
    version=0,
    block=0,
    shape=view["resolution"],
    dataset=ds,
)
```

To forward HPX flags from CLI scripts, use `argparse.parse_known_args()` and pass unknown args into `Runtime.from_parsed_args(...)` (as done by `scripts/plotfile_slice.py` and `scripts/plotfile_projection.py`).

## Histogram API

The pipeline also supports global histogram reductions:

```python
from analysis import cdf_from_histogram

hist1 = pipe.histogram1d(
    pipe.field("density"),
    hist_range=(1.0e-30, 1.0e-20),
    bins=128,
    out="density_hist",
)

hist2 = pipe.histogram2d(
    pipe.field("density"),
    pipe.field("temperature"),
    x_range=(1.0e-30, 1.0e-20),
    y_range=(1.0, 1.0e8),
    bins=(256, 256),
    weights=pipe.field("cell_mass"),
    out="phase_hist",
)

pipe.run()

h1 = rt.get_task_chunk_array(
    step=ds.step,
    level=ds.level,
    field=hist1.counts.field,
    block=0,
    shape=(hist1.bins,),
    dtype=float,
    dataset=ds,
)
cdf = cdf_from_histogram(h1)
```

Histogram accumulation is AMR-aware and uses global graph reductions across all blocks/levels.

## Data Sources

`open_dataset(...)` resolves these forms:
- AMReX plotfile directory paths (auto-resolved to `amrex://...`)
- `amrex://...`
- `openpmd://...`
- `parthenon://...` (including `.phdf/.h5/.hdf5` files)
- `memory://...` (for synthetic/in-memory testing)

## Dashboard

Run the local dashboard:

```bash
pixi run python scripts/kangaroo_dashboard.py
```

Run a workflow under the dashboard:

```bash
pixi run python scripts/kangaroo_dashboard.py \
  --run scripts/plotfile_slice.py -- /path/to/plotfile --var density
```

## Development Notes

- Python tests live in `tests/` and run with `pytest -q`.
- Main implementation areas:
  - Python API: `analysis/`
  - runtime/bindings: `cpp/`
- Additional design context: `PLAN.md` and `TRACKED_GAPS.md`.
