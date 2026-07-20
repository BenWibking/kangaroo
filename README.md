# Kangaroo [experimental]

![Kangaroo logo](docs/src/logo.webp)

Kangaroo is a prototype distributed analysis runtime with:
- a lazy scientific Python API in `kangaroo/`
- compatibility and advanced low-level interfaces in `analysis/`
- an HPX/C++ runtime in `cpp/`
- plotfile-focused workflows in `scripts/plotfile_*.py`

The recommended workflow is to `import kangaroo as kr`, open a dataset, build
expressions from named fields, and call `compute()` only at the desired local
materialization boundary. Explicit clients and the advanced IR/runtime modules
remain available for launchers, kernel development, and direct chunk access.

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

## Documentation

The HTML documentation is built with [mdBook](https://rust-lang.github.io/mdBook/):

```bash
mdbook serve docs --open
```

Use `mdbook build docs` for a one-off build. The generated site is written to
`docs/book/` and is published at <https://benwibking.github.io/kangaroo/> after
changes land on `main`.

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
- projection always uses AMR cell-average semantics.
- chunk dtype, shape, and physical strides travel with each chunk; projection output is `float64`.
- the high-level result owns its materialization coordinates and returns the typed array from `compute()`.

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

### 5) Gas Toomre-Q radial profiles

```bash
pixi run python scripts/plotfile_toomre_q.py /path/to/run/outputs \
  --output-dir /path/to/toomre_q
```

This workflow reads either one full 3D Quokka plotfile or the immediate
`plt*` children of an output directory. It writes one shared-axis PNG and one
CSV per plotfile for three gas stability measures:

- thermal + magnetic support,
- thermal + resolved radial turbulence,
- thermal + resolved radial turbulence + magnetic support.

The defaults target the MW_Gen1 Phase0 disk: `0.5 <= R <= 16 kpc` in
`0.25 kpc` bins and `|z| <= 4 kpc`. The epicyclic frequency is derived from
the gas-mass-weighted annular radial gradient of `gpot`. Full plotfiles are
required because the configured Quokka projection products do not contain the
momentum, internal energy, magnetic-field, and potential fields needed by the
calculation. Run with `--help` for field overrides, alternate geometry, and HPX
options.

## Lazy Python API Example (Recommended)

```python
import kangaroo as kr

ds = kr.open_dataset("/path/to/plotfile")
image = ds["density"].slice(axis="z", resolution=(512, 512))
array = image.compute()
```

Use `kr.compute(image, histogram, progress=True)` to materialize multiple outputs
through one shared graph execution. Unbounded AMR fields return an
`AMRChunkedArray` whose blocks retain their level, box, and geometry; use
`iter_chunks()`, `slice()`, or `project()` rather than concatenating the
hierarchy. Particle arrays remain linearly chunked and support explicit dense
gathering with `compute(gather=True, max_bytes=...)`.

For explicit runtime configuration, construct `kr.Client(hpx_args=[...])` or use
`kr.Client.from_parsed_args(...)`, then call `client.open_dataset(...)`.

## Histogram API

Lazy arrays also support global histogram reductions:

```python
hist1 = ds["density"].histogram(
    bins=128,
    range=(1.0e-30, 1.0e-20),
)

hist2 = ds["density"].histogram2d(
    ds["temperature"],
    bins=(256, 256),
    range=((1.0e-30, 1.0e-20), (1.0, 1.0e8)),
    weights=ds["cell_mass"],
)

hist1_result, hist2_result = kr.compute(hist1, hist2)
```

Histogram accumulation is AMR-aware and uses global graph reductions across all blocks/levels.

## Data Sources

`open_dataset(...)` resolves these forms:
- AMReX plotfile directory paths (auto-resolved to `amrex://...`)
- `amrex://...`
- `openpmd://...`
- `parthenon://...` (including `.phdf/.h5/.hdf5` files)
- `memory://...` (for synthetic/in-memory testing)

## Tracing (Perfetto UI)

Kangaroo can emit Perfetto traces for runtime execution. Set `KANGAROO_PERFETTO_TRACE`
to a trace output path before running your workflow, then open the generated trace in
Perfetto UI (`https://ui.perfetto.dev`).

```bash
KANGAROO_PERFETTO_TRACE=run.pftrace pixi run python scripts/plotfile_slice.py \
  /path/to/plotfile --var density
```

In distributed runs, each locality writes its own trace file (for example
`run.loc000.pftrace`, `run.loc001.pftrace`).

## Development Notes

- Python tests live in `tests/` and run with `pytest -q`.
- Main implementation areas:
  - recommended Python API: `kangaroo/`
  - compatibility and low-level Python API: `analysis/`
  - runtime/bindings: `cpp/`
- Additional design context: `PLAN.md` and `TRACKED_GAPS.md`.
