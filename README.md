# Kangaroo [experimental]

![Kangaroo logo](docs/src/logo.webp)

Kangaroo is a prototype distributed analysis runtime with:
- a lazy scientific Python API in `kangaroo/`
- compatibility and advanced low-level interfaces in `analysis/`
- an HPX/C++ runtime in `cpp/`
- plotfile-focused workflows in `scripts/plotfile_*.py`

Use the `kangaroo` CLI for the batch workflows below. For custom Python
analyses, `import kangaroo as kr`, open a dataset, build expressions from named
fields, and call `compute()` at the desired local materialization boundary.
Explicit clients and the advanced IR/runtime modules remain available for launchers, kernel development, and direct chunk access.

## Requirements

- Python 3.10-3.13
- CMake + C++ toolchain (installed by Pixi)
- On macOS, Apple Command Line Tools (`xcode-select --install`) for the system SDK
- HPX (Pixi-managed by default)

## Setup and Build

If you do not have [Pixi](https://pixi.sh/) installed, install it on Linux or
macOS with the [official installer](https://pixi.sh/latest/installation/):

```bash
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
pixi --version
```

The installer adds Pixi to your shell startup configuration. The `export` above
makes it available in the current shell; new terminals pick it up automatically.

Then, from the Kangaroo repository root:

The `pixi-hpx` environment installs HPX 1.11.0 and the matching compiler,
Boost, HDF5, and Python dependencies. No separate HPX installation is needed.

```bash
pixi install -e pixi-hpx
pixi run -e pixi-hpx configure
pixi run -e pixi-hpx build
pixi run -e pixi-hpx install
```

These tasks configure and build in `.pixi/build-pixi-hpx`, then install Kangaroo
into `.pixi/envs/pixi-hpx`. Use this same environment for Python API commands.
The plain `pixi install` command installs the separate `default`
environment, which does not include HPX.

Verify the installed Python API:

```bash
pixi run -e pixi-hpx python -c "import kangaroo; from analysis import _core, _plotfile; print('Kangaroo ready')"
```

Run tests:

```bash
pixi run -e pixi-hpx test
```

If `analysis._core` fails to import, install the extension module with `pixi run -e pixi-hpx install`.

On macOS, if the linker reports `libSystem.tbd: unknown architecture` with a
newer Apple SDK, select a compatible installed SDK before configuring. For
example, with the macOS 26.5 SDK installed:

```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
export KANGAROO_BUILD_DIR=.pixi/build-pixi-hpx-sdk26
pixi run -e pixi-hpx install
```

Use a fresh build directory when changing SDKs or compiler environments.

## Documentation

The HTML documentation is built with [mdBook](https://rust-lang.github.io/mdBook/):

```bash
mdbook serve docs --open
```

Use `mdbook build docs` for a one-off build. The generated site is written to
`docs/book/` and is published at <https://benwibking.github.io/kangaroo/> after
changes land on `main`.

## Plotfile Workflows with the CLI

Add the checkout's launcher to `PATH` from the repository root (repeat this in
new shells, or add the absolute path to your shell configuration):

```bash
export PATH="$PWD/scripts:$PATH"
kangaroo --help
```

The CLI submits the existing analysis jobs through Slurm. Submission requires
`sbatch` and the cluster modules, Spack HPX/MPI runtime, and paths configured in
`batch/*.submit`. The Pixi setup above builds the local Python API; the batch
scripts use their cluster-specific runtime.

Run these examples from the repository root so submission outputs and plotting
inputs share `results/<campaign>/<plotfile>/<analysis>/`. Set a campaign once:

```bash
export KANGAROO_CAMPAIGN=noCGM_0.1uG_Phase0
```

Alternatively, supply `--campaign NAME` on each command.

### Submit analyses

Each command takes one full plotfile directory. Preview a submission with
`--dry-run`, and pass Slurm options after `--`:

```bash
kangaroo toomre-q /path/to/plt000100
kangaroo enclosed-sfr /path/to/plt000100 -- --time=01:00:00
kangaroo cylindrical-flux /path/to/plt000100 --dry-run
kangaroo cylindrical-flux /path/to/plt000100
kangaroo cylindrical-radial-flux /path/to/plt000100
kangaroo spherical-flux /path/to/plt000100
kangaroo projection /path/to/plt000100
kangaroo jobs
```

`kangaroo jobs` lists pending and running Kangaroo jobs with their job ID,
name, state, elapsed time, node count, and assigned nodes or pending reason.
Relative plotfile paths are resolved before submission; jobs run from the
repository root. Outputs and Slurm logs go under `results/`. Existing batch
settings control scientific parameters and overwrite behavior. For example:

```bash
R_MIN_KPC=0.5 R_MAX_KPC=16 Z_MAX_KPC=4 kangaroo toomre-q /path/to/plt000100
```

`toomre-q` writes a shared-axis PNG and a CSV with thermal + magnetic,
thermal + resolved radial turbulence, and thermal + resolved radial turbulence
+ magnetic gas stability measures. Its batch defaults are `0 <= R <= 17.5 kpc`,
`0.25 kpc` bins, and `|z| <= 1 kpc`. Full plotfiles must contain momentum,
internal energy, magnetic fields, and `gpot`.

`projection` uses the stellar yz-map settings in `batch/make_projection.submit`
and writes `stellar_projection_yz.png`.

### Plot completed results

The plotting commands currently use the separate `default` Pixi environment
for NumPy and Matplotlib. Install it before plotting:

```bash
pixi install
```

After the corresponding analysis jobs finish:

```bash
kangaroo plot-cylindrical-flux plt000100
kangaroo plot-cylindrical-radial-flux /path/to/plt000100
kangaroo plot-spherical-flux plt000100
kangaroo plot-radial-flux-to-sfr plt000100
kangaroo plot-wind-disk-angular-momentum-flux plt000100
```

Plot commands accept an existing plotfile directory or its basename and read
`$PWD/results/<campaign>/<plotfile>/`. Single-analysis plots are written beside
their JSON. The wind/disk comparison is written in the plotfile's result
directory. The flux-to-SFR plot requires both cylindrical radial flux and
enclosed-SFR outputs; the wind/disk comparison also requires cylindrical wind
flux and Toomre-Q outputs.

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
KANGAROO_PERFETTO_TRACE=run.pftrace kangaroo toomre-q /path/to/plt000100
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
