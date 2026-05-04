# Kangaroo vs yt Performance Report

Date: February 7, 2026

## Scope
This report summarizes measured performance comparisons between:
- `scripts/plotfile_slice.py` (Kangaroo)
- `scripts/plotfile_slice_yt.py` (yt equivalent)

Dataset and geometry used for comparisons:
- Dataset: `../amrex-visit-plugin/example_data/DiskGalaxy/plt0000020`
- Slice settings: `--resolution 1000,1000 --coord 0.0 --zoom 20 --axis x`
- Field for apples-to-apples runs: `--var gasDensity`

## Important Notes
- Early runs included first-run Matplotlib/font cache overhead and were much slower.
- Final comparison numbers below come from warm-cache runs.
- For no-plot benchmarks, `--no-plot` was used to isolate compute/data-path cost.
- MPI launcher was not initially available; `mpich` was added to the pixi environment.
- All final benchmark runs were executed sequentially (no overlap).

## Commands Used
Kangaroo, 1 thread, no-plot:
```bash
env MPLCONFIGDIR=/tmp/mplconfig pixi run scripts/plotfile_slice.py \
  ../amrex-visit-plugin/example_data/DiskGalaxy/plt0000020 \
  --var gasDensity --resolution 1000,1000 --coord 0.0 --zoom 20 --axis x --no-plot
```

Kangaroo, 8 threads, no-plot:
```bash
env MPLCONFIGDIR=/tmp/mplconfig pixi run scripts/plotfile_slice.py \
  ../amrex-visit-plugin/example_data/DiskGalaxy/plt0000020 \
  --var gasDensity --resolution 1000,1000 --coord 0.0 --zoom 20 --axis x \
  --no-plot --hpx-config hpx.os_threads=8
```

yt, 1 rank, no-plot:
```bash
env MPLCONFIGDIR=/tmp/mplconfig pixi run scripts/plotfile_slice_yt.py \
  ../amrex-visit-plugin/example_data/DiskGalaxy/plt0000020 \
  --var gasDensity --resolution 1000,1000 --coord 0.0 --zoom 20 --axis x --no-plot
```

yt, 8 ranks, no-plot:
```bash
env MPLCONFIGDIR=/tmp/mplconfig pixi run mpiexec -n 8 python3 scripts/plotfile_slice_yt.py \
  ../amrex-visit-plugin/example_data/DiskGalaxy/plt0000020 \
  --var gasDensity --resolution 1000,1000 --coord 0.0 --zoom 20 --axis x --no-plot
```

## Measured Results
### Baseline exact user command (before warm cache)
| Workflow | Command shape | Real time |
|---|---|---:|
| Kangaroo | user-provided command (with plotting) | 15.62 s |

This run included first-time Matplotlib/font-cache setup and is not representative for steady-state.

### Warm-cache with plotting, 1 process/rank
| Workflow | Real time |
|---|---:|
| Kangaroo (`--var gasDensity`) | 0.96 s |
| yt (`--var gasDensity`) | 1.37 s |

Relative result:
- Kangaroo was about `1.43x` faster (`1.37 / 0.96`).

### Warm-cache no-plot, 1 thread/rank
| Workflow | Real time |
|---|---:|
| Kangaroo (HPX 1 thread) | 0.67 s |
| yt (MPI 1 rank) | 1.05 s |

Relative result:
- Kangaroo was about `1.57x` faster (`1.05 / 0.67`).

### Warm-cache no-plot, parallel settings requested
| Workflow | Parallel setting | Real time |
|---|---|---:|
| Kangaroo | HPX `hpx.os_threads=8` | 0.55 s |
| yt | MPI `-n 8` | 3.16 s |

Relative result:
- Kangaroo (8 threads) was about `5.75x` faster than yt (8 ranks) (`3.16 / 0.55`).

## Scaling Comparisons
### Kangaroo: 1 thread vs 8 threads
| Config | Real time |
|---|---:|
| 1 thread | 0.67 s |
| 8 threads | 0.55 s |

Observed speedup:
- `1.22x` (`0.67 / 0.55`).

### yt: 1 rank vs 8 ranks
| Config | Real time |
|---|---:|
| 1 rank | 1.05 s |
| 8 ranks | 3.16 s |

Observed scaling:
- 8 ranks was about `3.0x` slower (`3.16 / 1.05`) for this workload.

## yt Internal Timing (8-rank measured pass)
yt script reported rank-max stage timings:
- `imports`: 1.296088 s
- `yt/load`: 0.425525 s
- `yt/metadata`: 0.340740 s
- `yt/slice`: 0.004214 s
- `yt/frb`: 0.008427 s
- `yt/extract`: 0.067570 s
- `total`: 2.142564 s

Interpretation:
- Most 8-rank time is in process/import/load/metadata overhead, not in slice computation.
- For this single-slice workload, MPI overhead dominates and reduces performance.

## Additional Observations
- yt default field auto-selection can pick particle fields (for example `particle_cpu`) that fail this FRB slice path; explicit `--var gasDensity` is required for consistent comparison.
- Kangaroo defaults to `hpx.os_threads=1` unless `--hpx-config hpx.os_threads=<N>` is provided.

## Conclusion
For this benchmark case (single slice, `1000x1000`, `gasDensity`, warm cache):
- Kangaroo outperformed yt in both serial and requested parallel configurations.
- Kangaroo improved moderately from 1 to 8 threads.
- yt became slower at 8 MPI ranks due to fixed parallel overheads dominating this workload.

## Repro Checklist
- Repository commit: `a56fdb56a2f2ed4a889f054bbb031c9bbfd82997`
- Python: `3.13.12` (conda-forge)
- yt: `4.4.2`
- mpi4py: `4.1.1`
- numpy: `2.4.2`
- matplotlib: `3.10.8`
- MPI launcher: `mpiexec` from pixi env (`mpich >=4.3.2,<5`)
- HPX runtime build/config string available via:
  - `pixi run python3 -c "import analysis._core as c; print(c.hpx_configuration_string())"`

## Repro Script (Sequential, No Overlap)
```bash
#!/usr/bin/env bash
set -euo pipefail

PF="../amrex-visit-plugin/example_data/DiskGalaxy/plt0000020"
ARGS="--var gasDensity --resolution 1000,1000 --coord 0.0 --zoom 20 --axis x --no-plot"

mkdir -p /tmp/mplconfig

echo "=== Kangaroo 1 thread warm-up ==="
/usr/bin/time -p env MPLCONFIGDIR=/tmp/mplconfig \
  pixi run scripts/plotfile_slice.py "$PF" $ARGS

echo "=== Kangaroo 1 thread measured ==="
/usr/bin/time -p env MPLCONFIGDIR=/tmp/mplconfig \
  pixi run scripts/plotfile_slice.py "$PF" $ARGS

echo "=== Kangaroo 8 threads warm-up ==="
/usr/bin/time -p env MPLCONFIGDIR=/tmp/mplconfig \
  pixi run scripts/plotfile_slice.py "$PF" $ARGS --hpx-config hpx.os_threads=8

echo "=== Kangaroo 8 threads measured ==="
/usr/bin/time -p env MPLCONFIGDIR=/tmp/mplconfig \
  pixi run scripts/plotfile_slice.py "$PF" $ARGS --hpx-config hpx.os_threads=8

echo "=== yt 1 rank warm-up ==="
/usr/bin/time -p env MPLCONFIGDIR=/tmp/mplconfig \
  pixi run scripts/plotfile_slice_yt.py "$PF" $ARGS

echo "=== yt 1 rank measured ==="
/usr/bin/time -p env MPLCONFIGDIR=/tmp/mplconfig \
  pixi run scripts/plotfile_slice_yt.py "$PF" $ARGS

echo "=== yt 8 ranks warm-up ==="
/usr/bin/time -p env MPLCONFIGDIR=/tmp/mplconfig \
  pixi run mpiexec -n 8 python3 scripts/plotfile_slice_yt.py "$PF" $ARGS

echo "=== yt 8 ranks measured ==="
/usr/bin/time -p env MPLCONFIGDIR=/tmp/mplconfig \
  pixi run mpiexec -n 8 python3 scripts/plotfile_slice_yt.py "$PF" $ARGS
```
