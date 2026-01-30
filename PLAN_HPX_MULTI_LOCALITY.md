HPX Multi-Locality: Install + Test Plan

Goal
- Enable HPX networking so Kangaroo can run across multiple localities, and verify distribution.

Constraints Observed (Jan 30, 2026)
- Pixi/conda-forge HPX appears built without networking; `--hpx:localities` errors out.
- Multi-locality requires HPX built with networking enabled (`-DHPX_WITH_NETWORKING=On`), optionally MPI.

Plan A: Build HPX (local install, no MPI)
1) Configure HPX (outside Pixi or inside a separate build dir):
   - CMake flags:
     - `-DHPX_WITH_NETWORKING=On`
     - `-DHPX_WITH_PARCELPORT_TCP=On` (default for networking)
     - Optional: `-DHPX_WITH_CXX_STANDARD=20` to match Kangaroo build.
2) Build + install HPX to a known prefix (e.g., `$HOME/hpx-install`).
3) Point Kangaroo to that HPX:
   - Export `HPX_DIR=/path/to/hpx-install/lib/cmake/HPX` (or update `scripts/detect_hpx.sh`).
4) Rebuild Kangaroo:
   - `pixi run configure`
   - `pixi run build`
   - `pixi run install`

Plan B: Build HPX with MPI (for multi-node)
1) Install/locate MPI (OpenMPI or MPICH).
2) Configure HPX with:
   - `-DHPX_WITH_NETWORKING=On`
   - `-DHPX_WITH_PARCELPORT_MPI=On`
   - MPI compiler wrappers (e.g., `CC=mpicc CXX=mpicxx`).
3) Install to prefix and set `HPX_DIR` as above.
4) Rebuild Kangaroo with that HPX.

Testing Checklist
1) Build config string:
   - `pixi run python scripts/print_hpx_config.py`
   - Look for networking/parcelport settings; if absent, multi-locality won’t work.
2) Multi-locality run (single node):
   - `KANGAROO_LOG_LOCALITY=1 scripts/hpxrun_pixi.sh -l 2 scripts/slice_operator_demo.py`
   - Expect logs like:
     - `uniform_slice block=0 locality=0`
     - `uniform_slice block=1 locality=1`
3) MPI run (if enabled):
   - `mpirun -n 2 pixi run python scripts/slice_operator_demo.py`
   - Confirm mixed locality ids in logs.

Notes
- `scripts/hpxrun_pixi.sh` auto-prepends the Pixi Python, so it works with HPX’s launcher.
- If `--hpx:localities` errors out, HPX networking is still disabled.
