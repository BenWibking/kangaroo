# Repository Guidelines

## Project Structure & Module Organization
- `analysis/`: Python API and DSL (plans, datasets, runtime wrappers, operators).
- `cpp/`: C++20 runtime and nanobind extension modules (`_core`, `_plotfile`), with headers in `cpp/include/kangaroo/`.
- `tests/`: `pytest` suite for Python API, operators, plotfile access, and smoke tests.
- `scripts/`: runnable demos, dashboard entrypoints, and utility scripts (HPX detection, benchmarking).
- Root docs: `README.md`, `PLAN.md`, and performance/tracking notes for design context.

## Build, Test, and Development Commands
- `pixi install`: create/update the managed dev environment (Python, HPX, CMake, pytest, etc.).
- `pixi run configure`: configure CMake in `.pixi/build` with HPX and nanobind paths.
- `pixi run build`: compile C++ targets.
- `pixi run install`: install extension modules and Python package into the Pixi environment.
- `pixi run test`: run tests (`pytest -q`).
- `python scripts/kangaroo_dashboard.py`: launch the local dashboard (use `pixi run python ...` if using Pixi env binaries).

## Coding Style & Naming Conventions
- Python: PEP 8 style, 4-space indentation, type hints used throughout (`analysis/runtime.py` is a good reference).
- C++: C++20, 2-space indentation, braces on the same line, `snake_case` for functions/variables.
- Naming: tests use `tests/test_*.py`; script names are descriptive (`plotfile_projection.py`, `smoke_demo.py`).
- Keep modules focused and avoid cross-layer leakage between Python DSL concerns and C++ runtime internals.

## Testing Guidelines
- Framework: `pytest` (`pyproject.toml` sets `testpaths = ["tests"]`).
- Add unit tests next to related coverage area (runtime, operators, dataset, dashboard DAG, etc.).
- Prefer deterministic, small-input tests; include a smoke path for new end-to-end behavior when practical.
- Run `pixi run test` before opening a PR.

## Commit & Pull Request Guidelines
- Recent commits follow short, imperative subjects (for example: `add vorticity test`, `simplify python api`).
- Keep commit scope narrow; separate refactors from behavior changes.
- PRs should include:
  - concise problem/solution summary,
  - linked issue (if applicable),
  - exact validation commands run (for example `pixi run test`),
  - screenshots only for dashboard/UI changes.
