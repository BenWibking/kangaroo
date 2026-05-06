# Tracked Gaps vs PLAN.md / PROTOTYPE.md

Date: 2026-02-02

This list compares the current codebase against `PLAN.md` and `PROTOTYPE.md`. Items are grouped by severity and tagged with status, file references, and suggested next action.

Legend:
- Status: **Missing**, **Partial**, **Divergent**, **Risk**, **Done**
- Plane: **Chunk**, **Graph**, **Mixed**, **Infra**

## P0 — Correctness/Architecture Gaps

- **[Partial][Graph/Mixed] Graph-plane execution** — `Executor` implements `ExecPlane::Graph` stages via `run_graph_task_impl`. A `reduce` template with configurable `fan_in` is implemented. Next: expand template library to support more general DAG patterns (e.g., prefix sums, global reductions) and clarify `Mixed` plane semantics. Files: `cpp/include/kangaroo/plan_ir.hpp`, `cpp/src/executor.cpp`, `cpp/src/runtime.cpp`

## P1 — Functional Parity Gaps

- **[Divergent][Chunk] Adjacency algorithm** — Plan calls for face hashing with sharding; current adjacency does O(n²) scan and caches per block/face. Next: implement face-hash indexing and shard by ownership. Files: `cpp/src/adjacency.cpp`, `cpp/include/kangaroo/adjacency.hpp`

## P2 — Prototype Limitations (Expected but Track)

- **[Done][Infra] Dataset-backed IO** — `DatasetBackend` interface and `PlotfileBackend` implementation now exist, allowing direct reading of AMReX plotfiles via `PlotfileReader` with automatic axis transposition. Next: Ensure full Python DSL integration to instantiate these backends transparently. Files: `cpp/src/backend_plotfile.cpp`, `cpp/include/kangaroo/dataset_backend.hpp`, `cpp/src/plotfile_reader.cpp`

## P3 — Performance / Scaling Targets

- **[Missing][Infra] Caching/eviction** — No cache policy for chunk data (`DataServiceLocal`) or adjacency; unbounded growth for long runs. Next: add LRU or epoch-based invalidation. Files: `cpp/src/data_service_local.cpp`, `cpp/src/adjacency.cpp`
- **[Missing][Infra] Memory pooling** — Plan calls out pooling for performance; absent in prototype. Next: add a simple slab or reuse pool for HostView buffers. Files: `cpp/include/kangaroo/kernel.hpp`, `cpp/src/executor.cpp`, `cpp/src/data_service_local.cpp`

## Notes

- The Python DSL, msgpack serialization, IR structs, and basic HPX execution scaffolding align well with `PROTOTYPE.md`.
- `README.md` already reflects “prototype” status; consider adding a short “Known Gaps” section referencing this document.