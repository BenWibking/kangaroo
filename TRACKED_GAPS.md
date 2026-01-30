# Tracked Gaps vs PLAN.md / PROTOTYPE.md

Date: 2026-01-30

This list compares the current codebase against `PLAN.md` and `PROTOTYPE.md`. Items are grouped by severity and tagged with status, file references, and suggested next action.

Legend:
- Status: **Missing**, **Partial**, **Divergent**, **Risk**
- Plane: **Chunk**, **Graph**, **Mixed**, **Infra**

## P0 — Correctness/Architecture Gaps

- **[Missing][Graph/Mixed] Graph-plane execution** — ExecPlane includes `Graph`/`Mixed`, but executor throws on non-Chunk; no graph runtime or interface skeleton beyond the error path. Next: add graph-plane interface stubs (executor path + runtime services) or implement a minimal graph executor. Files: `cpp/include/kangaroo/plan_ir.hpp`, `cpp/src/executor.cpp`

## P1 — Functional Parity Gaps

- **[Divergent][Chunk] Adjacency algorithm** — Plan calls for face hashing with sharding; current adjacency does O(n²) scan and caches per block/face. Next: implement face-hash indexing and shard by ownership. Files: `cpp/src/adjacency.cpp`, `cpp/include/kangaroo/adjacency.hpp`

## P2 — Prototype Limitations (Expected but Track)

- **[Partial][Infra] Dataset-backed IO** — Dataset is now read-only and in-memory (seeded from Python), with optional preload; no file-backed backend or schema for typed fields. Next: add a real IO backend and field metadata. Files: `analysis/dataset.py`, `cpp/include/kangaroo/runtime.hpp`, `cpp/src/runtime.cpp`

## P3 — Performance / Scaling Targets

- **[Missing][Infra] Caching/eviction** — No cache policy for chunk data or adjacency; unbounded growth for long runs. Next: add LRU or epoch-based invalidation. Files: `cpp/src/data_service_local.cpp`, `cpp/src/adjacency.cpp`
- **[Missing][Infra] Memory pooling** — Plan calls out pooling for performance; absent in prototype. Next: add a simple slab or reuse pool for HostView buffers. Files: `cpp/include/kangaroo/kernel.hpp`, `cpp/src/executor.cpp`

## Notes

- The Python DSL, msgpack serialization, IR structs, and basic HPX execution scaffolding align well with `PROTOTYPE.md`.
- `README.md` already reflects “prototype” status; consider adding a short “Known Gaps” section referencing this document.
