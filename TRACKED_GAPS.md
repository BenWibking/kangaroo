# Tracked Gaps vs PLAN.md / PROTOTYPE.md

Date: 2026-01-30

This list compares the current codebase against `PLAN.md` and `PROTOTYPE.md`. Items are grouped by severity and tagged with status, file references, and suggested next action.

Legend:
- Status: **Missing**, **Partial**, **Divergent**, **Risk**
- Plane: **Chunk**, **Graph**, **Mixed**, **Infra**

## P0 — Correctness/Architecture Gaps

- **[Missing][Graph/Mixed] Graph-plane execution** — ExecPlane includes `Graph`/`Mixed`, but executor throws on non-Chunk; no graph runtime, interfaces, or placeholders in executor path. Next: stub `Graph` plane with a clear error path or implement interface skeleton. Files: `cpp/include/kangaroo/plan_ir.hpp`, `cpp/src/executor.cpp`
- **[Divergent][Chunk] Data-local scheduling** — Plan expects tasks to execute where data lives; current executor runs locally and fetches remote data instead. Next: schedule `run_block_task` on `home_rank` with HPX actions and forward results. Files: `cpp/src/executor.cpp`, `cpp/src/data_service_local.cpp`
- **[Risk][Infra] Field ID ownership split** — Dataset allocates field IDs independently from runtime’s allocator; IDs may diverge from the C++ runtime’s registry. Next: route all field ID creation through runtime and register them in Dataset. Files: `analysis/dataset.py`, `analysis/runtime.py`, `cpp/src/runtime.cpp`

## P1 — Functional Parity Gaps

- **[Partial][Chunk] Neighbor dependency width/faces ignored** — `deps.width` and `deps.faces` parsed but not honored; executor only checks `kind == FaceNeighbors`. Next: filter faces by `faces[]` and support `width > 1` in adjacency lookup. Files: `cpp/src/executor.cpp`, `cpp/src/plan_decode_msgpack.cpp`
- **[Divergent][Chunk] Adjacency algorithm** — Plan calls for face hashing with sharding; current adjacency does O(n²) scan and caches per block/face. Next: implement face-hash indexing and shard by ownership. Files: `cpp/src/adjacency.cpp`, `cpp/include/kangaroo/adjacency.hpp`
- **[Partial][Chunk] Stage dependencies** — `after` is serialized but not used for scheduling; executor relies on pre-topo order only. Next: perform topo sort in C++ or enforce/validate ordering. Files: `analysis/runtime.py`, `cpp/src/executor.cpp`

## P2 — Prototype Limitations (Expected but Track)

- **[Missing][Infra] Data allocation contract** — Outputs are default-constructed `HostView` and kernels are expected to fill them without a clear allocator contract. Next: add `DataService::alloc_host` or define output sizing in kernel descriptors. Files: `cpp/src/executor.cpp`, `cpp/include/kangaroo/kernel.hpp`
- **[Missing][Infra] Dataset-backed IO** — `DatasetHandle` is a stub; no actual read/initialize path for fields. Next: define dataset interface for loading/chunk access or provide a test in-memory backend. Files: `analysis/dataset.py`, `cpp/include/kangaroo/runtime.hpp`
- **[Partial][Chunk] Neighbor field selection** — Neighbor fetch uses only first input field; multi-input templates ignore neighbor data for other fields. Next: define neighbor fetch policy per input or use a dedicated field in deps. Files: `cpp/src/executor.cpp`

## P3 — Performance / Scaling Targets

- **[Missing][Infra] Caching/eviction** — No cache policy for chunk data or adjacency; unbounded growth for long runs. Next: add LRU or epoch-based invalidation. Files: `cpp/src/data_service_local.cpp`, `cpp/src/adjacency.cpp`
- **[Missing][Infra] Memory pooling** — Plan calls out pooling for performance; absent in prototype. Next: add a simple slab or reuse pool for HostView buffers. Files: `cpp/include/kangaroo/kernel.hpp`, `cpp/src/executor.cpp`

## Notes

- The Python DSL, msgpack serialization, IR structs, and basic HPX execution scaffolding align well with `PROTOTYPE.md`.
- `README.md` already reflects “prototype” status; consider adding a short “Known Gaps” section referencing this document.
