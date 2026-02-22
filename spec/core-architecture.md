# Core Architecture Specification

## 1. Runtime Responsibility

The runtime is responsible for transforming a decoded plan into executed tasks over dataset chunks and writing outputs back into runtime-addressable storage.

A conforming runtime MUST provide:
- Field ID allocation.
- Persistent field marking.
- Plan execution against explicit run metadata and dataset handle.
- Retrieval of produced chunk payloads by full chunk identity.
- Optional preloading of selected fields from dataset backend into runtime storage.

## 2. Execution Planes

The plan model recognizes three labels:
- `chunk`
- `graph`
- `mixed`

Execution compatibility rules:
- A stage labeled `chunk` MUST only contain `chunk` templates.
- A stage labeled `graph` MUST only contain `graph` templates.
- `mixed` MAY be accepted at decode time but is not required as an executable stage plane.
- Unsupported stage/template plane combinations MUST fail at execution.

## 3. Plan Structure

A plan is a list of stages. Each stage contains:
- `name`
- `plane`
- `after` (dependency indices)
- `templates`

Each template contains:
- Identity: `name`, `plane`, `kernel`
- Domain: `step`, `level`, optional `blocks`
- `inputs` and `outputs` as field references
- `output_bytes` (optional per-output size hints)
- `deps` dependency rule
- `params` payload

The runtime MUST interpret templates as declarative task generators, not as already-instantiated tasks.

## 4. Stage Dependency Graph

The stage dependency graph is defined by `after` index references.

Rules:
- Dependency indices MUST refer to valid stage indices.
- The graph MUST be acyclic.
- A stage MUST begin only after all dependencies finish successfully.
- Cycle detection MUST occur before stage execution begins.

## 5. Template Expansion

### 5.1 Chunk Stage Expansion

For a chunk template:
- If `domain.blocks` is omitted: instantiate for every block on `domain.step/domain.level`.
- If `domain.blocks` is present: instantiate only for listed blocks.

### 5.2 Graph Stage Expansion

Graph templates are reduction-group templates.

A conforming implementation MUST:
- Parse reduction grouping parameters from `params`.
- Compute number of groups from `num_inputs` and `fan_in`.
- Instantiate one graph task per group.

## 6. Task Locality and Ownership

Chunk ownership MUST be deterministic and stable for a fixed chunk identity.

Scheduling behavior:
- If template has inputs, task placement is based on first input chunk ownership.
- If template has no inputs, placement is based on first output chunk ownership.
- Remote fetch/put behavior MUST be transparent to operator semantics.

## 7. Data Service Contract

The runtime data service MUST support:
- Output buffer allocation by chunk identity and requested size.
- Asynchronous get by chunk identity.
- Asynchronous put by chunk identity.
- Subbox fetch by chunk identity plus request bounds.

Subbox contract:
- Returned box MUST be the geometric intersection of requested box and source chunk box.
- Returned payload MUST correspond exactly to returned box extents and bytes-per-value.
- No-overlap requests MUST return empty payload and invalid/empty bounds representation.

## 8. Neighbor Dependency Semantics

Dependency kinds:
- `None`
- `FaceNeighbors`

For `FaceNeighbors`:
- Template MUST have at least one input.
- `faces` is a six-entry mask in axis-face order: `Xm, Xp, Ym, Yp, Zm, Zp`.
- `width` is traversal depth over face-neighbor adjacency.
- `halo_inputs` selects which input indices receive neighbor collections.
- If `halo_inputs` is absent/empty, default target is input index `0`.
- Duplicate `halo_inputs` entries MAY be deduplicated.
- `width=0` MUST be valid and produce empty neighbor collections.
- All faces disabled MUST be valid and produce empty neighbor collections.

## 9. Input Domain Override Semantics

Each input field reference MAY supply a domain override.

Rules:
- If input override is absent: input domain equals template domain for that task instance.
- If override has a single block: that block is used.
- If override has multiple blocks: executing block MUST be contained in override list.
- If executing block is not in override list, execution MUST fail for that task.

## 10. Graph Reduction Parameters

Graph reduce templates are valid only when `params.graph_kind` is `reduce`.

Required fields/behavior:
- `num_inputs` MUST be positive unless recoverable from explicit `input_blocks`.
- `fan_in <= 0` MUST be coerced to `1`.
- If `input_blocks` is present, its length MUST equal `num_inputs`.
- Missing/malformed reduce params MUST fail before graph task execution.

## 11. Kernel Registry Contract

The runtime MUST maintain a name-keyed kernel registry.

Requirements:
- Registering kernels by name.
- Lookup by kernel name.
- Listing kernel descriptors.
- Lookup of unknown kernel name MUST fail explicitly.

Kernel name compatibility requirements are defined in `spec/operators.md`.

## 12. Output Allocation Semantics

If `output_bytes` is provided on a template:
- Its length MUST equal number of outputs.
- Each output buffer MUST be allocated with corresponding size.

If omitted:
- Runtime MAY allocate zero-sized output buffers initially.
- Kernel is then responsible for resizing output payload appropriately.

## 13. Event Logging

Runtime event logging MUST support:
- Configurable JSONL output path.
- Logging task lifecycle with start and terminal status (`end` or `error`).
- Per-event timing fields (`ts`, `start`, `end`).
- Task identity context (`id`, name, stage/template/block metadata).

Python-initiated task events MAY be logged via runtime bindings and SHOULD integrate with the same event sink.

## 14. Runtime Context Propagation

For distributed task execution, runtime context MUST be available wherever tasks execute:
- Active run metadata.
- Active dataset handle.
- Active plan object referenced by plan ID.
- Kernel registry reference.

This context MUST be synchronized before executing a new plan.

## 15. Runtime Construction and Startup

Runtime construction MUST support:
- Default constructor.
- Constructor receiving explicit runtime config/args.

Constraints:
- If backend runtime has already started, changing startup config MUST fail.
- If startup config was already fixed, attempting to set it again MUST fail.

## 16. Plan Lifecycle

For each `run_packed_plan` call, runtime MUST:
1. Decode and validate plan payload.
2. Ensure runtime backend is started/ready.
3. Broadcast/synchronize active context (run metadata, dataset, plan) to execution localities.
4. Execute the stage DAG.
5. Remove transient global plan context after completion.

If execution fails:
- Runtime MUST surface error to caller.
- Error events SHOULD be emitted for affected tasks when event logging is enabled.
