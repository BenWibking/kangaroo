# Streaming Executor Implementation Plan

## Objective

Build a general streaming executor that can run large Kangaroo plans with bounded I/O,
bounded memory, and bounded task fanout. The executor must not be specific to
`flux_surface_integral`; that workload is the first production validation case
because it exposes the current eager executor's failure mode.

The desired behavior is:

```text
small ready window -> admitted input requests -> kernel continuations -> output publish
                  -> release consumed temporaries -> admit more work
```

The existing eager executor should remain available as a fallback while the
streaming path is implemented and validated.

## Evidence Driving This Design

The production flux-surface plan currently creates `11152` tasks. On 4 nodes,
the eager executor rapidly issues thousands of block tasks. Most of those tasks
then sit in `wait_inputs` while the plotfile load queues drain.

The important observation is not specific to flux. Any operator with many blocks,
many input fields, large dataset-backed chunks, or reductions over large
intermediates can produce the same pattern:

- task fanout is much larger than useful active work,
- dataset load queues become thousands of entries deep,
- memory pressure is controlled indirectly by the filesystem queue,
- useful kernels run only after a large backlog of I/O has already accumulated.

The streaming executor should make the amount of active work proportional to
configured resources, not to total plan size.

## Non-Goals

- Do not add an operator-specific executor path.
- Do not change the Python pipeline API for normal users.
- Do not require all kernels to become streaming-aware before the first useful
  implementation.
- Do not remove the current eager executor until the streaming path has covered
  existing tests and production runs.
- Do not introduce `std::thread`, `std::mutex`, or standard-library concurrency
  primitives in runtime hot paths. Use HPX primitives.

## Target Architecture

### 1. Executor Modes

Add an executor mode selected by runtime option or environment variable:

```text
KANGAROO_EXECUTOR_MODE=eager|streaming
```

Initial default should remain `eager`. Once the streaming path has equivalent
coverage and production validation, switch the default to `streaming`.

Add a small options object on the C++ side:

```cpp
struct ExecutorOptions {
  std::string mode = "eager";
  int32_t max_active_tasks_per_locality = 128;
  int32_t max_active_tasks_per_stage = 0;      // 0 means derive from locality count.
  std::size_t max_input_bytes_per_locality = 0; // 0 means disabled initially.
  std::size_t max_output_bytes_per_locality = 0;
  bool enable_task_level_stage_overlap = false;
  bool enable_early_release = false;
};
```

Wire these options through `Runtime` into `Executor`.

Primary files:

- `cpp/include/kangaroo/executor.hpp`
- `cpp/src/executor.cpp`
- `cpp/src/runtime.cpp`

### 2. Compiled Task Instances

The current executor expands stage templates directly inside
`Executor::run_stage()`. For streaming, first compile each stage into explicit
task instances:

```cpp
enum class TaskKind {
  Chunk,
  Graph,
};

struct TaskInstance {
  TaskKind kind;
  int32_t stage_idx;
  int32_t tmpl_idx;
  int32_t block_or_group;
  int32_t target_locality;
  std::vector<ChunkRef> input_refs;
  std::vector<ChunkRef> output_refs;
  std::size_t estimated_input_bytes;
  std::size_t estimated_output_bytes;
};
```

This is generic plan metadata. It is not operator-specific.

For chunk tasks, `block_or_group` is the block id. For graph tasks,
`block_or_group` is the reduction group id.

Use existing helpers where possible:

- `resolve_input_location()`
- `graph_reduce_group_count()`
- `graph_reduce_output_block()`
- `DataService::home_rank()`

The first implementation can compute `estimated_output_bytes` from
`TaskTemplateIR::output_bytes` and leave `estimated_input_bytes` conservative or
unknown. Later phases should derive dataset chunk sizes from run metadata and
backend metadata.

### 3. Bounded Stage Launcher

Replace eager stage fanout with a bounded launcher in streaming mode.

Current eager behavior:

```text
for every task in stage:
  launch task future immediately
when_all(all task futures)
```

Streaming behavior:

```text
pending task queue
active task set

while pending is not empty and active < budget:
  launch next admitted task

when any active task completes:
  record completion
  release task slot
  launch next admitted task
```

This first slice should still honor existing stage barriers. In other words,
stage `N + 1` starts only after stage `N` completes unless the plan already
declares independent stages. That is less aggressive than the final design but
removes the worst fanout problem quickly.

Implementation notes:

- Use HPX futures and HPX synchronization primitives only.
- Avoid blocking worker threads waiting for task slots.
- Prefer a small local runner object whose state is owned by a `shared_ptr` and
  advanced through continuations.
- Preserve current task event logging, but add launcher-level events:
  - `streaming_stage_start`
  - `streaming_stage_admit_task`
  - `streaming_stage_task_done`
  - `streaming_stage_end`
  - active task count
  - pending task count

Primary files:

- `cpp/src/executor.cpp`
- `cpp/include/kangaroo/executor.hpp`

Validation:

- Existing executor tests pass in both eager and streaming modes.
- A synthetic many-block test shows bounded `wait_inputs` count.
- The full flux-surface progress run no longer reports thousands of tasks stuck
  in `wait_inputs`.

### 4. Per-Locality Task Partitions

After the bounded stage launcher works locally, change chunk-stage launch from
one remote action per task to one partition runner per target locality.

Current remote pattern:

```text
task -> resolve target -> remote run_block_task_action
```

Target pattern:

```text
stage -> build target-locality partitions
      -> run_stage_partition_action(locality, task instances)
      -> locality executes its partition with a bounded local window
```

This keeps scheduling and admission close to the data and reduces remote action
fanout from `O(blocks)` to `O(localities)` per stage.

Partition runner responsibilities:

- keep a local pending queue,
- admit at most `max_active_tasks_per_locality`,
- call `run_block_task_impl()` or `run_graph_task_impl()` for admitted tasks,
- return one future for the whole partition.

This remains general because partitioning is based on `ChunkRef` ownership, not
operator type.

Primary files:

- `cpp/src/executor.cpp`
- `cpp/include/kangaroo/executor.hpp`
- HPX action registration near existing `kangaroo_run_block_task_action`

Validation:

- Remote action count scales with localities, not blocks.
- Output equality matches eager mode on existing tests.
- Event logs show active tasks bounded independently on each locality.

### 5. Data-Service Admission and Coalescing

The executor can bound task launch, but dataset-backed I/O also needs admission
at the data-service layer. Otherwise a single admitted task with many inputs, or
many concurrently admitted tasks that share a block, can still create inefficient
field-level I/O.

Add a general read-admission layer to `DataServiceLocal`:

```text
ChunkRef requests -> group by dataset storage unit -> admit bounded physical reads
```

For plotfiles, the storage unit is currently:

```text
(dataset, chunk_store, step, level, version, block)
```

The queue entry owns all requested fields for that block. This is generic at the
data-service level: other backends can define their own storage-unit key.

Required behavior:

- `get_hosts(refs)` should group local unresolved refs before enqueueing.
- Multiple tasks requesting fields from the same unresolved storage unit should
  attach to the same in-flight load when possible.
- The load queue should have explicit limits:
  - physical reads in flight,
  - raw bytes in flight,
  - queued storage units.
- Blocking plotfile reads should run on the dedicated HPX I/O pool.

Primary files:

- `cpp/include/kangaroo/data_service.hpp`
- `cpp/include/kangaroo/data_service_local.hpp`
- `cpp/src/data_service_local.cpp`
- `cpp/src/backend_plotfile.cpp`

Validation:

- One task requesting nine fields from the same block creates one block-bundle
  queue entry, not nine independent physical reads.
- Queue depth is reported in storage units, not only field futures.
- Per-locality in-flight reads stay near the configured I/O budget.

### 6. Memory Lifetime and Early Release

Streaming only helps fully if large temporaries are released as soon as their
last consumer finishes.

Add plan analysis that computes consumer counts for every produced `ChunkRef`.
The executor decrements counts after each task has consumed its inputs. When a
count reaches zero, the chunk can be released if it is not externally visible.

Add output retention metadata:

```text
temporary    release after last consumer
materialized keep until run end
final        keep for user retrieval
dataset      cache policy controlled by data service
```

Initial implementation can infer:

- outputs from `ctx.temp_field(...)` are `temporary`,
- outputs from `ctx.output_field(...)` are `final`,
- dataset fields are `dataset`.

If that inference is not available in C++ yet, add an explicit field-retention
map to the serialized plan.

Data-service API addition:

```cpp
virtual hpx::future<void> release_host(const ChunkRef&);
```

The default implementation can be a no-op so other backends keep working.

Primary files:

- `analysis/plan.py`
- plan serialization and decode code
- `cpp/include/kangaroo/plan_ir.hpp`
- `cpp/src/plan_decode_msgpack.cpp`
- `cpp/include/kangaroo/data_service.hpp`
- `cpp/src/data_service_local.cpp`

Validation:

- Memory high-water mark is bounded by the active window plus retained outputs.
- Existing output retrieval tests still pass.
- A test with a temporary consumed by two downstream tasks releases only after
  the second consumer completes.

### 7. Task-Level Stage Overlap

The bounded stage launcher still treats stage dependencies as barriers. The
final streaming executor should allow downstream tasks to be admitted once their
specific inputs are available, even if the producer stage has not fully
completed.

This requires a task-level dependency view:

```text
producer output ChunkRef -> consumer TaskInstance ids
```

Scheduling rule:

- a task is structurally ready when all predecessor stages have been opened,
- a task is data ready when its produced-input futures are ready,
- dataset-backed inputs count as data-ready after admitted reads complete,
- barrier stages keep current behavior and wait for predecessor completion.

Add stage/template traits:

```text
streamable = true|false
barrier = true|false
associative_reduction = true|false
```

Initial defaults:

- chunk stages with normal block kernels: `streamable=true`
- graph reductions: `streamable=true` once input-group readiness is implemented
- stages with unknown side effects: `barrier=true`

This enables generic streaming reductions. For example, a reduce group can run
as soon as its input chunks exist instead of waiting for every block in the
producer stage.

Validation:

- A graph-reduce test verifies that early reduce groups run before all producer
  blocks complete.
- Final numeric outputs match eager mode.
- Event logs show overlap between producer kernels and reduction tasks.

### 8. Instrumentation and Autotuning

Add low-overhead counters that can be enabled without full JSON event logging.

Suggested counters per locality:

- pending tasks,
- active tasks,
- tasks completed,
- tasks waiting on input futures,
- dataset storage units queued,
- dataset storage units in flight,
- raw read bytes,
- published bytes,
- active input bytes,
- active output bytes,
- chunk-store bytes retained,
- chunks released early.

Expose a concise progress line for production runs:

```text
tasks done/total, active, input_wait, read_q, read_inflight, read_gib_s, store_gib
```

The JSON event log should remain available for short diagnostics, but production
timing sweeps should use the concise counters to avoid perturbing I/O.

Primary files:

- `cpp/include/kangaroo/runtime.hpp`
- `cpp/src/runtime.cpp`
- `cpp/src/executor.cpp`
- `cpp/src/data_service_local.cpp`
- Python progress plumbing if needed

Validation:

- Progress-only runs do not create event log files.
- Counter output is enough to diagnose whether a run is I/O-bound,
  compute-bound, or memory-budget-bound.

## Implementation Phases

### Phase 0: Scaffolding

- Add `ExecutorOptions`.
- Add `KANGAROO_EXECUTOR_MODE`.
- Keep eager mode as default.
- Add task-instance expansion helpers, but do not change execution behavior.
- Add unit tests for task expansion on chunk and graph stages.

Exit criteria:

- No behavior change in default mode.
- Existing tests pass.

### Phase 1: Bounded Stage Fanout

- Implement streaming mode with stage barriers preserved.
- Add bounded launcher for explicit `TaskInstance` lists.
- Use the existing per-task `run_block_task()` and `run_graph_task()` methods.
- Add tests proving that active task count is bounded.

Exit criteria:

- Existing tests pass with `KANGAROO_EXECUTOR_MODE=streaming`.
- A many-block synthetic plan does not create one future per block at once.
- Full flux production diagnostics show bounded `wait_inputs`, not thousands.

### Phase 2: Data-Service Storage-Unit Queue

- Change dataset loads from field-level queue entries to storage-unit queue
  entries.
- Preserve the `ChunkRef` future API for kernels.
- Add queue counters in storage-unit terms.
- Keep dedicated HPX I/O pool behavior.

Exit criteria:

- Existing data-service tests pass.
- Plotfile block reads coalesce fields by block.
- The flux workload no longer has thousands-deep plotfile load queues.

### Phase 3: Locality Partition Runner

- Add partition actions.
- Dispatch one stage partition per locality.
- Run bounded local windows inside each partition.
- Keep stage barriers for this phase.

Exit criteria:

- Remote action count is `O(localities)` per chunk stage.
- Per-node I/O utilization is stable.
- Flux progress improves versus Phase 1.

### Phase 4: Early Release

- Add output retention metadata.
- Add consumer-count analysis.
- Add `release_host()`.
- Enable early release behind an option.

Exit criteria:

- Memory high-water mark drops on multi-stage plans with large temporaries.
- Final outputs remain available after run completion.
- Dataset-backed cache behavior remains correct.

### Phase 5: Task-Level Stage Overlap

- Build producer-to-consumer task dependency indexes.
- Allow streamable downstream graph tasks to wait on specific input futures
  rather than whole-stage completion.
- Keep barrier fallback for unknown or unsafe stages.

Exit criteria:

- Streaming reductions overlap with producer stages.
- Existing graph-reduce tests pass.
- Production flux run reduces as blocks finish instead of after all block
  accumulations complete.

### Phase 6: Production Tuning

- Run progress-only sweeps over:
  - node count: `4`, `8`, `16`, `32`,
  - per-locality active task window,
  - plotfile read concurrency,
  - input-byte budget.
- Use JSON event logs only for short diagnostic runs.
- Pick conservative defaults from the first stable saturation point.

Exit criteria:

- Full `NBINS=50` flux-surface production run completes in a few minutes.
- Runtime counters show bounded queues and sustained per-node read bandwidth.
- Defaults do not regress small tests or single-node smoke runs.

## First Implementation Slice

The first useful patch should be deliberately small:

1. Add `ExecutorOptions` and `KANGAROO_EXECUTOR_MODE`.
2. Add task-instance expansion helpers.
3. Add `Executor::run_stage_streaming()` with bounded active task count and
   existing stage barriers.
4. Add a unit test that constructs a many-block stage and verifies the maximum
   concurrently admitted tasks stays under the configured window.
5. Run existing tests in eager and streaming modes.

This slice will not solve all production I/O by itself, but it will remove the
global task flood and create the control point needed for the later I/O and
memory-budget work.

