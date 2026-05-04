# Plotfile I/O and Executor Redesign Plan

## Context

The flux-surface run in `slurm-3268071.out` did not make meaningful task progress before the five-minute time limit. Event-log analysis showed that the runtime rapidly queued tens of thousands of dataset chunk loads, then spent the observed window dominated by plotfile input.

The direct plotfile I/O microbenchmark in `scripts/utils/plotfile_io_microbench.py` reproduced the bottleneck without running the full job. On the Lustre filesystem used for the test plotfile, one node showed strong scaling with multiple in-flight read calls up to about 16 concurrent block reads:

| In-flight reads | Raw bandwidth |
| ---: | ---: |
| 1 | ~82 MiB/s |
| 2 | ~155 MiB/s |
| 4 | ~291 MiB/s |
| 8 | ~616 MiB/s |
| 16 | ~1.05-1.14 GiB/s |
| 20-32 | flat or noisy, no reliable improvement |

For the current flux-surface input set, each AMReX FAB/block read pulls a contiguous component range. The requested fields are components `0,1,2,3,4,6,7,8,9`, so the current backend reads components `0..9`: about 640 MiB raw per block and 576 MiB of requested payload. Component 5 is currently wasted I/O.

The filesystem is expected to scale read bandwidth roughly linearly with node count, so the runtime should maximize delivered bandwidth by keeping every node near its per-node saturation point while avoiding duplicate block reads, remote raw-data movement, and unbounded queue growth.

## Goals

- Use all allocated nodes as independent Lustre readers.
- Keep each node near the measured saturation point of about 16 in-flight block reads.
- Read each plotfile FAB/block at most once per needed field set on its owning locality.
- Run kernels on the locality that owns the input block data.
- Preserve asynchronous task execution and avoid blocking HPX worker threads on I/O.
- Add enough instrumentation to verify per-node I/O utilization, queue depth, memory pressure, and delivered kernel input bandwidth.

## Non-Goals

- Do not redesign the AMReX plotfile format.
- Do not centralize raw input data on one reducer node.
- Do not split one block's fields across localities.
- Do not rely on radius-based decomposition if it causes rereads of the same block.
- Do not make global all-block preloading the default unless a memory budget explicitly allows it.

## Current Bottlenecks

### Per-field request explosion

`DataServiceLocal::get_local_shared_batch_impl()` receives field-level `ChunkRef` requests and enqueues one `DatasetLoadRequest` per field. The queue later tries to coalesce requests with the same dataset, chunk store, step, level, version, and block.

This works opportunistically, but the scheduler still exposes a per-field request stream to the load queue. In the flux-surface trace, this created a very large queue before useful work could complete.

### Task launch is not backpressured by I/O or memory

`Executor::run_stage()` launches one future per block for the whole stage. Each task independently discovers its inputs and triggers dataset loads. This lets a stage create far more outstanding input demand than the I/O subsystem or memory budget can satisfy efficiently.

### Blocking reads run inside HPX work

`PlotfileBackend::get_chunks()` performs blocking file reads. If these reads occupy normal HPX worker threads, the same worker pool must handle I/O, continuations, promise fulfillment, remote actions, and kernels.

### Coalescing is too coarse

`PlotfileBackend::get_chunks()` groups by `(level, block)` and reads the contiguous component range from `min_comp` to `max_comp`. For the flux fields, this reads one unused component. That is about 10% extra raw I/O for the current workload.

### Data layout conversion is part of the load path

After reading the FAB payload, the backend transposes each selected component into the runtime chunk layout. The microbenchmark measures the full `get_chunks()` path, so the observed bandwidth includes file I/O plus extraction and transpose cost.

## Target Architecture

### 1. Block-bundle load queue

Replace the field-level dataset load queue with a block-bundle queue keyed by:

```cpp
struct PlotfileBlockKey {
  const DatasetHandle* dataset;
  ChunkStore* chunk_store;
  int32_t step;
  int16_t level;
  int32_t version;
  int32_t block;
};
```

Each queue entry owns the union of requested fields for that block:

```cpp
struct PlotfileBlockLoad {
  PlotfileBlockKey key;
  std::vector<int32_t> fields;
  std::vector<ChunkRef> refs;
  double enqueue_time;
};
```

When a task asks for multiple fields from the same block, the data service should enqueue one block load. If later tasks request additional fields for a block already queued but not yet started, merge those fields into the same queue entry. If the read has already started, attach to existing per-field futures when available and enqueue a second narrow load only for genuinely missing fields.

Expected benefit:

- The load queue sees units that match the storage format.
- One I/O token corresponds to one FAB/block read.
- Queue depth and in-flight counts become meaningful predictors of bandwidth.

### 2. Per-node I/O token pool

Set the default plotfile dataset load concurrency to 16 block-bundle reads per physical node. If an HPX job starts more than one locality per node, divide the node budget across those localities or elect one locality-local I/O service for the node.

The concurrency limit should apply to blocking plotfile reads, not to field futures. Suggested environment variables:

- `KANGAROO_PLOTFILE_READ_CONCURRENCY=16`
- `KANGAROO_PLOTFILE_READ_MEMORY_BUDGET_GIB=<optional>`
- `KANGAROO_PLOTFILE_IO_POOL=plotfile_io`
- `KANGAROO_PLOTFILE_IO_POOL_THREADS=<optional, enables the dedicated HPX I/O pool>`

The current `KANGAROO_DATASET_LOAD_CONCURRENCY` can remain as a compatibility alias, but the plotfile path should use the more specific setting once implemented.

### 3. Dedicated I/O executor

Run blocking plotfile reads on a dedicated executor or thread pool. Normal HPX worker threads should submit work and process completions, but they should not spend long intervals inside `std::ifstream::read()`.

The basic flow should be:

1. Executor/task requests block fields.
2. Data service creates or finds field futures in the `ChunkStore`.
3. Block-bundle queue admits a read when an I/O token and memory budget are available.
4. Dedicated I/O thread performs `DatasetHandle::get_chunks()` or a lower-level plotfile block read.
5. Completion publishes all requested field futures through the chunk store.
6. Kernel continuations run on HPX workers after inputs are ready.

### 4. Locality-owned stage partitions

Change chunk-stage scheduling from "launch every block as an independent remote action" to "launch one partition runner per locality."

For each chunk stage:

1. Build the list of blocks for each locality using `DataServiceLocal::home_rank()` on the primary input block.
2. Send one `run_stage_partition` action to each locality with its block list.
3. On each locality, maintain a bounded window of blocks:
   - prefetch block bundles until I/O tokens or memory budget are full,
   - start a kernel as soon as that block's local inputs are ready,
   - publish outputs locally or through `put_host()`,
   - advance the window.

This turns the executor into a pipeline:

```text
local block queue -> bounded prefetch -> input-ready kernels -> local partial outputs
```

Expected benefit:

- I/O demand is shaped per locality.
- The executor no longer creates all block tasks and all input requests immediately.
- Memory pressure is bounded by the prefetch window and in-flight read budget.

### 5. Keep raw data local

The home rank for dataset chunks should remain based on `(step, level, block)` rather than field. All fields from a block should be read and consumed on the same locality.

For flux-surface style reductions:

1. Read and compute per-block contributions on the block's home locality.
2. Reduce contributions locally on each node.
3. Send only small partial reduction values to the final reducer.

Raw plotfile chunks should not move across the network unless the plan explicitly requires a neighbor block from another locality.

### 6. Smarter component reads

Teach the plotfile backend to read component runs instead of one `min_comp..max_comp` span.

For requested components:

```text
0,1,2,3,4,6,7,8,9
```

the backend should issue two payload reads:

```text
0..4
6..9
```

and publish the same field outputs. This saves component 5, about 64 MiB per tested level-8 block, or roughly 10% of raw I/O for the flux workload.

The implementation should keep coalescing when requested components are contiguous, because one larger sequential read can still be better than many tiny reads. The split policy should be simple at first:

- group requested components into contiguous runs,
- read each run,
- extract and transpose requested components from those run buffers,
- benchmark one-run versus run-split behavior before making it unconditional for all workloads.

### 7. Layout and copy improvements

The current load path allocates a raw FAB buffer and then creates one transposed `HostView` per component. After the block-bundle queue is in place, optimize this path in order:

1. Measure read time, extraction time, transpose time, and publish time separately.
2. Parallelize component transpose within a block if CPU time is significant.
3. Consider a block-bundle cache layout that stores all requested fields from one FAB together.
4. Consider letting kernels consume native plotfile component-major layout for kernels that can tolerate it.

These changes should come after the scheduler and queue changes, because they are second-order until I/O is kept saturated.

## Implementation Phases

### Phase 1: Make current behavior tunable and measurable

- Change the default plotfile load concurrency to 16 for Lustre runs.
- Keep the microbenchmark script and JSON outputs as a repeatable test path.
- Add event-log counters for:
  - block-bundle queue depth,
  - field futures waiting on each bundle,
  - in-flight block reads,
  - raw bytes read,
  - selected bytes published,
  - read duration,
  - transpose duration,
  - publish duration.
- Add a short README section or script help showing the standard saturation sweep.

Validation:

- `scripts/utils/plotfile_io_microbench.py` still shows ~1 GiB/s raw bandwidth around concurrency 16 on one node.
- Full flux-surface event logs show in-flight plotfile reads near 16 on each active locality.

### Phase 2: Introduce block-bundle queue

- Add a block-keyed queue entry type.
- Change `get_local_shared_batch_impl()` to group local unresolved refs by block before enqueueing.
- Merge queued field requests for the same block before the read starts.
- Fulfill all field futures for a block from one backend call.
- Preserve the current field-level `ChunkStore` API so kernels do not change yet.

Validation:

- Existing data-service async tests pass.
- A task requesting nine fields for one block creates one block-bundle load, not nine independent load requests.
- Event logs show queue depth in block units.

### Phase 3: Dedicated plotfile I/O executor

- Add a dedicated blocking-read executor for plotfile dataset loads.
- Ensure I/O threads are separate from kernel/continuation workers.
- Apply the 16-token limit to submitted blocking read jobs.
- Add optional memory budget admission control.

Validation:

- Under load, HPX worker threads continue to process promise fulfillment and task continuations.
- `put_outputs` and `wait_outputs` no longer stall behind large blocking-read waves.

### Phase 4: Locality partition runner

- Add a `run_stage_partition` action.
- In `Executor::run_stage()`, group chunk-stage blocks by target locality and dispatch one partition per locality.
- Implement a bounded local prefetch window.
- Start kernels as soon as each block's inputs are ready rather than waiting for the full partition.
- Keep graph stages unchanged initially, except that they should consume locality-reduced partials where possible.

Validation:

- Number of remote task actions becomes proportional to localities per stage, not blocks per stage.
- Per-node in-flight reads stay near the configured token count.
- Peak memory scales with window size and token count, not total block count.

### Phase 5: Smarter plotfile component runs

- Add an internal backend path that accepts a block and field list.
- Map field IDs to component IDs once per bundle.
- Split requested components into contiguous runs.
- Read only needed component runs.
- Preserve output ordering and missing-field behavior.

Validation:

- Flux fields read 576 MiB raw per level-8 block instead of 640 MiB, assuming two-run reads do not regress bandwidth enough to erase the savings.
- Microbench compares contiguous one-run mode with split-run mode.

### Phase 6: Reduction-aware execution

- For reduction-heavy plans, keep per-block computation on the data-owning locality.
- Add or use locality-level graph-reduce groups before final global reduction.
- Confirm final network traffic is small partial values rather than raw chunks.

Validation:

- Multi-node runtime improves approximately with node count until Lustre or reduction overhead dominates.
- Raw remote `get_host` traffic stays near zero for dataset-backed input chunks.

## Memory Budgeting

The measured flux field set uses about:

- 640 MiB raw read span per block with the current contiguous component range,
- 576 MiB selected field payload per block,
- additional transient memory during transpose and publication.

At 16 in-flight block reads, a node can transiently touch around 10 GiB of raw read buffers plus selected outputs, before accounting for kernels and cached chunks. The partition runner should therefore admit new prefetches using both:

- read-token availability,
- estimated memory availability.

Initial policy:

```text
max_inflight_blocks = min(read_concurrency, memory_budget / estimated_bytes_per_block)
```

For the current flux workload, use `estimated_bytes_per_block = raw_bytes + selected_bytes` until direct measurements show a tighter bound.

## Instrumentation Requirements

Every block-bundle read should emit enough data to compute:

- queue wait time,
- read duration,
- raw MiB/s,
- selected MiB/s,
- fields requested,
- components read,
- locality,
- I/O token count,
- memory-admission wait time,
- publish duration.

Every stage partition should emit:

- locality block count,
- active prefetch window size,
- kernels ready/running/completed,
- input wait time,
- output publish time,
- remote raw chunk fetch count.

These metrics should be visible in both JSONL event logs and Perfetto traces.

## Risks and Mitigations

- **Lustre variability can obscure small wins.** Use repeat sweeps and compare medians, not one timing.
- **More I/O concurrency can increase memory pressure.** Gate prefetch with a memory budget, not only read tokens.
- **Splitting component reads may add seek overhead.** Benchmark split-run mode before making it the default globally.
- **Dedicated I/O threads may oversubscribe CPU cores.** Make I/O thread count configurable and document expected HPX thread settings.
- **Partition scheduling may change task ordering.** Preserve stage dependency semantics and keep output `ChunkRef` ownership unchanged.
- **Neighbor-heavy kernels may still need remote chunks.** Keep raw data local for primary inputs and measure remote neighbor traffic separately.

## Success Criteria

- Single-node plotfile reads reach about 1.0 GiB/s raw bandwidth on the tested Lustre filesystem.
- Multi-node runs scale delivered input bandwidth roughly linearly with node count until a measured external limit appears.
- Full flux-surface runs no longer queue tens of thousands of field-level load requests up front.
- Event logs show each active node maintaining about 16 in-flight block reads during the input-heavy phase.
- Raw dataset-backed input chunks are read on their home locality and are not centralized.
- The flux workload reads no unnecessary component 5 bytes once split component runs are enabled and validated.
