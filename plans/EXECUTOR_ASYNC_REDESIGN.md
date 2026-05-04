# Executor Async Redesign Plan

## Context

The current executor mixes HPX futures with several blocking or globally synchronized code paths. That was already hard to reason about, and the recent change to `DataServiceLocal` made the problem visible by turning cache misses into blocking waits on a `std::condition_variable`. Under load, that can starve HPX workers and leave producers stuck in `wait_outputs` while consumers block in `wait_inputs`.

This document proposes a clean redesign of the executor-side runtime model so that:

- readiness is represented by HPX futures, not condition variables,
- task execution composes asynchronous dependencies instead of blocking on them,
- shared executor metadata is immutable or per-run,
- hot-path locks are reduced to short metadata lookups,
- setup and teardown barriers stay synchronous only at the public runtime boundary.

## Audit Summary

### High-priority problems

1. `DataServiceLocal` now blocks on cache miss.
   - `get_local_impl()` waits on a global `condition_variable`.
   - `get_host()` exposes an async API but performs a synchronous local wait.
   - `put_host()` uses `notify_all()`, which is coarse and unrelated to HPX scheduling.

2. `AdjacencyServiceLocal` is not thread-safe.
   - It mutates an internal cache from concurrent executor tasks without synchronization.
   - One `Executor` shares one adjacency service across many tasks on the same locality.

3. There is still internal synchronous waiting in execution paths.
   - Some code does `future.get()` inside kernel or helper paths rather than composing futures.
   - Local task launch wraps async work in a lambda that immediately calls `.get()`.

### Medium-priority structural problems

4. Task execution depends on process-global mutable state.
   - Run metadata, dataset handles, and plans are stored globally and accessed under `g_ctx_mutex`.
   - Remote task actions reconstruct execution from that global state.

5. The hot path still pays for repeated synchronized lookups.
   - Kernel lookup in `KernelRegistry` takes a mutex.
   - Parameter decode caches also take mutexes during task execution.

6. Some recent debug-oriented changes added unconditional work in projection kernels.
   - This is a performance regression, but it is not the main executor design issue.

### Acceptable synchronous boundaries

These are not the main redesign target:

- runtime-level setup barriers such as broadcasts before execution,
- waiting for the full plan to complete at the public API boundary,
- output retrieval after a run has finished,
- side-channel workers for event logs and Perfetto traces.

## Design Principles

1. **Inside plan execution, HPX futures and continuations are the only readiness mechanism.**
2. **Mutexes may protect metadata maps, but they must not represent data readiness.**
3. **A cache miss must produce an unresolved future, not a blocked worker thread.**
4. **Per-run state should be explicit and locality-owned, not process-global by default.**
5. **Adjacency and plan metadata should be immutable once execution starts.**
6. **Any synchronous wait that remains must be clearly outside the task scheduler hot path.**

## Target Architecture

### 1. Per-run `ExecutionContext`

Introduce a per-run execution object, installed on every locality before tasks start.

It should hold:

- `RunMeta`
- `DatasetHandle`
- compiled `PlanIR`
- immutable adjacency index
- kernel table or compiled kernel references
- event logging / tracing sinks
- locality-owned async chunk store

Instead of remote task actions reading `global_runmeta()`, `global_dataset()`, and `global_plan()`, they should take a `run_id` and look up the corresponding `ExecutionContext`.

Benefits:

- removes hidden coupling to process-global state,
- makes lifetime explicit,
- makes concurrent or overlapping runs easier to reason about later,
- allows task actions to use the same async services as local execution without rebuilding ad hoc state.

### 2. Async chunk store replacing blocking `DataServiceLocal`

The public `DataService` interface is already future-shaped and should remain so:

- `get_host() -> hpx::future<HostView>`
- `get_subbox() -> hpx::future<SubboxView>`
- `put_host() -> hpx::future<void>`

The implementation should become a locality-owned async store keyed by `ChunkRef`.

#### Slot model

Each `ChunkRef` gets one slot with:

- a small mutex or spinlock for metadata,
- a state enum such as `empty`, `loading`, `ready`, `failed`,
- a `hpx::shared_future<HostView>` representing readiness,
- a producer-side promise held only while the slot is unresolved,
- optional cached value / exception.

#### `get_host(ref)`

- Returns immediately with a future.
- Never blocks.
- If the data is already ready, returns a ready shared future.
- If the chunk is dataset-backed and not yet loading, initiates exactly one async load and fulfills the slot when done.
- If a load is already in flight, attaches to the same shared future.
- If the chunk is produced by another task, returns the unresolved slot future and lets the producer fulfill it later via `put_host()`.

#### `put_host(ref, view)`

- Fulfills the slot promise for exactly that `ChunkRef`.
- Returns an HPX future for any remote delivery work.
- Does not broadcast wakeups with `notify_all()`.
- Becomes idempotent or explicitly checks for duplicate producers.

#### `get_subbox(ref)`

- Implement as `get_host(ref.chunk).then(...)`.
- Slice the returned `HostView` in a continuation.
- No `get()` inside the helper.

#### Remote ownership

Home-locality ownership should remain, but future composition must be preserved:

- local get: return the local slot future directly,
- remote get: send an HPX action to the owning locality and `unwrap()` the returned future,
- remote put: send an action that fulfills the owning slot.

The owning locality is the only place where readiness is materialized.

### 3. Executor task flow should compose futures directly

The executor should stop wrapping async work inside lambdas that call `.get()`.

#### Current anti-pattern

Local task submission currently does the equivalent of:

```cpp
return hpx::async([...] {
  run_block_task_impl(...).get();
});
```

That is better than a raw thread block because HPX can suspend the HPX thread, but it still obscures the dependency graph and makes it easy to reintroduce blocking behavior.

#### Target model

- local task submission returns `run_block_task_impl(...)` directly,
- remote task submission returns the remote action future and unwraps it if needed,
- `run_stage()` returns `when_all(...)` over task futures,
- `run()` returns `when_all(...)` over stage futures.

Inside `run_block_task_impl()` and `run_graph_task_impl()`:

- request input futures,
- compose with `when_all`,
- decode ready results only inside continuations where they are known-ready,
- invoke kernels,
- chain output publication futures,
- finish with a single future representing task completion.

The only `get()` calls allowed inside these paths should be on futures that are already known-ready because they were produced by a completed `when_all` continuation.

### 4. Immutable adjacency index

`AdjacencyServiceLocal` should stop using a mutable cache.

Instead:

- precompute adjacency once per run or once per level during context construction,
- store it as immutable arrays or vectors,
- make `neighbors()` a const lookup with no mutation and no locks.

This removes a data race and makes neighbor queries deterministic and cheap.

### 5. Compile kernels and params before execution

Move per-task synchronized lookup work out of the hot path.

#### Kernel lookup

Instead of calling `KernelRegistry::get_by_name()` for every task:

- resolve kernel function pointers during plan setup,
- store them in compiled task templates,
- use direct references during execution.

#### Parameter decode

Instead of decoding or cache-checking typed params per task:

- parse typed params once during plan decode or plan compilation,
- attach typed, immutable parameter objects to compiled templates,
- pass references or shared pointers into kernel invocation.

This does not need to be fully generic on day one. Even moving the most frequent graph and projection params out of the hot path is worthwhile.

### 6. Keep runtime setup/teardown synchronization at the API boundary

The redesign does not require everything to become fully asynchronous from Python down.

It is still reasonable that:

- runtime setup broadcasts complete before execution starts,
- `Runtime::execute_plan()` waits for the executor future before returning,
- output retrieval waits for a single fetch future after the run is done.

The key rule is narrower:

> synchronous waiting is allowed at runtime boundaries, but not inside the task scheduler hot path.

### 7. Side-channel workers stay isolated

Event logging and Perfetto writing already use background worker threads. They can remain as they are, provided they do not become task readiness primitives.

They should continue to be observational only.

## Proposed API and Data Model Changes

### `ExecutionContext`

Introduce something like:

```cpp
struct ExecutionContext {
  int32_t run_id;
  RunMeta runmeta;
  DatasetHandle dataset;
  CompiledPlan plan;
  AdjacencyIndex adjacency;
  AsyncChunkStore chunk_store;
  CompiledKernelTable kernels;
  EventSink* events = nullptr;
};
```

Runtime keeps a map from `run_id` to `shared_ptr<ExecutionContext>` per locality.

### `CompiledPlan`

Convert `PlanIR` into an execution-ready form:

- resolved stage DAG,
- compiled task templates,
- resolved kernel references,
- parsed typed params,
- precomputed graph reduction metadata.

### `AsyncChunkStore`

Suggested internal API:

```cpp
class AsyncChunkStore {
 public:
  hpx::shared_future<HostView> get(const ChunkRef& ref);
  hpx::future<SubboxView> get_subbox(const ChunkSubboxRef& ref);
  hpx::future<void> put(const ChunkRef& ref, HostView view);
};
```

This can either replace `DataServiceLocal` directly or sit behind it.

## Migration Plan

### Current implementation checklist

- [x] Remove the blocking `condition_variable` wait from `DataServiceLocal`.
- [x] Add async chunk slots backed by `hpx::promise` / `hpx::shared_future`.
- [x] Make pending `get_host()` consumers attach to one unresolved slot future.
- [x] Make dataset-backed loads coalesce into one in-flight load per `ChunkRef`.
- [x] Reimplement `get_subbox()` as future composition.
- [x] Add regression coverage for many graph-reduce consumers waiting on a later producer.
- [x] Add direct local dispatch for chunk tasks.
- [x] Introduce per-run `ExecutionContext` registration on each locality.
- [x] Update remote task actions to resolve execution by `run_id`.
- [x] Precompute adjacency during run setup and remove executor-path cache mutation.
- [x] Resolve kernel references during plan preparation.
- [x] Prepare typed kernel params during plan preparation for kernels with registered preparers.
- [ ] Guard or remove remaining unconditional projection debug bookkeeping.
- [ ] Return the local graph task implementation future directly instead of wrapping it in `hpx::async(... unwrap ...)`.
- [ ] Make the adjacency API explicitly immutable, for example by making `neighbors()` `const`.
- [ ] Remove remaining kernel reliance on thread-local `current_runmeta()` / `current_dataset()`.
- [ ] Ensure helper-created `DataServiceLocal` instances inside execution use the active run context.
- [ ] Promote the prepared `PlanIR` metadata into an explicit `CompiledPlan` model.
- [ ] Audit remaining internal `.get()` calls and classify each as boundary wait, known-ready continuation unwrap, or a real sync wait to remove.
- [ ] Add multi-locality tests for remote get and remote put.
- [ ] Add a race-oriented test for concurrent adjacency queries.
- [ ] Re-run the original stalled projection workload and compare event/progress metrics.

### Phase 0: Remove known regressions before structural work

1. Remove the blocking `condition_variable` wait from `DataServiceLocal`.
2. Guard or remove unconditional projection debug bookkeeping.
3. Add a regression test that exercises concurrent graph-reduce consumers waiting on producer outputs.

This gets the system back to a safe baseline before the larger redesign lands.

### Phase 1: Introduce async-ready chunk slots

1. Refactor `DataServiceLocal` storage from `ChunkRef -> HostView` into `ChunkRef -> Slot`.
2. Implement slot readiness with `hpx::promise` / `hpx::shared_future`.
3. Make `get_host()` return immediately with a future on both hit and miss.
4. Make dataset-backed loads coalesce into one in-flight future per `ChunkRef`.
5. Reimplement `get_subbox()` as future composition.

Deliverable:

- no blocking waits inside `DataServiceLocal`,
- all data dependencies represented by HPX futures.

### Phase 2: Remove local wrapper waits in the executor

1. Update `Executor::run_block_task()` and `run_graph_task()` to return direct futures for local tasks.
2. Keep remote actions async and unwrap them.
3. Audit `run_block_task_impl()` and `run_graph_task_impl()` so every dependency transition is a continuation.

Deliverable:

- executor no longer uses “launch async then `.get()` immediately” for local tasks.

### Phase 3: Make adjacency immutable

1. Replace `AdjacencyServiceLocal` cache mutation with a precomputed adjacency index.
2. Build adjacency during run setup.
3. Make `neighbors()` lock-free and const.

Deliverable:

- no executor-path data races in adjacency lookup.

### Phase 4: Move execution to explicit per-run contexts

1. Introduce `ExecutionContext`.
2. Register one context per locality for each active run.
3. Update remote task actions to take `run_id` and resolve context from a locality map.
4. Remove executor reliance on `global_runmeta()`, `global_dataset()`, and `global_plan()`.

Deliverable:

- task execution depends on explicit run context rather than global mutable state.

### Phase 5: Compile kernels and params

1. Add a compilation pass from `PlanIR` to `CompiledPlan`.
2. Resolve kernel references once.
3. Parse typed params once and attach them to compiled templates.
4. Simplify runtime hot path to use compiled entries directly.

Deliverable:

- no per-task kernel registry mutex lookup,
- no per-task typed param cache traffic.

### Phase 6: Clean up remaining internal sync waits

1. Audit kernels and helper paths for `.get()` on data futures.
2. Convert those paths to continuations where they participate in task execution.
3. Keep only boundary waits in runtime entrypoints.

Deliverable:

- no internal blocking waits on data or task dependencies.

## Validation Plan

### Correctness tests

1. Add a unit test where many graph-reduce tasks request the same not-yet-produced chunk and verify no deadlock.
2. Add tests for:
   - producer-before-consumer,
   - consumer-before-producer,
   - multiple consumers on one chunk,
   - remote get and remote put across localities.
3. Add a race-oriented test for concurrent adjacency queries.
4. Add tests that `get_subbox()` works correctly when the backing chunk arrives asynchronously.

### Performance tests

1. Re-run the stalled projection workload that triggered the timeout.
2. Compare:
   - total runtime,
   - event log progress continuity,
   - time spent in `wait_inputs` and `wait_outputs`,
   - per-locality worker utilization.
3. Add a benchmark that isolates graph-reduce fan-in behavior under load.

### Instrumentation checks

1. Event logs should continue progressing until plan completion or a real error.
2. No long plateau where the event log stops while the job remains alive.
3. If a deadlock happens, it should be diagnosable from unresolved slot futures rather than hidden behind thread blocking.

## Acceptance Criteria

The redesign is successful when all of the following are true:

1. There are no `std::condition_variable` waits or equivalent blocking readiness waits in executor hot paths.
2. `DataService` readiness is fully represented by HPX futures.
3. Local executor dispatch does not call `.get()` inside wrapper tasks.
4. `AdjacencyService` is immutable or otherwise thread-safe without hot-path mutation.
5. Remote task actions run from explicit per-run context rather than implicit global plan state.
6. The original projection workload completes without the stalled-worker behavior seen in the timeout run.

## Open Questions

1. Should the async chunk store live behind `DataServiceLocal`, or should `DataService` grow a slightly richer API around slot ownership and readiness?
2. Do we want one `ExecutionContext` per plan run only, or a longer-lived locality context with multiple active runs inside it?
3. Should compiled params be type-erased behind a small virtual interface or stored as templated payloads on compiled task templates?
4. Do we want to preserve the current remote action granularity, or move toward longer-lived per-stage schedulers on each locality?

## Recommended Implementation Order

If this work is done incrementally, the order should be:

1. async chunk slots,
2. direct future composition in executor dispatch,
3. immutable adjacency,
4. explicit per-run contexts,
5. compiled kernels and params,
6. final cleanup of remaining internal waits.

That order addresses the deadlock risk first, then simplifies the architecture without forcing the whole runtime to be rewritten in one patch.
