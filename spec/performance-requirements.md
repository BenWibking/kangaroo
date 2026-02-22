# Performance, Scalability, and Memory Requirements

This document defines requirements for a *performant* Kangaroo implementation.

It is complementary to the functional specification. Functional conformance means behavior is correct. This document defines additional requirements for an implementation that is considered production-grade for large workloads.

## 1. Scope and Intent

A performant implementation is expected to:
- Preserve functional semantics from the core spec.
- Execute efficiently on modern heterogeneous systems.
- Scale across distributed-memory resources.
- Operate under explicit memory caps using streaming/out-of-core execution.

The requirements below are normative for the performance profile.

## 2. Performance Profile Levels

### 2.1 Baseline Functional Profile

- Satisfies behavior in functional spec.
- No guarantees on acceleration, scaling efficiency, or memory-capped execution.

### 2.2 Performant Profile

In addition to functional correctness, MUST satisfy:
- Native runtime support for GPU-offloaded kernel execution.
- Data-parallel chunk execution across distributed-memory processes.
- Streaming/out-of-core execution mode controlled by a memory usage cap.
- Remote-client deployment model where Python API control may run on a physically separate machine and different network from the native runtime cluster.

## 2.3 Remote Client / Cluster Deployment Model

A performant implementation MUST support control-plane operation where:
- The Python API process runs on a client machine that is physically separate from the compute cluster.
- The client machine may be on a different network from the cluster runtime processes.
- Connectivity may be provided through routed links or an SSH tunnel bridge.

In this model:
- Plan construction and orchestration requests originate from the Python client.
- Task execution and data-plane kernel work occur on the native runtime in the compute environment.
- Performance features in this profile (GPU offload, distributed chunk parallelism, memory-capped streaming) MUST remain available when connectivity is provided through the supported bridge path.

The implementation SHOULD tolerate control-plane latency and intermittent link variability without violating functional semantics.

## 3. GPU Offload Requirements

## 3.1 Kernel Execution Model

The native runtime MUST support executing kernels on at least one GPU backend.

This includes:
- Device execution path for data-parallel kernels.
- Device-compatible memory movement for chunk inputs/outputs.
- Correct handling of mixed host/device execution in one plan when required.

## 3.2 Correctness Under Offload

GPU execution MUST be semantically equivalent to CPU execution within defined floating-point tolerance for relevant operators.

Requirements:
- Same stage/template dependency ordering semantics.
- Same chunk identity and output placement semantics.
- Deterministic reduction semantics where deterministic mode is configured.

## 3.3 Data Movement Efficiency

Runtime SHOULD minimize host-device traffic by:
- Keeping intermediate buffers device-resident where possible.
- Fusing adjacent device-compatible operations when safe.
- Using asynchronous transfers and overlap where available.

## 3.4 Multi-GPU

A performant implementation SHOULD support multiple GPUs per node.

If multiple GPUs are present, scheduler SHOULD:
- Partition chunk tasks across devices.
- Avoid pathological imbalance when chunk costs are heterogeneous.
- Respect memory cap and per-device memory pressure.

## 4. Distributed-Memory Parallelism Requirements

## 4.1 Data-Parallel Chunk Execution

Runtime MUST support distributed execution where chunk tasks are placed across multiple processes/nodes.

Required behavior:
- Chunk ownership/sharding for data placement.
- Remote fetch/put for non-local chunk dependencies.
- Correct global execution of stage DAG and graph reductions.

## 4.2 Scaling Expectations

A performant implementation SHOULD demonstrate:
- Near-linear weak scaling for embarrassingly parallel chunk workloads until communication dominates.
- Strong scaling benefits for sufficiently large workloads.

The implementation SHOULD provide observability for:
- Task throughput.
- Communication volume/time.
- Synchronization wait time.

## 4.3 Communication and Serialization

Runtime SHOULD:
- Use compact payload formats for control messages and chunk metadata.
- Avoid redundant serialization of unchanged data.
- Batch small transfers when latency dominates.

## 5. Streaming / Out-of-Core Execution Requirements

## 5.1 Memory-Capped Mode

Runtime MUST provide an execution mode with explicit memory cap `M`.

In this mode:
- The scheduler MUST constrain in-memory working set to remain under `M` (subject to bounded control overhead).
- Chunks are streamed in from source (disk/network/memory service), processed, and streamed out.
- Intermediate retention MUST be bounded and eviction/backpressure aware.

## 5.2 Streaming Semantics

Under memory-capped mode, runtime MUST preserve functional equivalence with in-memory execution.

The implementation MUST support:
- Incremental loading of chunk inputs.
- Incremental emission of chunk outputs.
- Re-materialization or checkpoint-aware replay for values that cannot stay resident.

## 5.3 Backpressure and Flow Control

Runtime MUST implement backpressure between:
- Input streaming.
- Kernel execution.
- Output sinks.

Backpressure SHOULD prevent memory growth beyond configured cap and avoid unbounded task queue growth.

## 5.4 Spill and Eviction Policy

A performant implementation SHOULD provide:
- Configurable eviction strategy for cached chunks.
- Optional spill-to-disk for intermediate data when beneficial.
- Metrics for spill volume and cache hit/miss behavior.

## 6. Memory Usage Requirements

## 6.1 Memory Accounting

Runtime MUST track memory consumption for:
- Input chunk buffers.
- Output/intermediate chunk buffers.
- Neighbor/subbox buffers.
- Device memory buffers.
- Scheduler/task metadata overhead.

## 6.2 Bounded Growth

Runtime MUST avoid unbounded growth in:
- Task metadata structures.
- Buffered remote transfers.
- Deferred output queues.

## 6.3 Fragmentation and Reuse

A performant implementation SHOULD:
- Reuse buffers when safe.
- Pool allocations for common chunk sizes.
- Reduce allocation churn in steady-state streaming modes.

## 7. Scheduling and Throughput Requirements

## 7.1 Task Scheduling

Scheduler SHOULD optimize for:
- Data locality.
- Device affinity.
- Load balance across processes and devices.

## 7.2 Overlap

Implementation SHOULD overlap:
- I/O and communication with compute.
- Host-device transfer with kernel execution.
- Stage-level readiness checks with downstream preparation.

## 7.3 Reduction Efficiency

Graph reductions SHOULD use fan-in trees that avoid centralized bottlenecks and minimize depth/transfer volume for large input counts.

## 8. Observability and Performance Diagnostics

A performant implementation MUST expose enough telemetry to diagnose bottlenecks.

At minimum, SHOULD report:
- Stage and kernel runtimes.
- Queue wait times.
- Data transfer sizes and durations.
- Memory usage over time (host and device).
- Streaming throughput (in/out rates).

## 9. Configuration Requirements

A performant implementation SHOULD expose configuration controls for:
- Device enable/disable and backend selection.
- Distributed process topology parameters.
- Memory cap for out-of-core mode.
- Cache/spill policy parameters.
- Reduction fan-in tuning.

Configuration changes MUST NOT alter functional semantics.

## 10. Failure and Degradation Behavior

When performance features are unavailable at runtime (for example no GPU device present):
- Runtime MUST fail explicitly if feature is required by user configuration.
- Runtime MAY fall back to CPU path when fallback is permitted by configuration.

When memory cap is too low to execute even minimal working set:
- Runtime MUST fail explicitly with actionable error context.

## 11. Conformance Guidance for This Profile

To claim the performant profile, an implementation should provide:
- Evidence of GPU-offloaded kernel support in native runtime.
- Evidence of distributed chunk execution across processes.
- Evidence of memory-capped streaming execution that preserves correctness.
- Observability outputs that allow validating the above in practice.
