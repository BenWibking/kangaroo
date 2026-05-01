# Implementation Pitfalls

This document records non-normative implementation hazards discovered while building and tuning Kangaroo. These notes are not conformance requirements, but they describe failure modes that can preserve API behavior while making the runtime hang or collapse under production workloads.

## HPX Future Completion Reentrancy

Do not assume that `hpx::promise::set_value()` is a passive bookkeeping operation. Completing an HPX future may execute attached continuations immediately on the thread that calls `set_value()`.

This is dangerous in runtime plumbing such as dataset-load workers and output publication paths. If those paths complete futures inline, a worker that is still inside I/O or chunk-store delivery can re-enter task execution, run kernels, publish outputs, or trigger more dataflow before the original operation has unwound. With coalesced multi-field reads, this can produce partial progress followed by a deadlock-like stall: the event log stops, many tasks remain open in `wait_inputs`, and one or more tasks may be left in `kernel` or `put_outputs`.

The safe pattern is:

1. Update the shared chunk store state under its mutex.
2. Release the mutex.
3. Fulfill the promise asynchronously, for example with `hpx::post`.

This keeps arbitrary task continuations out of the critical I/O and output publication call stack, while still allowing later readers to observe that the chunk is ready in the store.
