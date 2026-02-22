# Kangaroo Design Overview (Non-Normative)

This document explains the architecture and design intent behind Kangaroo in plain terms.

Unlike the other specification files, this overview is descriptive, not normative. It is meant to give a new contributor or implementation agent the right mental model before reading the strict behavioral contracts.

## 1. What Kangaroo Is

Kangaroo is a hybrid analysis system with:
- A Python front-end for expressing workflows.
- A native runtime back-end for executing those workflows over AMR and particle data.

The front-end focuses on expressiveness and composition. The back-end focuses on data access, parallel execution, and numeric kernels.

## 2. Python/C++ Boundary

A useful way to think about the boundary is:
- Python decides *what work should happen* and in what dependency order.
- The native runtime decides *where and when work is executed* and performs the heavy data-touching operations.

### Python side responsibilities

The Python layer does the following:
- Builds symbolic workflow fragments via `Pipeline` operations.
- Allocates and wires field IDs.
- Encodes domain and dependency intent (`step`, `level`, blocks, neighbor deps, reduction structure).
- Produces a plan object and serializes it into a transportable representation.

### Native runtime side responsibilities

The native layer does the following:
- Decodes plan payloads.
- Validates stage and template structure.
- Schedules concrete task instances (per-block or per-reduction-group).
- Fetches input chunks/subboxes/neighbor data.
- Runs registered kernels.
- Writes outputs back to chunk-addressed storage.

This split keeps workflow construction ergonomic in Python while keeping core numerical and execution hot paths in native code.

## 3. Pipeline as Staged Compilation

Kangaroo’s pipeline API is best viewed as a lightweight compiler pipeline.

A core design choice is that the task DAG is constructed before execution begins. Rather than interleaving graph construction with runtime scheduling, Kangaroo first materializes an explicit whole-plan graph, then executes that graph.

### Step A: High-level operator composition

Users compose analysis with operations like:
- field expressions
- AMR slice/projection
- histograms
- particle filters/reductions

These calls return handles and append symbolic fragments.

### Step B: Lowering to plan fragments

Each operation lowers into one or more stages and templates. For example, one high-level operator may lower into:
- block-local accumulation stage(s)
- one or more graph reduction stage(s)
- output finalize stage

### Step C: Whole-pipeline plan assembly

Fragments are stitched together with explicit stage dependencies. The result is a plan DAG.

This assembly phase is intentionally complete: the runtime receives a fully formed dependency graph, not a partially discovered stream of tasks.

### Step D: Serialized IR handoff

The plan is serialized and handed to the runtime. At this point, Python has finished describing intent and the runtime takes over execution.

Because the graph is complete before execution, optimization passes can reason over the full dependency structure. That enables global transformations such as:
- fusing compatible kernels across adjacent graph regions
- splitting heavy kernels or wide stages into finer-grained tasks
- reordering independent subgraphs for better locality or overlap

## 4. Plan IR as the Contract Layer

The plan intermediate representation (IR) is the key contract between front-end and runtime.

It captures:
- Stage DAG structure.
- Template-level kernel calls.
- Domain scoping (`step`, `level`, optional blocks).
- Input/output field references.
- Dependency policy (including neighbor rules).
- Per-template params payload.

The IR is designed to be complete enough that the runtime can focus on execution mechanics rather than graph discovery logic.

This IR is intentionally compact and explicit. It is not a general programming language; it is a purpose-built execution description for analysis tasks.

## 5. Why Stages and Templates

The stage/template model separates concerns cleanly:
- A **stage** says "these templates can run after these other stages."
- A **template** says "instantiate this kernel over this domain."

This makes lowering predictable and makes execution orchestration straightforward:
- expand template domains
- place tasks
- run kernels
- materialize outputs

It also creates optimization boundaries that are explicit in the graph. Those boundaries can be merged (fusion) or subdivided (splitting) during planning/tuning without changing user-facing workflow code.

## 6. Chunk-Centric Data Model

Kangaroo data movement and persistence revolve around chunk identity:
- `(step, level, field, version, block)`

This identity is used for:
- all runtime reads/writes
- task input/output wiring
- ownership/locality decisions
- output retrieval

The model works uniformly for synthetic memory-backed workflows and file-backed workflows.

## 7. AMR-Aware Composition Pattern

A recurring pattern in Kangaroo operators is:
1. Do block-local work on each level.
2. Mask covered coarse regions to avoid coarse/fine double counting.
3. Reduce within level and across levels.
4. Finalize result into user-facing array form.

This pattern appears in slices, projections, and AMR histograms.

## 8. Mesh vs Particle Workflow Flavor

Kangaroo supports two distinct but related workflow styles.

### Mesh/field workflows

These are naturally block-structured and map directly onto AMR block domains.

### Particle workflows

These often look like array algebra and reductions. Kangaroo still executes them through staged tasks, often with virtual chunking and graph reductions.

The unified pipeline API allows both styles to coexist in one user workflow.

## 9. Runtime Kernel Registry Model

Kernels are identified by names that appear in plan templates.

Conceptually:
- The plan names *what kernel* to run.
- The runtime registry resolves that name to executable native logic.

This makes the plan portable across compatible runtime instances and decouples lowering from concrete function pointers.

## 10. Event Logging and Observability

Kangaroo includes a task-event stream designed for runtime introspection.

The dashboard consumes:
- task lifecycle events
- runtime metric events
- optional plan artifacts

This creates a feedback loop for understanding:
- DAG structure
- execution timing and worker utilization
- bottleneck patterns

## 11. Why This Architecture Exists

The architecture is a pragmatic compromise:
- Keep user experience high-level and iterative in Python.
- Keep data-plane execution and kernels in native code.
- Use an explicit IR boundary to avoid hidden coupling.
- Preserve AMR semantics while still supporting particle-centric analysis.
- Build the full DAG ahead of execution so the runtime stays simpler and optimization opportunities are maximized.

In short: Python is the workflow language, the plan IR is the contract, and the native runtime is the execution engine.
