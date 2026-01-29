Here’s a **clean, implementation-ready summary** of the entire design so you can paste it into a Codex CLI session and start building.

---

# Project Summary: Distributed Task-Based Analysis Runtime for AMR Data

## Goal

Build a **distributed, low-latency task runtime for AMR post-processing and in-situ analysis**, where:

* **C++ + HPX** handle execution, scheduling, and data movement.
* **Python provides a DSL** for operator authoring and workflow composition.
* Operators are written **mostly in Python**, except for **kernels that touch bulk data**, which are written in **C++**.
* Task graphs may be **dynamic and complex**, but the prototype implements **only the chunk plane** first.
* Must scale to **millions of AMR blocks**, multi-node, with **low overhead**.

---

# Core Architecture

## Execution Planes

We separate execution into **two conceptual planes**:

### Chunk Plane (implemented first)

* Data-parallel execution over AMR blocks.
* Regular communication patterns (face neighbors, halos).
* Stencil kernels, reductions, block-local transforms.
* Implemented now.

### Graph Plane (deferred)

* Irregular dependencies (TDA, feature tracking, union-find, global graph algorithms).
* Only reserve interfaces now (plane enum + placeholders).

All tasks declare which plane they use; executor only implements `Chunk` initially.

---

# Metadata & AMR Model

## Replicated Metadata (on all ranks)

We replicate only **minimal AMR metadata**, making block identity implicit:

```cpp
struct BlockBox { Int3 lo, hi; };

struct LevelMeta {
  LevelGeom geom;
  span<BlockBox> boxes;   // block i == boxes[i]
};

struct StepMeta {
  vector<LevelMeta> levels;
};

struct RunMeta {
  vector<StepMeta> steps;
};
```

### Key Design Choice

**Block identity is implicit**:
A block is simply `(step, level, block_index)`.

No explicit IDs or keys stored.

---

## Geometry Container

Physical coordinates are computed lazily:

```cpp
struct LevelGeom {
  double dx[3];
  double x0[3];
  int ref_ratio;
};
```

Block physical extents are computed from `(geom + logical box)` only when needed.

---

# Chunk Identity

Chunk references passed to the runtime:

```cpp
struct ChunkRef {
  int step;
  int level;
  int field;
  int version;
  int block;
};
```

This is all that’s needed for addressing data.

---

# Adjacency (Irregular Block Sizes)

Blocks **do not have uniform sizes**, so adjacency is computed **on demand**, not stored.

### Strategy

* **Replicate only block boxes**
* **Compute adjacency lazily**
* **Shard adjacency computation by block ownership**
* Cache only what is needed

### On-demand adjacency interface:

```cpp
enum class Face { Xm, Xp, Ym, Yp, Zm, Zp };

class AdjacencyService {
public:
  NeighborSpan neighbors(step, level, block, face);
};
```

### Implementation Strategy

Use **face hashing**:

* For each block face, generate:

  * `(face, plane coordinate, 2D rectangle)`
* Neighbors are blocks whose opposite face:

  * lies on the same plane
  * whose 2D face rectangles overlap

Adjacency can be:

* computed lazily
* cached per `(block, face)`
* sharded by home rank
* discarded when metadata epoch changes

---

# Data Service

Responsible for:

* distributed chunk storage
* caching
* remote fetch

```cpp
class DataService {
public:
  int home_rank(ChunkRef);
  future<HostView> get_host(ChunkRef);
  future<void> put_host(ChunkRef, HostView);
};
```

Ownership policy:

```cpp
home_rank = hash(step, level, block) % nranks
```

Tasks execute **where their data lives**.

---

# Execution Model (Chunk Plane)

## Task Templates

Python constructs **templates**, not tasks.

```cpp
struct TaskTemplate {
  string name;
  ExecPlane plane;
  string kernel;
  Domain domain;         // step, level, blocks
  vector<FieldRef> inputs;
  vector<FieldRef> outputs;
  DepRule deps;          // None or FaceNeighbors
  bytes params;          // msgpack/json blob
};
```

## Stage

```cpp
struct Stage {
  string name;
  ExecPlane plane;
  vector<int> after;     // dependencies
  vector<TaskTemplate> templates;
};
```

## Plan

```cpp
struct Plan {
  vector<Stage> stages;
};
```

Execution:

* Stages run in topological order.
* Templates expand to **one HPX task per block**.
* Dependencies handled via futures.
* Neighbor dependencies fetch halos using adjacency service.

---

# Kernel Model (C++ only)

Python **never touches bulk data**.

All data-touching code lives in C++ kernels:

```cpp
using KernelFn = future<void>(
    const LevelMeta& level,
    int block_index,
    span<const HostView> inputs,
    NeighborViews nbrs,
    span<HostView> outputs,
    span<const uint8_t> params
);
```

NeighborViews:

```cpp
struct NeighborViews {
  span<const HostView> xm, xp, ym, yp, zm, zp;
};
```

Kernels registered in:

```cpp
class KernelRegistry {
public:
  KernelID register_kernel(desc, fn);
  KernelID lookup(name);
  vector<KernelDesc> list();
};
```

---

# Python DSL (Operator Authoring Layer)

## Key Design

* **Operators written in Python**
* **Kernels in C++**
* Python builds a **plan**, submits once.

### Python Operator Example

```python
class VorticityMag(Op):
    def lower(self, ctx):
        dom = ctx.domain(step=ds.step, level=ds.level)

        grad = ctx.temp_field("gradU")
        vort = ctx.output_field("vortmag")

        s1 = ctx.stage("grad")
        s1.map_blocks(
            kernel="gradU_stencil",
            domain=dom,
            inputs=[vel],
            outputs=[grad],
            deps={"kind": "FaceNeighbors", "width": 1},
            params={"order": 2},
        )

        s2 = ctx.stage("vortmag", after=[s1])
        s2.map_blocks(
            kernel="vorticity_mag",
            domain=dom,
            inputs=[grad],
            outputs=[vort],
            deps={"kind": "None"},
            params={},
        )

        return [s1, s2]
```

---

# Python → C++ Interface

Python builds plan → serializes to msgpack → C++ decodes → executor runs.

```python
rt.run(plan, runmeta, dataset)
```

C++:

```cpp
void Runtime::run_packed_plan(bytes, runmeta_handle, dataset_handle);
```

---

# Prototype Implementation Roadmap

## Phase 1 — Core Runtime

* HPX executor
* Kernel registry
* DataService (basic, no GPU)
* Metadata replication

## Phase 2 — Chunk Plane

* Task template expansion
* Neighbor adjacency via face hashing
* Halo fetch
* Simple stencil kernels

## Phase 3 — Python DSL

* Plan builder
* Operator lowering API
* Plan serialization

## Phase 4 — Performance

* Caching
* Batched neighbor fetch
* Memory pooling
* GPU kernels (later)

---

# Key Design Principles

* **Python = control plane**
* **C++ = data plane**
* **No Python in hot execution loop**
* **No explicit task DAG materialization**
* **No explicit block IDs — array index is identity**
* **Adjacency is derived, cached, and sharded**
* **Execution always scheduled near data**

---

# Deferred Work (Future Graph Plane)

* Distributed entity table
* Distributed union-find
* Irregular graph algorithms
* TDA, feature tracking, segmentation

Design already supports this via:

```cpp
enum ExecPlane { Chunk, Graph, Mixed };
```

---

# Suggested Codex Prompt

> Implement a C++ + HPX distributed task runtime with Python DSL frontend, based on the following design:
> [paste this summary]

