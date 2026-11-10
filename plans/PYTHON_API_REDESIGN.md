# Python API Redesign

## Status

Implemented on 2026-07-19. The preferred surface is the `kangaroo` package;
`analysis` remains as the compatibility and low-level namespace. The normative
contract, including lazy execution, provenance-derived dependencies, explicit
gathering, persistence, diagnostics, and advanced namespaces, is recorded in
[`spec/python-api.md`](../spec/python-api.md).

## Summary

Kangaroo already has the right underlying pieces for a high-level lazy analysis
API: symbolic field handles, typed plans, self-describing chunk buffers, dataset
metadata, and a distributed runtime. The Python surface currently makes users
coordinate those pieces explicitly.

The redesigned API should follow Dask's execution model and PyTorch/NumPy's
expression ergonomics:

- scientific values are lazy objects with familiar methods and operators,
- dependencies are derived from value provenance rather than statement order,
- `compute()` is the explicit boundary that returns local Python results,
- `persist()` retains distributed results without gathering them,
- result objects own their materialization coordinates and metadata,
- runtime, plan, field-ID, and chunk-addressing details remain available through
  advanced submodules rather than the default workflow.

Dask is the primary model because Kangaroo is a distributed, lazy computation
system. PyTorch is a secondary influence for operator overloading, consistent
metadata, dtype conversion, concise representations, and composable callable
operations. Kangaroo should not copy APIs whose semantics do not fit AMR data.

## Goals

1. Make a first plotfile analysis require only dataset selection, expression
   construction, and `compute()`.
2. Give mesh fields, particle arrays, masks, reductions, and structured
   scientific results one consistent lazy execution model.
3. Allow independent expression branches to execute concurrently.
4. Make common operations discoverable through normal Python indexing,
   operators, methods, type hints, docstrings, and rich representations.
5. Preserve explicit access to distributed chunks and low-level plan building
   for advanced users.
6. Introduce the product namespace `kangaroo` without abruptly breaking the
   existing `analysis` imports.

## Non-goals

- Replacing the typed plan, kernel catalog, FlatBuffer handoff, or C++ runtime.
- Hiding chunked or distributed behavior when it affects correctness or memory
  use.
- Pretending that an AMR hierarchy is always one dense NumPy array.
- Copying PyTorch autograd, training modes, or device APIs that do not represent
  Kangaroo concepts.
- Making every low-level runtime operation a method of the high-level array
  type.

## Current API Assessment

### Strengths to preserve

- `Pipeline` provides a useful high-level lowering layer over `Plan` and
  `Stage`.
- Handles already retain domain metadata such as histogram edges, flux
  components, and Toomre-Q radial bins.
- Dataset backends share a single `open_dataset()` entry point.
- Operator construction validates important scientific and storage constraints
  before execution.
- Chunk descriptors carry authoritative dtype, shape, and layout information.
- The low-level plan API is useful for tests, kernel development, and expert
  workflows.

### Main usability problems

#### 1. Execution and materialization are detached from the result

A normal slice currently requires users to create a `Runtime`, extract
`RunMeta`, resolve an integer field ID, derive plane geometry, build a pipeline,
run the plan, and retrieve a chunk using `step`, `level`, `field`, `version`, and
`block`.

Those coordinates are valid runtime concepts, but a high-level result already
has enough provenance to own them. The standard boundary should be:

```python
image = ds["density"].slice(axis="z", resolution=(512, 512))
array = image.compute()
```

Direct chunk retrieval remains available as an advanced operation.

#### 2. Lazy semantics differ between mesh and particle operations

Mesh operators return handles and execute later. Particle reductions such as
`particle_sum()` and `particle_min()` execute while the method is called and
return concrete Python scalars. Particle `iter_chunks()` also triggers execution
implicitly.

Every scientific operation should construct a lazy value. Only an explicitly
named execution or materialization method should launch work:

```python
total_mass = particles["mass"].sum()  # lazy Scalar
value = total_mass.compute()          # executes and returns float
```

#### 3. The imperative frontier serializes independent expressions

The current pipeline appends every new fragment after its current frontier.
This imposes definition-order dependencies even when the new fragment does not
consume an earlier result.

The redesigned graph must derive edges from handle provenance. Two independent
histograms should remain independent branches and share only the source and
execution context that they actually have in common.

#### 4. Operations live on `Pipeline` instead of their values

Calls such as `pipe.field_add(a, b)`, `pipe.particle_gt(mass, 0.0)`, and
`pipe.particle_and(a, b)` are harder to discover and compose than familiar
Python expressions:

```python
pressure = density * temperature
mask = (mass > 0.0) & mass.isfinite()
selected = mass[mask]
```

The existing string-based `field_expr()` remains useful as an escape hatch, but
ordinary arithmetic, comparisons, masking, casts, and reductions belong on
lazy values.

#### 5. Dataset discovery is method-heavy and weakly typed

The current API exposes `list_meshes()`, `list_particle_types()`,
`resolve_field()`, `metadata_bundle()`, and a dictionary-valued
`plane_geometry()` result. This requires users to know method names and mapping
keys before interactive exploration.

Prefer browsable collections and typed metadata:

```python
ds.fields
ds["density"]
ds.meshes
ds.particles["stars"]["mass"]
ds.geometry.plane(axis="z")
```

`PlaneGeometry` should be a typed immutable object rather than a
`dict[str, Any]`.

#### 6. Runtime ownership leaks into every workflow

`open_dataset()` accepts `runtime=None`, but dataset field registration requires
a runtime field allocator. Pipeline construction then requires runtime,
runmeta, and dataset separately.

The default entry point should create or obtain one execution client:

```python
ds = kangaroo.open_dataset(path)
```

Explicit runtime configuration remains possible:

```python
client = kangaroo.Client(hpx_args=["--hpx:threads=8"])
ds = client.open_dataset(path)
```

Console-locality coordination should be owned by the client or launcher rather
than each analysis script.

#### 7. Package and public namespace do not match the product

The distribution is named `kangaroo`, while users import `analysis`. The
top-level package also mixes recommended user objects with low-level IR types
such as `Plan`, `Stage`, `TaskTemplate`, and `LoweringContext`.

The preferred import should become:

```python
import kangaroo as kr
```

Advanced interfaces should live under clear namespaces:

```python
from kangaroo import ir
from kangaroo import runtime
from kangaroo import backends
```

The `analysis` package should re-export compatible objects during a documented
deprecation period.

#### 8. Documentation and introspection are sparse

Public methods need docstrings, stable type annotations, examples, and concise
representations. A lazy value should expose predictable metadata without
executing:

```text
kangaroo.Array<name='density', dtype=float64, domain=AMR(3 levels), lazy=True>
```

Useful common properties include `name`, `dtype`, `domain`, `chunks`,
`dataset`, and `is_materialized`. A global dense `shape` should only be shown
when it is meaningful.

## Target Mental Model

The public model consists of four layers:

1. **Client** owns runtime configuration and distributed execution resources.
2. **Dataset** owns backend selection, metadata, named fields, particle species,
   and geometry.
3. **Lazy values** describe transformations and reductions without executing.
4. **Materialized results** contain local NumPy arrays, scalars, or typed
   structured scientific results.

```text
Client -> Dataset -> Lazy values -> compute/persist -> Results
                      |                    |
                      +---- expression DAG-+
```

Users should not need to construct a separate mutable pipeline. A private graph
context shared by values may continue to use the current lowering machinery.

## Proposed Public Types

### `Client`

Owns runtime initialization, configuration, tracing, and execution.

```python
client = kr.Client(
    hpx_args=["--hpx:threads=8"],
    progress=False,
)
ds = client.open_dataset(path)
```

The module-level `open_dataset()` uses a lazily created default client.

### `Dataset`

Provides named, typed access to source fields and particles:

```python
ds = kr.open_dataset(path, step=0)

rho = ds["density"]
stars = ds.particles["stars"]
mass = stars["mass"]
```

Suggested discovery surface:

- `ds.fields`: mapping-like field collection.
- `ds.particles`: mapping-like particle-species collection.
- `ds.meshes`: mapping-like mesh collection when supported.
- `ds.geometry`: typed geometry accessor.
- `ds.metadata`: typed common metadata plus backend-specific metadata under an
  explicit extension field.
- `ds.step` and `ds.level`: active selection.
- `ds.select(step=..., level=..., mesh=...)`: return a new selected dataset
  view rather than mutating the original dataset.

### `Array`

A lazy mesh or regular-array result. It carries graph provenance and output
metadata, and supports:

- arithmetic operators: `+`, `-`, `*`, `/`, `**`, unary `-`,
- comparisons and boolean operators where valid,
- `astype(dtype)`,
- `rename(name)`,
- `slice(...)`, `project(...)`, `histogram(...)`,
- `compute()`, `persist()`, `visualize()`, and `explain()`,
- explicit `iter_chunks()` for results that remain distributed.

AMR source fields should not claim a misleading global dense shape. Their
representation should show an AMR domain and per-level chunk information.

### `ParticleArray` and `Mask`

These use the same lazy base protocol as `Array`, with domain-appropriate
operators:

```python
mass = ds.particles["stars"]["mass"]
valid = mass.isfinite() & (mass > 0)
selected = mass[valid]
mean = selected.mean()
```

Their reductions return lazy `Scalar` objects. Filtering remains chunked and
does not imply global concatenation.

### `Scalar`

A lazy scalar reduction:

```python
total = mass.sum()
assert total.dtype == kr.float64
value = total.compute()
```

Implicit conversion through `float(total)`, `int(total)`, or `bool(total)`
should fail with a message directing the user to `compute()`. Silent execution
through Python conversion would make performance unpredictable.

### Structured result expressions

Operations such as histograms, flux integrals, and Toomre-Q profiles have more
metadata than one array. Their lazy expression objects should compute into
typed result dataclasses:

```python
hist = rho.histogram(bins=128, range=(1e-30, 1e-20))
result = hist.compute()

result.counts  # np.ndarray
result.edges   # np.ndarray
```

Suggested pairs include:

- `Histogram` -> `HistogramResult(counts, edges)`
- `Histogram2D` -> `Histogram2DResult(counts, x_edges, y_edges)`
- `FluxSurfaceIntegral` -> `FluxSurfaceIntegralResult`
- `ToomreQProfile` -> `ToomreQProfileResult`

The lazy object exposes static metadata such as bin edges when that metadata is
known before execution.

## Execution Semantics

### `compute()`

`compute()` executes the minimal graph needed for its requested outputs and
returns local Python values:

```python
image_array = image.compute()
image_array, hist_result = kr.compute(image, hist)
```

The multi-value form must merge compatible graphs so shared work executes once.
It should reject values owned by incompatible clients or dataset contexts with
an explicit error.

Gathering an unbounded distributed value into local memory must require either
a bounded result or an explicit opt-in. For example, a complete particle array
should not become a NumPy array merely because `compute()` was called. It may
return a chunked result or require `gather=True` with an optional byte limit.

### `persist()`

`persist()` executes the required graph but keeps results in Kangaroo's
distributed storage. It returns equivalent lazy values whose sources are the
persisted chunks:

```python
prepared = expensive_derived_field.persist()
hist_a, hist_b = kr.compute(
    prepared.histogram(...),
    prepared.project(...),
)
```

### `visualize()` and `explain()`

- `visualize()` produces a graph representation suitable for notebooks or a
  file.
- `explain()` returns or prints a textual summary of stages, domains, estimated
  tasks, expected storage, reductions, and locality decisions.

Neither method executes scientific kernels.

### `numpy()`

`numpy()` belongs on materialized numeric results, not general lazy values.
Calling it on a lazy value should either be unsupported or be a clearly
documented alias for `compute()`; the preferred design is to keep `compute()` as
the only implicit execution boundary.

### Configuration and scheduler policy

Scientific operator signatures should describe scientific behavior. Scheduler
details such as `reduce_fan_in` should normally move to execution configuration:

```python
result = hist.compute(reduction_fan_in=8)
```

or:

```python
with kr.config.set({"reduction.fan_in": 8}):
    result = hist.compute()
```

An expert-only per-operator override may remain where it is needed for
benchmarking or correctness experiments, but it should not dominate common
signatures.

## End-to-End Examples

### Slice and histogram

```python
import kangaroo as kr

ds = kr.open_dataset("/path/to/plotfile")

rho = ds["density"]
temperature = ds["temperature"]
pressure = (rho * temperature).rename("pressure")

image = pressure.slice(
    axis="z",
    coord=0.0,
    resolution=(512, 512),
)

hist = rho.histogram(
    bins=128,
    range=(1.0e-30, 1.0e-20),
    weights=ds["cell_mass"],
)

image_array, hist_result = kr.compute(image, hist, progress=True)
```

The dataset supplies full-domain bounds and a default center coordinate when
they are omitted. Advanced callers may provide an explicit rectangle and axis
bounds.

### Particle selection

```python
stars = ds.particles["StochasticStellarPop_particles"]
mass = stars["mass"]

selected = mass[mass.isfinite() & (mass > 0)]
hist = selected.histogram(bins=128, range=(1e30, 1e40))
total = selected.sum()

hist_result, total_mass = kr.compute(hist, total)
```

### Explicit client configuration

```python
import kangaroo as kr

client = kr.Client.from_parsed_args(args, unknown_args=unknown)
ds = client.open_dataset(args.plotfile)

projection = ds[args.var].project(
    axis=args.axis,
    bounds=args.axis_bounds,
    resolution=args.resolution,
)

array = projection.compute(progress=args.progress)
```

## Graph Construction Requirements

Each lazy value must record:

- its owning graph/client context,
- its producer node,
- its logical result kind,
- its dataset/domain identity,
- its known dtype and shape/chunk metadata,
- any structured-result metadata known before execution.

When an operation consumes lazy values, its dependencies are exactly their
producer nodes plus any explicit runtime constraints. Graph construction must
not add a dependency merely because one expression was created earlier.

The lowering layer may still emit ordered stages within one operator fragment.
The change is at fragment composition: roots connect to the producers of their
actual input handles rather than to a pipeline-global frontier.

Mesh and particle nodes should share one graph abstraction even if lowering
uses different domains or run metadata internally. A single `compute()` call
must be able to request both kinds of output.

## Errors and Safety

- Cross-client or cross-dataset expressions fail when constructed, with both
  owners identified.
- Unknown fields show the requested name and a short list of available or
  similarly named fields.
- Unsupported operations identify the value kind and supported alternatives.
- Materialization errors explain whether the value is opaque, distributed,
  unbounded, or too large to gather.
- Calling `compute()` during an active incompatible run fails explicitly.
- Lazy scalar truth testing fails instead of launching work or returning an
  arbitrary truth value.
- Optional dtype and shape assertions continue to be checked against the
  authoritative chunk descriptor.

## Namespace Layout

Suggested package structure:

```text
kangaroo/
  __init__.py       # Client, Dataset, Array, Scalar, compute, open_dataset
  array.py          # common lazy value operations
  dataset.py        # dataset and collection facade
  results.py        # materialized and structured result types
  config.py         # execution configuration
  diagnostics.py    # visualize and explain
  ir/               # Plan, Stage, TaskTemplate, buffer contracts
  runtime/          # Runtime and low-level result retrieval
  backends/         # backend-specific extensions
analysis/            # temporary compatibility re-exports
```

The exact internal module split is not normative. The important rule is that
the default namespace contains the small user workflow, while implementation
and expert concepts have explicit homes.

## Compatibility Strategy

1. Implement the new lazy types as wrappers over the existing pipeline and
   lowering machinery.
2. Add `kangaroo` as the preferred namespace while keeping `analysis` working.
3. Add `compute()` to current handles where practical, backed by the same result
   materializer used by the new types.
4. Migrate repository scripts to the new API. Treat those scripts as usability
   tests and examples.
5. Emit targeted deprecation warnings only after equivalent new functionality
   is available.
6. Keep `analysis.plan`, `analysis.runtime`, and related low-level modules
   usable until downstream callers have a documented replacement.
7. Update `spec/python-api.md` in the release that makes the new surface the
   compatibility target.

## Implementation Stages

### Stage 1: Result-owned materialization

- Add a common lazy-value protocol.
- Give mesh handles `compute()` and explicit `iter_chunks()` behavior.
- Add `kangaroo.compute(*values)` with graph/context validation.
- Introduce typed materialized results.
- Remove the need for normal users to call `Runtime.get_task_chunk_array()`.

This stage may use the current pipeline graph unchanged.

### Stage 2: Uniform laziness

- Make particle scalar reductions, histograms, and top-k operations return lazy
  results.
- Replace implicit execution in particle materialization helpers with the
  common execution path.
- Execute mesh and particle requests through one public `compute()` boundary.

### Stage 3: Dependency-derived graph composition

- Record producer provenance on every lazy value.
- Replace pipeline-global frontier chaining with input-derived dependencies.
- Merge graphs for multi-output `compute()`.
- Verify that independent branches can execute concurrently and shared branches
  execute once.

### Stage 4: Ergonomic value and dataset APIs

- Add dataset indexing and mapping-like discovery collections.
- Add arithmetic, comparisons, masks, reductions, `astype()`, and `rename()`.
- Add typed geometry objects and dataset-derived defaults for slices and
  projections.
- Add rich `repr` implementations and complete docstrings.

### Stage 5: Namespace and advanced API organization

- Add the `kangaroo` namespace.
- Move low-level concepts under `kangaroo.ir`, `kangaroo.runtime`, and
  `kangaroo.backends`.
- Provide `analysis` compatibility re-exports and migration documentation.
- Migrate examples, scripts, and tests in bounded slices.

### Stage 6: Persistence and diagnostics

- Add `persist()` using distributed runtime storage.
- Add textual `explain()` output.
- Add graph `visualize()` output.
- Expose resource estimates and optimization controls without leaking internal
  plan encoding.

## Acceptance Criteria

The redesign is successful when:

1. A new user can open a supported dataset, select a field, define a slice or
   projection, and receive a NumPy array in no more than four conceptual lines.
2. No recommended example manually passes `step`, `level`, `field`, `version`,
   or `block` to retrieve an operator result.
3. All reductions are lazy until `compute()` or `persist()`.
4. Independent expressions do not acquire definition-order dependencies.
5. Multi-output `compute()` shares common graph work.
6. Unbounded particle gathering remains explicit and memory-safe.
7. Every top-level public object and method has a docstring and stable type
   annotations.
8. Current low-level workflows remain possible through documented advanced
   modules during the compatibility period.
9. The existing numerical, descriptor, backend, and runtime tests continue to
   pass in the dedicated `pixi-hpx` environment.

## Historical Implementation Order

Stages 1 and 2 were implemented before changing package names or adding broad operator
syntax. Result-owned `compute()` and uniform laziness remove the largest user
burden while preserving the current kernels, typed plans, and packed-plan
format. They also establish the execution contract needed by all later API
work. All six stages are now implemented.
