# Python API Specification

## High-level `kangaroo` contract

`kangaroo` is the recommended product namespace. Its default surface MUST expose
`Client`, `Dataset`, `Array`, `ParticleArray`, `ParticleMask`, `Scalar`, the
structured lazy result expressions, their typed materialized results,
`open_dataset()`, and `compute()`.

### Client and datasets

- `Client` MUST own one low-level runtime and explicit HPX configuration.
- Module-level `open_dataset()` MUST use a lazily created process-local default
  client.
- `Dataset` MUST expose mapping-like `fields`, `particles`, and `meshes`
  collections, named field indexing, typed `geometry`, read-only metadata, and
  immutable selection through `select()`.
- Unknown names MUST report the requested name, available names, and close
  matches when available.
- Values from different selected dataset contexts MUST fail when combined.

### Lazy values

- Mesh arithmetic, comparisons, casts, renaming, slices, projections,
  histograms, vorticity, flux integrals, and Toomre-Q profiles MUST construct
  lazy values without running scientific kernels.
- Particle arithmetic, scalar comparisons, masks, filtering, casts, top-k,
  histograms, and scalar reductions MUST use the same lazy execution boundary.
- `float()`, `int()`, and truth testing of a lazy `Scalar` MUST fail with an
  error directing the caller to `compute()`.
- Public lazy values MUST expose stable `name`, `dtype`, `domain`, `dataset`,
  and `is_materialized` metadata. AMR source fields MUST describe per-level
  chunks rather than advertise a misleading global dense shape.

### Execution and materialization

- `value.compute()` and `kangaroo.compute(*values)` are the explicit local
  materialization boundaries.
- Multi-output compute MUST validate one owning dataset context, union requested
  producer ancestry, and submit each compatible mesh or particle graph once so
  common work is shared.
- Stage dependencies MUST derive from the producers of actual input fields.
  Independent branches MUST NOT acquire definition-order edges.
- Bounded mesh results MUST return NumPy arrays using authoritative runtime
  descriptors. Histograms and scientific multi-component operations MUST return
  typed result dataclasses.
- AMR and particle values MUST return `ChunkedArray` by default. Dense particle
  gathering MUST require `gather=True`; `max_bytes` MUST reject an oversized
  gather before concatenation.
- `persist()` MUST execute and retain the requested distributed fields, then
  treat those fields as sources for subsequently built work.
- `explain()` and `visualize()` MUST describe the graph without execution.

### Advanced namespaces and compatibility

Typed plan and buffer concepts MUST be available from `kangaroo.ir`. Direct
runtime and chunk retrieval MUST be available from `kangaroo.runtime`. Backend
interfaces MUST be available from `kangaroo.backends`. The `analysis` package
and its low-level modules remain supported during the compatibility period and
MUST re-export the broadly useful high-level entry points.

## 1. Compatibility Package Export Contract

The `analysis` package MUST expose the following public symbols (directly or via lazy attribute loading):
- Runtime: `Runtime`, runtime configuration string helper.
- Plan model: `Plan`, `Stage`, `TaskTemplate`, `Domain`, `FieldRef`, `OutputRef`.
- Buffer model: `BufferSpec`, dtype/init enums, and closed shape-rule types.
- Lowering context: `LoweringContext`.
- Dataset: `Dataset`, `open_dataset`.
- Run metadata model: `RunMeta`, `StepMeta`, `LevelMeta`, `LevelGeom`, `BlockBox`, metadata loader helper.
- Plotfile reader: `PlotfileReader`.
- Pipeline API: `Pipeline`, `FieldHandle`, `Histogram1DHandle`, `Histogram2DHandle`, `ParticleArrayHandle`, `ParticleMaskHandle`, `pipeline`.
- Histogram/CDF helpers.

Unknown attribute access MUST raise `AttributeError`.

## 2. `Runtime` API

### 2.1 Construction

`Runtime` MUST support:
- Default construction with runtime defaults.
- Construction from explicit runtime config and runtime command-line args.
- Construction from parsed argparse namespace plus unknown passthrough args.

`from_parsed_args` behavior:
- Reads recognized runtime config options when present on parsed args.
- Appends unknown args to runtime command-line args in argv-compatible form.
- If no runtime args/config are present, equivalent to default constructor.

### 2.2 Core Methods

- `alloc_field_id(name)` returns a new integer field ID.
- `mark_field_persistent(fid, name)` marks field for persistent output identity.
- `run(plan, runmeta, dataset)` executes the plan.
- `preload(runmeta, dataset, fields)` preloads selected field chunks where available.
- `get_task_chunk_bytes(...)` returns raw bytes for debugging and opaque payloads.
- `get_task_chunk_array(...)` materializes a numeric ndarray from its runtime descriptor.
- `set_event_log_path(path)` updates runtime event sink path.
- `kernels` property exposes registry/list API.

Execution visibility rules:
- `run(...)` MUST be a completion boundary for output visibility.
- The Python runtime MUST NOT provide output-buffer access while a plan execution is in progress.
- Calls to `get_task_chunk(...)` or `get_task_chunk_array(...)` that target outputs from an in-flight run MUST fail explicitly.
- Output retrieval APIs MUST only expose data after successful completion of the corresponding plan execution.

### 2.3 `get_task_chunk_array` Semantics

Required behavior:
- The runtime descriptor is authoritative for dtype, shape, and strides.
- Optional caller-provided `dtype` or `shape` values are assertions and mismatches MUST fail.
- Opaque payloads MUST fail numeric materialization and use raw-byte retrieval instead.

## 3. Plan Model API

### 3.1 `Domain`

Contains `step`, `level`, optional `blocks`.

### 3.2 `FieldRef`

Contains `field`, `version`, optional domain override.

### 3.3 `Stage`

Contains `name`, `plane`, `after`, `templates`.

`map_blocks(...)` MUST append a new `TaskTemplate` preserving stage plane.

### 3.4 `Plan`

Contains `stages`.

`topo_stages()` MUST produce dependency-respecting order for DAG traversal and serialization.

## 4. `Dataset` API

### 4.1 URI Resolution (`open_dataset`)

`open_dataset(path_or_uri, ...)` MUST resolve in this order:
- If input already has recognized scheme (`amrex://`, `openpmd://`, `parthenon://`, `file://`, `memory://`), keep as is.
- If input has an unknown scheme (contains `://`), pass through unchanged.
- Else path MUST exist locally, or `FileNotFoundError`.
- Existing directory containing AMReX `Header` resolves to AMReX URI.
- Existing `.phdf/.h5/.hdf5` file resolves to Parthenon URI.
- Otherwise resolves to openPMD URI.

### 4.2 Dataset State

Dataset instance MUST track:
- Source URI.
- Active `step` and `level`.
- Runtime reference.
- Backend handle.
- Field name to field ID cache.

### 4.3 Automatic Field Registration

Dataset SHOULD auto-register backend-known fields when metadata supports discovery:
- AMReX variable names from plotfile header.
- openPMD variable names and mesh components.
- Parthenon variable names and component labels.

### 4.4 Field Registration and Allocation

- `register_field(name, fid)` binds name->ID mapping.
- `field_id(name)` returns cached ID or allocates through runtime and caches result.

### 4.5 Metadata Access

- `metadata` returns backend metadata dictionary.
- Metadata failures MUST surface explicit errors with backend context.

### 4.6 Run Metadata Construction

`get_runmeta(periodic=None)` MUST:
- Return provided explicit runmeta when set.
- Else synthesize runmeta from dataset metadata.
- Carry periodicity from metadata unless explicit override provided.
- Include particle species chunk counts when detectable.

### 4.7 Geometry Helper

`plane_geometry(axis, level, coord, zoom, resolution)` MUST return mapping containing:
- `coord`
- `rect`
- `resolution`
- `labels`
- `plane`
- `axis_index`
- `axis_bounds`

Validation:
- Axis must be valid.
- Zoom must be strictly positive.
- Resolution override values must be strictly positive integers.

### 4.8 Field Resolution

`resolve_field(var)` behavior:
- Non-openPMD: if `var` absent, choose first available variable name; error if none.
- openPMD: support plain variable names, mesh selection names, and `mesh/component` syntax with explicit validation.
- Unknown requested field MUST raise explicit error.

### 4.9 Chunk Utilities

- `set_chunk(...)` accepts NumPy arrays, explicitly typed/shaped raw bytes, or explicitly opaque bytes.

## 5. `PlotfileReader` API

`PlotfileReader` MUST provide:
- `header()`
- `metadata()`
- `num_levels()`
- `num_fabs(level)`
- `read_fab(level, fab, comp_start, comp_count, return_ndarray=False)`
- Particle APIs: type list, field list, chunk count, and chunked field read

Array conversion behavior:
- `return_ndarray=True` converts payload bytes to NumPy array using reported dtype and shape.

## 6. `Pipeline` API

`Pipeline` is an imperative DAG builder and execution facade.

### 6.1 Construction

Requires runtime, runmeta, dataset.

If runtime provides dataset-binding hook, pipeline MUST bind dataset on construction.

### 6.2 Handles

- `FieldHandle`: symbolic mesh field reference.
- `Histogram1DHandle`: histogram result field plus range/bin metadata and computed edges.
- `Histogram2DHandle`: 2D histogram result field plus range/bin metadata and computed edges.
- `ParticleArrayHandle`: lazy materialized numeric particle array.
- `ParticleMaskHandle`: lazy materialized boolean particle mask.

### 6.3 Field Operators

Pipeline MUST support:
- `field(name_or_id)`
- `field_expr(expression, variables, out=None, dtype="f64")`
- Arithmetic helpers (`field_add`, `field_subtract`, `field_multiply`, `field_divide`)
- Derived field registration and retrieval with caching controls.

### 6.4 AMR Operators

Pipeline MUST support:
- `vorticity_mag(...)`
- `uniform_slice(...)`
- `uniform_projection(...)`
- `particle_cic_projection(...)`
- `histogram1d(...)`
- `histogram2d(...)`

Each operator MUST append its lowered fragment with dependency linkage to the
producer stages of its actual input fields. Definition order alone MUST NOT add
an edge.

### 6.5 Particle Operators

Pipeline MUST support:
- Particle field loading.
- Mask creators (`equals`, `isin`, `isfinite`, threshold comparisons).
- Mask combinator (`and`).
- Filtering and arithmetic operators.
- Distance, sum, length, min, max, count.
- Top-k modes.
- Particle histogram.

Particle operators MAY use a dedicated internal stage list and virtual particle runmeta over chunk indices.

### 6.6 Plan and Execution

- `plan()` returns current mesh-stage plan.
- `run()` executes mesh-stage plan (if non-empty) and particle-stage plan (if present).

## 7. Helper Functions

`histogram_edges_1d`, `histogram_edges_2d`, `cdf_from_histogram`, `cdf_from_samples` MUST satisfy the behavior in `spec/operators.md` and validation in `spec/validation-and-errors.md`.
