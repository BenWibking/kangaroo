# Data Model Specification

## 1. Core Metadata Objects

### 1.1 `RunMeta`

`RunMeta` is the authoritative geometry/topology description for plan execution.

It contains:
- Ordered `steps` list.
- Optional particle species map of `species_name -> chunk_count`.

### 1.2 `StepMeta`

A step entry contains:
- Integer `step` identifier.
- Ordered `levels` list.

### 1.3 `LevelMeta`

A level entry contains:
- `LevelGeom` geometry.
- List of `BlockBox` block extents.

### 1.4 `LevelGeom`

`LevelGeom` contains:
- `dx`: physical cell size per axis.
- `x0`: physical origin anchor.
- `index_origin`: integer index-space origin offset.
- `is_periodic`: periodicity flags per axis.
- `ref_ratio`: refinement ratio to next finer level.

### 1.5 `BlockBox`

Each block box defines inclusive integer index extents:
- `lo = (ilo, jlo, klo)`
- `hi = (ihi, jhi, khi)`

Cell count for one block is `(ihi-ilo+1)*(jhi-jlo+1)*(khi-klo+1)` and MUST be positive for executable blocks.

## 2. Chunk Identity Model

A chunk is uniquely identified by five integers:
- `step`
- `level`
- `field`
- `version`
- `block`

All runtime storage, task I/O references, and retrieval APIs MUST use this identity model.

## 3. Domain Model

A domain has:
- `step`
- `level`
- Optional `blocks`

Semantics:
- Omitted `blocks` means all blocks at that level.
- Explicit `blocks` means only listed blocks.
- Domain on input references can override template domain for that input only.

## 4. Field and Version Semantics

- `field` is an integer namespace allocated by runtime.
- `version` distinguishes multiple revisions of same field ID.
- Persistent outputs are field IDs explicitly marked by name.

No implicit field-name lookup occurs during task execution; kernels operate on numeric IDs and template wiring.

## 5. Coordinate Conventions

### 5.1 Index Space

Block boxes and coverage masks are specified in integer index space.

### 5.2 Physical Space

Physical coordinate interpretation uses `x0`, `dx`, and `index_origin`.

### 5.3 Cell-Center Convention

Field values for mesh-based operators are interpreted as cell-centered values unless explicitly noted otherwise by backend constraints.

## 6. Axis Semantics

Accepted axis forms:
- Named: `x`, `y`, `z`
- Indexed: `0`, `1`, `2`

Mapping:
- `x -> 0`
- `y -> 1`
- `z -> 2`

Invalid axis values MUST raise errors in user-facing APIs.

## 7. Plane and View Conventions

For slice/projection output plane labeling:
- Axis `z` produces `xy` plane.
- Axis `y` produces `xz` plane.
- Axis `x` produces `yz` plane.

Associated axis labels MUST match output plane orientation.

## 8. Byte-Width and Dtype Semantics

Chunk payloads are raw bytes. Consumers interpret data type using context.

Required widths for standardized array conversion paths:
- 4 bytes per value (`float32`)
- 8 bytes per value (`float64`)

Unsupported inferred widths MUST fail explicitly in ndarray conversion helpers.

## 9. AMR Coverage Semantics

For composite operations (slice, projection, histograms):
- Coarse contributions in regions covered by finer levels MUST be excluded.
- Covered regions are represented as index-space boxes on the level being masked.
- If no covered boxes are provided for a level, no masking is applied at that level.

## 10. Particle Metadata Semantics

`RunMeta.particle_species` (when present) provides expected chunk counts per particle species.

Pipeline particle execution MAY use this map to determine virtual chunk domains for particle graph operations.

If absent, chunk count MAY be queried from dataset particle metadata APIs.

## 11. Subbox Data Model

A subbox request includes:
- Source chunk reference.
- Source chunk index bounds.
- Requested index bounds.
- Bytes-per-value.

A subbox response includes:
- Returned overlap bounds.
- Returned payload bytes corresponding exactly to overlap region.
- Bytes-per-value used for packing.

