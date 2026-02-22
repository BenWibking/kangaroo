# Backends and I/O Specification

## 1. Supported Dataset Source Classes

A conforming system MUST support the following source classes:
- AMReX plotfile datasets.
- openPMD datasets.
- Parthenon datasets stored in `.phdf`, `.h5`, or `.hdf5` files.
- In-memory datasets for synthetic/test workflows.

Support MAY be build-conditional for openPMD and Parthenon, but unsupported builds MUST fail explicitly when those backends are requested.

## 2. Backend Selection Contract

Backend selection MUST follow URI/path rules in `spec/python-api.md`.

Observed behavior requirement:
- Backend selection is deterministic for the same input URI/path.

## 3. Common Backend Interface Requirements

Each backend MUST provide:
- `get_chunk(chunk_ref)` returning optional raw payload.
- `has_chunk(chunk_ref)`.
- `get_metadata()` with backend-appropriate geometry/domain descriptors.

Backends MAY expose additional capabilities (mesh selection, particle readers) through dataset/reader APIs.

## 4. AMReX Plotfile Backend Behavior

### 4.1 Field Mapping

- Supports mapping runtime field IDs to plotfile component indices.
- `get_chunk` for mapped field reads the corresponding component.
- Unmapped field requests MUST return no chunk rather than fabricated data.

### 4.2 FAB Data Layout

On read:
- Source FAB payload is read from plotfile storage.
- Returned chunk payload MUST be transposed to runtime canonical axis order expected by operators.

### 4.3 Metadata Exposure

Metadata MUST include enough information for:
- Variable name discovery.
- Level count and per-level boxes.
- Problem bounds and domain boxes.
- Refinement ratios and cell size by level.

### 4.4 Particle Capabilities

Plotfile backend MUST expose:
- Particle type list.
- Particle field list per type.
- Whole-field particle reads.
- Chunk-count and chunk-wise field reads.

## 5. Memory Backend Behavior

Memory backend is mutable key-value chunk storage for tests/synthetic workflows.

Requirements:
- `set_chunk` writes payload by full chunk identity.
- `get_chunk` returns exact written bytes for same key.
- Missing keys return no chunk.
- Multiple levels/steps/fields/versions MAY coexist independently.

## 6. openPMD Backend Behavior

If enabled, openPMD backend MUST:
- Discover meshes and selectable active mesh.
- Expose mesh and field metadata including variable names.
- Support field registration by string field names.
- Provide chunk payloads for registered fields.

Error behavior:
- Unsupported record topology or unsupported data layout constraints MUST raise explicit metadata/access errors.

## 7. Parthenon Backend Behavior

If enabled, Parthenon backend MUST:
- Open and parse file-level metadata and variable descriptors.
- Expose variable names and per-variable component metadata.
- Support field registration by name and component labels.
- Provide chunk payloads for registered fields.

Error behavior:
- Missing required metadata groups/datasets or invalid metadata shapes MUST raise explicit errors.

## 8. Plotfile Reader Contract

`PlotfileReader` MUST provide:
- Header dictionary with version/components/levels/variable names and core metadata.
- Metadata dictionary containing bounds, refinement, level boxes, domain boxes, and per-level cell size.
- FAB read method returning payload bytes, dtype tag, and logical shape.
- Particle type and particle field discovery methods.
- Particle field read method returning dtype/count/bytes.
- Particle chunk count and per-chunk particle field reads.

### 8.1 Reader Dtype Contract

Supported particle dtype tags:
- `float32`
- `float64`
- `int64`

Unsupported dtype tags MUST fail explicitly in conversion layers.

## 9. Dataset Particle API Contract

Dataset particle methods MUST:
- Return empty lists if backend cannot provide particle metadata list methods.
- Raise explicit errors for particle reads when backend lacks required particle read capability.

## 10. Subbox Fetch Contract

Subbox fetch MUST:
- Intersect requested index box with source chunk index box.
- Return only overlapping extents and corresponding payload.
- Preserve contiguous memory packing in overlap-local index order.
- Respect requested bytes-per-value when slicing payload.

No-overlap behavior MUST return empty payload and invalid/empty overlap bounds.

## 11. Backend Mutability Contract

- Mutable chunk write (`set_chunk`) is only valid for mutable backends (memory backend).
- Calling mutable write on read-only backends MUST fail explicitly.

## 12. Serialization/Distribution Contract

When dataset handles are propagated to distributed runtime contexts:
- Source URI, step, and level identity MUST remain intact.
- Backend rehydration MUST preserve backend type and source location.
- Unsupported backend rehydration in current build MUST fail explicitly.
