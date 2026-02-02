# I/O Implementation Plan: Dataset-Backed Storage

This document outlines the architectural changes required to transition Kangaroo from an in-memory prototype to a robust, dataset-backed I/O system.

## 1. Unified C++ Backend Interface
Refactor the existing `DatasetHandle` into a polymorphic backend system to support multiple storage formats.

- **`DatasetBackend` (Abstract Base Class)**: Define the core interface for fetching data.
    - `virtual std::optional<HostView> get_chunk(const ChunkRef& ref) = 0;`
    - `virtual bool has_chunk(const ChunkRef& ref) = 0;`
    - `virtual DatasetMetadata get_metadata() = 0;`
- **Implementations**:
    - **`MemoryBackend`**: Wraps the existing `std::unordered_map` for Python-seeded data.
    - **`PlotfileBackend`**: Integrates `PlotfileReader` to pull AMReX data directly from disk on demand.

## 2. On-Demand (Lazy) Loading
Integrate the `DatasetBackend` with the `DataService` to enable transparent I/O.

- **`DataServiceLocal` Integration**:
    - Update `get_local_impl` to query the active `DatasetBackend` if a `ChunkRef` is not found in the local cache.
    - Ensure this mechanism respects locality; only the "home" rank for a chunk should trigger a disk read.
- **Workflow Change**:
    - Move from "Preload-then-Run" to "Run-with-On-Demand-Fetch".
    - Manual `preload` remains an optimization for warm-starting caches, not a requirement for execution.

## 3. Schema and Metadata Support
Provide a formal structure for dataset discovery and typed access.

- **`FieldDescriptor`**: Add metadata for fields including:
    - Data type (float32, float64).
    - Component names and counts.
    - Units and scaling factors.
- **`DatasetMetadata`**: Centralize global information:
    - Geometry (prob_lo, prob_hi).
    - Coordinate system and periodicity.
    - Refinement hierarchy (ref_ratio).

## 4. Python API Enhancements
Improve the DSL to handle different dataset types via URI-based dispatch.

- **URI Dispatch**:
    - `kangaroo.open_dataset("amrex://path/to/plt")` -> Returns a Plotfile-backed dataset.
    - `kangaroo.open_dataset("memory://local")` -> Returns an in-memory dataset.
- **Auto-Registration**:
    - When opening a `PlotfileBackend`, automatically register field IDs based on the variable names found in the `Header` file.

## Implementation Phases

1.  **Phase 1**: Define `DatasetBackend` interface and migrate `MemoryBackend`.
2.  **Phase 2**: Implement `PlotfileBackend` using existing `PlotfileReader` logic.
3.  **Phase 3**: Hook `DatasetBackend` into `DataServiceLocal` for lazy loading.
4.  **Phase 4**: Add metadata/schema support and update Python bindings.
