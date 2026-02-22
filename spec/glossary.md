# Kangaroo Glossary

This glossary defines terms used throughout `spec/` that may be unfamiliar or overloaded.

## A

`after`  
Stage dependency list in a plan. Each value is the index of a prerequisite stage.

`AMR` (Adaptive Mesh Refinement)  
A hierarchy of grid levels with different resolutions, where finer levels refine selected regions of coarser levels.

`axis bounds`  
Two physical coordinates defining the integration extent along the projection axis.

## B

`backend`  
Dataset implementation for a concrete storage format (for example AMReX plotfile, openPMD, Parthenon, in-memory).

`block`  
A logical AMR subdomain (box) on a level. In many APIs this is identified by a zero-based block index within that level.

`BlockBox`  
Run metadata structure containing integer lower/upper corners (`lo`, `hi`) of a block in index space.

`bytes_per_value`  
Element width (typically 4 or 8 bytes) used to interpret raw chunk payload bytes.

## C

`cell-centered`  
Data values represent cell-center samples (not node/corner values).

`chunk`  
Runtime unit of stored data addressed by `(step, level, field, version, block)`. A chunk payload is raw bytes.

`ChunkRef`  
Identifier tuple used to locate a chunk in runtime storage.

`ChunkSubboxRef`  
Request structure describing a subregion fetch from a source chunk box.

`CIC` (Cloud-In-Cell)  
Particle deposition/interpolation scheme distributing particle mass to nearby grid cells or pixels by overlap weights.

`covered boxes`  
Coarse-level regions overlain by finer AMR levels; coarse contributions in these regions must be excluded in composite results.

## D

`Dataset`  
Python wrapper around an underlying backend handle plus runtime context (`step`, `level`, field registration cache).

`DatasetHandle`  
C++ runtime-side dataset object holding URI, step/level selection, and backend instance.

`deps` (dependency rule)  
Template-local dependency mode (`None` or `FaceNeighbors`) controlling neighbor halo fetch behavior.

`Domain` / `DomainIR`  
Execution scope for a template or field reference: `step`, `level`, and optional explicit block list.

## E

`Exec plane`  
Template/stage execution category (`chunk`, `graph`, `mixed` label support in decoding).

## F

`FAB`  
AMReX "Fortran Array Box": one on-disk data block for one box, with one or more components. In spec/docs, FAB reads are per-level/per-fab-index component data fetches.

`FaceNeighbors`  
Dependency mode requesting neighbor halos by traversing face adjacency (`Xm, Xp, Ym, Yp, Zm, Zp`) up to configured width.

`field`  
Integer field ID allocated by the runtime and used in chunk addressing and plan I/O references.

`FieldRef` / `FieldRefIR`  
Reference to a field/version, optionally with an input domain override.

`frontier`  
Pipeline-internal set of latest stages used to connect newly appended fragments in imperative chaining order.

## G

`graph stage`  
Stage whose templates run as grouped reduction tasks rather than per-block chunk tasks.

`graph reduce parameters`  
Template params (`graph_kind=reduce`, `fan_in`, `num_inputs`, optional `input_blocks`, etc.) that define reduction grouping.

## H

`halo`  
Neighbor data region needed from adjacent blocks for stencil or neighborhood operations.

`halo_inputs`  
Indices of template inputs that should receive `FaceNeighbors` halo collections.

`HostView`  
C++ runtime buffer wrapper containing raw byte storage for one chunk payload.

## I

`index_origin`  
Per-level integer offset defining how integer indices map to physical coordinates.

`input domain override`  
Per-input `FieldRef.domain` that can differ from the template's own domain for cross-level/block reads.

## K

`kernel`  
Named runtime operation executed for a template instance. Kernels consume input chunks/neighbor views and produce output chunks.

`KernelRegistry`  
Runtime registry mapping kernel names to executable implementations and descriptor metadata.

## L

`LevelGeom`  
Run metadata geometry for one AMR level (`dx`, `x0`, `index_origin`, `is_periodic`, `ref_ratio`).

`LevelMeta`  
Run metadata level entry containing `LevelGeom` and block boxes.

## M

`metadata_bundle`  
Dataset helper returning both backend metadata and constructed run metadata in one object.

`mixed plane`  
Accepted plan label in decoding; not required as an executable stage plane in current behavior.

## O

`output_bytes`  
Optional template hint listing byte sizes for each output chunk allocation.

## P

`Parthenon`  
AMR data format/ecosystem commonly stored in `.phdf`/`.h5` files, supported via dedicated backend.

`persistent field`  
Field ID explicitly marked for persistent output retrieval by name.

`Plan` / `PlanIR`  
Top-level execution graph object containing stages and their templates.

`plotfile`  
AMReX directory-based output format containing metadata header(s) and per-level FAB data files.

`projection`  
Integration of field values through an axis interval into a 2D output plane.

## R

`ref_ratio`  
Integer refinement ratio between adjacent AMR levels.

`RunMeta`  
Primary run metadata object containing steps, each step's AMR levels/geometry/boxes, and optional particle species chunk-count metadata.

## S

`slice`  
2D extraction/aggregation at a specified coordinate along one axis.

`Stage` / `StageIR`  
Plan node containing templates and stage-level dependencies.

`step`  
Simulation output index/time-slice entry in run metadata.

`StepMeta`  
Run metadata entry for one step, containing level list.

`subbox`  
Requested overlap region inside a chunk's index-space bounds.

`SubboxView`  
Result payload for subbox fetch (data bytes, returned overlap box, bytes-per-value).

## T

`Task template` (`TaskTemplate` / `TaskTemplateIR`)  
Declarative per-stage operation definition expanded into concrete executions over blocks or graph groups.

`template plane`  
Execution plane declared on a task template; must match containing stage plane for execution.

`topo order`  
Topological order of stages respecting dependency edges.

## U

`URI scheme`  
Dataset source prefix indicating backend resolution (`amrex://`, `openpmd://`, `parthenon://`, `memory://`, etc.).

## V

`version` (field version)  
Integer version namespace for a field in chunk addressing.

`VisMF`  
AMReX internal file/header convention used for multi-fab storage metadata in plotfiles.
