# Validation and Error Semantics Specification

This document defines mandatory rejection behavior and failure cases. A conformant implementation MUST fail explicitly for these conditions.

## 1. Plan Payload Validation

### 1.1 Structural map/list validation

Errors required for:
- Missing required keys in plan/stage/template/domain entries.
- Non-map where map is required.
- Non-array where array is required (`stages`, `after`, `templates`, field lists, output bytes).

### 1.2 Plane validation

Errors required for:
- Unknown plane label in decode path.
- Stage/template plane mismatch at execution.
- Unsupported stage plane execution.

### 1.3 Dependency rule validation

Errors required for:
- `faces` not length 6.
- `faces` entries not coercible to bool/int.
- `halo_inputs` not an array.
- `FaceNeighbors` with zero inputs.
- `halo_inputs` index outside input range.

## 2. Stage Graph Validation

Errors required for:
- `after` dependency index out of range.
- Dependency graph cycles.

## 3. Template Execution Validation

Errors required for:
- Missing required outputs on templates.
- `output_bytes` length mismatch with output count.
- Graph templates with non-`None` deps.
- Graph templates missing required input/output wiring.
- Invalid graph reduce parameter combinations.

## 4. Domain and Block Validation

Errors required for:
- Input domain with explicit multiple blocks that does not contain executing task block.
- Domain blocks malformed or semantically invalid where required by executor.

## 5. Runtime Startup/Configuration Validation

Errors required for:
- Reconfiguring runtime startup options after runtime backend startup.
- Conflicting repeated startup option initialization.
- Attempting Python-side output-buffer retrieval for a plan run that has not yet completed.

## 6. Dataset and URI Validation

Errors required for:
- Missing path when local path is required.
- Invalid URI/path resolution inputs.
- Backend initialization attempts for unsupported optional backends in current build.

## 7. API Argument Validation

### 7.1 Axis/geometry validation

Errors required for:
- Invalid axis names or indices.
- Non-positive zoom.
- Non-positive resolution dimensions.
- Invalid axis-bound parse formats.

### 7.2 Array conversion validation

Errors required for:
- Non-positive total element count in shape for array conversion.
- Unsupported bytes-per-value for inferred dtype conversion.

## 8. Field Expression Validation

Errors required for:
- Empty expression.
- Empty variable mapping.
- Empty/invalid variable names.
- Variable/input count mismatch.
- More than supported max variable count.
- Unsupported input/output byte widths.
- Parse/evaluation setup failures.

## 9. Histogram Validation

Errors required for:
- Non-positive bin counts.
- Non-finite histogram range boundaries where finite required.
- Non-increasing ranges.
- Invalid edge arrays for particle histogram (rank/length).
- Missing `hist_range` when particle histogram is requested with integer bins.

## 10. Particle API Validation

Errors required for:
- Particle handle from different pipeline used in current pipeline operation.
- Chunk-count mismatch across particle operands where matching is required.
- `particle_and` invoked without any masks.
- Unsupported particle dtype conversion.
- Particle read requests on backend lacking particle support.

## 11. Backend Mutability Validation

Errors required for:
- Attempting mutable chunk writes on read-only backends.

## 12. Dashboard/Workflow Validation

Errors required for:
- Invalid thread-count argument for dashboard workflow launcher.
- Invalid/malformed DAG-plan payloads that cannot be interpreted for node/edge construction.

## 13. Script Exit/Error Semantics

Scripts MUST provide explicit user-visible error messages and non-zero exit status for invalid configuration or runtime failure.

For `smoke_demo.py`, non-zero statuses MUST distinguish initialization failure from post-initialization execution failure.

## 14. Test-Critical Behavioral Guarantees

Conformance requires reproducing behavior validated by repository tests, including:
- Coarse-fine correctness for gradient and vorticity workflows.
- Correct AMR covered-cell masking in projections/histograms.
- Proper plan topology and fragment chaining behavior.
- Correct dashboard blockwise DAG edge mapping.
- Accepted and rejected neighbor dependency configurations.
- Correct cross-level subbox fetch behavior.
