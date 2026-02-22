# Operator Specification

## 1. Kernel Compatibility Set

A conforming runtime MUST expose and execute the kernel names listed below, with semantics compatible with this specification:
- `amr_subbox_fetch_pack`
- `gradU_stencil`
- `plotfile_load`
- `uniform_slice`
- `uniform_slice_cellavg_accumulate`
- `uniform_projection_accumulate`
- `uniform_slice_add`
- `uniform_slice_reduce`
- `uniform_slice_finalize`
- `field_expr`
- `vorticity_mag`
- `histogram1d_accumulate`
- `histogram2d_accumulate`
- `particle_load_field_chunk_f64`
- `particle_cic_grid_accumulate`
- `particle_cic_projection_accumulate`
- `particle_eq_mask`
- `particle_isin_mask`
- `particle_isfinite_mask`
- `particle_abs_lt_mask`
- `particle_le_mask`
- `particle_gt_mask`
- `particle_and_mask`
- `particle_filter`
- `particle_subtract`
- `particle_distance3`
- `particle_sum`
- `particle_count`
- `particle_len_f64`
- `particle_min`
- `particle_max`
- `particle_histogram1d`
- `particle_topk_modes`
- `particle_int64_sum_reduce`
- `particle_scalar_min_reduce`
- `particle_scalar_max_reduce`

## 2. AMR Neighbor and Gradient Workflow

### 2.1 `amr_subbox_fetch_pack`

Purpose:
- Collect nearby patches/subboxes (including cross-level where needed) around a target block.

Contract:
- Accepts target field identity and halo width.
- Produces packed neighbor patch payload for downstream stencil use.
- Returns empty payload if no valid contributing patches are found.

### 2.2 `gradU_stencil`

Purpose:
- Compute per-cell spatial gradients over scalar field input, using local and packed neighbor data.

Contract:
- Output stores three gradient components per cell.
- Supports both same-level and coarse-fine interface neighborhoods.
- Must produce correct gradients for affine fields across refinement interfaces.

## 3. Vorticity Magnitude

`vorticity_mag` operator semantics:
- If provided three gradient fields corresponding to velocity components, compute curl magnitude from component derivatives.
- If provided one gradient field, fallback behavior may compute gradient magnitude of that input.

Compatibility requirements:
- For affine velocity definitions with constant curl, output magnitude MUST be constant and match analytical value.
- On periodic smooth velocity fields with mesh refinement, aggregate error MUST decrease with refinement.

## 4. Uniform Slice (AMR Cell-Average Path)

Pipeline-level uniform slice semantics MUST follow this flow:
1. Per-level, per-block accumulation of weighted sum and weighted area into output image bins.
2. Masking of coarse covered cells using level-specific covered boxes.
3. Intra-level reduction over block contributions.
4. Inter-level reduction over level contributions.
5. Final normalization by pixel area.

Numerical behavior:
- Sum accumulation is performed in floating-point accumulation buffers.
- Final output dtype follows requested/inferred byte width.
- Pixels with zero contributing area MUST become NaN.

Validation:
- Resolution dimensions MUST be positive.
- Axis selection MUST be valid.

## 5. Uniform Projection (AMR Cell-Average)

Uniform projection MUST:
- Integrate values through axis range over the selected rectangle.
- Apply covered-cell masking to avoid coarse/fine double counting.
- Aggregate across blocks and levels using reduction stages.

Output semantics:
- Output is accumulated column quantity per image pixel (pre any script-level unit conversions).
- Output buffer element type for reduction accumulators is double-precision.

Validation:
- Resolution dimensions MUST be positive.
- `amr_cell_average=False` mode is unsupported and MUST fail explicitly.

## 6. Field Expression Operator

`field_expr` MUST:
- Parse and evaluate scalar expression over named variables.
- Bind variables to input fields by provided variable order.
- Evaluate element-wise for overlapping valid input length.

Constraints:
- 1 to 8 variables inclusive.
- Input and output byte widths limited to 4 or 8.

Failure conditions:
- Empty expression.
- Empty variable list.
- Variable/input count mismatch.
- Unsupported byte width.
- Parse failure.

## 7. Histograms Over Mesh Fields

### 7.1 Histogram1D

Behavior:
- Bin scalar values over finite increasing range.
- Optional explicit weights input.
- Exclude covered coarse cells via covered-box masking.
- Reduce across all blocks and levels.

### 7.2 Histogram2D

Behavior:
- Jointly bin `x_field` and `y_field`.
- Optional explicit weight field.
- If no explicit weights, support mode-specific implicit weights:
  - `input`: unit weights.
  - `cell_mass`: per-cell mass proxy from field/cell geometry semantics.
  - `cell_volume`: geometric cell volume.
- Exclude covered coarse cells via covered-box masking.

### 7.3 Helper Behavior

- 1D edge helper returns `bins+1` edges from inclusive low to inclusive high boundary definition.
- 2D edge helper applies 1D edge helper independently to each axis.
- CDF from histogram returns cumulative running sum, optionally normalized by total if total is positive.
- CDF from samples returns sorted values and empirical cumulative fractions.

## 8. Particle Field Access and Conversion

### 8.1 `particle_load_field_chunk_f64`

Behavior:
- Reads one particle field chunk by chunk index.
- Converts source numeric dtypes (`float32`, `float64`, `int64`) to `float64` output.
- Errors on unsupported dtype or short payload.

### 8.2 Particle dataset requirements

Particle kernels that read backend particle fields directly require plotfile-backed datasets.

If backend does not support required particle access, operator MUST fail explicitly.

## 9. Particle Boolean/Filter Operators

Required semantics:
- `particle_eq_mask`: exact equality against scalar.
- `particle_isin_mask`: membership against provided scalar set.
- `particle_isfinite_mask`: true where finite.
- `particle_abs_lt_mask`: true where absolute value is less than scalar.
- `particle_le_mask`: true where value <= scalar.
- `particle_gt_mask`: true where value > scalar.
- `particle_and_mask`: elementwise logical and for mask bytes.
- `particle_filter`: select values where mask is true.

Mask encoding:
- Non-zero byte means true; zero means false.
- API-level materialization converts mask bytes to boolean arrays.

## 10. Particle Numeric Operators

Required semantics:
- `particle_subtract`: elementwise subtraction.
- `particle_distance3`: Euclidean distance from two 3D point tuples.
- `particle_sum`: scalar sum of values in one chunk/group.
- `particle_len_f64`: count of values.
- `particle_count`: count of true bytes in mask.
- `particle_min`/`particle_max`: scalar extrema with optional finite filtering.

Reduction semantics:
- Multi-chunk scalar reductions MUST support graph reduction trees via configured reduce kernels.

## 11. Particle Histogram and Mode Operators

### 11.1 `particle_histogram1d`

Behavior:
- Supports explicit edges.
- Supports optional density normalization.
- Supports optional explicit per-sample weights.
- Ignores non-finite samples and non-finite weights.

### 11.2 `particle_topk_modes`

Behavior:
- Counts exact value frequencies.
- Returns top `k` values by descending count.
- Tie-break by descending value.
- Output layout contains values and counts vectors of length `k`.

## 12. Particle CIC AMR Deposition Operators

### 12.1 `particle_cic_grid_accumulate`

Behavior:
- Deposits particle mass on native AMR cell grid for each block.
- Applies axis-range filtering and optional upper mass threshold.
- Performs CIC weighting in in-plane axes.
- Converts mass deposition to density-like quantity using cell volume.
- Excludes cells in covered regions.

### 12.2 `particle_cic_projection_accumulate`

Behavior:
- Performs native deposition then distributes to output pixels by overlap.
- Applies axis-range filtering and optional mass threshold.
- Excludes covered regions.
- Produces 2D accumulated mass map.

## 13. Reduction Helper Kernels

### 13.1 `uniform_slice_add` and `uniform_slice_reduce`

Behavior:
- Elementwise summation over input buffers.
- Supports 4-byte and 8-byte floating-point accumulation modes.

### 13.2 `uniform_slice_finalize`

Behavior:
- Combines sum and area buffers.
- Produces normalized output using provided pixel area.
- Emits NaN where area is zero.

### 13.3 Particle scalar reductions

- `particle_int64_sum_reduce` sums int64 scalars.
- `particle_scalar_min_reduce` computes finite minimum over scalar inputs.
- `particle_scalar_max_reduce` computes finite maximum over scalar inputs.

