# Flux Surface Integration Operator Plan

## Goal

Implement a Kangaroo operator for spherical surface integration of mass flux,
passive scalar flux, and energy flux. The numerical algorithm must match the
Quokka implementation in:

`/autofs/nccs-svm1_home1/wibking/quokka/src/problems/DiskGalaxy/testDiskGalaxy.cpp`
starting at line 773, including its helper
`quokka::math::sphericalSectionAreaInCell` from
`src/math/spherical_geometry.hpp`.

The one intentional deviation from the Quokka implementation is magnetic-field
sampling. Quokka reconstructs cell-centered magnetic fields by averaging
face-centered fields. Kangaroo plotfiles will only provide cell-centered
magnetic fields, so the Kangaroo operator will read `Bx`, `By`, and `Bz`
directly at cell centers and otherwise preserve the reference algorithm.

The operator should produce these four double-precision scalar outputs for a
chosen radius:

- `mass_flux_sphere`
- `hydro_energy_flux_sphere`
- `mhd_energy_flux_sphere`
- `passive_scalar_flux_sphere`

## Reference Algorithm

For every AMR level and every valid cell not covered by a finer level:

1. Compute cell edges from level geometry:
   - `x0 = prob_lo[0] + i * dx[0]`, `x1 = x0 + dx[0]`
   - same for `y` and `z`
2. Compute cell center and radius:
   - `x = prob_lo[0] + (i + 0.5) * dx[0]`
   - `r = sqrt(x*x + y*y + z*z)`
3. Skip the cell when `r <= 0` or `rho <= 0`.
4. Compute velocity and radial velocity:
   - `vx = momx / rho`, `vy = momy / rho`, `vz = momz / rho`
   - `vr = (x*momx + y*momy + z*momz) / (rho*r)`
   - `rhat = (x/r, y/r, z/r)`
5. Read cell-centered conserved fields:
   - density
   - x/y/z momentum
   - total energy density
   - passive scalar density
6. Read cell-centered magnetic field components:
   - `Bx = Bx(i,j,k)`
   - `By = By(i,j,k)`
   - `Bz = Bz(i,j,k)`
   This is the only deliberate deviation from the Quokka reference, because the
   Kangaroo plotfile inputs are cell-centered rather than face-centered.
7. Compute pressure, magnetic energy, and hydrodynamic energy:
   - `Pgas` must match the Quokka EOS pressure calculation.
   - `Emag` must match `HydroSystem<DiskGalaxy>::ComputeMagneticEnergy`.
   - `Ehydro = energy_density - Emag`.
8. Compute flux densities:
   - mass: `rho * vr`
   - hydro energy: `(Ehydro + Pgas) * vr`
   - MHD energy: `(energy_density + Pgas + Emag) * vr - Bdotv * Br`
   - passive scalar: `scalar_density * vr`
9. Compute area with the same approximation as Quokka:
   - reject cells when `R^2` is outside the cell min/max distance range,
   - approximate the sphere/cell section by the exact tangent-plane/box
     intersection through the cell, using the sphere normal at the cell center,
   - use the same polygon construction, tolerance, angular sort, and shoelace
     area as `sphericalSectionAreaInCell`.
10. Add `flux_density * area` to each component.
11. Sum across blocks, levels, and localities.

## Public Python API

Add a pipeline-level method:

```python
flux = pipe.flux_surface_integral(
    density,
    momentum=(momx, momy, momz),
    energy=energy,
    passive_scalar=scalar0,
    radius=flux_sphere_radius,
    pressure_model="ideal_gas",
    gamma=5.0 / 3.0,
    magnetic_field=(bx, by, bz),
    out="flux_sphere",
)
```

Return a small handle, for example `FluxSurfaceIntegralHandle`, exposing the
single output field plus component names. The output layout should be a
`float64[4]` buffer in this order:

1. mass flux
2. hydro energy flux
3. MHD energy flux
4. passive scalar flux

The API should also support a hydrodynamic mode without magnetic inputs only if
that mode is explicitly useful. For the Quokka-equivalent DiskGalaxy operator,
require all three cell-centered magnetic inputs and fail early if they are
missing.

## Parameters and Field Inputs

The lowering layer should pass these kernel parameters:

- `radius`: physical radius in the same units as `RunMeta` geometry.
- `bytes_per_value`: `4` or `8`; the first implementation can require `8`.
- `covered_boxes`: coarse cells covered by finer levels, using Kangaroo's
  existing shared `covered_boxes_ref` hoisting.
- `pressure_model`: initially `ideal_gas`.
- `gamma`: ideal-gas ratio of specific heats.
- Optional component labels for debugging/output metadata.

The kernel should take nine input fields:

1. `rho`
2. `momx`
3. `momy`
4. `momz`
5. `energy`
6. `passive_scalar`
7. `Bx` cell-centered x component
8. `By` cell-centered y component
9. `Bz` cell-centered z component

That is nine inputs total. If Kangaroo's `KernelDesc::n_inputs` supports this
without additional executor changes, keep the direct multi-field interface. If
the current input count assumptions are too narrow, add the least invasive
runtime support rather than packing fields in Python.

## Magnetic Field Representation

Kangaroo will use cell-centered magnetic field payloads with the same block
shape as the conserved fields:

- `Bx`: `(nx, ny, nz)`
- `By`: `(nx, ny, nz)`
- `Bz`: `(nx, ny, nz)`

This differs from Quokka only in how `Bx`, `By`, and `Bz` are sampled. Quokka
averages staggered face fields onto the cell center before computing `Bdotv`,
`Br`, and magnetic energy. Kangaroo will read the already cell-centered values
directly and then use the same flux formulas.

Plan:

1. Support memory-backed tests by storing magnetic chunks with the same shape
   as the hydrodynamic conserved fields.
2. Use the existing canonical cell-centered indexing helper for all magnetic
   inputs.
3. Keep the operator contract strict: the MHD mode requires all three magnetic
   component fields.
4. Document this deviation in `spec/operators.md` so future validation against
   Quokka does not mistake expected magnetic-field sampling differences for a
   runtime bug.

## C++ Runtime Work

Add a new kernel in `cpp/src/runtime.cpp`, or move the implementation to a
dedicated source file once it grows:

- kernel name: `flux_surface_integral_accumulate`
- inputs: conserved cell-centered fields plus cell-centered magnetic fields
- outputs: one `float64[4]` block-local accumulator
- params: radius, byte widths, pressure settings, covered boxes

Implementation details:

- Initialize the output buffer to four zero doubles.
- Iterate over all cells in the block.
- Skip any global cell index contained by `covered_boxes`.
- Decode scalar input values with the configured byte width.
- Use canonical Kangaroo cell-centered indexing `(i * ny + j) * nz + k`.
- Use the same cell-centered indexing for each magnetic field.
- Compute `area = spherical_section_area_in_cell(radius, x0, x1, y0, y1, z0, z1)`.
- Skip cells with `area <= 0`.
- Accumulate the four products.

Add helper functions near the projection/geometry helpers or in a new
`cpp/include/kangaroo/spherical_geometry.hpp`:

- `min_dist_sq_to_interval`
- `max_dist_sq_to_interval`
- `plane_box_section_area`
- `spherical_section_area_in_cell`

These should be a direct C++20/host equivalent of Quokka's helper, with names
and comments adjusted for Kangaroo. Preserve the algorithm, tolerance, point
deduplication, angular sort, and final area calculation.

## Python Lowering Work

Add a class in `analysis/ops.py`, for example `FluxSurfaceIntegral`, following
the reduction style used by `Histogram1D`, `Histogram2D`, and
`UniformProjection`.

Lowering flow:

1. Validate `radius > 0`.
2. Resolve input fields from `FieldHandle`/field IDs in `analysis/pipeline.py`.
3. For each AMR level, build `covered_boxes` by coarsening every finer-level
   box into the current level, matching the existing histogram/projection
   masking behavior.
4. Optionally cull blocks to those whose physical extents can intersect the
   sphere:
   - include block when `radius^2` lies between the block min/max squared
     distance to the origin.
   - conservative inclusion is acceptable; false negatives are not.
5. Emit per-block `flux_surface_integral_accumulate` tasks with `output_bytes =
   [4 * 8]`.
6. Reduce per-level block outputs with `uniform_slice_reduce` and
   `bytes_per_value = 8`.
7. Reduce across levels with `uniform_slice_add` or `uniform_slice_reduce`,
   preserving the existing graph-reduce conventions.
8. Finalize into a persistent output field named by `out`.

Add `Pipeline.flux_surface_integral(...)` and export the handle from
`analysis/__init__.py` if public API parity with other operator handles is
desired.

## Pressure and Energy Semantics

The only ambiguous part of the Quokka port is pressure. In Quokka,
`HydroSystem<DiskGalaxy>::ComputePressure` uses the problem physics/EOS and
magnetic fields after Quokka's face-to-center reconstruction. Kangaroo should
not hide this behind an underspecified formula; it should use the same pressure
formula with the cell-centered magnetic inputs available from plotfiles.

Initial implementation options:

1. Ideal-gas pressure from hydrodynamic internal energy:
   - `kinetic = 0.5 * (momx*momx + momy*momy + momz*momz) / rho`
   - `Emag = 0.5 * (Bx*Bx + By*By + Bz*Bz)` in the same unit convention as
     Quokka's `ComputeMagneticEnergy`
   - `Ehydro = energy - Emag`
   - `Pgas = (gamma - 1) * (Ehydro - kinetic)`
2. A pressure-field mode:
   - accept a precomputed pressure field and use it directly.
   - this is useful if exact DiskGalaxy pressure requires physics not already
     available in Kangaroo.

For "same algorithm except cell-centered magnetic sampling" parity with the
DiskGalaxy reference, add a validation step against Quokka output before
treating the ideal-gas path as complete. If DiskGalaxy pressure includes
additional physics or unit conventions, prefer the pressure-field mode or
implement a named `disk_galaxy` pressure model with the same formula.

## Testing Plan

### Python Lowering Tests

Add tests in `tests/test_pipeline_api.py` or a new
`tests/test_flux_surface_integral.py`:

- Lowering emits one accumulation template per intersecting block.
- Each accumulation template uses `flux_surface_integral_accumulate`.
- Output bytes are `32`.
- Reduce stages use `uniform_slice_reduce` with `bytes_per_value = 8`.
- Coarse-level templates include coarsened `covered_boxes`; finest-level
  templates use an empty mask.
- `radius <= 0` raises `ValueError`.
- Missing cell-centered magnetic fields in MHD mode raises `ValueError`.

### Geometry Unit Tests

Add C++-backed runtime tests through memory datasets:

- `spherical_section_area_in_cell` returns zero when the sphere cannot
  intersect the cell.
- For a plane-aligned tangent section through a unit cube, the area matches the
  expected rectangle area.
- Symmetry checks: sign/permutation changes of cell coordinates preserve area
  when geometry is symmetric.
- For a uniform Cartesian grid shell, total summed area approaches `4*pi*R^2`
  under refinement. This is a convergence check for the Quokka approximation,
  not an exact equality test.

### Single-Level Flux Tests

Use `memory://` datasets with simple fields:

- Constant density, zero velocity: all four fluxes are zero.
- Constant density and radial velocity `v = a * rhat`: mass flux should be
  approximately `rho * a * 4*pi*R^2`, with convergence as resolution increases.
- Constant passive scalar density with the same radial velocity: passive scalar
  flux should scale as `scalar_density * a * area`.
- Constant magnetic field and velocity cases where `Bdotv * Br` has a known
  analytic value should validate the MHD energy term.
- Negative or zero density cells are skipped.

### AMR Masking Tests

Build two-level synthetic AMR metadata:

- Level 0 covers a coarse domain.
- Level 1 refines part of the spherical surface.
- Fill coarse and fine fields with deliberately different constants.
- Verify the result equals "fine contribution over refined cells plus coarse
  contribution elsewhere", proving coarse covered cells are masked.
- Add a regression test where the sphere intersects only refined cells to catch
  double counting.

### Multi-Block and Graph Reduction Tests

- Split a single-level domain into several blocks and compare against one-block
  output.
- Use `reduce_fan_in=2` to force multiple graph reduction rounds.
- Run with a fake multi-locality home-rank runtime in lowering tests to verify
  grouped block reduction metadata is stable.

### Plotfile Integration Tests

- Run the operator on a small DiskGalaxy plotfile.
- Compare the four output scalars against a Quokka-derived reference for the
  same `flux_sphere_radius_kpc`, accounting only for the known magnetic-field
  sampling deviation.
- Record the exact validation command and tolerance.
- Include a skip condition when the fixture plotfile is unavailable.

### Validation Commands

During implementation, run:

```bash
pixi run test tests/test_pipeline_api.py
pixi run test tests/test_flux_surface_integral.py
pixi run test
```

Run `pixi run build` before runtime tests that require the new C++ kernel.

## Implementation Milestones

1. Add the planning-level public API and lowering tests.
2. Add `FluxSurfaceIntegral` lowering and handle type.
3. Add spherical geometry helpers with focused tests.
4. Add `flux_surface_integral_accumulate` for memory-backed cell-centered fields.
5. Add runtime numerical tests for single-level and AMR synthetic cases.
6. Validate plotfile cell-centered magnetic field component access.
7. Validate against Quokka DiskGalaxy stats, or against an adjusted Quokka
   reference that uses cell-centered magnetic fields, and document tolerances.
8. Update `spec/operators.md` with the final operator contract.

## Open Questions

- What exact field names should Kangaroo use for DiskGalaxy cell-centered
  magnetic fields in plotfiles?
- Does DiskGalaxy pressure reduce to the ideal-gas formula above for the target
  outputs, or does Kangaroo need a named DiskGalaxy pressure model?
- Should the public API return one `float64[4]` field, four scalar handles, or
  both?
- Should radius unit conversion from kpc to cgs live in user scripts, or should
  the operator accept unit-labeled convenience parameters?
