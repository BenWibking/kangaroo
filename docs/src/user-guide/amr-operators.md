# AMR operators

This page covers every high-level operation currently available on a mesh field
returned by `ds["field"]`.

Kangaroo has two useful kinds of mesh array:

- An **AMR field** covers the dataset's hierarchy of levels and blocks. Its
  `shape` is `None`.
- A **regular array** is a bounded rectangular result, such as a slice or
  projection. It has a concrete `shape` and materializes as one NumPy array.

Most element-wise operations work on either kind. Geometry-aware operations
such as slicing, projecting, vorticity, flux surfaces, and Toomre-Q start from
an AMR field.

## Operator overview

| Operation | Called on | Result |
|---|---|---|
| `+`, `-`, `*`, `/`, `**`, unary `-` | AMR or regular numeric array | Lazy `Array` with the same domain |
| `<`, `<=`, `>`, `>=`, `==`, `!=` | AMR or regular numeric array | Lazy boolean `MeshMask` |
| `mask1 & mask2` | Two matching mesh masks | Combined `MeshMask` |
| `rename(name)` | AMR or regular array | Same calculation with a new display name |
| `astype(dtype)` | AMR or regular numeric array | Lazy float32 or float64 `Array` |
| `slice(...)` | AMR field | Regular two-dimensional `Array` |
| `project(...)` | AMR field | Regular two-dimensional `Array` |
| `histogram(...)` | AMR or regular array | `Histogram` |
| `histogram2d(...)` | Two matching AMR or regular arrays | `Histogram2D` |
| `vorticity(y, z)` | Three AMR velocity fields | AMR vorticity-magnitude `Array` |
| `flux_surface_integral(...)` | Density AMR field plus MHD fields | `FluxSurfaceIntegral` |
| `cylindrical_flux_surface_integral(...)` | Density AMR field plus MHD fields | `CylindricalFluxSurfaceIntegral` |
| `toomre_q_profile(...)` | Density AMR field plus disk fields | `ToomreQProfile` |
| `iter_chunks()` | AMR or regular array | Iterator over NumPy arrays |

## Arithmetic and derived fields

Mesh arrays support normal Python arithmetic with finite numeric scalars and
with other arrays:

```python
density = ds["density"]
momentum_x = ds["xmom"]

velocity_x = (momentum_x / density).rename("velocity_x")
density_squared = density**2
shifted = 1.0 + density
negative = -density
```

The available operators are addition, subtraction, multiplication, division,
power, and unary negation. Scalars can appear on either side of addition,
subtraction, multiplication, and division; exponentiation uses
`array ** exponent`. Array-to-array expressions must use fields from the same
dataset view and the same physical domain. This prevents, for example,
accidentally adding two same-shaped slices taken at different coordinates.

Use `astype()` to choose floating-point precision:

```python
density_f32 = density.astype(kr.float32)
density_f64 = density_f32.astype("float64")
```

Mesh conversion currently supports only `float32` and `float64`.

## Comparisons and masks

Comparisons are lazy and return a `kr.MeshMask`:

```python
interesting = (ds["density"] > 1.0e-24) & (ds["temperature"] < 1.0e5)
mask_chunks = interesting.compute()
```

The available comparisons are `<`, `<=`, `>`, `>=`, `==`, and `!=`. Combine
two masks with `&`. Both masks must describe the same dataset and physical
domain. Mesh masks are materialized as boolean NumPy arrays, either inside an
`AMRChunkedArray` or as one bounded array.

At present, masks are useful as outputs and diagnostics; mesh-array indexing by
a mask is not part of the high-level API.

## Slices

`slice()` samples an AMR field on a plane perpendicular to `axis`:

```python
image = ds["density"].slice(
    axis="z",
    coord=0.0,
    resolution=(1024, 768),
    zoom=2.0,
)

pixels = image.compute()
```

Parameters:

- `axis`: `"x"`, `"y"`, `"z"`, or the corresponding index `0`, `1`, `2`.
- `coord`: physical coordinate along that axis. If omitted, Kangaroo chooses a
  cell-centered coordinate near the middle of the selected level.
- `resolution`: `(width, height)` or the string `"width,height"`. If omitted,
  the selected level's in-plane cell counts are used. The resulting NumPy shape
  is `(height, width)`.
- `zoom`: zoom around the center of the default view. Values greater than one
  show a smaller region.
- `rect`: optional physical in-plane bounds `(u_min, v_min, u_max, v_max)` in
  the two displayed axes. It replaces the rectangle chosen from `zoom`.

The output is a regular two-dimensional array. Arithmetic, comparisons, casts,
and histograms can be applied to it, but it cannot be sliced or projected again.

## Projections

`project()` integrates a field along an axis and places the result on a regular
two-dimensional grid:

```python
column_density = ds["density"].project(
    axis="z",
    bounds=(-2.0e21, 2.0e21),
    resolution=(1024, 1024),
    zoom=1.5,
)
```

Parameters:

- `axis`: projection direction, as `"x"`, `"y"`, `"z"`, `0`, `1`, or `2`.
- `bounds`: physical lower and upper limits along the projection direction. The
  full axis is used when omitted.
- `resolution`, `zoom`, and `rect`: control the output plane in the same way as
  `slice()`.
- `amr_cell_average`: keeps the default AMR cell-average treatment when `True`.

Projection results use `float64` and materialize as NumPy arrays.

## One-dimensional histograms

`histogram()` counts values in evenly spaced bins:

```python
hist = ds["density"].histogram(
    bins=128,
    range=(1.0e-30, 1.0e-20),
)
result = hist.compute()

counts = result.counts
edges = result.edges
```

Set `weights` to another array from the same dataset and domain for a weighted
histogram:

```python
mass_by_temperature = ds["temperature"].histogram(
    bins=100,
    range=(1.0, 1.0e8),
    weights=ds["cell_mass"],
)
```

Histograms over an AMR hierarchy leave out coarse cells covered by finer data,
so refined regions are not counted twice. Histograms on a slice or projection
operate on that bounded result only.

The lazy `Histogram` exposes `edges` before execution. `compute()` returns
`kr.HistogramResult` with `counts` and `edges` arrays.

## Two-dimensional histograms

`histogram2d()` bins two arrays together:

```python
phase = ds["density"].histogram2d(
    ds["temperature"],
    bins=(256, 256),
    range=((1.0e-30, 1.0e-20), (1.0, 1.0e8)),
    weights=ds["cell_mass"],
)

result = phase.compute()
counts, density_edges, temperature_edges = (
    result.counts,
    result.x_edges,
    result.y_edges,
)
```

The first array supplies the x values and the second supplies the y values.
`bins` and `range` contain one entry per axis. `weights` is optional. All input
arrays must share one dataset view and physical domain.

AMR coverage is handled in the same way as for one-dimensional histograms.

## Vorticity magnitude

Call `vorticity()` on the x velocity component and pass the y and z components:

```python
vorticity = ds["velocity_x"].vorticity(
    ds["velocity_y"],
    ds["velocity_z"],
)
```

The result is an AMR field containing the magnitude of the curl of the velocity
vector. It can feed directly into slices, projections, histograms, arithmetic,
or chunk-wise processing:

```python
vorticity_image = vorticity.slice(axis="z", resolution=(1024, 1024))
```

All three velocity components must come from the same dataset view.

## Spherical flux surfaces

`flux_surface_integral()` calculates inward and outward fluxes through one or
more spheres centered at the physical origin. Call it on density and supply the
other hydrodynamic and magnetic fields:

```python
flux = ds["density"].flux_surface_integral(
    momentum=(ds["xmom"], ds["ymom"], ds["zmom"]),
    energy=ds["Etot"],
    passive_scalar=ds["metal_density"],
    magnetic_field=(ds["Bx"], ds["By"], ds["Bz"]),
    radius=(1.0e21, 2.0e21, 4.0e21),
    gamma=5.0 / 3.0,
)

result = flux.compute()
```

`radius` may be one positive radius or a sequence. The result identifies its
radii and component names alongside the `values` array. Components cover mass,
hydrodynamic energy, MHD energy, and passive-scalar flux, each split into
negative (inward) and positive (outward) contributions.

To split each contribution into temperature intervals, provide both a
temperature field and increasing bin edges:

```python
flux_by_temperature = ds["density"].flux_surface_integral(
    momentum=(ds["xmom"], ds["ymom"], ds["zmom"]),
    energy=ds["Etot"],
    passive_scalar=ds["metal_density"],
    magnetic_field=(ds["Bx"], ds["By"], ds["Bz"]),
    radius=2.0e21,
    temperature=ds["temperature"],
    temperature_bins=(0.0, 1.0e4, 1.0e6, 1.0e9),
)
```

Radii must intersect the mesh. `gamma` must be greater than one.

## Cylindrical flux surfaces

`cylindrical_flux_surface_integral()` measures flux through a cylinder aligned
with the z axis. It reports the endcaps and cylindrical walls separately:

```python
flux = ds["density"].cylindrical_flux_surface_integral(
    momentum=(ds["xmom"], ds["ymom"], ds["zmom"]),
    energy=ds["Etot"],
    passive_scalar=ds["metal_density"],
    magnetic_field=(ds["Bx"], ds["By"], ds["Bz"]),
    radius=8.0e21,
    height=(1.0e21, 2.0e21, 4.0e21),
)

result = flux.compute()
print(result.geometric_sections)  # ("endcaps", "walls")
```

`radius` is one fixed positive value. `height` may be one positive half-height
or a sequence of half-heights. Temperature bins and `gamma` work as they do for
spherical flux surfaces. The returned result also carries `radius`, `heights`,
component names, and temperature-bin edges.

## Gas Toomre-Q profiles

`toomre_q_profile()` accumulates the radial information needed to calculate gas
Toomre-Q profiles in a disk centered on `center` and oriented in the xy plane:

```python
profile = ds["density"].toomre_q_profile(
    momentum=(ds["xmom"], ds["ymom"]),
    internal_energy=ds["gasInternalEnergy"],
    magnetic_field=(ds["Bx"], ds["By"], ds["Bz"]),
    potential=ds["gpot"],
    z_bounds=(-4.0e21, 4.0e21),
    radial_range=(0.5e21, 16.0e21),
    bins=64,
    center=(0.0, 0.0, 0.0),
    gamma=5.0 / 3.0,
)

result = profile.compute()
```

The operator accepts either evenly spaced bins through `radial_range` plus
`bins`, or an explicit sequence of `radial_edges`:

```python
profile = ds["density"].toomre_q_profile(
    momentum=(ds["xmom"], ds["ymom"]),
    internal_energy=ds["gasInternalEnergy"],
    magnetic_field=(ds["Bx"], ds["By"], ds["Bz"]),
    potential=ds["gpot"],
    z_bounds=(-4.0e21, 4.0e21),
    radial_edges=(0.5e21, 1.0e21, 2.0e21, 4.0e21, 8.0e21),
)
```

Do not combine `radial_edges` with `radial_range` or `bins`.

The result contains the accumulated `moments`, their component names, the
actual radial edges, the vertical bounds, center, and gamma. The seven moment
components are mass, internal energy, magnetic-field-squared volume term, radial
momentum, radial-velocity second moment, radial gravity, and sampled volume.
These are the ingredients used by the plotting workflow to derive the final
thermal, turbulent, and magnetic Toomre-Q curves; the result is intentionally
more reusable than a single precomputed Q array.

## Materializing AMR data safely

An unbounded AMR field returns `kr.AMRChunkedArray`:

```python
hierarchy = ds["density"].compute()

for chunk in hierarchy:
    print(chunk.level, chunk.block, chunk.box)
    consume(chunk.values)
```

There is no `gather=True` mode for an AMR hierarchy because refinement levels
may overlap and blocks do not have a single linear ordering. Use one of these
approaches instead:

- `array.iter_chunks()` for block-wise processing;
- `array.slice(...)` for a regular plane;
- `array.project(...)` for an axis-integrated regular image;
- `array.histogram(...)` or `array.histogram2d(...)` for global distributions.
