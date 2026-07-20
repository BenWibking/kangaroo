# High-level Python reference

The `kangaroo` module is the main entry point for interactive analysis and
application code:

```python
import kangaroo as kr
```

It provides a lazy, array-like interface. Opening a dataset and combining fields
does not immediately read and process the whole dataset. Kangaroo records the
work and starts it when you call `compute()` or `persist()`.

## Opening a dataset

```python
ds = kr.open_dataset(
    "/path/to/plotfile",
    step=0,
    level=0,
)
```

`kr.open_dataset()` accepts a local path or one of the URIs described in
[Data sources](data-sources.md). The optional arguments are:

- `step`: the time step to use. The default is `0`.
- `level`: the AMR level used for plane geometry and reduced outputs. When it is
  omitted, the backend chooses its default level.
- `runmeta`: explicit run metadata, mainly useful for in-memory datasets and
  tests.
- `client`: a specific `kr.Client`. Without one, Kangaroo creates and reuses a
  process-local default client.

The returned `kr.Dataset` is a view of the selected step and level. It has these
useful attributes:

| Attribute | What it provides |
|---|---|
| `ds.fields` | A browsable mapping of mesh field names to lazy `kr.Array` objects. |
| `ds.particles` | A mapping of particle-species names to `kr.ParticleSpecies` objects. |
| `ds.meshes` | Names of meshes exposed by the backend. |
| `ds.metadata` | Read-only dataset and backend metadata. |
| `ds.geometry` | Helpers for choosing physical planes. |
| `ds.step`, `ds.level` | The active selection. |

Mesh fields can be accessed directly or through `fields`:

```python
density = ds["density"]
temperature = ds.fields["temperature"]

print(list(ds.fields))
print(ds)
```

Unknown field and particle names produce an error that lists available names
and, when possible, close matches.

Use `select()` to create another view without changing the original dataset:

```python
next_step = ds.select(step=1)
fine_view = ds.select(level=2)
mesh_view = ds.select(mesh="hydro")
```

## Clients and runtime options

For a simple script, the default client is usually enough. Create a client
explicitly when you need HPX arguments, HPX configuration, or progress output:

```python
client = kr.Client(
    hpx_args=["--hpx:threads=8"],
    progress=True,
)
ds = client.open_dataset("/path/to/plotfile")
```

Command-line applications can forward arguments they do not recognize:

```python
args, unknown = parser.parse_known_args()
client = kr.Client.from_parsed_args(args, unknown_args=unknown)
```

`kr.get_default_client()` returns the current default client.
`kr.set_default_client(client)` replaces it, and
`kr.set_default_client(None)` makes the next `kr.open_dataset()` create a fresh
one.

## Working with lazy values

All lazy values expose a small common interface:

| Member | Purpose |
|---|---|
| `value.name` | Display name used in diagnostics. |
| `value.dtype` | NumPy-style dtype name. |
| `value.domain` | Short description such as `AMR(2 levels)` or `Regular(shape=(512, 512))`. |
| `value.is_materialized` | Whether this object has already crossed an execution boundary. |
| `value.compute()` | Run the required work and return a local result. |
| `value.persist()` | Run the work and keep its distributed output for reuse. |
| `value.explain()` | Summarize stages, tasks, storage, reductions, and kernels without running them. |
| `value.visualize()` | Return the graph as Graphviz DOT; pass a filename to save it. |

For example:

```python
hot_density = (ds["density"] * 2.0).rename("double_density")

print(hot_density.explain())
hot_density.visualize("density-plan.dot")

prepared = hot_density.persist(progress=True)
image = prepared.slice(axis="z", resolution=(1024, 1024)).compute()
```

`persist()` is useful when several later calculations reuse an expensive
derived field. It returns the same lazy value, now backed by data retained in
the runtime.

## Computing one or more results

Use the method form for one result:

```python
image = ds["density"].slice(axis="z", resolution=(512, 512))
image_data = image.compute()
```

Use `kr.compute()` when results share input work:

```python
density = ds["density"]
image = density.slice(axis="z", resolution=(512, 512))
histogram = density.histogram(bins=128, range=(1.0e-30, 1.0e-20))

image_data, histogram_data = kr.compute(image, histogram, progress=True)
```

All values in one `kr.compute()` call must come from the same dataset view.
Kangaroo combines their work into one execution where possible.

## Mesh arrays and their results

A field such as `ds["density"]` is an unbounded `kr.Array`: it represents the
AMR hierarchy rather than one rectangular NumPy array. Its `shape` is `None`,
and `chunks` reports `(level, block count, cell count)` for each level.

Calling `compute()` on it returns `kr.AMRChunkedArray`. Each `kr.AMRChunk`
contains:

- `values`: the block's NumPy array;
- `step`, `level`, and `block`: where the block came from;
- `box` and `geometry`: enough information to locate it in the hierarchy.

AMR blocks can overlap across refinement levels, so they cannot be safely
concatenated. Iterate them, or first produce a regular slice or projection:

```python
for chunk in density.compute():
    print(chunk.level, chunk.box, chunk.values.shape)

for values in density.iter_chunks():
    process(values)
```

A slice or projection is a bounded regular `kr.Array`. It has a concrete
`shape`, and `compute()` returns a NumPy array directly.

See [AMR operators](amr-operators.md) for every operation available on mesh
arrays, including arithmetic, masks, slices, projections, histograms,
vorticity, flux surfaces, and Toomre-Q profiles.

## Particles

Particle species and fields use the same browsable mapping style as mesh fields:

```python
stars = ds.particles["stars"]
mass = stars["mass"]
print(list(stars.fields))
```

Particle calculations are lazy and chunked. They include arithmetic, masks and
filtering, scalar reductions, histograms, top-k modes, and cloud-in-cell mass
projections. See [Particle operators](particle-operators.md) for the complete
guide.

## Result objects

Reductions return small named result objects instead of anonymous tuples:

| Lazy expression | Result |
|---|---|
| `Histogram` | `HistogramResult(counts, edges)` |
| `Histogram2D` | `Histogram2DResult(counts, x_edges, y_edges)` |
| `FluxSurfaceIntegral` | `FluxSurfaceIntegralResult(values, radii, components, temperature_bins)` |
| `CylindricalFluxSurfaceIntegral` | `CylindricalFluxSurfaceIntegralResult(values, radius, heights, geometric_sections, components, temperature_bins)` |
| `ToomreQProfile` | `ToomreQProfileResult(moments, radial_edges, components, z_bounds, center, gamma)` |
| `TopK` | `TopKResult(values, counts)` |
| Particle `Scalar` | Python `float` or `int` |

Histogram edges are available before execution through `histogram.edges`.
Lazy particle scalars deliberately cannot be converted with `float()`, `int()`,
or a truth test; call `compute()` first.

## Geometry helpers

`ds.geometry.plane()` shows the geometry Kangaroo will use for a slice or
projection without creating the operation:

```python
plane = ds.geometry.plane(
    axis="z",
    coord=0.0,
    zoom=2.0,
    resolution=(1024, 1024),
)

print(plane.rect, plane.labels, plane.axis_bounds)
```

The returned `kr.PlaneGeometry` contains the chosen coordinate, in-plane
rectangle, output resolution, axis labels, plane name, numeric axis index, and
full bounds along the viewing axis.

## Dtypes and configuration

The module exports NumPy dtype aliases `kr.bool_`, `kr.float32`, `kr.float64`,
and `kr.int64`. Mesh and particle `astype()` currently accept `float32` and
`float64`.

`kr.config.set()` temporarily changes graph-building options in the current
Python context. For example, this changes the reduction fan-in for operations
built inside the block:

```python
with kr.config.set({"reduction.fan_in": 8}):
    histogram = ds["density"].histogram(
        bins=128,
        range=(1.0e-30, 1.0e-20),
    )
```
