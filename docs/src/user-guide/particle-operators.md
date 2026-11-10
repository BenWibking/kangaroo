# Particle operators

Kangaroo exposes particle data as lazy, chunked arrays. You can build filters,
derived values, and reductions without first collecting every particle into one
large NumPy array.

## Finding species and fields

`ds.particles` is a mapping of particle-species names:

```python
print(list(ds.particles))

stars = ds.particles["stars"]
print(list(stars.fields))

mass = stars["mass"]
x = stars["x"]
```

Each field is a `kr.ParticleArray`. Its `chunks` property reports the number of
distributed chunks, while `domain` gives a short description suitable for
interactive inspection.

Unknown species and field names produce an error with available names and close
matches.

## Operator overview

| Operation | Result |
|---|---|
| `+`, `-`, `*`, `/`, `**`, unary `-` | Derived `ParticleArray` |
| `<`, `<=`, `>`, `>=`, `==`, `!=` | `ParticleMask` |
| `isfinite()` | Mask selecting finite values |
| `mask1 & mask2` | Combined `ParticleMask` |
| `values[mask]` | Filtered `ParticleArray` |
| `rename(name)` | Same lazy value with a new display name |
| `astype(dtype)` | Particle array materialized as float32 or float64 |
| `sum()`, `min()`, `max()`, `mean()` | Lazy `Scalar` |
| `mask.count()` | Lazy number of selected particles |
| `histogram(...)` | Lazy `Histogram` |
| `topk(k)` | Lazy `TopK` mode counts |
| `iter_chunks()` | Iterator over NumPy arrays |
| `species.project(...)` | Regular cloud-in-cell mass image |

Like other Kangaroo values, particle arrays also provide `compute()`,
`persist()`, `explain()`, and `visualize()`.

## Arithmetic

Particle arrays support element-wise addition, subtraction, multiplication,
division, power, and negation:

```python
mass_solar = stars["mass"] / solar_mass
specific_energy = stars["energy"] / stars["mass"]
radius_squared = stars["x"] ** 2 + stars["y"] ** 2 + stars["z"] ** 2
```

Scalars can appear on either side of a binary operator:

```python
shifted = 1.0 + mass
inverse = 1.0 / mass
exponential = 2.0 ** stars["evolution_stage"]
```

Arrays combined in one expression must come from the same dataset, particle
species, and current selection. Kangaroo also keeps track of filtering history,
so two arrays filtered in different ways cannot be combined accidentally.

## Comparisons, masks, and filtering

Compare a particle field with a numeric scalar to create a mask:

```python
mass = stars["mass"]
valid = mass.isfinite() & (mass > 0.0) & (mass <= 1.0e36)
selected_mass = mass[valid]
```

The available comparisons are `<`, `<=`, `>`, `>=`, `==`, and `!=`.
`isfinite()` removes NaN and infinite values. Combine masks with `&`, then use a
mask inside square brackets to filter any aligned field from the same species:

```python
selected_mass = stars["mass"][valid]
selected_x = stars["x"][valid]
selected_y = stars["y"][valid]
```

Reusing one mask preserves the alignment of the filtered arrays. Masks from
different species or independently filtered selections cannot be combined.

`kr.ParticleMask` is also available as the shorter alias `kr.Mask`.

## Names and materialization dtype

`rename()` changes the label used by representations and diagnostics without
changing the calculation:

```python
mass_solar = (mass / solar_mass).rename("stellar_mass_msun")
```

`astype()` chooses the floating-point dtype used when chunks are materialized:

```python
mass_f32 = mass.astype(kr.float32)
mass_f64 = mass.astype("float64")
```

Particle conversion currently supports `float32` and `float64`.

## Scalar reductions

Particle reductions remain lazy until computed:

```python
valid_mass = mass[mass.isfinite() & (mass > 0.0)]

total = valid_mass.sum()
minimum = valid_mass.min()
maximum = valid_mass.max()
average = valid_mass.mean()
number = (mass > 0.0).count()

total_value, min_value, max_value, mean_value, count_value = kr.compute(
    total,
    minimum,
    maximum,
    average,
    number,
)
```

`min()` and `max()` ignore non-finite values by default. Pass
`finite_only=False` to include them in the reduction.

The result of a scalar reduction is a `kr.Scalar` until execution and a Python
`float` or `int` afterward. A lazy scalar cannot be used in `float()`, `int()`,
an `if` statement, or another truth test; call `compute()` first.

## Histograms

For evenly spaced bins, pass a bin count and range:

```python
histogram = valid_mass.histogram(
    bins=64,
    range=(0.0, 1.0e36),
)
result = histogram.compute()

counts = result.counts
edges = result.edges
```

Explicit edges allow nonuniform bins:

```python
import numpy as np

histogram = valid_mass.histogram(
    bins=np.geomspace(1.0e30, 1.0e36, 49),
)
```

Use another aligned particle array as weights:

```python
weighted = stars["temperature"][valid].histogram(
    bins=100,
    range=(1.0, 1.0e8),
    weights=stars["mass"][valid],
)
```

Set `density=True` to normalize the counts by their total and bin widths.
`histogram.edges` is available before execution. Computing the histogram returns
`kr.HistogramResult(counts, edges)`.

## Top-k modes

`topk()` finds the most common values in a field, which is useful for discrete
particle properties such as state or evolution-stage identifiers:

```python
modes = stars["evolution_stage"].topk(5).compute()

print(modes.values)
print(modes.counts)
```

The result is `kr.TopKResult(values, counts)`. This operation needs a field that
comes directly from the particle backend; it is not available on an arithmetic
or filtered expression. Renaming a source field does not change its provenance.

## Chunked and gathered results

Computing a particle array returns `kr.ChunkedArray` by default:

```python
chunks = valid_mass.compute()

print(len(chunks), chunks.nbytes)
for values in chunks:
    process(values)
```

`iter_chunks()` is a convenient shorthand when you only need the arrays:

```python
for values in valid_mass.iter_chunks():
    process(values)
```

When a library requires one contiguous array, request gathering explicitly and
set a memory budget:

```python
values = valid_mass.compute(
    gather=True,
    max_bytes=512 * 1024 * 1024,
)
```

Kangaroo raises `MemoryError` before concatenation if the chunks exceed
`max_bytes`. A `ParticleMask` follows the same chunked and gathered
materialization pattern, with boolean arrays as its output.

## Cloud-in-cell mass projections

Call `project()` on a particle species to deposit its particle masses onto a
regular two-dimensional plane:

```python
stellar_mass = stars.project(
    axis="z",
    bounds=(-2.0e21, 2.0e21),
    resolution=(1024, 1024),
    zoom=1.5,
    mass_max=1.0e36,
)

image = stellar_mass.compute()
```

Parameters:

- `axis`: projection direction as `"x"`, `"y"`, `"z"`, `0`, `1`, or `2`;
- `bounds`: optional physical lower and upper limits along that axis;
- `resolution`: output `(width, height)` or `"width,height"`;
- `zoom`: zoom around the center of the default in-plane view;
- `mass_max`: optional upper mass threshold for particles included in the
  projection.

The result is a regular `float64` `kr.Array`, so `compute()` returns a NumPy
array. It can participate in regular-array arithmetic, comparisons, casts, and
histograms. Combining it with another bounded mesh result requires both arrays
to describe the same physical plane.

## Sharing one execution

Use `kr.compute()` to calculate compatible particle results together:

```python
valid = mass.isfinite() & (mass > 0.0)
selected = mass[valid]

total, histogram, count = kr.compute(
    selected.sum(),
    selected.histogram(bins=64, range=(0.0, 1.0e36)),
    valid.count(),
    progress=True,
)
```

All values in one call must belong to the same dataset view. Kangaroo shares
the particle graph work used by those results.
