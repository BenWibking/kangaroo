# Offline total-potential reconstruction: design

This first representation fits **the saved total `gpot` once**. It is not a
halo/disk decomposition. Arora et al. (2024),
https://arxiv.org/abs/2407.12932, and Vasiliev (2019),
https://arxiv.org/abs/1802.08239, motivate angular harmonics and smooth spatial
coefficients; their component-separated models are not reproduced here.

## Estimator and domain (written before implementation)

Use equally spaced azimuthal quadrature on a configurable cylindrical grid in a
fixed, explicitly supplied frame. Keep signed z and both sine and cosine modes.
All modes reuse each potential sample. Discrete Fourier normalization is 1/N for
m=0 and 2/N otherwise; require N >= 4*mmax+4, and assess angular aliasing by
increasing N. mmax=4 is a starting choice, not an accuracy guarantee.

At each quadrature position reconstruct the cell-centered saved scalar by a
local quadratic least-squares polynomial in Cartesian coordinates. The support
is an axis-aligned cube in simulation coordinates with user-selected half-width
h. For each uncovered valid AMR cell intersecting it, weight by the exact volume
of intersection divided by the cube volume. Evaluate the polynomial basis at the
cell center using coordinates scaled by h. Accumulate the Gram matrix, right-hand
side, finite/bad volume, and sample count in C++, then tree-reduce compact
statistics. Exclude all cells covered by finer levels, including nonadjacent
levels. No ghost values, particle collection, or full mesh enters Python.

This is a volume-weighted moving least-squares estimator, not an interpolating
AMR stencil and not a broad-bin harmonic volume average. Refinement changes its
quadrature error, not its continuum weight. It exactly reproduces quadratic
cell-center fields when full rank. Reject incomplete cube coverage, nonfinite
input, fewer than ten cells, and ill-conditioned scaled Gram matrices. There is
no fallback to covered coarse data. The support must fit inside available data;
the fitted cylinder therefore needs a surrounding data margin. Decrease h with
mesh spacing to test convergence; a fixed h retains a smoothing bias. Statistics
scale with requested samples, not mesh size; this initial implementation is not
claimed to be production-performance tuned.

## Smooth representation and boundaries

Write the scalar as C0(s,z) + sum [Re(w^m) Cm(s,z) + Im(w^m) Dm(s,z)],
w=(x+iy)/Rmax, s=|w|^2. Away from the axis this is the requested Fourier expansion
with Am=(R/Rmax)^m Cm and Bm=(R/Rmax)^m Dm. Tensor cubic splines of s and signed z
provide smooth coefficients. Regularity is explicit: m=0 is even in R and each
higher mode vanishes as R^m. The normalized higher-mode values at s=0 are
extrapolated from the first four positive-radius samples, not divided by zero.
This assumes a smooth Cartesian potential; noisy high modes near the axis can
be amplified by R^-m and must be checked by force convergence. Singular point
potentials are outside this regular representation's scope.

Use not-a-knot cubic end conditions, with no exterior physical boundary
condition implied. The cylinder can cover disk and halo, but wide dynamical
ranges need appropriately spaced knots and force tests; spherical expansions
may be more economical for a distant halo. Evaluation outside the closed fitted
cylinder raises an error. No silent extrapolation, symmetry folding, asymptotic
halo extension, or masked holes. Acceleration is the analytic negative Cartesian
gradient of the same scalar spline, including its axis limit and frame rotation.
Subtract the axisymmetric value at (R=0,z=zmin) before fitting and store that
additive offset. Restoring it affects potential only.

## Validation and time

Test analytic forces as well as potential, spatial convergence, sine/cosine
modes, odd z, finite-difference derivatives, axis behavior, coverage failures,
AMR layout/refinement equivalence, serialization, and runtime-free imports.
Native extraction tests must use the dedicated pixi-hpx toolchain. Test distributed
reductions where localities are available; distinguish that from multi-node proof.
Production validation must compare held-out forces with an independent derivative
of saved AMR gpot, stratified by radius, signed height, and refinement interfaces.

Each snapshot is independent. Read its time from Header metadata, never its name
or presumed cadence. Store units explicitly, fixed frame, actual level geometry,
source Header hash/path, and user phase tags. Preserve the Phase 1 / Phase 2
resolution transition in the supplied campaign manifest. Temporal interpolation,
moving-frame inertial forces, orbit integration, and stream generation are deferred.

The [stream sensitivity pilot](potential_stream_pilot.md) specifies a future
0.5 Gyr experiment to measure preserved widths, gaps and heating at fixed force
accuracy. It is a protocol, not an executed stream-validation result.

## Implementation and usage

`analysis.ops.PotentialSamples` / `Pipeline.potential_samples` lower a typed
`PotentialSamplesParams` kernel through the existing block I/O, covered-box
sharing, block reduction and cross-level reduction machinery. The C++ kernel is
`cpp/src/kernels/potential_kernels.cpp`. `analysis.potential_extract.extract_samples`
batches requests (128 unique positions by default) and solves the reduced
statistics. Duplicate axis positions are extracted once. All Fourier modes are
projected together from the same samples in `potential_replay/fit.py`; neither
field I/O nor an extraction graph is repeated per mode. No density decomposition
or particle access is involved.

For each sample q, define u=(cell_center-q)/h and
b=(1,ux,uy,uz,ux²,ux*uy,ux*uz,uy²,uy*uz,uz²). The estimator minimizes
sum_cells w [b.c - gpot_cell]², where w is the exact intersected cell volume
fraction of the support cube. Accumulated statistics are G=sum w b bᵀ and
f=sum w b gpot. Python equilibrates G by its diagonal, rejects eigenvalue
condition ratios above the configured limit (default 1e10), and solves for c.
The sample is c[0]. This Gram formulation squares the design matrix condition
number; the bound refers to the equilibrated **Gram** matrix. No regularization
or silently discarded small eigenvalues are used. Coverage must equal one to
absolute tolerance 1e-8, bad volume must be zero, and at least ten contributing
finite cells are required. A support narrower than a few cells is usually rank
deficient. A larger support smooths over more structure: it is an independent
numerical choice, not a consequence of increasing the spline knot count.

Build/install with the repository's `pixi-hpx` commands. For one plotfile in CGS:

```sh
pixi run -e pixi-hpx python scripts/plotfile_potential.py /path/to/plt0000123 \
  --output-dir models --field gpot --center 0 0 0 \
  --rmax 9.257032744474102e22 \
  --zmin=-3.702813097789641e22 --zmax=3.702813097789641e22 \
  --nr 33 --nz 33 --nphi 32 --mmax 4 --support 9.257032744474102e21 \
  --length-to-cm 1 --time-to-s 1 --potential-to-cm2-s2 1
```

This example means R≤30 kpc, -12≤z≤12 kpc, and h=3 kpc. These are example
choices, **not** a recommended accuracy setting for the production run. Units
are explicitly required because plotfiles do not establish physical units.
`--center` is required, orientation defaults explicitly to the identity; use
`--orientation` with nine row-major entries of a proper orthonormal matrix to
select a different fixed orientation. Its columns are model basis vectors in
simulation/world coordinates: world=center+Q*local. Use ordinary decimal notation
for negative entries in multi-value options (argparse can treat negative
scientific notation as an option). Single-value negative bounds accept `--zmin=-1e22`.

Use multiple positional paths, or `--manifest inputs.csv` / `--manifest inputs.json`.
CSV needs a `plotfile` column and may include `phase` and other provenance columns.
The supplied `combined_disk_mass_timeseries.csv` can be passed directly when its
Frontier paths are accessible. JSON accepts a list of paths or objects with
`plotfile`, or an object with a `snapshots` list. Relative paths resolve against
the input manifest's directory. Time columns in input manifests are retained
only as provenance. Actual Header times determine the sorted output time series;
duplicate times or paths are rejected. Different field selections, frames, or
unit systems require separate invocations.

For nonuniform spatial sampling use `--knots knots.json` instead of domain bounds:

```json
{"radii": [0, 1, 2, 4, 8, 16], "heights": [-8, -4, -1, 0, 2, 8]}
```

These numbers are in **native length units**. At least five radial knots
(including zero) and four signed-height knots are required. `--nr` / `--nz`
otherwise generate uniformly spaced native R / z knots. No reflection symmetry
is assumed. Sampling half-width remains in simulation Cartesian coordinates even
when the model frame is rotated. The current plotfile metadata adapter assumes
isotropic Cartesian 3D cells; the CLI explicitly rejects anisotropic or
non-Cartesian files rather than applying incorrect geometry. The kernel itself
uses all three spacings.

Output names include input sequence index to avoid basename collisions. Existing
outputs are never overwritten. All inputs/metadata are preflighted; if a later
snapshot fails sampling, already completed models remain inspectable but no
completed time-series manifest is written. Use a new output directory for reruns.
A successfully written `timeseries.json` marks completion. `--hpx-config` and
`--hpx-arg` forward explicit HPX settings; distributed execution otherwise follows
the existing launcher/runtime. The script runs on the console locality.

## Portable version 1 schema

Copy the `potential_replay/` directory to an independent Python environment with
NumPy. It has no Kangaroo, HPX, AMReX, SciPy, or plotfile dependency:

```python
from potential_replay import PotentialModel
model = PotentialModel.load("snapshot_0000_plt0000123.npz")
phi, acceleration = model.evaluate([[1e22, 0, 0]])
original_gauge_phi = model.potential([[1e22, 0, 0]], restore_offset=True)
```

NPZ is a ZIP of standard NPY arrays, loaded with `allow_pickle=False`. It contains
exactly these members:

| Member | Type / shape | Meaning |
|---|---|---|
| `s_knots` | float64 `(nr,)` | increasing (R/Rmax)² from 0 to 1 |
| `z_knots` | float64 `(nz,)` | increasing signed native height |
| `coefficients` | float64 `(1+2*mmax,nr-1,nz-1,4,4)` | ascending bicubic power coefficients |
| `metadata` | scalar Unicode JSON | version, frame, time, units, gauge, provenance |

For interval (i,j), u=(s-s[i])/(s[i+1]-s[i]) and
v=(z-z[j])/(z[j+1]-z[j]), coefficient function k is
sum_(a,b=0..3) coefficients[k,i,j,a,b] u^a v^b. Channel order is
C0,C1,D1,C2,D2,... with Phi=C0+sum Re(w^m)Cm+Im(w^m)Dm and
w=(local_x+i*local_y)/Rmax. Cm and Dm are regularized coefficients, **not** Am
and Bm themselves. The basis and normalization strings in metadata are explicit.

Differentiate these polynomials in u,v with chain factors 1/ds and 1/dz.
For the planar harmonic factor d(w^m)/dx=m*w^(m-1)/Rmax and
d(w^m)/dy=i*m*w^(m-1)/Rmax; ds/dx=2x/Rmax² and ds/dy=2y/Rmax².
These expressions have finite axis limits without dividing by R. Acceleration
is minus this gradient, rotated back with Q. The evaluator returns native
potential units and native potential/length acceleration units, respectively.
Multiplication by `potential_to_cm2_s2 / length_to_cm` converts acceleration to
cm/s². It does not perform conversions implicitly.

Required metadata includes `schema="kangaroo.total-potential"`, `version=1`,
`basis`, `mmax`, `nphi`, `rmax`, `time`, `units` (`length_to_cm`, `time_to_s`,
`potential_to_cm2_s2`), `center`, `world_from_model`, `frame="fixed"`,
`potential_offset`, `gauge`, `normalization`, `spline`, `axis`, and
`outside="error"`. Unknown schema/version/basis/boundary policy is rejected.
`provenance` contains source path, field, Header SHA256, original input record
(including phase), geometry (spacings, refinement ratios, domain origins/bounds,
box counts and layout hash), support/conditioning/coverage choices, batch size,
and diagnostic extrema. Full block lists are hashed rather than duplicated into
every compact model; extraction still uses all original valid boxes.

The native statistics layout is `(N,114)` float64: columns 0–99 are row-major G,
100–109 are f, 110 coverage fraction (including bad values), 111 nonfinite-value
volume fraction, 112 finite contributing cell count, 113 volume-weighted maximum
cell width divided by h. The extraction API returns per-sample coverage,
conditioning, count, and resolution diagnostics; the CLI stores their extrema.

`timeseries.json` uses schema `kangaroo.potential-timeseries`, version 1. It stores
sorted snapshots with relative model filenames, actual native `time`, converted
`time_seconds`, phase, source path, spacings and Header hash, plus the common
units and fixed frame. No constant cadence, interpolation rule, or continuity
across a resolution change is implied. A future moving-frame consumer must use
each model's center/orientation and the frame's time derivatives when adding
non-inertial accelerations. This implementation supplies gravitational forces
in world coordinates and does not infer those derivatives.

## Force diagnostics

`scripts/validate_potential.py` takes a saved model, matching source plotfile,
and an NPY array of held-out world points. It checks the source Header hash and
uses a separate, bounded-cache PlotfileReader implementation to differentiate
cell-center trilinear interpolation. Missing fine stencil centers are prolonged
from coarser levels. Stencils with changing refinement level are flagged. This
reference has its own discretization and interface errors; it is not a replay
of Quokka's exact gravity solver. It never gathers all FABs or particles.

```sh
pixi run -e pixi-hpx python scripts/validate_potential.py model.npz /path/to/plt \
  --points heldout.npy --radial-edges 0,1.5e22,4.6e22,9.3e22 \
  --height-edges=-3.7e22,-1.5e22,0,1.5e22,3.7e22 --output force-errors.json
```

Reports include absolute RMS force error, median and 95th-percentile vector
relative error ||a_model-a_ref||/||a_ref||, counts, signed-height/radius bins,
and refinement-interface versus interior subsets. Zero reference forces are
excluded from relative summaries and counted explicitly. These are spatial
comparisons only. Increase knots and azimuthal sampling separately, vary support
and mesh resolution, and assess reference convergence before selecting a
production tolerance. Potential residuals alone cannot validate forces.

## Validation recorded on 2026-11-01

Synthetic tests cover quadratic MLS exactness, asymmetric signed height, m=1..4
sine and cosine forces, nonzero axis force, rotated/translated frames, spline
joins and finite-difference derivatives, missing coverage/nonfinite/rank failures,
nonzero mesh index origins, split blocks, coarse/fine layouts with poisoned covered
coarse values, batching, portable serialization, and imports with runtime modules
explicitly forbidden. CLI tests use real synthetic plotfiles with irregular Header
times, deliberately wrong CSV times, reversed filename/time ordering, and different
resolution across phases.

For a softened monopole plus m=1..4 perturbations, 500 fixed held-out points gave:

| R/z knots | maximum potential error | maximum vector force error |
|---|---:|---:|
| 9/9 | 1.404e-4 | 1.341e-3 |
| 17/17 | 5.599e-6 | 1.380e-4 |
| 33/33 | 3.267e-7 | 1.481e-5 |

These are dimensionless analytic-sample spline errors. End-to-end native sampling
of a softened monopole plus xy and odd-z perturbations on 12³, 24³ and 48³ meshes,
with h=2*dx and a fixed 13×13×20 sample grid, gave RMS absolute force errors
0.13756, 0.019982 and 0.0021402. This separate test includes extraction error.
Exact quadratic end-to-end forces agree within the 2e-10 test tolerance.

A **different, local example**, not the requested campaign, was available:
`/Users/benwibking/amrex_codes/amrex-visit-plugin/example_data/DiskGalaxy/plt0000020`.
Its actual Header time is 77667901972768.156 s. A 17×17×32 fit with mmax=4,
Rmax=30 kpc, z∈[-12,12] kpc, h=3 kpc, center zero, identity orientation and CGS
units produced a 281,482-byte model. On 1,080 held-out points its median relative
vector force error was **2.979%**, and its 95th percentile was **7.937%**; RMS
absolute error was 6.345e-10 cm/s². Of these, 548 reference stencils touched a
refinement transition (median 2.982%, p95 7.231%); 532 interior stencils had
median 2.970%, p95 8.501%. The reference is the separate trilinear derivative
above, so these discrepancies combine reconstruction and reference errors.
No claim of reproducing the simulation force to these tolerances is made.

| R (kpc) | signed z (kpc) | N | median force error (%) | p95 (%) |
|---|---|---:|---:|---:|
| 0–5 | -12 to -5 | 13 | 3.843 | 6.212 |
| 0–5 | -5 to 0 | 6 | 5.221 | 7.205 |
| 0–5 | 0 to 5 | 12 | 4.513 | 17.059 |
| 0–5 | 5 to 12 | 9 | 2.958 | 4.016 |
| 5–15 | -12 to -5 | 101 | 3.061 | 7.398 |
| 5–15 | -5 to 0 | 84 | 3.450 | 9.915 |
| 5–15 | 0 to 5 | 101 | 3.455 | 8.633 |
| 5–15 | 5 to 12 | 95 | 2.724 | 5.969 |
| 15–30 | -12 to -5 | 129 | 2.730 | 6.074 |
| 15–30 | -5 to 0 | 191 | 2.762 | 7.807 |
| 15–30 | 0 to 5 | 193 | 3.122 | 9.526 |
| 15–30 | 5 to 12 | 146 | 2.599 | 5.903 |

The machine-readable [force report](validation/potential_diskgalaxy_force_errors.json)
includes source Header hash and paths. The generated model and held-out points
are in `/private/tmp/kangaroo-potential-real/`. Points use RNG seed 1701: 600
uniform-area random cylinder points, 320 near z=±4.6875/±9.375 kpc interfaces,
and 160 near x=±18.75 kpc interfaces. Sparse central bins are not strong accuracy
bounds. This smoke test does not establish halo accuracy outside 30 kpc or spatial
convergence on real data.

The requested campaign CSV contains 56 selected snapshots, spanning 0 to
550.5740565445049 Myr in the table. None of its 56 plotfile Headers is accessible
locally at the recorded paths. The table transitions from Phase 1 at
459.94996222814274 Myr to Phase 2 at 470.18640623597094 Myr. The input configurations
use base grids 256³ and 512³ respectively (both max level 8), request `gpot`, and
set a 600 Myr stop time. These configuration/table values are provenance, not
substitutes for missing actual plotfile metadata.

Remaining production checks are: extract those full plotfiles and verify their
actual times/geometry/units; converge support, knots, and nphi/mmax over the chosen
disk **and halo** domain; compare held-out forces by signed height/radius and every
refinement transition using a converged independent reference; compare both sides
of the phase resolution change; and run serial/multiple-locality agreement and
memory/performance tests with the appropriate cluster launcher. Temporal cadence
error must be studied separately by a future time-dependent consumer. This task
provides neither that consumer nor a temporal error estimate.

Native validation used a fresh `.pixi/build-potential-sdk26` tree. The default
macOS 27 SDK could not be read by the installed Pixi linker; configuration selected
`-DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`.
C/C++ compilers, HPX, Boost, HDF5 and hwloc were checked to resolve inside
`.pixi/envs/pixi-hpx`; linked libraries use that environment's rpath. No Homebrew
or Spack libraries were used. HPX reports `HPX_WITH_NETWORKING=OFF`, so local
block/tree reductions were executed but multiple-locality and Frontier execution
were not tested.

Validation commands and outcomes:

```sh
KANGAROO_BUILD_DIR=.pixi/build-potential-sdk26 pixi run -e pixi-hpx test
# 342 passed, 1 xfailed (39.67 s); includes configure, build, install and full pytest

pixi run -e pixi-hpx pytest -q tests/test_potential_model.py \
  tests/test_potential_extraction.py tests/test_kernel_catalog.py \
  tests/test_python_packaging.py
# Focused numerical, native, CLI, standalone and packaging checks
```

The existing expected failure remains expected. `ruff check` on all new Python
modules/tests and `git diff --check` pass. Multi-node performance and temporal
accuracy remain unmeasured.
