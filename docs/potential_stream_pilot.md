# Pilot: stream features preserved by approximate force replay

## Question and status

For a fixed reconstruction, what stream widths, gap lengths and heating signals
remain accurate after **0.5 Gyr**, and on which orbits? Measure the limits rather
than interpreting a force percentile as a stream-resolution guarantee. Retain
track-position accuracy as a separate gate.

This is an experimental protocol, not a measured sensitivity result. Kangaroo
currently supplies extraction, snapshot models and spatial force validation.
Temporal interpolation, orbit integration and stream generation remain outside
the implemented pipeline. No stream integrations have been run for this protocol.

The [isolated MW run scope](isolated_mw_stream_pilot.md) evaluates a 64-node,
4–5 Gyr live-gas rerun and proposes disk/halo resolution qualification for
GD-1-like orbital radii. It does not establish a measured stream-resolution limit.

## What the Arora benchmark establishes

[Arora et al. (2024), sections II.3 and V](https://arxiv.org/html/2407.12932v2)
use angular orders four and approximately 25 Myr snapshot spacing. Their force
diagnostics include signed fractional errors in individual Cartesian components
and in force magnitude. The headline 95%-within-4% result is not the same
statistic as Kangaroo's vector-norm relative error. Their roughly 2,000-particle,
10^7-solar-mass dwarf stream example does not measure minimum preserved cold
stream widths, gap lengths or heating amplitudes.

Actual Arora TEMP models and independent reference data would test that method
and galaxy; their availability has not been established here. Kangaroo models
of total Quokka `gpot` with comparable force errors instead test **Arora-level
accuracy in this reconstruction**. They do not reproduce the component-separated
Arora representation.

Use mmax=4 as the baseline. Measure component, magnitude and vector residuals on
both a declared volume sample and stream-occupied regions. Record bias, tails,
spatial and temporal correlations. Mask and report near-zero denominators for
component ratios, retaining absolute errors there. Do not tune on validation
points or add independent 4% force noise to emulate the paper: identical
marginal errors can have very different tidal effects.

The current local smoke fit has vector-error p95=7.937%, median=2.979% against
a trilinear AMR reference. It is neither an Arora-equivalent calibration nor a
converged reference. Its 6 kpc-wide sampling support is an independent smoothing
choice, not a requirement of angular order four.

## Paired experiments

Use identical initial phase-space samples and release histories in reference and
replay, with shared random seeds. Begin with already-unbound tracer streams to
isolate host-force replay. Bound progenitor dynamics require a separate later
stream-formation test. Use an independently validated downstream integration
consumer; this protocol does not add an integrator to Kangaroo.

1. **Smooth controls:** evolve without an injected encounter. Measure track
   drift, width bias, false gaps and artificial heating/cooling.
2. **Feature survival:** put identical density deficits or velocity perturbations
   into both initial streams. Compare with the evolving reference feature, not
   its initial amplitude or length. This tests transport of an existing feature,
   not reconstruction of its generating force. A feature that disappears in the
   reference cannot establish a preservation threshold.
3. **Feature formation:** add a resolved moving perturber to a controlled
   reference scalar potential, sample the total potential and reconstruct it.
   Compare perturber-on/off pairs in both reference and replay. The perturbation
   must pass through the same extraction, harmonic and temporal approximations
   as the background. Adding an exact perturber only at evaluation would bypass
   the relevant failure mode. Use the total potential once, without double counting.

Begin with analytic time-dependent potentials, including non-axisymmetry and
signed-z asymmetry, to obtain a controlled reference. Follow with Quokka data
when a sufficiently long series is accessible. A finer fit alone is not proof
of reference convergence: use independent spatial derivatives and finer temporal
sampling where possible. Distinguish agreement with saved AMR data from agreement
with simulation dynamics.

## Initial sweep: proposed search ranges, not sensitivity claims

| Variable | Initial values |
| --- | --- |
| Initial transverse RMS width | 10, 30, 100, 300, 1000 pc |
| Target reference gap length | 30, 100, 300, 1000, 3000 pc |
| Reference fractional gap depth | 0.1, 0.3, 0.5 |
| Heating amplitude sqrt(H), H an added one-component velocity variance | 0.03, 0.1, 0.3, 1, 3 km/s |
| Evolution duration | 0.5 Gyr; diagnostics also at 0.1 and 0.25 Gyr |

First scan widths on smooth controls, then gap/heating scales at representative
passing and marginal widths. This is not a full Cartesian product. Extend ranges
or bisect near pass/fail transitions. Use at least 32 independent encounter or
orbital-phase realizations near each reported transition and publish confidence
intervals. Shared tracer seeds reduce noise but do not replace convergence in
tracer count.

Cover nearly circular halo, eccentric halo and inclined disk-crossing orbits,
with guiding/apocentric scales around 5, 15 and 30 kpc where the fitted domain
permits. Enlarge the domain to contain the entire stream and force support;
do not extrapolate the current 30 kpc smoke model. Stratify by pericenter,
inclination, encounter phase, signed height and AMR transitions. Orbit families
are separate strata, not interchangeable realizations of one population.
Preserve Phase 1/Phase 2 provenance and diagnostics on each side of the change.

## Operational meaning of preserved

Use fixed physical units and shared reference-stream coordinates. Measure local
cross-sectional centroids and both transverse widths about each stream's own
centroid, resolving principal directions. Keep along-track phase errors separately;
do not align curves to erase track drift. Fix reference-derived binning and
feature definitions before inspecting replay. Converge bins, tracer counts and
diagnostic output cadence.

Use a provisional **10% fidelity criterion**, retaining raw errors so 5% and
20% thresholds can also be evaluated:

* Width: |w_replay - w_reference| / w_reference <= 0.1 in both transverse
  directions. Publish initial and evolved reference widths.
* Track: transverse centroid displacement <= 0.1 times local reference width.
  Report absolute displacement and along-track phase errors too. Publish
  width-only and joint track-plus-width limits separately.
* Gap: depth and length errors <=10% of reference values; gap-center displacement
  <=10% of reference length. Define depth relative to the matched no-gap or
  no-encounter control and length as the full width at half maximum of the
  density deficit for isolated gaps. Report merged/complex gaps separately.
  Also report residual density power versus wavelength. Gap length is not the
  length scale of the perturbing force.
* Heating: H = sigma_v,on^2 - sigma_v,off^2, after subtracting local mean
  streaming velocity. Require |H_replay - H_reference| <=0.1 H_reference for
  positive reference heating. Report each velocity component, signed variance
  differences and the control artificial-heating/cooling floor. sqrt(H) is a
  reporting amplitude, not the change in velocity dispersion.

Require an observed pass fraction >=90% in each orbit stratum; publish binomial
confidence intervals. Thirty-two trials do not establish a precise population
guarantee. For track and width, use the 95th percentile over fixed stream-length
and time sampling for the criterion; also report maxima and localized failures.
Test gap/heating at each diagnostic epoch where the reference signal exists.

Run the same gap search on controls, with a provisional <=5% probability of any
target-sized false gap per tested stream. Report its uncertainty and searched
length. For weak signals, increase the ensemble until sampling/reference
uncertainty is below one third of the 10% tolerance, or label the case
inconclusive rather than preserved or destroyed.

## Separate error sources

Hold the baseline reconstruction fixed while scanning stream scales. Then vary
support, knots, azimuthal sampling and harmonic order separately to identify the
controlling error. Measure differential force residuals between nearby stream
stars and correlations along reference trajectories as explanatory diagnostics.
They do not substitute for paired stream measurements.

For analytic cases, start cadence refinement at 25, 12.5, 6.25 and 3.125 Myr;
refine further relative to encounter duration as needed. Record the temporal
interpolation rule. Converge integration steps and reference sampling separately.
For Quokka use actual Header times. Interpolating existing snapshots does not
create a finer temporal reference. Downsampling can demonstrate degradation but
cannot prove the finest available cadence adequate.

A stream can be narrower than an AMR cell in a smooth, resolved tidal field:
cell width is not automatically a minimum tracer-stream width. Conversely,
unresolved perturbers or rapid encounters cannot be recovered by increasing
tracer count or spline resolution. Record force resolution and encounter
timescales, and distinguish these limits.

## Deliverables and execution dependencies

Publish response curves and pass/fail/inconclusive tables conditional on orbit,
gap depth, width, cadence and encounter history. Where a monotonic transition is
bracketed, report the narrowest passing width, shortest passing gap at each depth
and weakest passing heating with brackets and uncertainty. Otherwise report
disconnected passing ranges. If all scales pass or fail, report a bound and
extend the sweep. Keep survival and formation limits separate.

Archive model hashes, force samples, reference definition, initial conditions,
seeds, actual snapshot times, units/frame, AMR phase, integration settings,
tracer counts, binning and observable errors.

Execution needs a validated downstream consumer and analytic test streams or
accessible trajectories/time series. The requested campaign's plotfile paths
were unavailable during local implementation validation; the single local
example snapshot cannot establish 0.5 Gyr sensitivity. The next empirical result
must be labelled a controlled synthetic pilot until production inputs and their
reference checks are available.

Scale-dependent density/track diagnostics are motivated by
[Bovy, Erkal & Sanders (2017)](https://arxiv.org/abs/1606.03470). Numerical search
ranges, fidelity cutoffs and pass fractions above are pilot design choices.
