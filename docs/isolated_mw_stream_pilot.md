# Scoping a 4–5 Gyr isolated Milky Way force-replay pilot

Scope: 64 Frontier nodes, live star-forming magnetized gas disk, streams reaching
at least GD-1-like Galactocentric radii. This document proposes a convergence
ladder and estimates cost; it does not certify a coarsest acceptable resolution
or report a Frontier benchmark. Keep the experiment distinct from reproducing
cosmological accretion, satellite populations, or the exact FIRE galaxy.

## Recommendation

The study is plausible on this allocation. Start from the existing HI particle
ICs and a 1.2 Mpc box. Add radius-dependent halo refinement, and compare disk
resolutions 18.31, 36.62 and 73.24 pc in short runs before selecting a long run.
The 36.62 pc model is the first cost-saving candidate; 73.24 pc is an aggressive
screening case requiring explicit disk-physics validation. The existing setup's
only documented permitted lower limit is 18 pc. Do not label a coarser unchanged
input deck safe. Keep the live gas, cooling, MHD, star formation and feedback.

Use a target orbital region to r=50 kpc with refinement/fit margins. GD-1 is
roughly 8 kpc from the Sun, but this is not the relevant halo radius: a published
orbit fit has Galactocentric pericenter 14.2 kpc, apocenter 27.0 kpc and inclination
about 40 degrees ([Li et al. 2018](https://arxiv.org/abs/1811.06427)). These are
planning scales, not a uniquely determined orbit in our different galaxy.

## Verified existing configuration

Read-only inspection used `incite-2026/inputs/MW_Gen1/noCGM_1uG/Phase1.in`,
`Phase2.in`, their README, the parent README and `pull_container.sh`. The parent
README pins container commit `16e5b96f3f6bb43cf8ea835eda91d8f2e4346aac`.
Current Quokka source at `d080ce1528f608261449add739bce8888302efb0` was also
inspected; it has unrelated local work, which was not changed. Key refinement,
gravity-supercycling and Jeans-creation behavior was checked in the pinned source.
A new run must record its exact executable/container and any intentional changes.

* Box: [-600,600] kpc cubed, nonperiodic; base grid 256 cubed in Phase 1 and
  512 cubed in Phase 2, with eight factor-two refinements. Finest spacings are
  18.3105 and 9.1553 pc, respectively.
* Static tagging refines the **same cylinder R<16 kpc, |z|<4 kpc on every level**.
  There is no independently specified spherical halo ladder. Nesting adds some
  surrounding coverage; actual box extents require a generated hierarchy.
* `do_subcycle=0`, `regrid_interval=-1`, CFL=0.3, gravity solved every ten steps.
  Coarsening the finest grid therefore changes the timestep of the whole hierarchy.
* Gas disk mass 8.59322e9 solar masses, radial scale 3.43218 kpc, initial vertical
  scale 0.343218 kpc, initial temperature 10^4 K, initial field amplitude 1 microgauss.
  Cooling reaches a 10 K floor; eps_ff=0.3; low-mass composite star cap=299 solar
  masses. Early-feedback parameters are 377 km/s, 3.3 Myr, alpha=1 and a
  3000 km/s velocity cap; terminal SN momentum is 5e5 solar-mass km/s.
* The gas atmosphere is tenuous rather than absent. The setup README explicitly
  says: "Any resolution lower than Phase1.in (256^3 base grid, dx=18pc on the
  finest level) should NOT be used. Bad things will happen." It supplies no
  diagnosis of that warning; the scope therefore requires new validation.
* Streaming the complete compressed HI particle archive, checking gzip completion
  and matching its header count, found 21,249,982 particles: 9,999,984 of mass
  125,477 solar masses and 11,249,998 of mass 3,438 solar masses. These are the
  halo and stellar populations inferred from the AGORA setup; the ASCII records
  themselves provide mass/phase space rather than separate species labels.
  No particle population was gathered into memory. Keep this mass resolution.

## Halo resolution as a function of radius

For a smooth background, propose a three-point ladder

    dx_halo <= q r,  q = 0.01 (fine), 0.02 (fiducial), 0.04 (coarse).

This is a starting convergence bracket, not an empirically established bound.
For fixed angular order, the angular wavelength grows with radius, making a
constant dx/r efficient. As an illustration only, a centered derivative of a
resolved sinusoid with local wavenumber k=m/r attenuates its gradient by
sin(k dx)/(k dx). For m=4 the gradient errors at these q values are 0.027%,
0.107%, and 0.426%. This calculation assumes exact nodal potentials: it excludes
Poisson discretization, particle sampling, grid interpolation, AMR boundaries,
temporal errors and the other spatial modes. It is not a prediction of Quokka
force accuracy or a bound on accumulated stream errors over 5 Gyr.

Round spacing downward to an available factor-two level. For the 256-cubed base
and 1.2 Mpc box, outside regions with a finer disk/perturber requirement:

| r (kpc) | Fine q=.01 (pc) | Fiducial q=.02 (pc) | Coarse q=.04 (pc) |
| --- | ---: | ---: | ---: |
| 5 | 36.6 | 73.2 | 146.5 |
| 10 | 73.2 | 146.5 | 293.0 |
| 15 | 146.5 | 293.0 | 585.9 |
| 20 | 146.5 | 293.0 | 585.9 |
| 30 | 293.0 | 585.9 | 1171.9 |
| 50 | 293.0 | 585.9 | 1171.9 |
| 100 (optional extension) | 585.9 | 1171.9 | 2343.8 |

Apply the cap at the inner edge of each cell/patch, with proper nesting and a
central disk/bulge resolution cap. The exact hierarchy will not be identical to
this pointwise table because of block alignment. GD-1-like radii need roughly
0.3–0.6 kpc fiducial halo cells; the 0.6–1.2 kpc case is the coarsest **candidate
to test**, not a recommendation already shown to preserve narrow streams.

The local requirement is more generally

    dx <= min(q r, disk requirement, L_encounter / N),  N initially 4–8.

For a softened encounter, a useful force-variation scale is
L_encounter ~ sqrt(b^2 + a_perturber^2), with impact parameter b and perturber
scale a_perturber. The 4–8-cell requirement is itself to be converged. An
encounter varying over 0.1 kpc would require approximately 12.5–25 pc locally;
the proposed background halo mesh cannot resolve it. Gap length is not the same
as this encounter scale. Conversely, tracer-stream width need not exceed cell
size if the relevant tidal field is smooth. No radius-only rule can guarantee
small-gap or weak-heating fidelity.

The present cylindrical tagging function needs a level-aware spherical rule
combined with disk refinement. These are proposed changes, not existing input
parameters. Begin with static nested meshes; moving refinement would add an
independent source of time-dependent force error and is incompatible with the
current Poisson supercycling path when the interval exceeds one.

## Disk resolution and cost

Keep the existing R<16 kpc, |z|<4 kpc fine cylinder in the first comparisons so
resolution is the changed variable. Its ideal volume is pi*16^2*8 kpc cubed.
Ignoring AMR block padding, ghosts and coarser grids:

| Finest dx (pc) | max_level on base 256 | Ideal finest cells | Disk cell-update work for 5 Gyr / Phase-1 0.6 Gyr |
| --- | ---: | ---: | ---: |
| 18.31 | 8 | 1.048 billion | 8.33 |
| 36.62 | 7 | 131.0 million | 0.521 |
| 73.24 | 6 | 16.38 million | 0.0326 |
| 146.48 | 5 | 2.05 million | 0.00203 |

The last row is cost context, not a proposed live-disk production setting.
The estimates use work proportional to duration*dx^-4 at fixed refined volume
and fixed limiting signal speed. They are NOT runtime forecasts: particles,
new star creation, global Poisson solves, base-grid work, extra halo refinement,
communication, memory and output will limit these ideal savings. They do show
why trying 36.6 pc is worthwhile before changing the physical model drastically.

The initial gas scale height has about 19, 9.4, 4.7 and 2.3 cells at these
resolutions. The cold disk can become thinner than its initial scale height.
Thus 73 pc may be useful for global morphology, but cannot be presumed to retain
the same clouds, vertical tides or feedback coupling.

In the inspected creation model Lambda_J=cs/sqrt(G rho), and eligibility is
Lambda_J<0.5 dx. At fixed sound speed, changing dx by two lowers the corresponding
density threshold by four and raises the mass at threshold proportional to dx.
Cold gas cooling, stochastic birth masses and feedback are consequently affected
even if every input parameter is unchanged. Diagnose the README warning first;
compare 18/36/73 pc for gas surface density, scale height, SFR, cloud/clump masses,
birth-mass distribution, bar/spiral amplitudes, rotation curve and tidal field.
Do not silently retune eps_ff to hide disagreement. Any pressure floor or revised
subgrid prescription defines a new model and needs a separate documented comparison.

The AGORA isolated-disk comparison used 80 pc with a Jeans pressure floor and a
different star-formation/feedback prescription over 500 Myr; it establishes that
coarse live disks are possible, not that this 10 K Quokka setup is validated at
80 pc or for 5 Gyr ([Kim et al. 2016](https://arxiv.org/abs/1610.03066)).

If 36 pc fails, a second cost-saving direction is to retain 18 pc only where
dense gas and feedback require it, with a graded vertical mesh. Reducing the
fine half-height from 4 to 1 kpc cuts the ideal fine volume by four, but cannot
be done blindly: check star-formation/feedback restrictions at coarse levels,
off-plane SNe, fountain evolution and coarse-fine force artifacts. Keep this
geometry experiment separate from the first resolution comparison.

## Particle noise limits the small-gap interpretation

Do not downgrade the HI ICs or merge particles solely to reduce cost.
[Banik & Bovy (2021)](https://arxiv.org/abs/2101.12201) evolved GD-1-like streams
for 4 Gyr and found artificial stream-density perturbations with live halo
particle masses 10^4–10^5 solar masses even without substantial overall heating.
Their force softening and solver differ from Quokka's particle mesh, so their
quantitative resolution thresholds cannot be transplanted directly. The mass
range nevertheless overlaps the measured HI halo particle mass and makes a
particle-noise control essential.

Use the same evolved galaxy for direct-force and replay comparisons to measure
reconstruction fidelity. Separately assess physical credibility with independent
particle realizations/mass-resolution tests and controlled perturbers. A model
can faithfully replay numerical structure or smooth it away; neither by itself
identifies real subhalo gaps. The isolated initial galaxy also does not supply
a cosmological population of subhalos automatically. Coarser halo cells can
reduce particle noise while removing physical encounters, so finer is not
monotonically better without mass-resolution convergence.

## Other parameters required for a long run

* **Duration and initialization:** 4/5 Gyr correspond to stop_time=1.262304e17 /
  1.57788e17 s with a Julian year. Allow an explicitly defined settling interval
  before releasing streams; a full 5 Gyr stream experiment after 0.5 Gyr settling
  needs a 5.5 Gyr galaxy run. Keep fixed resolution through a production trajectory.
* **Gas supply:** the existing analysis table, in its disk aperture, falls from
  8.133e9 solar masses at t=0 to 2.321e9 at 550.574 Myr. These are table values,
  not new plotfile measurements or a total-box gas census. With no deliberate
  fuel supply, expect a gas-depleting galaxy. Keeping star formation enabled
  does not ensure a steady MW-like SFR for 5 Gyr. If sustained gas-rich evolution
  is required, specify an equilibrium CGM/recycling or inflow model and mass/
  angular-momentum budget; the separate Mdot_0.1 setup is marked unvalidated in
  the parent README. Do not adopt it without testing. A simple gas budget is
  dMgas/dt=inflow+returns-SFR-outflow, to be measured rather than extrapolated.
* **Boundaries:** retain the 1.2 Mpc box initially. DiskGalaxy's gas/MHD boundaries
  are reflecting, while its nonperiodic gravity uses OpenBCSolver. Wind travel
  to a 600 kpc face is about 0.59 Gyr at 1000 km/s; reflection is a plausible
  long-run contaminant. Test an appropriate outflow/diode MHD boundary or show
  reflected material cannot affect the science region. Nonperiodic geometry
  alone does not change the gas boundary. Monitor floor-injected mass too.
* **Timesteps and gravity:** retain CFL=.3 and synchronized levels initially.
  At dx=36.6 pc and limiting speed 1000 km/s, dt is approximately 0.0107 Myr,
  or 465,000 steps over 5 Gyr; a speed of 3000 km/s triples this estimate.
  These speeds are illustrations, not measured timestep limits. Compare gravity
  update intervals 1, 5, 10 at fixed mesh; do not transfer a ten-step setting
  without considering the changed physical interval. Compare Poisson tolerances
  1e-5 and 1e-7 using force/tidal errors, not residual norms alone.
* **Potential timing:** save full 3D gpot at physical intervals, initially 2 Myr
  (plottime_interval=6.31152e13 s), with 1 Myr short-window reference output and
  5/10/25 Myr downsample tests. This targets the smooth background; fast small
  encounters may require much finer output or explicit perturber histories.
  At 200 km/s, 0.1 kpc is crossed in 0.49 Myr. Derive encounter output cadence
  from its resolved duration, not from Arora's global-orbit cadence.
* **Fresh potential at output:** existing gravity supercycling reuses phi between
  solves. Verify output phi was solved at the recorded snapshot time, or record
  its actual solve time; schedule a fresh solve at output for the new pilot.
  Header time alone does not establish the age of a cached gravitational field.
* **Storage:** 5 Gyr at 2 Myr cadence means approximately 2501 snapshots. The
  36.6 pc fine cylinder alone needs about 2.62 TB for one float64 potential per
  snapshot; at 18.3 pc it is about 21.0 TB. These exclude all other fields, coarse
  levels, padding, particles and checkpoints. Scope a dedicated potential output
  path plus occasional full-state archives, or budget actual measured full
  plotfile sizes. Compact models cannot replace the only independent reference.
* **Frame/reference:** record center, orientation, units and actual times. Track
  galaxy drift without imposing unrecorded recentering. A future live tracer
  comparison should record the same force gather used by simulation particles;
  fluid advection tracers are not automatically collisionless stellar tracers.

## Qualification on 64 Frontier nodes

No wall-clock benchmark was run. A node count is not a total node-hour allocation.

### Conditional node-hour budget for 5 Gyr

A follow-up inspection found historical timing data in
`incite-2026/scaling_tests/walltime_scaling_data.csv`: the 16-node, Phase-1-size
hydro case took 0.514381 seconds per step after the recorded read-time adjustment.
The renewal performance discussion reports an approximately 2.7-fold MHD cost
increase. Applying this gives a **derived**, not directly measured, 1.38883
seconds per step anchor. These tests are not the proposed halo-refined pilot,
and the old gravity-supercycling/physics configuration differs.

For a transparent fixed-64-node sensitivity calculation, let s=dx/18.3105 pc:

    t_step(64, s) = 1.38883 s * [f + (1-f)/(4 s^3)]
    N_step = 5000 Myr / (0.3 dx / v_signal)
    budget = 1.25 * 64 * N_step * t_step / 3600.

Here f=0.1–0.3 is an ASSUMED nonshrinking per-step fraction, not a measured
particle or communication profile. The remaining work is assumed to scale
linearly with cells and from 16 to 64 nodes. Use v_signal=3000–6000 km/s;
3000 km/s is the updated empirical input documented in the project simulation
planner, while 6000 is a factor-two sensitivity case, not a measured bound.
Central values are f=.2 and v=4500 km/s. The 25% reserve allows for output,
restarts and unmodelled costs; it does not guarantee coverage of new halo
refinement, long-term particle growth or poor strong scaling.

| Disk dx | Low scenario | Central scenario | High scenario | Central active days on 64 nodes |
| --- | ---: | ---: | ---: | ---: |
| 18.31 pc | 28,012 node-h | 51,714 node-h | 81,881 node-h | 33.7 |
| 36.62 pc | 5,522 node-h | 14,545 node-h | 27,742 node-h | 9.47 |
| 73.24 pc | 2,231 node-h | 6,565 node-h | 13,046 node-h | 4.27 |

Use rounded working allowances of approximately **52k, 15k and 7k node-hours**,
respectively, subject to short-run calibration. The ranges are conditional
scenarios, not confidence intervals or upper/lower performance bounds. They
assume the intended fiducial halo mesh fits within the unmeasured overhead
allowance; benchmark that assumption. This is more cautious than pure dx^-4
scaling but is still a model. A much smaller timestep or severe particle growth
can exceed it. With other parameters fixed, cost scales linearly with effective
signal speed and elapsed simulated time.

Queue waiting increases calendar duration, not charged node-hours. The values
exclude a separate qualification campaign and downstream reconstruction/replay.
A 0.5 Gyr settling interval before a full 5 Gyr replay adds approximately 10%
under the same cost model. The old 1000 km/s timestep example above was purely
illustrative and is not the timestep assumed in these budgets.

Machine-readable assumptions and calculations:
[isolated_mw_pilot_node_hours.json](validation/isolated_mw_pilot_node_hours.json).

### Throughput targets and qualification

For 5 Gyr, the following are required throughputs, excluding queueing and retries:

| Elapsed allocation | Node-hours at 64 nodes | Required simulated Myr / wall hour |
| --- | ---: | ---: |
| 7 days | 10,752 | 29.8 |
| 14 days | 21,504 | 14.9 |
| 30 days | 46,080 | 6.94 |

Run a 200–500 Myr qualification with timed windows before, during and after
the starburst, using 18/36/73 pc disk variants and q=.02 halo. Check short-window
q=.01/.04 and gravity/output cadence changes independently. Use a matched
reference and statistical comparisons, since different-resolution live galaxies
will diverge in individual clouds and orbits. Measure valid cells and allocated
boxes by level, peak memory, hydro/Poisson/particle/I/O time, timestep distribution,
Myr/hour, new-particle growth and projected storage. Check scaling at 16/32/64
nodes rather than assuming 64 always fastest for a small mesh.

Promote the coarsest disk model that passes the disk/force checks. Then use the
[stream pilot protocol](potential_stream_pilot.md) to measure preserved widths,
gaps and heating, first over 0.5 Gyr and eventually over the full 4–5 Gyr. Longer
duration requires its own measurement; a 0.5 Gyr pass is not a 5 Gyr guarantee.
If only 18 pc passes, retain it and change the refined volume or budget instead
of claiming a coarser setting is physically equivalent.

The scope leaves the ICs and production inputs untouched. Calculation inputs,
particle mass counts and radius/AMR tables are saved in
[isolated_mw_pilot_estimates.json](validation/isolated_mw_pilot_estimates.json).
Counts/costs use V/dx^3 and dt=CFL*dx/v; they are reproducible dimensional
estimates, not an executed mesh build or performance measurement.
