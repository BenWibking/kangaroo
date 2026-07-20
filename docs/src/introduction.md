# Kangaroo

Kangaroo is an experimental distributed analysis runtime for scientific data. It
combines a lazy Python interface with an HPX/C++ execution engine designed for
adaptive mesh refinement (AMR) and particle workflows.

The repository has three main layers:

- `kangaroo/` provides the recommended high-level Python API.
- `analysis/` provides compatibility and advanced low-level interfaces.
- `cpp/` contains the native runtime, data backends, and numerical kernels.

Python describes what to compute as a lazy graph. At an explicit `compute()`
boundary, Kangaroo lowers that graph to a typed execution plan and runs it over
independently transportable data chunks.

Kangaroo currently supports AMReX plotfiles, openPMD, Parthenon HDF5, and
in-memory datasets.

> Kangaroo is a prototype. Public interfaces and serialized plan details may
> change as the runtime evolves.
