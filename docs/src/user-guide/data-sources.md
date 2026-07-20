# Data sources

`open_dataset()` accepts local paths or explicit URIs. Kangaroo recognizes:

| Source | URI form |
|---|---|
| AMReX plotfile | `/path/to/plotfile` or `amrex:///path/to/plotfile` |
| openPMD | `openpmd:///path/to/series` |
| Parthenon HDF5 | `/path/to/output.phdf` or `parthenon:///path/to/output.phdf` |
| Synthetic/in-memory data | `memory://name` |

Existing local paths are inspected to choose a backend. An AMReX plotfile is
identified by its `Header`; `.phdf`, `.h5`, and `.hdf5` files select the
Parthenon backend. Use an explicit URI when automatic detection is ambiguous.

Backends provide field discovery, metadata, and chunk access through a common
runtime contract. See [Backends and I/O](../reference/backends-and-io.md) for the
normative backend behavior.
