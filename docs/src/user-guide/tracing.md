# Tracing

Set `KANGAROO_PERFETTO_TRACE` to record runtime execution as a Perfetto trace:

```console
$ KANGAROO_PERFETTO_TRACE=run.pftrace \
    pixi run python scripts/plotfile_slice.py \
    /path/to/plotfile --var density
```

Open the resulting file in the [Perfetto UI](https://ui.perfetto.dev/). In a
distributed run, each locality writes a separate trace such as
`run.loc000.pftrace` or `run.loc001.pftrace`.

Traces contain task lifecycle and runtime metric events and are useful for
examining graph structure, worker utilization, and bottlenecks.
