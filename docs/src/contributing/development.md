# Development workflow

Python code lives in `kangaroo/` and `analysis/`; the C++20 runtime and nanobind
extensions live in `cpp/`. Tests are under `tests/` and use pytest.

Use the repository's Pixi tasks for a consistent HPX toolchain:

```console
$ pixi run test
```

To build and preview this documentation locally:

```console
$ mdbook build docs
$ mdbook serve docs --open
```

The generated HTML in `docs/book/` is ignored by Git. Pull requests build the
book in CI, and pushes to `main` deploy it to GitHub Pages.

Keep changes focused, add deterministic tests beside the related coverage area,
and include the exact validation commands in pull requests.
