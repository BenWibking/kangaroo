# Getting started

## Requirements

- Python 3.10 through 3.13
- [Pixi](https://pixi.sh/)
- CMake and a C++20 compiler
- HPX (provided by the project Pixi environment)

## Build and test

Clone the repository, then create the environment and build the native
extensions:

```console
$ pixi install
$ pixi run configure
$ pixi run build
$ pixi run install
$ pixi run test
```

The default tasks use the dedicated `pixi-hpx` environment so HPX, Boost, HDF5,
and the compiler come from one consistent toolchain.

## First analysis

```python
import kangaroo as kr

dataset = kr.open_dataset("/path/to/plotfile")
image = dataset["density"].slice(axis="z", resolution=(512, 512))
array = image.compute()
```

Operations remain lazy until `compute()` is called. To share common work between
multiple results, compute them together:

```python
image_result, histogram_result = kr.compute(image, histogram, progress=True)
```

For explicit runtime configuration, create a `kr.Client` and open datasets
through it.
