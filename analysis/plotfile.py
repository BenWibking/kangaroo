from __future__ import annotations

from typing import Any

import importlib.util
import sys
from pathlib import Path

try:
    from . import _plotfile as _plotfile  # type: ignore
except ImportError as exc:
    # If running from the repo, the extension may live in site-packages.
    candidates: list[Path] = []
    for base in sys.path:
        try:
            base_path = Path(base)
        except OSError:
            continue
        if not base_path.is_dir():
            continue
        candidates.extend(base_path.glob("analysis/_plotfile*.so"))
    if not candidates:
        raise ImportError(
            "analysis._plotfile extension not found. "
            "Run `pixi run install` to install the C++ bindings."
        ) from exc
    ext_path = candidates[0]
    spec = importlib.util.spec_from_file_location("analysis._plotfile", ext_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load analysis._plotfile from {ext_path}") from exc
    module = importlib.util.module_from_spec(spec)
    sys.modules["analysis._plotfile"] = module
    spec.loader.exec_module(module)
    _plotfile = module  # type: ignore


class PlotfileReader:
    def __init__(self, path: str) -> None:
        self._reader = _plotfile.PlotfileReader(path)

    def header(self) -> dict[str, Any]:
        return self._reader.header()

    def metadata(self) -> dict[str, Any]:
        return self._reader.metadata()

    def num_levels(self) -> int:
        return self._reader.num_levels()

    def num_fabs(self, level: int) -> int:
        return self._reader.num_fabs(level)

    def particle_types(self) -> list[str]:
        return list(self._reader.particle_types())

    def particle_fields(self, particle_type: str) -> list[str]:
        return list(self._reader.particle_fields(particle_type))

    def read_particle_field(
        self,
        particle_type: str,
        field_name: str,
        *,
        return_ndarray: bool = False,
    ) -> dict[str, Any]:
        payload = self._reader.read_particle_field(particle_type, field_name)
        if not return_ndarray:
            return payload

        import numpy as np

        dtype = payload["dtype"]
        np_dtype: Any
        if dtype == "float32":
            np_dtype = np.float32
        elif dtype == "float64":
            np_dtype = np.float64
        elif dtype == "int64":
            np_dtype = np.int64
        else:
            raise RuntimeError(f"unsupported particle dtype '{dtype}'")
        payload["data"] = np.frombuffer(payload["data"], dtype=np_dtype)
        return payload

    def read_fab(
        self,
        level: int,
        fab: int,
        comp_start: int,
        comp_count: int,
        *,
        return_ndarray: bool = False,
    ) -> dict[str, Any]:
        payload = self._reader.read_fab(level, fab, comp_start, comp_count)
        if not return_ndarray:
            return payload

        import numpy as np

        data = payload["data"]
        dtype = payload["dtype"]
        shape = payload["shape"]
        np_dtype = np.float32 if dtype == "float32" else np.float64
        arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
        payload["data"] = arr
        return payload
