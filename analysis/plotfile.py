from __future__ import annotations

from typing import Any

try:
    from . import _plotfile as _plotfile  # type: ignore
except ModuleNotFoundError:
    from . import _core as _plotfile  # type: ignore


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
