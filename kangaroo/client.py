"""Runtime ownership for the high-level Kangaroo API."""

from __future__ import annotations

import sys
from typing import Any, Sequence

from analysis.dataset import open_dataset as _open_backend_dataset
from analysis.runtime import Runtime


class Client:
    """Own Kangaroo runtime resources and open datasets in that context."""

    def __init__(
        self,
        *,
        hpx_args: Sequence[str] | None = None,
        hpx_config: Sequence[str] | None = None,
        progress: bool = False,
        runtime: Runtime | None = None,
    ) -> None:
        self.runtime = runtime or Runtime(
            hpx_args=None if hpx_args is None else list(hpx_args),
            hpx_config=None if hpx_config is None else list(hpx_config),
        )
        self.progress = bool(progress)

    @classmethod
    def from_parsed_args(
        cls,
        parsed_args: Any,
        *,
        unknown_args: Sequence[str] | None = None,
        argv0: str | None = None,
        progress: bool = False,
    ) -> "Client":
        """Create a client from argparse output and unparsed HPX arguments."""

        runtime = Runtime.from_parsed_args(
            parsed_args,
            unknown_args=None if unknown_args is None else list(unknown_args),
            argv0=argv0 or sys.argv[0],
        )
        return cls(runtime=runtime, progress=progress)

    def open_dataset(
        self,
        uri: str,
        *,
        runmeta: Any | None = None,
        step: int = 0,
        level: int | None = None,
    ) -> "Dataset":
        """Open a supported dataset and return its high-level lazy facade."""

        from .dataset import Dataset

        backend = _open_backend_dataset(
            uri,
            runmeta=runmeta,
            step=step,
            level=level,
            runtime=self.runtime,
        )
        return Dataset(backend, self)

