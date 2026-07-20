"""Process-context execution configuration for high-level graph construction."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Mapping

_settings: ContextVar[dict[str, Any]] = ContextVar("kangaroo_config", default={})


def get(key: str, default: Any = None) -> Any:
    """Return one active configuration value."""

    return _settings.get().get(key, default)


class set:
    """Temporarily set configuration values in the current Python context."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("config.set expects a mapping")
        updated = dict(_settings.get())
        updated.update(values)
        self._token: Token[dict[str, Any]] | None = _settings.set(updated)

    def __enter__(self) -> "set":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._token is not None:
            _settings.reset(self._token)
            self._token = None

