from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import _core  # type: ignore


@dataclass(frozen=True)
class BlockBox:
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]


@dataclass(frozen=True)
class LevelGeom:
    dx: tuple[float, float, float]
    x0: tuple[float, float, float]
    ref_ratio: int
    index_origin: tuple[int, int, int] = (0, 0, 0)


@dataclass
class LevelMeta:
    geom: LevelGeom
    boxes: List[BlockBox]


@dataclass
class StepMeta:
    step: int
    levels: List[LevelMeta]


@dataclass
class RunMeta:
    steps: List[StepMeta]
    _h: Any = field(init=False)

    def __post_init__(self) -> None:
        steps_payload: List[Dict[str, Any]] = []
        for step in self.steps:
            levels_payload = []
            for lvl in step.levels:
                levels_payload.append(
                    {
                        "geom": {
                            "dx": lvl.geom.dx,
                            "x0": lvl.geom.x0,
                            "index_origin": lvl.geom.index_origin,
                            "ref_ratio": lvl.geom.ref_ratio,
                        },
                        "boxes": [(b.lo, b.hi) for b in lvl.boxes],
                    }
                )
            steps_payload.append({"step": step.step, "levels": levels_payload})
        self._h = _core.RunMetaHandle(steps_payload)


def load_runmeta_from_dict(payload: Dict[str, Any]) -> RunMeta:
    steps: List[StepMeta] = []
    for step_entry in payload.get("steps", []):
        levels: List[LevelMeta] = []
        for lvl_entry in step_entry.get("levels", []):
            geom = LevelGeom(
                dx=tuple(lvl_entry["geom"]["dx"]),
                x0=tuple(lvl_entry["geom"]["x0"]),
                index_origin=tuple(lvl_entry["geom"].get("index_origin", (0, 0, 0))),
                ref_ratio=int(lvl_entry["geom"]["ref_ratio"]),
            )
            boxes = [BlockBox(tuple(lo), tuple(hi)) for lo, hi in lvl_entry.get("boxes", [])]
            levels.append(LevelMeta(geom=geom, boxes=boxes))
        steps.append(StepMeta(step=int(step_entry["step"]), levels=levels))
    return RunMeta(steps=steps)
