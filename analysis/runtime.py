from __future__ import annotations

from typing import Any

import msgpack

from .plan import Plan

from . import _core  # type: ignore


class Runtime:
    def __init__(self, core_runtime: Any | None = None) -> None:
        if core_runtime is not None:
            self._rt = core_runtime
        else:
            self._rt = _core.Runtime()

    @property
    def kernels(self):
        return self._rt.kernels()

    def alloc_field_id(self, name: str) -> int:
        return self._rt.alloc_field_id(name)

    def mark_field_persistent(self, fid: int, name: str) -> None:
        self._rt.mark_field_persistent(fid, name)

    def run(self, plan: Plan, *, runmeta, dataset) -> None:
        packed = msgpack.packb(plan_to_dict(plan), use_bin_type=True)
        self._rt.run_packed_plan(packed, runmeta._h, dataset._h)

    def preload(self, *, runmeta, dataset, fields: list[int]) -> None:
        self._rt.preload_dataset(runmeta._h, dataset._h, list(fields))


def plan_to_dict(plan: Plan) -> dict:
    stages = []
    topo = plan.topo_stages()
    stage_ids = {id(s): i for i, s in enumerate(topo)}
    for stage in topo:
        stages.append(
            {
                "name": stage.name,
                "plane": stage.plane,
                "after": [stage_ids[id(parent)] for parent in stage.after],
                "templates": [
                    {
                        "name": tmpl.name,
                        "plane": tmpl.plane,
                        "kernel": tmpl.kernel,
                        "domain": {
                            "step": tmpl.domain.step,
                            "level": tmpl.domain.level,
                            "blocks": list(tmpl.domain.blocks)
                            if tmpl.domain.blocks is not None
                            else None,
                        },
                    "inputs": [
                        {"field": ref.field, "version": ref.version}
                        for ref in tmpl.inputs
                    ],
                    "outputs": [
                        {"field": ref.field, "version": ref.version}
                        for ref in tmpl.outputs
                    ],
                    "output_bytes": list(tmpl.output_bytes),
                    "deps": tmpl.deps,
                    "params": tmpl.params,
                }
                    for tmpl in stage.templates
                ],
            }
        )
    return {"stages": stages}
