#!/usr/bin/env python3
from __future__ import annotations

from analysis import Plan, Runtime
from analysis.ctx import LoweringContext
from analysis.ops import VorticityMag
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta
from analysis.dataset import open_dataset


def main() -> int:
    try:
        rt = Runtime()
    except Exception as exc:
        print("Runtime init failed (is the C++ module built?):", exc)
        return 1

    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(
                            dx=(1.0, 1.0, 1.0),
                            x0=(0.0, 0.0, 0.0),
                            index_origin=(0, 0, 0),
                            ref_ratio=1,
                        ),
                        boxes=[
                            BlockBox((0, 0, 0), (7, 7, 7)),
                            BlockBox((8, 0, 0), (15, 7, 7)),
                        ],
                    )
                ],
            )
        ]
    )

    ds = open_dataset("memory://example", runmeta=runmeta, step=0, level=0, runtime=rt)
    vel = ds.field_id("vel")

    op = VorticityMag(vel_field=vel)
    ctx = LoweringContext(runtime=rt._rt, runmeta=runmeta._h, dataset=ds)
    plan = Plan(stages=op.lower(ctx))

    try:
        rt.run(plan, runmeta=runmeta, dataset=ds)
    except Exception as exc:
        print("Runtime executed but raised (kernels may be missing):", exc)
        return 2

    print("Runtime completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
