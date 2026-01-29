from __future__ import annotations

from .ctx import LoweringContext
from .plan import FieldRef


class VorticityMag:
    def __init__(self, vel_field: int, out_name: str = "vortmag") -> None:
        self.vel_field = vel_field
        self.out_name = out_name

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        dom = ctx.domain(step=ds.step, level=ds.level)

        grad_u = ctx.temp_field("gradU")
        vort = ctx.output_field(self.out_name)

        s1 = ctx.stage("gradients")
        s1.map_blocks(
            name="gradU",
            kernel="gradU_stencil",
            domain=dom,
            inputs=[FieldRef(self.vel_field)],
            outputs=[grad_u],
            deps={"kind": "FaceNeighbors", "width": 1, "faces": [1, 1, 1, 1, 1, 1]},
            params={"order": 2},
        )

        s2 = ctx.stage("vortmag", after=[s1])
        s2.map_blocks(
            name="vortmag",
            kernel="vorticity_mag",
            domain=dom,
            inputs=[grad_u],
            outputs=[vort],
            deps={"kind": "None"},
            params={},
        )

        return ctx.fragment([s1, s2])
