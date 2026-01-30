from __future__ import annotations

from typing import Iterable

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


def _axis_index(axis: str | int) -> int:
    if isinstance(axis, int):
        if axis in (0, 1, 2):
            return axis
        raise ValueError("axis must be 0, 1, or 2")
    axis_l = axis.lower()
    if axis_l == "x":
        return 0
    if axis_l == "y":
        return 1
    if axis_l == "z":
        return 2
    raise ValueError("axis must be 'x', 'y', 'z', or 0/1/2")


def _bounds_1d(lo: int, hi: int, x0: float, dx: float) -> tuple[float, float]:
    return x0 + lo * dx, x0 + (hi + 1) * dx


def _overlaps_1d(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)


class UniformSlice:
    def __init__(
        self,
        field: int,
        *,
        axis: str | int,
        coord: float,
        rect: tuple[float, float, float, float],
        resolution: tuple[int, int],
        out_name: str = "slice",
        bytes_per_value: int = 4,
    ) -> None:
        self.field = field
        self.axis = axis
        self.coord = coord
        self.rect = rect
        self.resolution = resolution
        self.out_name = out_name
        self.bytes_per_value = bytes_per_value

    def _intersecting_blocks(self, ctx: LoweringContext) -> Iterable[int]:
        ds = ctx.dataset
        level_meta = ctx.runmeta.steps[ds.step].levels[ds.level]
        axis_idx = _axis_index(self.axis)
        u_axis, v_axis = [i for i in range(3) if i != axis_idx]

        u0, v0, u1, v1 = self.rect
        umin, umax = (u0, u1) if u0 <= u1 else (u1, u0)
        vmin, vmax = (v0, v1) if v0 <= v1 else (v1, v0)

        geom = level_meta.geom
        for i, box in enumerate(level_meta.boxes):
            ax0, ax1 = _bounds_1d(box.lo[axis_idx], box.hi[axis_idx], geom.x0[axis_idx], geom.dx[axis_idx])
            if not (ax0 <= self.coord <= ax1):
                continue

            u0_b, u1_b = _bounds_1d(box.lo[u_axis], box.hi[u_axis], geom.x0[u_axis], geom.dx[u_axis])
            v0_b, v1_b = _bounds_1d(box.lo[v_axis], box.hi[v_axis], geom.x0[v_axis], geom.dx[v_axis])
            if not (_overlaps_1d(u0_b, u1_b, umin, umax) and _overlaps_1d(v0_b, v1_b, vmin, vmax)):
                continue
            yield i

    def lower(self, ctx: LoweringContext):
        ds = ctx.dataset
        axis_idx = _axis_index(self.axis)
        u_axis, v_axis = [i for i in range(3) if i != axis_idx]
        nx, ny = self.resolution
        if nx <= 0 or ny <= 0:
            raise ValueError("resolution must be positive")

        blocks = list(self._intersecting_blocks(ctx))
        if not blocks:
            return ctx.fragment([])

        out_field = ctx.output_field(self.out_name)
        out_bytes = nx * ny * self.bytes_per_value

        stage = ctx.stage("uniform_slice")
        for block in blocks:
            dom = ctx.domain(step=ds.step, level=ds.level, blocks=[block])
            stage.map_blocks(
                name=f"uniform_slice_b{block}",
                kernel="uniform_slice",
                domain=dom,
                inputs=[FieldRef(self.field)],
                outputs=[out_field],
                output_bytes=[out_bytes],
                deps={"kind": "None"},
                params={
                    "axis": axis_idx,
                    "coord": self.coord,
                    "rect": list(self.rect),
                    "resolution": [nx, ny],
                    "plane_axes": [u_axis, v_axis],
                    "bytes_per_value": self.bytes_per_value,
                },
            )

        return ctx.fragment([stage])
