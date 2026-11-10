"""Block culling must never reject a block over an ulp.

The flux and profile operators skip blocks whose extent misses the surface,
using _bounds_1d; the kernels compute cell faces in their own floating-point
order, fused multiply-adds included. A cylinder of height 2 kpc on a grid whose
level-2 block face lay one ulp above the unfused bound lost its end caps over
the block's whole footprint.
"""

from __future__ import annotations

import math
import random

from analysis.ops import _bounds_1d

KPC = 1.0e3 * 3.0856775814913673e18


def test_bounds_enclose_fused_faces() -> None:
    rng = random.Random(7)
    for _ in range(20000):
        origin = rng.randint(-50, 50)
        lo = rng.randint(-10**6, 10**6)
        hi = lo + rng.randint(0, 4096)
        dx = 10.0 ** rng.uniform(-3, 22)
        x0 = rng.uniform(-1.0, 1.0) * 10.0 ** rng.uniform(-3, 25)
        lower, upper = _bounds_1d(lo, hi, x0, dx, origin)
        for face, bound, inside in ((lo - origin, lower, True), (hi + 1 - origin, upper, False)):
            plain = x0 + face * dx
            fused = math.fma(face, dx, x0)
            if inside:
                assert bound <= min(plain, fused)
            else:
                assert bound >= max(plain, fused)


def test_height_on_a_block_face_keeps_the_block() -> None:
    # The MHD fixture of amrexplorer's validation: level 2 of 64 cells over
    # +-8 kpc, a block whose upper z face is 2 kpc from the center.
    x0 = -8.0 * KPC
    dx = 16.0 * KPC / 64.0
    height = 2.0 * KPC
    lower, upper = _bounds_1d(24, 39, x0, dx, 0)
    assert lower <= -height <= upper or lower <= height <= upper
    assert upper >= math.fma(40, dx, x0)
