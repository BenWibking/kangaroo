from __future__ import annotations

import math
import struct
import numpy as np

from analysis import Plan, Runtime
from analysis.ctx import LoweringContext
from analysis.dataset import open_dataset
from analysis.ops import VorticityMag
from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta


def _pack_linear_field_double(box_lo, box_hi, geom, ax=0.0, ay=0.0, az=0.0, c0=0.0) -> bytes:
    nx = box_hi[0] - box_lo[0] + 1
    ny = box_hi[1] - box_lo[1] + 1
    nz = box_hi[2] - box_lo[2] + 1
    out = bytearray()
    for i in range(nx):
        gi = box_lo[0] + i
        x = geom["x0"][0] + (gi - geom["index_origin"][0] + 0.5) * geom["dx"][0]
        for j in range(ny):
            gj = box_lo[1] + j
            y = geom["x0"][1] + (gj - geom["index_origin"][1] + 0.5) * geom["dx"][1]
            for k in range(nz):
                gk = box_lo[2] + k
                z = geom["x0"][2] + (gk - geom["index_origin"][2] + 0.5) * geom["dx"][2]
                f = c0 + ax * x + ay * y + az * z
                out.extend(struct.pack("<d", f))
    return bytes(out)


def test_vorticity_mag_from_three_component_gradients_amr() -> None:
    rt = Runtime()
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=2),
                        boxes=[BlockBox((0, 0, 0), (3, 3, 3))],
                    ),
                    LevelMeta(
                        geom=LevelGeom(dx=(0.5, 0.5, 0.5), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox((2, 0, 0), (7, 7, 7))],
                    ),
                ],
            )
        ]
    )
    ds = open_dataset("memory://local", runmeta=runmeta, step=0, level=0, runtime=rt)

    geom0 = {"dx": (1.0, 1.0, 1.0), "x0": (0.0, 0.0, 0.0), "index_origin": (0, 0, 0)}
    geom1 = {"dx": (0.5, 0.5, 0.5), "x0": (0.0, 0.0, 0.0), "index_origin": (0, 0, 0)}

    # Velocity components: u=y, v=z, w=x => curl = (-1,-1,-1), |curl| = sqrt(3).
    fu = ds.field_id("vel_x")
    fv = ds.field_id("vel_y")
    fw = ds.field_id("vel_z")
    for level, geom, lo, hi in [
        (0, geom0, (0, 0, 0), (3, 3, 3)),
        (1, geom1, (2, 0, 0), (7, 7, 7)),
    ]:
        ds._h.set_chunk_ref(0, level, fu, 0, 0, _pack_linear_field_double(lo, hi, geom, ay=1.0))
        ds._h.set_chunk_ref(0, level, fv, 0, 0, _pack_linear_field_double(lo, hi, geom, az=1.0))
        ds._h.set_chunk_ref(0, level, fw, 0, 0, _pack_linear_field_double(lo, hi, geom, ax=1.0))

    op = VorticityMag((fu, fv, fw), out_name="vort")
    ctx = LoweringContext(runtime=rt._rt, runmeta=runmeta, dataset=ds)
    plan = Plan(stages=op.lower(ctx))
    kernels = [tmpl.kernel for stage in plan.stages for tmpl in stage.templates]
    assert "amr_subbox_fetch_pack" in kernels
    assert "gradU_stencil" in kernels
    assert "vorticity_mag" in kernels
    rt.run(plan, runmeta=runmeta, dataset=ds)

    vort_field = plan.stages[-1].templates[0].outputs[0].field
    raw = rt.get_task_chunk(step=0, level=0, field=vort_field, version=0, block=0)
    vals = struct.unpack(f"<{len(raw) // 8}d", raw)

    expected = math.sqrt(3.0)
    nx = ny = nz = 4
    for i in (0, 1, 2, 3):
        idx = (i * ny + 2) * nz + 2
        assert abs(vals[idx] - expected) < 1e-8


def test_vorticity_mag_closed_form_on_cell_centers_two_level_amr() -> None:
    rt = Runtime()
    step = 1
    runmeta = RunMeta(
        steps=[
            StepMeta(
                step=0,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=2),
                        boxes=[BlockBox((0, 0, 0), (3, 3, 3))],
                    ),
                    LevelMeta(
                        geom=LevelGeom(dx=(0.5, 0.5, 0.5), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox((2, 0, 0), (7, 7, 7))],
                    ),
                ],
            ),
            StepMeta(
                step=step,
                levels=[
                    LevelMeta(
                        geom=LevelGeom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), ref_ratio=2),
                        boxes=[BlockBox((0, 0, 0), (3, 3, 3))],
                    ),
                    LevelMeta(
                        geom=LevelGeom(dx=(0.5, 0.5, 0.5), x0=(0.0, 0.0, 0.0), ref_ratio=1),
                        boxes=[BlockBox((2, 0, 0), (7, 7, 7))],
                    ),
                ],
            )
        ]
    )
    ds = open_dataset("memory://local", runmeta=runmeta, step=step, level=0, runtime=rt)

    geom0 = {"dx": (1.0, 1.0, 1.0), "x0": (0.0, 0.0, 0.0), "index_origin": (0, 0, 0)}
    geom1 = {"dx": (0.5, 0.5, 0.5), "x0": (0.0, 0.0, 0.0), "index_origin": (0, 0, 0)}

    # Affine velocity:
    # u = 2y + 3z, v = 7x + 5z, w = 11x + 13y
    # curl = (13-5, 3-11, 7-2) = (8, -8, 5), |curl| = sqrt(153)
    fu = ds.field_id("vel_u")
    fv = ds.field_id("vel_v")
    fw = ds.field_id("vel_w")
    for level, geom, lo, hi in [
        (0, geom0, (0, 0, 0), (3, 3, 3)),
        (1, geom1, (2, 0, 0), (7, 7, 7)),
    ]:
        ds._h.set_chunk_ref(
            step, level, fu, 0, 0, _pack_linear_field_double(lo, hi, geom, ay=2.0, az=3.0)
        )
        ds._h.set_chunk_ref(
            step, level, fv, 0, 0, _pack_linear_field_double(lo, hi, geom, ax=7.0, az=5.0)
        )
        ds._h.set_chunk_ref(
            step, level, fw, 0, 0, _pack_linear_field_double(lo, hi, geom, ax=11.0, ay=13.0)
        )

    op = VorticityMag((fu, fv, fw), out_name="vort_cf")
    ctx = LoweringContext(runtime=rt._rt, runmeta=runmeta, dataset=ds)
    plan = Plan(stages=op.lower(ctx))
    rt.run(plan, runmeta=runmeta, dataset=ds)

    vort_field = plan.stages[-1].templates[0].outputs[0].field
    raw = rt.get_task_chunk(step=step, level=0, field=vort_field, version=0, block=0)
    vals = struct.unpack(f"<{len(raw) // 8}d", raw)

    expected = math.sqrt(153.0)
    nx = ny = nz = 4
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = (i * ny + j) * nz + k
                assert abs(vals[idx] - expected) < 1e-8


def _pack_periodic_velocity_component_double(
    box_lo,
    box_hi,
    geom,
    *,
    component: str,
) -> bytes:
    nx = box_hi[0] - box_lo[0] + 1
    ny = box_hi[1] - box_lo[1] + 1
    nz = box_hi[2] - box_lo[2] + 1
    out = bytearray()
    for i in range(nx):
        gi = box_lo[0] + i
        x = geom["x0"][0] + (gi - geom["index_origin"][0] + 0.5) * geom["dx"][0]
        for j in range(ny):
            gj = box_lo[1] + j
            y = geom["x0"][1] + (gj - geom["index_origin"][1] + 0.5) * geom["dx"][1]
            for k in range(nz):
                gk = box_lo[2] + k
                z = geom["x0"][2] + (gk - geom["index_origin"][2] + 0.5) * geom["dx"][2]
                twopi = 2.0 * math.pi
                # Periodic analytic velocity:
                # u = sin(2π y), v = sin(2π z), w = sin(2π x)
                if component == "u":
                    f = math.sin(twopi * y)
                elif component == "v":
                    f = math.sin(twopi * z)
                elif component == "w":
                    f = math.sin(twopi * x)
                else:
                    raise ValueError(component)
                out.extend(struct.pack("<d", f))
    return bytes(out)


def test_vorticity_mag_closed_form_variable_field_two_level_amr() -> None:
    def run_vorticity_on_two_level_amr(step: int, n0: int) -> float:
        rt = Runtime()
        dx0 = 1.0 / float(n0)
        levels = [
            LevelMeta(
                geom=LevelGeom(
                    dx=(dx0, dx0, dx0),
                    x0=(0.0, 0.0, 0.0),
                    ref_ratio=2,
                    is_periodic=(True, True, True),
                ),
                boxes=[BlockBox((0, 0, 0), (n0 - 1, n0 - 1, n0 - 1))],
            ),
            LevelMeta(
                geom=LevelGeom(
                    dx=(0.5 * dx0, 0.5 * dx0, 0.5 * dx0),
                    x0=(0.0, 0.0, 0.0),
                    ref_ratio=1,
                    is_periodic=(True, True, True),
                ),
                boxes=[BlockBox((n0, 0, 0), (2 * n0 - 1, 2 * n0 - 1, 2 * n0 - 1))],
            ),
        ]
        steps = [StepMeta(step=s, levels=levels) for s in range(step + 1)]
        runmeta = RunMeta(steps=steps)
        ds0 = open_dataset("memory://local", runmeta=runmeta, step=step, level=0, runtime=rt)
        ds1 = open_dataset("memory://local", runmeta=runmeta, step=step, level=1, runtime=rt)
        geom0 = {"dx": (dx0, dx0, dx0), "x0": (0.0, 0.0, 0.0), "index_origin": (0, 0, 0)}
        geom1 = {"dx": (0.5 * dx0, 0.5 * dx0, 0.5 * dx0), "x0": (0.0, 0.0, 0.0), "index_origin": (0, 0, 0)}
        # Populate both dataset handles (each has an independent memory backend).
        fu0 = ds0.field_id(f"vel_u_var0_{step}")
        fv0 = ds0.field_id(f"vel_v_var0_{step}")
        fw0 = ds0.field_id(f"vel_w_var0_{step}")
        fu1 = ds1.field_id(f"vel_u_var1_{step}")
        fv1 = ds1.field_id(f"vel_v_var1_{step}")
        fw1 = ds1.field_id(f"vel_w_var1_{step}")
        for ds, fu, fv, fw in [(ds0, fu0, fv0, fw0), (ds1, fu1, fv1, fw1)]:
            for level, geom, lo, hi in [
                (0, geom0, (0, 0, 0), (n0 - 1, n0 - 1, n0 - 1)),
                (1, geom1, (n0, 0, 0), (2 * n0 - 1, 2 * n0 - 1, 2 * n0 - 1)),
            ]:
                ds._h.set_chunk_ref(
                    step, level, fu, 0, 0, _pack_periodic_velocity_component_double(lo, hi, geom, component="u")
                )
                ds._h.set_chunk_ref(
                    step, level, fv, 0, 0, _pack_periodic_velocity_component_double(lo, hi, geom, component="v")
                )
                ds._h.set_chunk_ref(
                    step, level, fw, 0, 0, _pack_periodic_velocity_component_double(lo, hi, geom, component="w")
                )

        op0 = VorticityMag((fu0, fv0, fw0), out_name=f"vort_var0_{step}")
        op1 = VorticityMag((fu1, fv1, fw1), out_name=f"vort_var1_{step}")
        ctx0 = LoweringContext(runtime=rt._rt, runmeta=runmeta, dataset=ds0)
        ctx1 = LoweringContext(runtime=rt._rt, runmeta=runmeta, dataset=ds1)
        plan0 = Plan(stages=op0.lower(ctx0))
        plan1 = Plan(stages=op1.lower(ctx1))
        rt.run(plan0, runmeta=runmeta, dataset=ds0)
        rt.run(plan1, runmeta=runmeta, dataset=ds1)

        vort0 = plan0.stages[-1].templates[0].outputs[0].field
        vort1 = plan1.stages[-1].templates[0].outputs[0].field
        raw0 = rt.get_task_chunk(step=step, level=0, field=vort0, version=0, block=0)
        raw1 = rt.get_task_chunk(step=step, level=1, field=vort1, version=0, block=0)
        arr0 = np.frombuffer(raw0, dtype=np.float64).copy().reshape((n0, n0, n0))
        arr1 = np.frombuffer(raw1, dtype=np.float64).copy().reshape((n0, 2 * n0, 2 * n0))

        def exact_vortmag(x: float, y: float, z: float) -> float:
            twopi = 2.0 * math.pi
            cx = math.cos(twopi * x)
            cy = math.cos(twopi * y)
            cz = math.cos(twopi * z)
            # curl = (-2π cos(2π z), -2π cos(2π x), -2π cos(2π y))
            return twopi * math.sqrt(cx * cx + cy * cy + cz * cz)

        # Composite valid cells:
        # - level 1: all cells
        # - level 0: only non-covered cells (x < n0/2)
        sum_abs = 0.0
        count = 0
        for i in range(n0 // 2):
            x = (i + 0.5) * dx0
            for j in range(n0):
                y = (j + 0.5) * dx0
                for k in range(n0):
                    z = (k + 0.5) * dx0
                    expected = exact_vortmag(x, y, z)
                    sum_abs += abs(float(arr0[i, j, k]) - expected)
                    count += 1
        dx1 = 0.5 * dx0
        for i in range(n0):
            x = (n0 + i + 0.5) * dx1
            for j in range(2 * n0):
                y = (j + 0.5) * dx1
                for k in range(2 * n0):
                    z = (k + 0.5) * dx1
                    expected = exact_vortmag(x, y, z)
                    sum_abs += abs(float(arr1[i, j, k]) - expected)
                    count += 1
        return sum_abs / float(count)

    # Periodic field:
    # u = sin(2π y), v = sin(2π z), w = sin(2π x)
    # |curl| = 2π sqrt(cos^2(2πx)+cos^2(2πy)+cos^2(2πz))
    eh = run_vorticity_on_two_level_amr(step=20, n0=8)
    e2h = run_vorticity_on_two_level_amr(step=21, n0=16)
    e4h = run_vorticity_on_two_level_amr(step=22, n0=32)

    # Error should decay with refinement.
    assert e4h < e2h < eh

    # Richardson-style observed order from error ratios.
    p12 = math.log(eh / e2h, 2.0)
    p24 = math.log(e2h / e4h, 2.0)
    p_obs = 0.5 * (p12 + p24)

    # Composite AMR + coarse-fine transitions reduce formal order here;
    # require clear, consistent convergence.
    assert p_obs > 0.45
    assert abs(p12 - p24) < 0.8
