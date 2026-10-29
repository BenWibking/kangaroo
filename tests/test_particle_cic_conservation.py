"""The particle CIC projection keeps every particle's mass.

Weight crossing into a neighboring block, landing in a cell a finer level
covers, or leaving the domain used to be dropped. These plotfiles put
particles right beside block faces, coarse-fine boundaries and domain walls,
and check that the projected image holds the whole mass and, on one level,
matches a direct NumPy deposit.
"""

from __future__ import annotations

import os
import struct

import numpy as np

DESCRIPTOR = "((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))"


def _box(lo, hi):
    return f"(({lo[0]},{lo[1]},{lo[2]}) ({hi[0]},{hi[1]},{hi[2]}) (0,0,0))"


def _write_level(root, lev, boxes):
    directory = root / f"Level_{lev}"
    os.makedirs(directory, exist_ok=True)
    data = bytearray()
    offsets = []
    for lo, hi in boxes:
        offsets.append(len(data))
        data += f"FAB {DESCRIPTOR}{_box(lo, hi)} 1\n".encode()
        cells = int(np.prod(np.subtract(hi, lo) + 1))
        data += struct.pack(f"<{cells}d", *([1.0] * cells))
    (directory / "Cell_D_00000").write_bytes(bytes(data))
    lines = ["1", "1", "1", "0", f"({len(boxes)} 0"]
    lines += [_box(lo, hi) for lo, hi in boxes]
    lines += [")", str(len(boxes))] + [f"FabOnDisk: Cell_D_00000 {o}" for o in offsets]
    lines += ["", f"{len(boxes)},1"] + ["1.0,"] * len(boxes)
    lines += ["", f"{len(boxes)},1"] + ["1.0,"] * len(boxes) + [""]
    (directory / "Cell_H").write_text("\n".join(lines) + "\n")


def _write_plotfile(root, levels, particles_by_level, n0=8):
    """A [0, 1]^3 plotfile with `levels` (lists of boxes, ratio 2) and one
    `Stars_particles` species, each particle stored on its level in the first
    grid of that level that contains it."""
    os.makedirs(root)
    nlev = len(levels)
    header = ["HyperCLaw-V1.1", "1", "rho", "3", "0.0", str(nlev - 1), "0 0 0", "1 1 1",
              " ".join(["2"] * (nlev - 1)),
              " ".join(_box((0, 0, 0), (n0 * 2**l - 1,) * 3) for l in range(nlev)),
              " ".join(["0"] * nlev)]
    header += [" ".join([repr(1.0 / (n0 * 2**l))] * 3) for l in range(nlev)]
    header += ["0", "0"]
    for lev, boxes in enumerate(levels):
        dx = 1.0 / (n0 * 2**lev)
        header += [f"{lev} {len(boxes)} 0.0", "0"]
        for lo, hi in boxes:
            for a in range(3):
                header.append(f"{lo[a] * dx!r} {(hi[a] + 1) * dx!r}")
        header.append(f"Level_{lev}/Cell")
        _write_level(root, lev, boxes)
    (root / "Header").write_text("\n".join(header) + "\n")

    species = root / "Stars_particles"
    os.makedirs(species)
    total = sum(len(p) for p in particles_by_level)
    lines = ["Version_Two_Dot_Zero_double", "3", "1", "mass", "0", "1", str(total),
             str(total + 1), str(nlev - 1)] + [str(len(b)) for b in levels]
    next_id = 1
    for lev, boxes in enumerate(levels):
        os.makedirs(species / f"Level_{lev}")
        dx = 1.0 / (n0 * 2**lev)
        grids = [[] for _ in boxes]
        for x, y, z, m in particles_by_level[lev] if lev < len(particles_by_level) else []:
            cell = (int(x / dx), int(y / dx), int(z / dx))
            for g, (lo, hi) in enumerate(boxes):
                if all(lo[a] <= cell[a] <= hi[a] for a in range(3)):
                    grids[g].append((x, y, z, m))
                    break
            else:
                raise AssertionError("particle outside every grid of its level")
        data = bytearray()
        for g, entries in enumerate(grids):
            offset = len(data)
            for _ in entries:
                data += struct.pack("<ii", next_id, 0)
                next_id += 1
            for x, y, z, m in entries:
                data += struct.pack("<4d", x, y, z, m)
            lines.append(f"0 {len(entries)} {offset if entries else 0}")
        (species / f"Level_{lev}" / "DATA_00000").write_bytes(bytes(data))
        (species / f"Level_{lev}" / "Particle_H").write_text(
            f"({len(boxes)} 0\n" + "\n".join(_box(lo, hi) for lo, hi in boxes) + "\n)\n")
    (species / "Header").write_text("\n".join(lines) + "\n")
    return str(root)


def _project(path, resolution=(8, 8), rect=(0.0, 0.0, 1.0, 1.0)):
    from analysis import Runtime
    from analysis.dataset import open_dataset
    from analysis.pipeline import pipeline

    rt = Runtime()
    ds = open_dataset(path, level=0, runtime=rt)
    pipe = pipeline(runtime=rt, runmeta=ds.metadata_bundle().runmeta, dataset=ds)
    out = pipe.particle_cic_projection(particle_type="Stars_particles", axis="z",
                                       axis_bounds=(0.0, 1.0), rect=rect,
                                       resolution=resolution, out="stars")
    pipe.run()
    arr = rt.get_task_chunk_array(step=0, level=0, field=out.field, version=0,
                                  block=0, dataset=ds)
    return np.asarray(arr, dtype=np.float64).reshape(resolution[1], resolution[0])


def _reference(points, n):
    image = np.zeros((n, n))
    for x, y, _z, m in points:
        su, sv = x * n - 0.5, y * n - 0.5
        iu, iv = int(np.floor(su)), int(np.floor(sv))
        fu, fv = su - iu, sv - iv
        for du, dv, f in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)),
                          (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
            image[np.clip(iv + dv, 0, n - 1), np.clip(iu + du, 0, n - 1)] += m * f
    return image


def test_cic_keeps_mass_across_blocks_and_walls(tmp_path) -> None:
    rng = np.random.default_rng(4)
    points = [(*p, m) for p, m in zip(rng.uniform(0, 1, (200, 3)), rng.uniform(1, 2, 200))]
    # Beside the block faces at x = 0.5 and y = 0.5, and the domain walls.
    points += [(0.49, 0.3, 0.2, 1.0), (0.51, 0.7, 0.4, 2.0), (0.3, 0.501, 0.6, 1.5),
               (0.01, 0.99, 0.5, 1.0), (0.999, 0.001, 0.1, 3.0)]
    boxes = [((0, 0, 0), (3, 7, 7)), ((4, 0, 0), (7, 3, 7)), ((4, 4, 0), (7, 7, 7))]
    path = _write_plotfile(tmp_path / "plt", [boxes], [points])
    image = _project(path)
    np.testing.assert_allclose(image, _reference(points, 8), rtol=1e-12, atol=1e-12)
    assert np.isclose(image.sum(), sum(p[3] for p in points), rtol=1e-14)


def test_cic_keeps_mass_across_coarse_fine_boundaries(tmp_path) -> None:
    rng = np.random.default_rng(5)
    coarse = [(*p, m) for p, m in zip(rng.uniform(0, 1, (150, 3)), rng.uniform(1, 2, 150))]
    coarse = [p for p in coarse if not all(0.25 <= c < 0.75 for c in p[:3])]
    coarse += [(0.24, 0.5, 0.5, 1.0), (0.5, 0.76, 0.3, 2.0)]   # CIC into covered cells
    fine = [(*p, m) for p, m in zip(rng.uniform(0.25, 0.75, (150, 3)), rng.uniform(1, 2, 150))]
    fine += [(0.251, 0.5, 0.5, 1.0), (0.749, 0.749, 0.5, 2.0)]  # CIC out of the fine grid
    levels = [[((0, 0, 0), (7, 7, 7))], [((4, 4, 4), (11, 11, 11))]]
    path = _write_plotfile(tmp_path / "plt2", levels, [coarse, fine])
    for resolution in ((16, 16), (13, 7)):
        image = _project(path, resolution=resolution)
        total = sum(p[3] for p in coarse) + sum(p[3] for p in fine)
        assert np.isclose(image.sum(), total, rtol=1e-13)
        assert np.all(image >= 0.0)
