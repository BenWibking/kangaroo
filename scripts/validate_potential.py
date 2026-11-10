#!/usr/bin/env python3
"""Held-out force diagnostics against a separate AMR trilinear reference.

Reads only requested FABs through a bounded LRU cache; never assembles a mesh.
The reference is not the simulation's gravity solver: it differentiates a
cell-center trilinear interpolant, prolonging missing fine stencil centers from
coarser levels. Reports interface points separately. No temporal error estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from potential_replay import PotentialModel


class AMRReference:
    def __init__(self, reader, field):
        self.reader = reader
        self.meta = reader.metadata()
        self.component = self.meta["var_names"].index(field)
        self.boxes = [np.asarray(b) for b in self.meta["level_boxes"]]
        self.origins = np.asarray([b[0] for b in self.meta["prob_domain"]])
        self.dx = np.asarray(self.meta["cell_size"])
        self.x0 = np.asarray(self.meta["prob_lo"])
        self._cache = OrderedDict()

    def owner(self, level, index):
        boxes = self.boxes[level]
        matches = np.flatnonzero(
            np.all((index >= boxes[:, 0]) & (index <= boxes[:, 1]), axis=1)
        )
        if len(matches) > 1:
            raise ValueError("overlapping same-level valid boxes")
        return int(matches[0]) if len(matches) else None

    def level_at(self, point):
        for level in range(len(self.boxes) - 1, -1, -1):
            index = (
                np.floor((point - self.x0) / self.dx[level]).astype(int)
                + self.origins[level]
            )
            if self.owner(level, index) is not None:
                return level
        raise ValueError("reference point outside AMR data")

    def block(self, level, block):
        # PlotfileReader arrays are component,z,y,x; native runtime input grids
        # use x,y,z. This independent reference explicitly uses the reader order.
        key = (level, block)
        if key not in self._cache:
            self._cache[key] = self.reader.read_fab(
                level, block, self.component, 1, return_ndarray=True
            )["data"][0]
            if len(self._cache) > 8:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def interpolate(self, point, level):
        u = (point - self.x0) / self.dx[level] + self.origins[level] - 0.5
        low = np.floor(u).astype(int)
        fraction = u - low
        value, gradient, interface = 0.0, np.zeros(3), False
        for corner in itertools.product((0, 1), repeat=3):
            index = low + corner
            block = self.owner(level, index)
            position = self.x0 + (index - self.origins[level] + 0.5) * self.dx[level]
            if block is None:
                if level == 0:
                    raise ValueError("reference stencil outside base valid domain")
                node = self.interpolate(position, level - 1)[0]
                interface = True
            else:
                local = index - self.boxes[level][block, 0]
                node = self.block(level, block)[local[2], local[1], local[0]]
                interface |= self.level_at(position) != level
            if not np.isfinite(node):
                raise ValueError("nonfinite reference potential")
            factors = np.where(corner, fraction, 1 - fraction)
            value += node * np.prod(factors)
            for axis in range(3):
                gradient[axis] += (
                    node
                    * (1 if corner[axis] else -1)
                    * np.prod(np.delete(factors, axis))
                    / self.dx[level, axis]
                )
        return value, gradient, interface

    def evaluate(self, point):
        point = np.asarray(point)
        level = self.level_at(point)
        value, grad, interface = self.interpolate(point, level)
        return value, -grad, level, interface


def force_report(model, points, reference, radii, heights):
    computed = model.acceleration(points)
    truth = np.array([item[1] for item in reference])
    error = np.linalg.norm(computed - truth, axis=1)
    norm = np.linalg.norm(truth, axis=1)
    # Report absolute errors everywhere; exclude zero forces from relative errors.
    relative = np.divide(error, norm, out=np.full_like(error, np.nan), where=norm > 0)
    local = (points - model.center) @ model.rotation
    radius = np.linalg.norm(local[:, :2], axis=1)
    interface = np.array([item[3] for item in reference], dtype=bool)

    def summary(mask):
        rel = relative[mask & np.isfinite(relative)]
        return {
            "count": int(mask.sum()),
            "zero_reference_count": int(np.sum(mask & (norm == 0))),
            "rms_absolute": float(np.sqrt(np.mean(error[mask] ** 2)))
            if np.any(mask)
            else None,
            "median_relative": float(np.median(rel)) if len(rel) else None,
            "p95_relative": float(np.percentile(rel, 95)) if len(rel) else None,
        }

    bins = []
    for ri, (rlo, rhi) in enumerate(itertools.pairwise(radii)):
        for zi, (zlo, zhi) in enumerate(itertools.pairwise(heights)):
            mask = (
                (radius >= rlo)
                & (radius <= rhi if ri == len(radii) - 2 else radius < rhi)
                & (local[:, 2] >= zlo)
                & (local[:, 2] <= zhi if zi == len(heights) - 2 else local[:, 2] < zhi)
            )
            bins.append(dict(radius=[rlo, rhi], height=[zlo, zhi], **summary(mask)))
    return {
        "all": summary(np.ones(len(points), dtype=bool)),
        "refinement_interface": summary(interface),
        "interior": summary(~interface),
        "bins": bins,
        "reference": "gradient of cell-center trilinear AMR potential; coarse prolongation at missing fine stencil centers",
        "temporal_error": "not assessed",
        "reference_discretization_error": "not removed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("plotfile", type=Path)
    parser.add_argument(
        "--points",
        required=True,
        type=Path,
        help="NPY (N,3) held-out world coordinates in native units",
    )
    parser.add_argument(
        "--radial-edges",
        type=lambda s: [float(x) for x in s.split(",")],
        required=True,
        help="comma-separated native radii",
    )
    parser.add_argument(
        "--height-edges",
        type=lambda s: [float(x) for x in s.split(",")],
        required=True,
        help="comma-separated signed native heights",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    model = PotentialModel.load(args.model)
    provenance = model.metadata["provenance"]
    if (
        hashlib.sha256((args.plotfile / "Header").read_bytes()).hexdigest()
        != provenance["header_sha256"]
    ):
        raise ValueError("reference Header does not match model provenance")
    points = np.load(args.points, allow_pickle=False)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("points must be finite nonempty (N,3)")
    for edges in (args.radial_edges, args.height_edges):
        if (
            len(edges) < 2
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError("diagnostic edges must be finite and increasing")
    from analysis import PlotfileReader

    reference = AMRReference(PlotfileReader(str(args.plotfile)), provenance["field"])
    samples = [reference.evaluate(p) for p in points]
    report = force_report(model, points, samples, args.radial_edges, args.height_edges)
    report.update(
        model=str(args.model),
        plotfile=str(args.plotfile),
        points=str(args.points),
        header_sha256=provenance["header_sha256"],
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {k: report[k] for k in ("all", "refinement_interface", "interior")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
