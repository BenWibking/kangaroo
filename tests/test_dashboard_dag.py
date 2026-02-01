from __future__ import annotations

from typing import Dict, Tuple

from analysis.dashboard import DashboardApp, DashboardConfig


def _edge_names(app: DashboardApp) -> set[Tuple[str, str]]:
    node_data = app._dag_source.data
    edge_data = app._dag_edge_source.data
    coords_to_name: Dict[Tuple[float, float], str] = {}
    for x, y, name in zip(node_data["x"], node_data["y"], node_data["name"]):
        coords_to_name[(x, y)] = name

    edges = set()
    for x0, y0, x1, y1 in zip(
        edge_data["x0"], edge_data["y0"], edge_data["x1"], edge_data["y1"]
    ):
        parent = coords_to_name[(x0, y0)]
        child = coords_to_name[(x1, y1)]
        edges.add((parent, child))
    return edges


def test_dag_uses_input_domain_for_block_edges() -> None:
    app = DashboardApp(DashboardConfig())
    stages = [
        {
            "name": "load",
            "plane": "chunk",
            "after": [],
            "templates": [
                {
                    "name": "load",
                    "plane": "chunk",
                    "kernel": "plotfile_load",
                    "domain": {"step": 0, "level": 0, "blocks": [0, 1]},
                    "inputs": [],
                    "outputs": [{"field": 1, "version": 0}],
                    "output_bytes": [],
                    "deps": {"kind": "None"},
                    "params": {},
                }
            ],
        },
        {
            "name": "slice",
            "plane": "chunk",
            "after": [0],
            "templates": [
                {
                    "name": "slice",
                    "plane": "chunk",
                    "kernel": "uniform_slice",
                    "domain": {"step": 0, "level": 0, "blocks": [0, 1]},
                    "inputs": [
                        {
                            "field": 1,
                            "version": 0,
                            "domain": {"step": 0, "level": 0, "blocks": [0, 1]},
                        }
                    ],
                    "outputs": [{"field": 2, "version": 0}],
                    "output_bytes": [],
                    "deps": {"kind": "None"},
                    "params": {},
                }
            ],
        },
    ]

    app._update_dag_sources(stages)
    edges = _edge_names(app)

    assert ("load:b0", "slice:b0") in edges
    assert ("load:b1", "slice:b1") in edges
    assert ("load:b0", "slice:b1") not in edges
    assert ("load:b1", "slice:b0") not in edges
