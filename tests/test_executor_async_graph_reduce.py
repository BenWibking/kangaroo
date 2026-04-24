from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_many_graph_reduce_consumers_wait_for_later_producer_without_deadlock() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import struct

        from analysis import Runtime
        from analysis.dataset import open_dataset
        from analysis.plan import Domain, FieldRef, Plan, Stage
        from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta

        rt = Runtime()
        runmeta = RunMeta(
            steps=[
                StepMeta(
                    step=0,
                    levels=[
                        LevelMeta(
                            geom=LevelGeom(
                                dx=(1.0, 1.0, 1.0),
                                x0=(0.0, 0.0, 0.0),
                                ref_ratio=1,
                            ),
                            boxes=[BlockBox((0, 0, 0), (0, 0, 0))],
                        )
                    ],
                )
            ]
        )
        ds = open_dataset("memory://async-graph-reduce", runmeta=runmeta, step=0, level=0, runtime=rt)

        source_field = 31001
        produced_field = 31002
        reduced_field = 31003
        consumer_count = 96
        value = 123456789

        ds._h.set_chunk_ref(0, 0, source_field, 0, 0, struct.pack("<q", value))

        # This stage is deliberately independent and first in the plan. It creates
        # many consumers of the same not-yet-produced chunk before the producer
        # stage below publishes that chunk.
        graph = Stage(name="early_graph_consumers", plane="graph")
        graph.map_blocks(
            name="same_chunk_consumers",
            kernel="particle_int64_sum_reduce",
            domain=Domain(step=0, level=0),
            inputs=[FieldRef(produced_field)],
            outputs=[FieldRef(reduced_field)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={
                "graph_kind": "reduce",
                "fan_in": 1,
                "num_inputs": consumer_count,
                "input_blocks": [0] * consumer_count,
                "output_base": 0,
            },
        )

        producer = Stage(name="late_producer", plane="chunk")
        producer.map_blocks(
            name="publish_shared_input",
            kernel="particle_int64_sum_reduce",
            domain=Domain(step=0, level=0, blocks=[0]),
            inputs=[FieldRef(source_field)],
            outputs=[FieldRef(produced_field)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={},
        )

        rt.run(Plan(stages=[graph, producer]), runmeta=runmeta, dataset=ds)

        for block in range(consumer_count):
            raw = rt.get_task_chunk(
                step=0,
                level=0,
                field=reduced_field,
                version=0,
                block=block,
                dataset=ds,
            )
            got = struct.unpack("<q", raw)[0]
            if got != value:
                raise AssertionError((block, got, value))
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=True,
        timeout=30,
    )


def test_graph_reduce_group_offsets_select_variable_input_groups() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import struct

        from analysis import Runtime
        from analysis.dataset import open_dataset
        from analysis.plan import Domain, FieldRef, Plan, Stage
        from analysis.runmeta import BlockBox, LevelGeom, LevelMeta, RunMeta, StepMeta

        rt = Runtime()
        runmeta = RunMeta(
            steps=[
                StepMeta(
                    step=0,
                    levels=[
                        LevelMeta(
                            geom=LevelGeom(
                                dx=(1.0, 1.0, 1.0),
                                x0=(0.0, 0.0, 0.0),
                                ref_ratio=1,
                            ),
                            boxes=[
                                BlockBox((0, 0, 0), (0, 0, 0)),
                                BlockBox((1, 0, 0), (1, 0, 0)),
                                BlockBox((2, 0, 0), (2, 0, 0)),
                                BlockBox((3, 0, 0), (3, 0, 0)),
                            ],
                        )
                    ],
                )
            ]
        )
        ds = open_dataset("memory://graph-reduce-offsets", runmeta=runmeta, step=0, level=0, runtime=rt)

        source_field = 32001
        reduced_field = 32002
        for block, value in enumerate([1, 2, 3, 4]):
            ds._h.set_chunk_ref(0, 0, source_field, 0, block, struct.pack("<q", value))

        graph = Stage(name="offset_reduce", plane="graph")
        graph.map_blocks(
            name="offset_groups",
            kernel="particle_int64_sum_reduce",
            domain=Domain(step=0, level=0),
            inputs=[FieldRef(source_field)],
            outputs=[FieldRef(reduced_field)],
            output_bytes=[8],
            deps={"kind": "None"},
            params={
                "graph_kind": "reduce",
                "fan_in": 4,
                "num_inputs": 4,
                "input_blocks": [0, 2, 1, 3],
                "output_blocks": [10, 20],
                "group_offsets": [0, 2, 4],
                "output_base": 0,
            },
        )

        rt.run(Plan(stages=[graph]), runmeta=runmeta, dataset=ds)

        got_10 = struct.unpack(
            "<q",
            rt.get_task_chunk(step=0, level=0, field=reduced_field, version=0, block=10, dataset=ds),
        )[0]
        got_20 = struct.unpack(
            "<q",
            rt.get_task_chunk(step=0, level=0, field=reduced_field, version=0, block=20, dataset=ds),
        )[0]
        if (got_10, got_20) != (4, 6):
            raise AssertionError((got_10, got_20))
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=True,
        timeout=30,
    )
