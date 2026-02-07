from __future__ import annotations

from dataclasses import dataclass

from analysis.pipeline import FieldHandle, Pipeline


class _FakeCoreRuntime:
    def __init__(self) -> None:
        self._next_field = 1
        self.persistent: dict[int, str] = {}

    def alloc_field_id(self, name: str) -> int:
        fid = self._next_field
        self._next_field += 1
        return fid

    def mark_field_persistent(self, fid: int, name: str) -> None:
        self.persistent[fid] = name


class _FakeRuntime:
    def __init__(self) -> None:
        self._rt = _FakeCoreRuntime()
        self.submitted = []

    def run(self, plan, *, runmeta, dataset) -> None:
        self.submitted.append((plan, runmeta, dataset))


class _FakeDataset:
    def __init__(self, runtime: _FakeRuntime, *, step: int = 0, level: int = 0) -> None:
        self.runtime = runtime
        self.step = step
        self.level = level
        self._fields: dict[str, int] = {}

    def field_id(self, name: str) -> int:
        if name not in self._fields:
            self._fields[name] = self.runtime._rt.alloc_field_id(name)
        return self._fields[name]


@dataclass(frozen=True)
class _Box:
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]


@dataclass(frozen=True)
class _Geom:
    dx: tuple[float, float, float]
    x0: tuple[float, float, float]
    index_origin: tuple[int, int, int]
    ref_ratio: int = 1


@dataclass(frozen=True)
class _Level:
    geom: _Geom
    boxes: list[_Box]


@dataclass(frozen=True)
class _Step:
    step: int
    levels: list[_Level]


@dataclass(frozen=True)
class _RunMeta:
    steps: list[_Step]


def _single_level_runmeta() -> _RunMeta:
    return _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), index_origin=(0, 0, 0)),
                        boxes=[_Box((0, 0, 0), (7, 7, 7))],
                    )
                ],
            )
        ]
    )


def test_pipeline_vorticity_fragment_and_run_submission() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_runmeta()
    ds = _FakeDataset(rt)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    u = pipe.field("vel_x")
    v = pipe.field("vel_y")
    w = pipe.field("vel_z")

    vort = pipe.vorticity_mag((u, v, w), out="vort")
    assert isinstance(vort, FieldHandle)

    plan = pipe.plan()
    topo = plan.topo_stages()
    assert [stage.name for stage in topo] == ["neighbor_fetch", "gradients", "vortmag"]
    assert topo[1].after == [topo[0]]
    assert topo[2].after == [topo[1]]

    pipe.run()
    assert len(rt.submitted) == 1


def test_pipeline_imperative_chaining_adds_cross_fragment_edge() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_runmeta()
    ds = _FakeDataset(rt)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    u = pipe.field("vel_x")
    v = pipe.field("vel_y")
    w = pipe.field("vel_z")

    vort = pipe.vorticity_mag((u, v, w), out="vort")
    sliced = pipe.uniform_slice(
        vort,
        axis="z",
        coord=0.5,
        rect=(0.0, 0.0, 8.0, 8.0),
        resolution=(16, 16),
        out="slice_vort",
    )
    assert isinstance(sliced, FieldHandle)

    plan = pipe.plan()
    topo = plan.topo_stages()
    vort_stage = next(stage for stage in topo if stage.name == "vortmag")
    slice_stage = next(stage for stage in topo if stage.name == "uniform_slice")
    assert vort_stage in slice_stage.after
