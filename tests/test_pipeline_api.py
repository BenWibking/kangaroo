from __future__ import annotations

from dataclasses import dataclass

from analysis.pipeline import FieldHandle, Histogram1DHandle, Histogram2DHandle, Pipeline


class _FakeCoreRuntime:
    def __init__(self, *, num_localities: int = 1) -> None:
        self._next_field = 1
        self.persistent: dict[int, str] = {}
        self._num_localities = max(1, int(num_localities))

    def alloc_field_id(self, name: str) -> int:
        fid = self._next_field
        self._next_field += 1
        return fid

    def mark_field_persistent(self, fid: int, name: str) -> None:
        self.persistent[fid] = name

    def num_localities(self) -> int:
        return self._num_localities

    def chunk_home_rank(self, step: int, level: int, block: int) -> int:
        return int(block) % self._num_localities


class _FakeRuntime:
    def __init__(self, *, num_localities: int = 1) -> None:
        self._rt = _FakeCoreRuntime(num_localities=num_localities)
        self.submitted = []

    def run(self, plan, *, runmeta, dataset, progress_bar: bool = False) -> None:
        self.submitted.append((plan, runmeta, dataset, bool(progress_bar)))


class _LevelHomeCoreRuntime(_FakeCoreRuntime):
    def chunk_home_rank(self, step: int, level: int, block: int) -> int:
        return int(level) % self._num_localities


class _LevelHomeRuntime(_FakeRuntime):
    def __init__(self, *, num_localities: int = 1) -> None:
        self._rt = _LevelHomeCoreRuntime(num_localities=num_localities)
        self.submitted = []


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


def _single_level_two_block_runmeta() -> _RunMeta:
    return _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), index_origin=(0, 0, 0)),
                        boxes=[
                            _Box((0, 0, 0), (3, 7, 7)),
                            _Box((4, 0, 0), (7, 7, 7)),
                        ],
                    )
                ],
            )
        ]
    )


def _single_level_many_block_runmeta(nblocks: int) -> _RunMeta:
    return _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), index_origin=(0, 0, 0)),
                        boxes=[
                            _Box((i, 0, 0), (i, 0, 0))
                            for i in range(nblocks)
                        ],
                    )
                ],
            )
        ]
    )


def _multi_level_one_block_runmeta(nlevels: int) -> _RunMeta:
    return _RunMeta(
        steps=[
            _Step(
                step=0,
                levels=[
                    _Level(
                        geom=_Geom(dx=(1.0, 1.0, 1.0), x0=(0.0, 0.0, 0.0), index_origin=(0, 0, 0)),
                        boxes=[_Box((0, 0, 0), (0, 0, 0))],
                    )
                    for _ in range(nlevels)
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
    assert rt.submitted[0][3] is False


def test_pipeline_run_forwards_progress_bar_flag() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_runmeta()
    ds = _FakeDataset(rt)

    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)
    pipe.field_expr("a + b", {"a": pipe.field("x"), "b": pipe.field("y")}, out="sum")
    pipe.run(progress_bar=True)

    assert len(rt.submitted) == 1
    assert rt.submitted[0][3] is True


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


def test_pipeline_histogram1d_lowering_and_result_shape() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_two_block_runmeta()
    ds = _FakeDataset(rt)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)

    scalar = pipe.field("scalar")
    hist = pipe.histogram1d(
        scalar,
        hist_range=(0.0, 1.0),
        bins=8,
        out="hist_scalar",
    )
    assert isinstance(hist, Histogram1DHandle)
    assert hist.bins == 8
    assert len(hist.edges) == 9

    plan = pipe.plan()
    templates = [tmpl for stage in plan.topo_stages() for tmpl in stage.templates]
    acc = [tmpl for tmpl in templates if tmpl.kernel == "histogram1d_accumulate"]
    red = [tmpl for tmpl in templates if tmpl.kernel == "uniform_slice_reduce"]
    assert len(acc) == 2
    assert red
    assert all(tmpl.params["bins"] == 8 for tmpl in acc)
    assert all(tmpl.params["range"] == [0.0, 1.0] for tmpl in acc)


def test_pipeline_histogram2d_weighted_input_wiring() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_runmeta()
    ds = _FakeDataset(rt)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)

    x = pipe.field("density")
    y = pipe.field("temperature")
    w = pipe.field("mass")
    hist = pipe.histogram2d(
        x,
        y,
        x_range=(0.0, 2.0),
        y_range=(1.0, 3.0),
        bins=(4, 5),
        weights=w,
        out="phase",
    )
    assert isinstance(hist, Histogram2DHandle)
    x_edges, y_edges = hist.edges
    assert len(x_edges) == 5
    assert len(y_edges) == 6

    plan = pipe.plan()
    templates = [tmpl for stage in plan.topo_stages() for tmpl in stage.templates]
    acc = [tmpl for tmpl in templates if tmpl.kernel == "histogram2d_accumulate"]
    assert acc
    assert all(len(tmpl.inputs) == 3 for tmpl in acc)
    assert all(tmpl.params["bins"] == [4, 5] for tmpl in acc)


def test_pipeline_histogram2d_weight_mode_wiring() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_runmeta()
    ds = _FakeDataset(rt)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)

    x = pipe.field("density")
    y = pipe.field("temperature")
    pipe.histogram2d(
        x,
        y,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        bins=(8, 8),
        weight_mode="cell_mass",
        out="phase_mass",
    )
    plan = pipe.plan()
    acc = [
        tmpl
        for stage in plan.topo_stages()
        for tmpl in stage.templates
        if tmpl.kernel == "histogram2d_accumulate"
    ]
    assert acc
    assert all(len(tmpl.inputs) == 2 for tmpl in acc)
    assert all(tmpl.params["weight_mode"] == "cell_mass" for tmpl in acc)


def test_pipeline_particle_cic_projection_lowering_wiring() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_two_block_runmeta()
    ds = _FakeDataset(rt)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)

    out = pipe.particle_cic_projection(
        particle_type="StochasticStellarPop_particles",
        axis="z",
        axis_bounds=(0.0, 8.0),
        rect=(0.0, 0.0, 8.0, 8.0),
        resolution=(16, 16),
        out="stars",
    )
    assert isinstance(out, FieldHandle)

    plan = pipe.plan()
    templates = [tmpl for stage in plan.topo_stages() for tmpl in stage.templates]
    acc = [tmpl for tmpl in templates if tmpl.kernel == "particle_cic_projection_accumulate"]
    red = [tmpl for tmpl in templates if tmpl.kernel == "uniform_slice_reduce"]
    assert len(acc) == 1
    assert red
    assert acc[0].domain.blocks == [0, 1]
    assert all(tmpl.params["resolution"] == [16, 16] for tmpl in acc)
    assert all(tmpl.params["particle_type"] == "StochasticStellarPop_particles" for tmpl in acc)
    assert all(tmpl.params["level_index"] == 0 for tmpl in acc)


def test_pipeline_projection_reduce_carries_group_output_blocks() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_many_block_runmeta(16)
    ds = _FakeDataset(rt)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)

    scalar = pipe.field("scalar")
    pipe.uniform_projection(
        scalar,
        axis="z",
        axis_bounds=(0.0, 1.0),
        rect=(0.0, 0.0, 16.0, 1.0),
        resolution=(4, 4),
        out="proj",
    )

    plan = pipe.plan()
    reductions = [
        tmpl
        for stage in plan.topo_stages()
        for tmpl in stage.templates
        if tmpl.kernel == "uniform_slice_reduce"
        and tmpl.name.startswith("uniform_projection_sum_reduce")
    ]

    assert len(reductions) == 2
    assert reductions[0].params["fan_in"] == 4
    assert reductions[0].params["input_blocks"] == list(range(16))
    assert reductions[0].params["output_blocks"] == [0, 4, 8, 12]
    assert reductions[0].params["group_offsets"] == [0, 4, 8, 12, 16]
    assert reductions[1].params["input_blocks"] == [0, 4, 8, 12]
    assert reductions[1].params["output_blocks"] == [0]
    assert reductions[1].params["group_offsets"] == [0, 4]


def test_pipeline_projection_reduce_groups_by_locality() -> None:
    rt = _FakeRuntime(num_localities=2)
    runmeta = _single_level_many_block_runmeta(16)
    ds = _FakeDataset(rt)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)

    scalar = pipe.field("scalar")
    pipe.uniform_projection(
        scalar,
        axis="z",
        axis_bounds=(0.0, 1.0),
        rect=(0.0, 0.0, 16.0, 1.0),
        resolution=(4, 4),
        out="proj",
    )

    plan = pipe.plan()
    reductions = [
        tmpl
        for stage in plan.topo_stages()
        for tmpl in stage.templates
        if tmpl.kernel == "uniform_slice_reduce"
        and tmpl.name.startswith("uniform_projection_sum_reduce")
    ]

    assert len(reductions) == 3
    assert reductions[0].params["fan_in"] == 4
    assert reductions[0].params["input_blocks"] == [
        0, 2, 4, 6,
        8, 10, 12, 14,
        1, 3, 5, 7,
        9, 11, 13, 15,
    ]
    assert reductions[0].params["output_blocks"] == [0, 8, 1, 9]
    assert reductions[0].params["group_offsets"] == [0, 4, 8, 12, 16]
    assert reductions[1].params["input_blocks"] == [0, 8, 1, 9]
    assert reductions[1].params["output_blocks"] == [0, 1]
    assert reductions[1].params["group_offsets"] == [0, 2, 4]
    assert reductions[2].params["input_blocks"] == [0, 1]
    assert reductions[2].params["output_blocks"] == [0]
    assert reductions[2].params["group_offsets"] == [0, 2]


def test_pipeline_projection_final_adds_group_levels_by_locality() -> None:
    rt = _LevelHomeRuntime(num_localities=2)
    runmeta = _multi_level_one_block_runmeta(4)
    ds = _FakeDataset(rt)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)

    scalar = pipe.field("scalar")
    pipe.uniform_projection(
        scalar,
        axis="z",
        axis_bounds=(0.0, 1.0),
        rect=(0.0, 0.0, 1.0, 1.0),
        resolution=(4, 4),
        out="proj",
    )

    plan = pipe.plan()
    adds = [
        tmpl
        for stage in plan.topo_stages()
        for tmpl in stage.templates
        if tmpl.kernel == "uniform_slice_add"
        and tmpl.name.startswith("uniform_projection_add")
    ]

    assert [[ref.domain.level for ref in tmpl.inputs] for tmpl in adds[:2]] == [[2, 0], [3, 1]]
    assert [tmpl.domain.level for tmpl in adds[:2]] == [2, 3]
    assert [tmpl.params["output_blocks"] for tmpl in adds[:2]] == [[0], [0]]


def test_pipeline_field_expr_lowering_wiring() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_runmeta()
    ds = _FakeDataset(rt)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)

    rho = pipe.field("density")
    mx = pipe.field("xmom")
    vx = pipe.field_expr("mx / rho", {"mx": mx, "rho": rho}, out="velx")
    assert isinstance(vx, FieldHandle)

    plan = pipe.plan()
    expr_templates = [
        tmpl
        for stage in plan.topo_stages()
        for tmpl in stage.templates
        if tmpl.kernel == "field_expr"
    ]
    assert expr_templates
    assert all(tmpl.params["expression"] == "mx / rho" for tmpl in expr_templates)
    assert all(tmpl.params["variables"] == ["mx", "rho"] for tmpl in expr_templates)


def test_pipeline_register_derived_field_cached() -> None:
    rt = _FakeRuntime()
    runmeta = _single_level_runmeta()
    ds = _FakeDataset(rt)
    pipe = Pipeline(runtime=rt, runmeta=runmeta, dataset=ds)

    pipe.register_derived_field(
        "velocity_x",
        lambda p: p.field_expr("mx / rho", {"mx": p.field("xmom"), "rho": p.field("density")}, out="velocity_x"),
    )
    v1 = pipe.derived_field("velocity_x")
    v2 = pipe.derived_field("velocity_x")
    v3 = pipe.field("velocity_x")

    assert isinstance(v1, FieldHandle)
    assert v1.field == v2.field
    assert v1.field == v3.field

    plan = pipe.plan()
    expr_templates = [
        tmpl
        for stage in plan.topo_stages()
        for tmpl in stage.templates
        if tmpl.kernel == "field_expr"
    ]
    assert len(expr_templates) == 1
