from __future__ import annotations

import numpy as np
import pytest

from analysis import _core
from analysis.buffer import (
    BlockShape,
    BufferSpec,
    DType,
    DynamicShape,
    DynamicUpperBound,
    FixedShape,
    LikeInputShape,
    dtype_from_numpy,
    numpy_dtype,
)
from analysis.dataset import Dataset


@pytest.mark.parametrize(
    ("dtype", "size"),
    [("u8", 1), ("i64", 8), ("f32", 4), ("f64", 8)],
)
def test_contiguous_descriptors(dtype: str, size: int) -> None:
    desc = _core.test_chunk_buffer_descriptor(dtype, [2, 3], "contiguous")
    assert desc == {
        "dtype": dtype,
        "rank": 2,
        "extents": [2, 3],
        "strides_bytes": [3 * size, size],
        "elements": 6,
        "bytes": 6 * size,
    }


@pytest.mark.parametrize("extents", [(2, 1, 1), (2, 1, 3)])
def test_contiguous_descriptors_accept_singleton_axes(
    extents: tuple[int, ...],
) -> None:
    desc = _core.test_chunk_buffer_descriptor("f64", list(extents), "contiguous")
    assert desc["extents"] == list(extents)
    assert desc["elements"] == 2 * extents[-1]


def test_opaque_descriptor_is_rank_one_bytes() -> None:
    assert _core.test_chunk_buffer_descriptor("opaque", [7], "contiguous")["bytes"] == 7


def test_plotfile_and_runtime_layouts_have_equal_logical_values() -> None:
    values = _core.test_chunk_buffer_layout_values()
    assert len(values) == 24
    assert all(runtime == plotfile for runtime, plotfile in values)


def test_grid_region_copy_uses_logical_indices_across_layouts() -> None:
    assert _core.test_chunk_buffer_grid_region() == [110, 111, 120, 121, 210, 211, 220, 221]


def test_layout_copy_converts_scalar_type() -> None:
    assert _core.test_chunk_buffer_layout_copy_converts_dtype() == (1.25, 2.5, 3.75, 5.0)


def test_amr_patch_codec_roundtrips_geometry_descriptor_and_data() -> None:
    assert _core.test_amr_patch_codec_roundtrip() == (1, 2, 4, 0.125, True, 7.5, "f32")


def test_amr_patch_codec_rejects_malformed_payload() -> None:
    with pytest.raises(RuntimeError, match="root must be a map"):
        _core.test_amr_patch_codec_rejects_malformed()


def test_mutating_a_copy_detaches_storage() -> None:
    assert _core.test_chunk_buffer_cow() == (7, 11)


def test_init_policy_selects_uninitialized_or_zeroed_storage() -> None:
    assert _core.test_chunk_buffer_init_policy(4096) == (True, False, True)


def test_dynamic_extent_commit() -> None:
    assert _core.test_chunk_buffer_dynamic(10, 3) == (3, 24, 80)
    with pytest.raises(RuntimeError, match="upper bound"):
        _core.test_chunk_buffer_dynamic(2, 3)


def test_dynamic_typed_writer_commits_only_visible_values() -> None:
    assert _core.test_chunk_buffer_dynamic_write(5, [1.5, 2.5]) == [1.5, 2.5]
    with pytest.raises(IndexError, match="TensorView index out of range"):
        _core.test_chunk_buffer_dynamic_write(1, [1.0, 2.0])


@pytest.mark.parametrize("extent", [0, 3])
def test_committed_dynamic_buffer_roundtrips_by_visible_size(extent: int) -> None:
    visible_bytes = extent * 8
    assert _core.test_chunk_buffer_dynamic_roundtrip(10, extent) == (
        extent,
        visible_bytes,
        visible_bytes,
    )


@pytest.mark.parametrize(
    ("alias", "canonical", "numpy_type"),
    [
        ("uint8", "u8", np.uint8),
        ("int64", "i64", np.int64),
        ("float32", "f32", np.float32),
        ("float64", "f64", np.float64),
    ],
)
def test_dataset_set_chunk_accepts_dtype_aliases(
    alias: str, canonical: str, numpy_type: type[np.generic]
) -> None:
    class Handle:
        calls: list[tuple[object, ...]] = []

        def set_chunk(self, *args: object) -> None:
            self.calls.append(args)

    dataset = Dataset.__new__(Dataset)
    dataset._h = Handle()
    array = np.asarray([1, 2], dtype=numpy_type)

    dataset.set_chunk(field=1, block=2, data=array, dtype=alias)
    dataset.set_chunk(
        field=1,
        block=2,
        data=array.tobytes(),
        dtype=alias,
        shape=array.shape,
    )

    assert [call[4] for call in dataset._h.calls] == [canonical, canonical]


def test_backend_chunk_bounds_follow_particle_records_and_reduce_inputs() -> None:
    records = 1_048_577
    assert _core.test_backend_chunk_dynamic_capacity(
        "f64", "particle_load_field_chunk_f64", records, []
    ) == records
    assert _core.test_backend_chunk_dynamic_capacity(
        "opaque", "particle_topk_modes_map", records, []
    ) == 8 + 16 * records

    input_bytes = [6 << 20, 4 << 20]
    assert _core.test_backend_chunk_dynamic_capacity(
        "opaque", "particle_value_counts_reduce", 0, input_bytes
    ) == sum(input_bytes)


def test_amr_subbox_bound_follows_requested_source_payload() -> None:
    source_bytes = 65 << 20
    capacity = _core.test_amr_subbox_dynamic_capacity(source_bytes)
    assert capacity > source_bytes
    assert capacity > 64 << 20


def test_buffer_specs_encode_closed_shape_language() -> None:
    assert BufferSpec(DType.F64, BlockShape(3)).to_dict() == {
        "dtype": "f64",
        "shape": {"kind": "block", "components": 3},
        "init": "uninitialized",
    }
    assert BufferSpec(DType.F32, FixedShape((8, 9))).to_dict()["shape"] == {
        "kind": "fixed",
        "extents": [8, 9],
    }
    assert BufferSpec(DType.U8, LikeInputShape(1)).to_dict()["shape"] == {
        "kind": "like_input",
        "input_index": 1,
    }
    assert BufferSpec(
        DType.OPAQUE,
        DynamicShape(DynamicUpperBound.literal(128)),
    ).to_dict()["shape"] == {
        "kind": "dynamic",
        "upper_bound": {"kind": "literal", "value": 128},
    }


def test_shared_numpy_dtype_mapping() -> None:
    assert numpy_dtype(DType.F32).name == "float32"
    assert dtype_from_numpy("int64") is DType.I64
    with pytest.raises(TypeError, match="opaque"):
        numpy_dtype(DType.OPAQUE)
