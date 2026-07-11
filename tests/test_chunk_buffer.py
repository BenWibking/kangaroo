from __future__ import annotations

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


def test_opaque_descriptor_is_rank_one_bytes() -> None:
    assert _core.test_chunk_buffer_descriptor("opaque", [7], "contiguous")["bytes"] == 7


def test_plotfile_and_runtime_layouts_have_equal_logical_values() -> None:
    values = _core.test_chunk_buffer_layout_values()
    assert len(values) == 24
    assert all(runtime == plotfile for runtime, plotfile in values)


def test_mutating_a_copy_detaches_storage() -> None:
    assert _core.test_chunk_buffer_cow() == (7, 11)


def test_init_policy_selects_uninitialized_or_zeroed_storage() -> None:
    assert _core.test_chunk_buffer_init_policy(4096) == (True, False, True)


def test_dynamic_extent_commit() -> None:
    assert _core.test_chunk_buffer_dynamic(10, 3) == (3, 24, 80)
    with pytest.raises(RuntimeError, match="upper bound"):
        _core.test_chunk_buffer_dynamic(2, 3)


@pytest.mark.parametrize("extent", [0, 3])
def test_committed_dynamic_buffer_roundtrips_by_visible_size(extent: int) -> None:
    visible_bytes = extent * 8
    assert _core.test_chunk_buffer_dynamic_roundtrip(10, extent) == (
        extent,
        visible_bytes,
        visible_bytes,
    )


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
