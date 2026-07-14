from __future__ import annotations


EXPECTED_DEFAULT_KERNELS = {
    "amr_subbox_fetch_pack",
    "cylindrical_flux_surface_integral_accumulate",
    "field_expr",
    "flux_surface_integral_accumulate",
    "gradU_stencil",
    "histogram1d_accumulate",
    "histogram2d_accumulate",
    "particle_abs_lt_mask",
    "particle_and_mask",
    "particle_cic_grid_accumulate",
    "particle_cic_projection_accumulate",
    "particle_count",
    "particle_distance3",
    "particle_eq_mask",
    "particle_filter",
    "particle_gt_mask",
    "particle_histogram1d",
    "particle_histogram1d_weighted",
    "particle_int64_sum_reduce",
    "particle_isfinite_mask",
    "particle_isin_mask",
    "particle_le_mask",
    "particle_len_f64",
    "particle_load_field_chunk_f64",
    "particle_max",
    "particle_min",
    "particle_scalar_max_reduce",
    "particle_scalar_min_reduce",
    "particle_subtract",
    "particle_sum",
    "particle_topk_modes_finalize",
    "particle_topk_modes_map",
    "particle_value_counts_reduce",
    "plotfile_load",
    "uniform_projection_accumulate",
    "uniform_slice",
    "uniform_slice_add",
    "uniform_slice_cellavg_accumulate",
    "uniform_slice_finalize",
    "uniform_slice_reduce",
    "vorticity_mag",
}


def test_default_kernel_catalog_is_complete_and_unique() -> None:
    from analysis import Runtime

    runtime = Runtime()
    descriptors = runtime.kernels.list()
    names = [descriptor.name for descriptor in descriptors]

    assert len(names) == len(set(names))
    assert set(names) == EXPECTED_DEFAULT_KERNELS
