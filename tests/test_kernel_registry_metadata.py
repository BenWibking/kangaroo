from analysis.runtime import Runtime


def test_kernel_registry_exposes_descriptor_metadata() -> None:
    runtime = Runtime()
    kernels = {desc.name: desc for desc in runtime.kernels.list()}

    assert kernels["plotfile_load"].n_inputs == 0
    assert kernels["plotfile_load"].n_outputs == 1
    assert kernels["plotfile_load"].needs_neighbors is False

    assert kernels["particle_eq_mask"].n_inputs == 1
    assert kernels["particle_eq_mask"].n_outputs == 1
    assert kernels["particle_eq_mask"].param_schema_json == ""

    assert kernels["uniform_slice_finalize"].n_inputs == 2
    assert kernels["uniform_slice_finalize"].n_outputs == 1
    assert kernels["uniform_slice_finalize"].needs_neighbors is False

    assert kernels["gradU_stencil"].n_inputs == 1
    assert kernels["gradU_stencil"].n_outputs == 1
    assert kernels["gradU_stencil"].needs_neighbors is False
