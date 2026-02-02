#include "kangaroo/backend_plotfile.hpp"

#include <iostream>

namespace kangaroo {

namespace {

template <typename InT, typename OutT>
void transpose_plotfile_axes(const InT* in, OutT* out, int nx, int ny, int nz) {
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t in_idx = (static_cast<std::size_t>(k) * ny + j) * nx + i;
        const std::size_t out_idx = (static_cast<std::size_t>(i) * ny + j) * nz + k;
        out[out_idx] = static_cast<OutT>(in[in_idx]);
      }
    }
  }
}

}  // namespace

PlotfileBackend::PlotfileBackend(std::string plotfile_dir)
    : plotfile_dir_(std::move(plotfile_dir)), reader_(plotfile_dir_) {}

std::optional<HostView> PlotfileBackend::get_chunk(const ChunkRef& ref) {
  if (ref.level < 0 || ref.level >= reader_.num_levels()) {
    return std::nullopt;
  }
  if (ref.block < 0 || ref.block >= reader_.num_fabs(ref.level)) {
    return std::nullopt;
  }

  int32_t comp = get_component_index(ref.field);
  if (comp < 0) {
    // Field not registered or not found
    return std::nullopt;
  }

  try {
    auto fab = reader_.read_fab(ref.level, ref.block, comp, 1);
    HostView view;
    // Resize view to match size (bytes match, layout differs)
    view.data.resize(fab.bytes.size());

    if (fab.type == plotfile::RealType::kFloat32) {
        const auto* in = reinterpret_cast<const float*>(fab.bytes.data());
        auto* out = reinterpret_cast<float*>(view.data.data());
        transpose_plotfile_axes(in, out, fab.nx, fab.ny, fab.nz);
    } else {
        const auto* in = reinterpret_cast<const double*>(fab.bytes.data());
        auto* out = reinterpret_cast<double*>(view.data.data());
        transpose_plotfile_axes(in, out, fab.nx, fab.ny, fab.nz);
    }
    
    return view;
  } catch (const std::exception& e) {
    std::cerr << "PlotfileBackend: failed to read chunk: " << e.what() << "\n";
    return std::nullopt;
  }
}

bool PlotfileBackend::has_chunk(const ChunkRef& ref) const {
  if (ref.level < 0 || ref.level >= reader_.num_levels()) {
    return false;
  }
  if (ref.block < 0 || ref.block >= reader_.num_fabs(ref.level)) {
    return false;
  }
  return get_component_index(ref.field) >= 0;
}

DatasetMetadata PlotfileBackend::get_metadata() const {
  DatasetMetadata meta;
  const auto& h = reader_.header();
  meta.prob_lo = h.prob_lo;
  meta.prob_hi = h.prob_hi;
  meta.ref_ratio = h.ref_ratio;
  return meta;
}

void PlotfileBackend::register_field(int32_t field_id, int32_t component_index) {
  std::lock_guard<std::mutex> lock(map_mutex_);
  field_map_[field_id] = component_index;
}

int32_t PlotfileBackend::get_component_index(int32_t field_id) const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  auto it = field_map_.find(field_id);
  if (it != field_map_.end()) {
    return it->second;
  }
  return -1;
}

}  // namespace kangaroo
