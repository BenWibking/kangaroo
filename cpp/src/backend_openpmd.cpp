#include "kangaroo/backend_openpmd.hpp"

#ifdef KANGAROO_USE_OPENPMD

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <sstream>

namespace kangaroo {

namespace {

bool approx_equal(float a, float b) {
  return std::abs(a - b) < 1e-5f;
}

std::string to_lower(std::string value) {
  for (auto& c : value) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return value;
}

}  // namespace

OpenPMDBackend::OpenPMDBackend(std::string uri) : uri_(std::move(uri)) {
  constexpr const char* prefix = "openpmd://";
  if (uri_.rfind(prefix, 0) == 0) {
    std::string rest = uri_.substr(std::string(prefix).size());
    auto pos = rest.find('?');
    if (pos != std::string::npos) {
      path_ = rest.substr(0, pos);
      // Ignore query parameters; mesh selection is driven by Python API.
    } else {
      path_ = rest;
    }
  } else {
    path_ = uri_;
  }

  series_ = std::make_unique<openPMD::Series>(path_, openPMD::Access::READ_ONLY);
  iteration_indices_.reserve(series_->iterations.size());
  for (const auto& entry : series_->iterations) {
    iteration_indices_.push_back(entry.first);
  }
  std::sort(iteration_indices_.begin(), iteration_indices_.end());
}

std::optional<HostView> OpenPMDBackend::get_chunk(const ChunkRef& ref) {
  const auto& cache = get_cache(ref.step);
  if (ref.level < 0 || ref.level >= static_cast<int32_t>(cache.levels.size())) {
    return std::nullopt;
  }
  const auto& level = cache.levels.at(ref.level);
  if (ref.block < 0 || ref.block >= static_cast<int32_t>(level.patches.size())) {
    return std::nullopt;
  }

  FieldSpec spec;
  {
    std::lock_guard<std::mutex> lock(field_mutex_);
    auto it = field_map_.find(ref.field);
    if (it == field_map_.end()) {
      return std::nullopt;
    }
    spec = it->second;
  }

  if (iteration_indices_.empty()) {
    return std::nullopt;
  }
  if (ref.step < 0 || ref.step >= static_cast<int32_t>(iteration_indices_.size())) {
    return std::nullopt;
  }

  try {
    auto iteration = series_->iterations.at(iteration_indices_.at(ref.step));
    const auto& patch = level.patches.at(ref.block);
    auto mesh_it = iteration.meshes.find(patch.mesh_name);
    if (mesh_it == iteration.meshes.end()) {
      return std::nullopt;
    }
    openPMD::Mesh mesh = mesh_it->second;
    if (!mesh.scalar() && spec.component_name.empty()) {
      return std::nullopt;
    }
    openPMD::MeshRecordComponent component =
        mesh.scalar() ? mesh[openPMD::MeshRecordComponent::SCALAR]
                      : mesh[spec.component_name];

    const uint64_t nx = patch.extent_xyz[0] == 0 ? 1 : patch.extent_xyz[0];
    const uint64_t ny = patch.extent_xyz[1] == 0 ? 1 : patch.extent_xyz[1];
    const uint64_t nz = patch.extent_xyz[2] == 0 ? 1 : patch.extent_xyz[2];
    const uint64_t elem_count = nx * ny * nz;
    if (elem_count == 0) {
      return std::nullopt;
    }

    HostView view;
    if (component.getDatatype() == openPMD::Datatype::DOUBLE) {
      view.data.resize(static_cast<size_t>(elem_count * sizeof(double)));
      auto* data = reinterpret_cast<double*>(view.data.data());
      component.loadChunkRaw(data, patch.storage_offset, patch.storage_extent);
      series_->flush();
      scale_values<double>(data, static_cast<size_t>(elem_count), component.unitSI());
      remap_to_xyz_layout<double>(data, static_cast<size_t>(elem_count), patch);
      return view;
    }
    if (component.getDatatype() == openPMD::Datatype::FLOAT) {
      view.data.resize(static_cast<size_t>(elem_count * sizeof(float)));
      auto* data = reinterpret_cast<float*>(view.data.data());
      component.loadChunkRaw(data, patch.storage_offset, patch.storage_extent);
      series_->flush();
      scale_values<float>(data, static_cast<size_t>(elem_count), component.unitSI());
      remap_to_xyz_layout<float>(data, static_cast<size_t>(elem_count), patch);
      return view;
    }

    std::cerr << "OpenPMDBackend: unsupported datatype for field\n";
  } catch (const std::exception& e) {
    std::cerr << "OpenPMDBackend: failed to read chunk: " << e.what() << "\n";
  }

  return std::nullopt;
}

bool OpenPMDBackend::has_chunk(const ChunkRef& ref) const {
  const auto& cache = get_cache(ref.step);
  if (ref.level < 0 || ref.level >= static_cast<int32_t>(cache.levels.size())) {
    return false;
  }
  const auto& level = cache.levels.at(ref.level);
  if (ref.block < 0 || ref.block >= static_cast<int32_t>(level.patches.size())) {
    return false;
  }
  std::lock_guard<std::mutex> lock(field_mutex_);
  return field_map_.find(ref.field) != field_map_.end();
}

DatasetMetadata OpenPMDBackend::get_metadata() const {
  DatasetMetadata meta;
  const auto& cache = get_cache(0);
  meta.prob_lo.assign(cache.prob_lo.begin(), cache.prob_lo.end());
  meta.prob_hi.assign(cache.prob_hi.begin(), cache.prob_hi.end());
  meta.ref_ratio.reserve(cache.ref_ratio.size());
  for (const auto& ratio : cache.ref_ratio) {
    meta.ref_ratio.push_back(ratio[0]);
  }
  return meta;
}

OpenPMDMetadata OpenPMDBackend::metadata(int32_t step) const {
  const auto& cache = get_cache(step);
  OpenPMDMetadata out;
  out.selected_mesh = cache.selected_mesh;
  out.mesh_names = cache.mesh_names;
  out.fields = cache.fields;
  out.finest_level = cache.finest_level;
  out.prob_lo = cache.prob_lo;
  out.prob_hi = cache.prob_hi;
  out.ref_ratio = cache.ref_ratio;
  out.cell_size.reserve(cache.levels.size());
  out.level_boxes.reserve(cache.levels.size());
  out.prob_domain.reserve(cache.levels.size());

  for (const auto& level : cache.levels) {
    out.cell_size.push_back(level.cell_size);
    std::vector<std::pair<std::array<int32_t, 3>, std::array<int32_t, 3>>> boxes;
    boxes.reserve(level.patches.size());
    for (const auto& patch : level.patches) {
      boxes.emplace_back(patch.logical_lower, patch.logical_upper);
    }
    out.level_boxes.push_back(std::move(boxes));
    out.prob_domain.push_back({level.domain_lo, level.domain_hi});
  }

  return out;
}

void OpenPMDBackend::register_field(int32_t field_id, const std::string& name) {
  FieldSpec spec;
  auto sep = name.find('/');
  if (sep == std::string::npos) {
    spec.component_name.clear();
  } else {
    spec.component_name = name.substr(sep + 1);
  }
  std::lock_guard<std::mutex> lock(field_mutex_);
  field_map_[field_id] = std::move(spec);
}

std::vector<std::string> OpenPMDBackend::list_meshes(int32_t step) const {
  const auto& cache = get_cache(step);
  return cache.mesh_names;
}

void OpenPMDBackend::select_mesh(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    selected_mesh_ = name;
    cache_by_step_.clear();
  }
  std::lock_guard<std::mutex> lock(field_mutex_);
  field_map_.clear();
}

const OpenPMDBackend::Cache& OpenPMDBackend::get_cache(int32_t step) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  auto it = cache_by_step_.find(step);
  if (it != cache_by_step_.end()) {
    return it->second;
  }
  Cache built = build_cache(step);
  auto [inserted_it, inserted] = cache_by_step_.emplace(step, std::move(built));
  return inserted_it->second;
}

OpenPMDBackend::Cache OpenPMDBackend::build_cache(int32_t step) const {
  Cache cache;
  if (iteration_indices_.empty()) {
    return cache;
  }

  if (step < 0 || step >= static_cast<int32_t>(iteration_indices_.size())) {
    return cache;
  }

  auto iteration = series_->iterations.at(iteration_indices_.at(step));

  std::unordered_map<std::string, std::vector<std::pair<int, std::string>>> groups;
  for (const auto& entry : iteration.meshes) {
    auto parsed = parse_mesh_level(entry.first);
    groups[parsed.first].push_back({parsed.second, entry.first});
  }

  if (groups.empty()) {
    return cache;
  }

  std::string selected = selected_mesh_;
  if (selected.empty()) {
    selected = groups.begin()->first;
  } else {
    auto parsed = parse_mesh_level(selected);
    if (groups.find(parsed.first) == groups.end()) {
      selected = groups.begin()->first;
    } else {
      selected = parsed.first;
    }
  }

  auto& levels = groups.at(selected);
  std::sort(levels.begin(), levels.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  cache.selected_mesh = selected;
  cache.mesh_names.reserve(groups.size());
  for (const auto& entry : groups) {
    cache.mesh_names.push_back(entry.first);
  }

  cache.levels.reserve(levels.size());
  for (const auto& entry : levels) {
    LevelInfo level_info;
    level_info.mesh_name = entry.second;
    cache.levels.push_back(std::move(level_info));
  }
  cache.finest_level = static_cast<int32_t>(cache.levels.size()) - 1;

  // Field discovery based on the selected mesh (level 0 is representative).
  {
    auto mesh_it = iteration.meshes.find(cache.levels.front().mesh_name);
    if (mesh_it != iteration.meshes.end()) {
      openPMD::Mesh mesh = mesh_it->second;
      if (mesh.scalar()) {
        OpenPMDFieldInfo info;
        info.name = selected;
        info.mesh_name = selected;
        info.component_name.clear();
        info.type = datatype_string(mesh[openPMD::MeshRecordComponent::SCALAR].getDatatype());
        cache.fields.push_back(std::move(info));
      } else {
        for (const auto& comp : mesh) {
          OpenPMDFieldInfo info;
          info.name = selected + "/" + comp.first;
          info.mesh_name = selected;
          info.component_name = comp.first;
          info.type = datatype_string(comp.second.getDatatype());
          cache.fields.push_back(std::move(info));
        }
      }
    }
  }

  for (size_t level_idx = 0; level_idx < levels.size(); ++level_idx) {
    const auto& entry = levels[level_idx];
    auto& level_info = cache.levels[level_idx];
    auto mesh_it = iteration.meshes.find(entry.second);
    if (mesh_it == iteration.meshes.end()) {
      continue;
    }
    openPMD::Mesh mesh = mesh_it->second;
    auto rep_component = mesh.scalar() ? mesh[openPMD::MeshRecordComponent::SCALAR]
                                       : mesh.begin()->second;
    auto data_order = mesh.dataOrder();
    auto geom = get_geometry_xyz(mesh, &rep_component, data_order);

    std::vector<std::string> raw_axis_labels = mesh.axisLabels();
    for (auto& label : raw_axis_labels) {
      label = to_lower(label);
    }
    if (data_order == openPMD::Mesh::DataOrder::F) {
      std::reverse(raw_axis_labels.begin(), raw_axis_labels.end());
    }

    openPMD::ChunkTable chunks = rep_component.availableChunks();
    if (chunks.empty()) {
      openPMD::WrittenChunkInfo whole(openPMD::Offset(rep_component.getExtent().size(), 0),
                                      rep_component.getExtent());
      chunks.push_back(whole);
    }

    double unit_si = mesh.gridUnitSI();
    for (int axis = 0; axis < 3; ++axis) {
      double spacing = 0.0;
      if (axis < static_cast<int>(geom.grid_spacing.size())) {
        spacing = geom.grid_spacing[axis];
      }
      if (spacing == 0.0) {
        spacing = 1.0;
      }
      level_info.cell_size[axis] = unit_si * spacing;
      level_info.grid_origin[axis] = (axis < static_cast<int>(geom.grid_origin.size()))
                                         ? unit_si * geom.grid_origin[axis]
                                         : 0.0;
      level_info.position[axis] = (axis < static_cast<int>(geom.position.size()))
                                      ? geom.position[axis]
                                      : 0.0;
    }

    const bool cell_centered = is_cell_centered(mesh, &rep_component);
    if (!cell_centered) {
      std::ostringstream msg;
      msg << "OpenPMDBackend: mesh '" << entry.second
          << "' is not cell-centered; node/other centerings are unsupported";
      throw std::runtime_error(msg.str());
    }
    level_info.node_centered = false;

    for (const auto& chunk : chunks) {
      Patch patch;
      patch.mesh_name = entry.second;
      patch.storage_offset = chunk.offset;
      patch.storage_extent = chunk.extent;
      patch.storage_to_xyz = geom.storage_to_xyz;
      patch.data_order = data_order;
      patch.node_centered = level_info.node_centered;

      const size_t storage_rank = geom.storage_axis_labels.size();
      patch.storage_extent_canonical.assign(storage_rank, 1);
      std::vector<uint64_t> storage_offset_canonical(storage_rank, 0);
      for (size_t idx = 0; idx < raw_axis_labels.size(); ++idx) {
        const auto& label = raw_axis_labels[idx];
        auto iter = std::find(geom.storage_axis_labels.begin(),
                              geom.storage_axis_labels.end(), label);
        if (iter == geom.storage_axis_labels.end()) {
          continue;
        }
        size_t storage_index = static_cast<size_t>(std::distance(geom.storage_axis_labels.begin(), iter));
        if (idx < chunk.offset.size()) {
          storage_offset_canonical[storage_index] = chunk.offset[idx];
        }
        if (idx < chunk.extent.size()) {
          patch.storage_extent_canonical[storage_index] = chunk.extent[idx];
        }
      }

      for (int axis = 0; axis < 3; ++axis) {
        int storage_index = patch.storage_to_xyz[axis];
        uint64_t extent_value = 1;
        uint64_t offset_value = 0;
        if (storage_index >= 0 &&
            storage_index < static_cast<int>(patch.storage_extent_canonical.size())) {
          extent_value = patch.storage_extent_canonical[static_cast<size_t>(storage_index)];
          offset_value = storage_offset_canonical[static_cast<size_t>(storage_index)];
        }
        patch.extent_xyz[axis] = extent_value == 0 ? 1 : extent_value;
        patch.logical_lower[axis] = static_cast<int32_t>(offset_value);

        int32_t cells = static_cast<int32_t>(patch.extent_xyz[axis]);
        if (cells <= 0) {
          cells = 1;
        }
        patch.logical_upper[axis] = patch.logical_lower[axis] + cells - 1;
      }

      level_info.patches.push_back(std::move(patch));
    }

    if (!level_info.patches.empty()) {
      level_info.domain_lo = level_info.patches.front().logical_lower;
      level_info.domain_hi = level_info.patches.front().logical_upper;
      for (const auto& patch : level_info.patches) {
        for (int axis = 0; axis < 3; ++axis) {
          level_info.domain_lo[axis] =
              std::min(level_info.domain_lo[axis], patch.logical_lower[axis]);
          level_info.domain_hi[axis] =
              std::max(level_info.domain_hi[axis], patch.logical_upper[axis]);
        }
      }
    }
  }

  cache.ref_ratio.clear();
  if (cache.levels.size() > 1) {
    for (size_t idx = 1; idx < cache.levels.size(); ++idx) {
      std::array<int32_t, 3> ratio{{1, 1, 1}};
      const auto& coarse = cache.levels[idx - 1].cell_size;
      const auto& fine = cache.levels[idx].cell_size;
      for (int axis = 0; axis < 3; ++axis) {
        if (fine[axis] > 0.0 && coarse[axis] > 0.0) {
          int32_t r = static_cast<int32_t>(std::round(coarse[axis] / fine[axis]));
          ratio[axis] = std::max<int32_t>(1, r);
        }
      }
      cache.ref_ratio.push_back(ratio);
    }
  }

  if (!cache.levels.empty()) {
    const auto& level0 = cache.levels.front();
    const double center_offset = 0.5;
    for (int axis = 0; axis < 3; ++axis) {
      const double spacing = level0.cell_size[axis];
      const double origin = level0.grid_origin[axis];
      const double position = level0.position[axis];
      const int32_t domain_lo = level0.domain_lo[axis];
      const int32_t domain_hi = level0.domain_hi[axis];
      const int32_t cells = std::max<int32_t>(1, domain_hi - domain_lo + 1);
      cache.prob_lo[axis] = origin + (static_cast<double>(domain_lo) + position - center_offset) * spacing;
      cache.prob_hi[axis] = cache.prob_lo[axis] + spacing * static_cast<double>(cells);
    }
  }

  return cache;
}

std::pair<std::string, int> OpenPMDBackend::parse_mesh_level(const std::string& mesh_name) {
  const std::string suffix = "_lvl";
  std::size_t pos = mesh_name.rfind(suffix);
  if (pos == std::string::npos) {
    return {mesh_name, 0};
  }
  std::size_t digits_begin = pos + suffix.size();
  std::size_t digits_end = digits_begin;
  while (digits_end < mesh_name.size() && std::isdigit(mesh_name[digits_end])) {
    ++digits_end;
  }
  if (digits_end == digits_begin) {
    return {mesh_name, 0};
  }
  int level = std::stoi(mesh_name.substr(digits_begin, digits_end - digits_begin));
  std::string base = mesh_name.substr(0, pos);
  return {base, level};
}

OpenPMDBackend::GeometryInfo OpenPMDBackend::get_geometry_xyz(
    const openPMD::Mesh& mesh, const openPMD::MeshRecordComponent* rep,
    openPMD::Mesh::DataOrder order) {
  GeometryInfo geom;
  std::vector<std::string> axis_labels = mesh.axisLabels();
  for (auto& label : axis_labels) {
    label = to_lower(label);
  }

  auto extent = mesh.getExtent();
  std::vector<double> grid_spacing = mesh.gridSpacing<double>();
  std::vector<double> grid_origin = mesh.gridGlobalOffset();
  std::vector<double> position;
  try {
    position = mesh.position<double>();
  } catch (const openPMD::Error&) {
    position.clear();
  }
  if (position.empty() && rep != nullptr) {
    try {
      position = rep->position<double>();
    } catch (const openPMD::Error&) {
      position.clear();
    }
  }

  if (order == openPMD::Mesh::DataOrder::F) {
    std::reverse(axis_labels.begin(), axis_labels.end());
    std::reverse(extent.begin(), extent.end());
    std::reverse(grid_spacing.begin(), grid_spacing.end());
    std::reverse(grid_origin.begin(), grid_origin.end());
    std::reverse(position.begin(), position.end());
  }

  const std::vector<std::string> canonical_axes = {"z", "y", "x"};
  for (const auto& axis : canonical_axes) {
    if (std::find(axis_labels.begin(), axis_labels.end(), axis) == axis_labels.end()) {
      axis_labels.insert(axis_labels.begin(), axis);
      extent.insert(extent.begin(), 1);
      grid_spacing.insert(grid_spacing.begin(), 0.0);
      grid_origin.insert(grid_origin.begin(), 0.0);
      position.insert(position.begin(), 0.0);
    }
  }

  geom.storage_axis_labels = axis_labels;
  geom.storage_to_xyz = {0, 1, 2};
  auto transpose = axis_transpose(axis_labels, {"x", "y", "z"});
  if (transpose.size() == 3) {
    geom.storage_to_xyz = {transpose[0], transpose[1], transpose[2]};
  }

  auto apply_transpose = [&](auto& vec) {
    if (transpose.empty()) {
      return;
    }
    std::vector<std::decay_t<decltype(vec[0])>> copy = vec;
    for (size_t i = 0; i < transpose.size() && i < vec.size(); ++i) {
      int idx = transpose[i];
      if (idx >= 0 && static_cast<size_t>(idx) < copy.size()) {
        vec[i] = copy[static_cast<size_t>(idx)];
      }
    }
  };

  apply_transpose(axis_labels);
  apply_transpose(extent);
  apply_transpose(grid_spacing);
  apply_transpose(grid_origin);
  apply_transpose(position);

  geom.extent = extent;
  geom.grid_spacing = grid_spacing;
  geom.grid_origin = grid_origin;
  geom.position = position;
  return geom;
}

std::vector<int> OpenPMDBackend::axis_transpose(const std::vector<std::string>& src,
                                                const std::vector<std::string>& dst) {
  auto index_of = [&](const std::string& axis) {
    auto iter = std::find(src.begin(), src.end(), axis);
    if (iter == src.end()) {
      return -1;
    }
    return static_cast<int>(std::distance(src.begin(), iter));
  };

  std::vector<int> out;
  out.reserve(dst.size());
  for (const auto& axis : dst) {
    out.push_back(index_of(axis));
  }
  return out;
}

bool OpenPMDBackend::is_cell_centered(const openPMD::Mesh& mesh,
                                      const openPMD::MeshRecordComponent* rep) {
  std::vector<float> centering;
  try {
    centering = mesh.position<float>();
  } catch (const openPMD::Error&) {
    centering.clear();
  }
  if (centering.empty() && rep != nullptr) {
    try {
      centering = rep->position<float>();
    } catch (const openPMD::Error&) {
      centering.clear();
    }
  }
  if (centering.empty()) {
    return false;
  }
  bool all_zero = true;
  bool all_half = true;
  for (float value : centering) {
    all_zero = all_zero && approx_equal(value, 0.0f);
    all_half = all_half && approx_equal(value, 0.5f);
  }
  if (all_half) {
    return true;
  }
  if (all_zero) {
    return false;
  }
  return false;
}

std::string OpenPMDBackend::datatype_string(openPMD::Datatype dtype) {
  if (dtype == openPMD::Datatype::FLOAT) {
    return "float32";
  }
  if (dtype == openPMD::Datatype::DOUBLE) {
    return "float64";
  }
  return "unknown";
}

template <typename T>
void OpenPMDBackend::remap_to_xyz_layout(T* data, size_t count, const Patch& patch) {
  if (data == nullptr || count == 0) {
    return;
  }

  bool requires_remap = patch.data_order == openPMD::Mesh::DataOrder::F;
  for (int axis = 0; axis < 3; ++axis) {
    if (patch.storage_to_xyz[axis] != axis) {
      requires_remap = true;
      break;
    }
  }
  if (!requires_remap) {
    return;
  }

  const size_t rank = patch.storage_extent_canonical.size();
  if (rank == 0) {
    return;
  }

  std::vector<size_t> storage_dims(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    uint64_t dim = patch.storage_extent_canonical[i];
    storage_dims[i] = dim == 0 ? 1 : static_cast<size_t>(dim);
  }

  size_t storage_count = 1;
  for (auto dim : storage_dims) {
    storage_count *= dim;
  }
  if (storage_count == 0) {
    return;
  }
  storage_count = std::min(storage_count, count);

  std::array<size_t, 3> xyz_dims{{1, 1, 1}};
  xyz_dims[0] = patch.extent_xyz[0] == 0 ? 1 : static_cast<size_t>(patch.extent_xyz[0]);
  xyz_dims[1] = patch.extent_xyz[1] == 0 ? 1 : static_cast<size_t>(patch.extent_xyz[1]);
  xyz_dims[2] = patch.extent_xyz[2] == 0 ? 1 : static_cast<size_t>(patch.extent_xyz[2]);

  std::vector<size_t> storage_strides(rank, 1);
  if (patch.data_order == openPMD::Mesh::DataOrder::F) {
    for (size_t i = 1; i < rank; ++i) {
      storage_strides[i] = storage_strides[i - 1] * storage_dims[i - 1];
    }
  } else {
    storage_strides[rank - 1] = 1;
    for (size_t i = rank - 1; i > 0; --i) {
      storage_strides[i - 1] = storage_strides[i] * storage_dims[i];
    }
  }

  std::vector<T> copy(data, data + storage_count);
  std::vector<size_t> coords_storage(rank, 0);

  for (size_t z = 0; z < xyz_dims[2]; ++z) {
    for (size_t y = 0; y < xyz_dims[1]; ++y) {
      for (size_t x = 0; x < xyz_dims[0]; ++x) {
        std::fill(coords_storage.begin(), coords_storage.end(), 0);
        std::array<size_t, 3> coords_xyz{{x, y, z}};
        for (size_t axis = 0; axis < 3; ++axis) {
          int storage_index = patch.storage_to_xyz[axis];
          if (storage_index < 0 || storage_index >= static_cast<int>(coords_storage.size())) {
            continue;
          }
          coords_storage[static_cast<size_t>(storage_index)] = coords_xyz[axis];
        }

        size_t storage_index_linear = 0;
        for (size_t axis = 0; axis < rank; ++axis) {
          storage_index_linear += coords_storage[axis] * storage_strides[axis];
        }
        if (storage_index_linear >= storage_count) {
          continue;
        }
        size_t xyz_index = x + xyz_dims[0] * (y + xyz_dims[1] * z);
        data[xyz_index] = copy[storage_index_linear];
      }
    }
  }
}

template void OpenPMDBackend::remap_to_xyz_layout<float>(float*, size_t, const Patch&);
template void OpenPMDBackend::remap_to_xyz_layout<double>(double*, size_t, const Patch&);

}  // namespace kangaroo

#endif  // KANGAROO_USE_OPENPMD
