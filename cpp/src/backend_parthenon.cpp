#include "kangaroo/backend_parthenon.hpp"

#ifdef KANGAROO_USE_PARTHENON_HDF5

#include <hdf5.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace kangaroo {

namespace {

std::string dtype_name(hid_t t) {
  if (H5Tequal(t, H5T_IEEE_F32LE) > 0 || H5Tequal(t, H5T_IEEE_F32BE) > 0 ||
      H5Tequal(t, H5T_NATIVE_FLOAT) > 0) {
    return "float32";
  }
  if (H5Tequal(t, H5T_IEEE_F64LE) > 0 || H5Tequal(t, H5T_IEEE_F64BE) > 0 ||
      H5Tequal(t, H5T_NATIVE_DOUBLE) > 0) {
    return "float64";
  }
  return "unknown";
}

bool read_attr(hid_t obj, const char* name, hid_t mem_type, void* out) {
  if (H5Aexists(obj, name) <= 0) {
    return false;
  }
  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  if (attr < 0) {
    return false;
  }
  const herr_t err = H5Aread(attr, mem_type, out);
  H5Aclose(attr);
  return err >= 0;
}

}  // namespace

ParthenonBackend::ParthenonBackend(std::string path) : path_(std::move(path)) {
  file_id_ = H5Fopen(path_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id_ < 0) {
    throw std::runtime_error("failed to open Parthenon HDF5 file: " + path_);
  }
  load_metadata();
}

ParthenonBackend::~ParthenonBackend() {
  if (file_id_ >= 0 && H5Iis_valid(file_id_) > 0) {
    H5Fclose(file_id_);
  }
  file_id_ = -1;
}

std::vector<std::string> ParthenonBackend::parse_string_attr(hid_t obj, const char* name) const {
  std::vector<std::string> out;
  if (H5Aexists(obj, name) <= 0) {
    return out;
  }

  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  if (attr < 0) {
    return out;
  }
  hid_t type = H5Aget_type(attr);
  hid_t space = H5Aget_space(attr);
  if (type < 0 || space < 0) {
    if (space >= 0) H5Sclose(space);
    if (type >= 0) H5Tclose(type);
    H5Aclose(attr);
    return out;
  }

  hssize_t npts = H5Sget_simple_extent_npoints(space);
  if (npts <= 0) {
    H5Sclose(space);
    H5Tclose(type);
    H5Aclose(attr);
    return out;
  }

  const bool is_vlen = H5Tis_variable_str(type) > 0;
  if (is_vlen) {
    hid_t mem_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(mem_type, H5T_VARIABLE);
    std::vector<char*> raw(static_cast<size_t>(npts), nullptr);
    if (H5Aread(attr, mem_type, raw.data()) >= 0) {
      out.reserve(static_cast<size_t>(npts));
      for (auto* s : raw) {
        out.emplace_back(s ? s : "");
      }
      H5Dvlen_reclaim(mem_type, space, H5P_DEFAULT, raw.data());
    }
    H5Tclose(mem_type);
  } else {
    const std::size_t width = static_cast<std::size_t>(std::max<hsize_t>(1, H5Tget_size(type)));
    std::vector<char> raw(static_cast<size_t>(npts) * width, '\0');
    if (H5Aread(attr, type, raw.data()) >= 0) {
      out.reserve(static_cast<size_t>(npts));
      for (hssize_t i = 0; i < npts; ++i) {
        const char* start = raw.data() + static_cast<size_t>(i) * width;
        std::size_t len = width;
        while (len > 0 && (start[len - 1] == '\0' || start[len - 1] == ' ')) {
          --len;
        }
        out.emplace_back(start, len);
      }
    }
  }
  H5Sclose(space);
  H5Tclose(type);
  H5Aclose(attr);
  return out;
}

std::vector<int64_t> ParthenonBackend::parse_i64_attr(hid_t obj, const char* name) const {
  std::vector<int64_t> out;
  if (H5Aexists(obj, name) <= 0) {
    return out;
  }
  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  if (attr < 0) {
    return out;
  }
  hid_t space = H5Aget_space(attr);
  if (space < 0) {
    H5Aclose(attr);
    return out;
  }
  hssize_t npts = H5Sget_simple_extent_npoints(space);
  if (npts > 0) {
    out.resize(static_cast<size_t>(npts));
    if (H5Aread(attr, H5T_NATIVE_LLONG, out.data()) < 0) {
      out.clear();
    }
  }
  H5Sclose(space);
  H5Aclose(attr);
  return out;
}

std::vector<double> ParthenonBackend::parse_f64_attr(hid_t obj, const char* name) const {
  std::vector<double> out;
  if (H5Aexists(obj, name) <= 0) {
    return out;
  }
  hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
  if (attr < 0) {
    return out;
  }
  hid_t space = H5Aget_space(attr);
  if (space < 0) {
    H5Aclose(attr);
    return out;
  }
  hssize_t npts = H5Sget_simple_extent_npoints(space);
  if (npts > 0) {
    out.resize(static_cast<size_t>(npts));
    if (H5Aread(attr, H5T_NATIVE_DOUBLE, out.data()) < 0) {
      out.clear();
    }
  }
  H5Sclose(space);
  H5Aclose(attr);
  return out;
}

void ParthenonBackend::load_metadata() {
  hid_t info = H5Gopen2(file_id_, "/Info", H5P_DEFAULT);
  if (info < 0) {
    throw std::runtime_error("Parthenon file missing /Info group: " + path_);
  }

  const auto var_names = parse_string_attr(info, "OutputDatasetNames");
  const auto num_components_raw = parse_i64_attr(info, "NumComponents");
  const auto component_names = parse_string_attr(info, "ComponentNames");

  meta_.var_names = var_names;
  meta_.prob_lo = {0.0, 0.0, 0.0};
  meta_.prob_hi = {1.0, 1.0, 1.0};

  double time_val = 0.0;
  if (read_attr(info, "Time", H5T_NATIVE_DOUBLE, &time_val)) {
    meta_.time = time_val;
  }

  std::vector<int64_t> meshblock_size = parse_i64_attr(info, "MeshBlockSize");
  if (meshblock_size.size() < 3) {
    meshblock_size = {1, 1, 1};
  }

  const auto root_grid_domain = parse_f64_attr(info, "RootGridDomain");
  if (root_grid_domain.size() >= 9) {
    meta_.prob_lo = {root_grid_domain[0], root_grid_domain[3], root_grid_domain[6]};
    meta_.prob_hi = {root_grid_domain[1], root_grid_domain[4], root_grid_domain[7]};
  }

  std::vector<int64_t> root_grid_size = parse_i64_attr(info, "RootGridSize");
  if (root_grid_size.size() < 3) {
    root_grid_size = {1, 1, 1};
  }

  H5Gclose(info);

  hid_t levels_ds = H5Dopen2(file_id_, "/Levels", H5P_DEFAULT);
  hid_t llocs_ds = H5Dopen2(file_id_, "/LogicalLocations", H5P_DEFAULT);
  if (levels_ds < 0 || llocs_ds < 0) {
    if (levels_ds >= 0) H5Dclose(levels_ds);
    if (llocs_ds >= 0) H5Dclose(llocs_ds);
    throw std::runtime_error("Parthenon file missing Levels or LogicalLocations datasets: " +
                             path_);
  }

  hid_t levels_space = H5Dget_space(levels_ds);
  hssize_t nblocks = H5Sget_simple_extent_npoints(levels_space);
  levels_.resize(static_cast<size_t>(std::max<hssize_t>(0, nblocks)));
  if (!levels_.empty()) {
    H5Dread(levels_ds, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, levels_.data());
  }
  H5Sclose(levels_space);

  hid_t llocs_space = H5Dget_space(llocs_ds);
  int ndims = H5Sget_simple_extent_ndims(llocs_space);
  if (ndims != 2) {
    H5Sclose(llocs_space);
    H5Dclose(levels_ds);
    H5Dclose(llocs_ds);
    throw std::runtime_error("LogicalLocations must be rank-2 in " + path_);
  }
  hsize_t dims[2] = {0, 0};
  H5Sget_simple_extent_dims(llocs_space, dims, nullptr);
  if (dims[1] != 3 || dims[0] != levels_.size()) {
    H5Sclose(llocs_space);
    H5Dclose(levels_ds);
    H5Dclose(llocs_ds);
    throw std::runtime_error("LogicalLocations must have shape (num_blocks,3) in " + path_);
  }
  std::vector<int64_t> raw_llocs(static_cast<size_t>(dims[0] * dims[1]), 0);
  if (!raw_llocs.empty()) {
    H5Dread(llocs_ds, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw_llocs.data());
  }
  logical_locations_.resize(static_cast<size_t>(dims[0]));
  for (size_t i = 0; i < logical_locations_.size(); ++i) {
    logical_locations_[i] = {raw_llocs[3 * i + 0], raw_llocs[3 * i + 1], raw_llocs[3 * i + 2]};
  }
  H5Sclose(llocs_space);
  H5Dclose(levels_ds);
  H5Dclose(llocs_ds);

  int32_t max_level = -1;
  for (size_t b = 0; b < levels_.size(); ++b) {
    const int32_t lev = static_cast<int32_t>(levels_[b]);
    max_level = std::max(max_level, lev);
    level_to_global_blocks_[lev].push_back(static_cast<int32_t>(b));
  }
  meta_.finest_level = max_level;

  const int nlevels = std::max(0, meta_.finest_level + 1);
  meta_.ref_ratio.assign(static_cast<size_t>(nlevels), 2);
  if (!meta_.ref_ratio.empty()) {
    meta_.ref_ratio[0] = 1;
  }

  meta_.cell_size.resize(static_cast<size_t>(nlevels), std::vector<double>(3, 1.0));
  for (int lev = 0; lev < nlevels; ++lev) {
    const double scale = static_cast<double>(1 << lev);
    for (int d = 0; d < 3; ++d) {
      meta_.cell_size[static_cast<size_t>(lev)][static_cast<size_t>(d)] =
          (meta_.prob_hi[static_cast<size_t>(d)] - meta_.prob_lo[static_cast<size_t>(d)]) /
          std::max<double>(1.0, static_cast<double>(root_grid_size[static_cast<size_t>(d)]) *
                                    scale);
    }
  }

  meta_.level_boxes.resize(static_cast<size_t>(nlevels));
  meta_.prob_domain.resize(static_cast<size_t>(nlevels));
  for (int lev = 0; lev < nlevels; ++lev) {
    auto& out = meta_.level_boxes[static_cast<size_t>(lev)];
    auto it = level_to_global_blocks_.find(lev);
    if (it == level_to_global_blocks_.end()) {
      meta_.prob_domain[static_cast<size_t>(lev)] = {{0, 0, 0}, {-1, -1, -1}};
      continue;
    }
    out.reserve(it->second.size());
    std::array<int32_t, 3> dom_lo{{std::numeric_limits<int32_t>::max(),
                                    std::numeric_limits<int32_t>::max(),
                                    std::numeric_limits<int32_t>::max()}};
    std::array<int32_t, 3> dom_hi{{std::numeric_limits<int32_t>::min(),
                                    std::numeric_limits<int32_t>::min(),
                                    std::numeric_limits<int32_t>::min()}};

    for (int32_t gb : it->second) {
      const auto& ll = logical_locations_.at(static_cast<size_t>(gb));
      std::array<int32_t, 3> lo{{static_cast<int32_t>(ll[0] * meshblock_size[0]),
                                  static_cast<int32_t>(ll[1] * meshblock_size[1]),
                                  static_cast<int32_t>(ll[2] * meshblock_size[2])}};
      std::array<int32_t, 3> hi{{lo[0] + static_cast<int32_t>(meshblock_size[0]) - 1,
                                  lo[1] + static_cast<int32_t>(meshblock_size[1]) - 1,
                                  lo[2] + static_cast<int32_t>(meshblock_size[2]) - 1}};
      out.push_back({lo, hi});
      for (int d = 0; d < 3; ++d) {
        dom_lo[d] = std::min(dom_lo[d], lo[d]);
        dom_hi[d] = std::max(dom_hi[d], hi[d]);
      }
    }
    meta_.prob_domain[static_cast<size_t>(lev)] = {dom_lo, dom_hi};
  }

  std::vector<int32_t> ncomps(var_names.size(), 1);
  for (size_t i = 0; i < num_components_raw.size() && i < ncomps.size(); ++i) {
    ncomps[i] = std::max<int32_t>(1, static_cast<int32_t>(num_components_raw[i]));
  }

  size_t cname_offset = 0;
  for (size_t i = 0; i < var_names.size(); ++i) {
    hid_t dset = H5Dopen2(file_id_, var_names[i].c_str(), H5P_DEFAULT);
    if (dset < 0) {
      continue;
    }
    hid_t dtype = H5Dget_type(dset);
    hid_t dspace = H5Dget_space(dset);
    int rank = H5Sget_simple_extent_ndims(dspace);
    std::vector<hsize_t> hdims(static_cast<size_t>(std::max(rank, 0)), 0);
    if (rank > 0) {
      H5Sget_simple_extent_dims(dspace, hdims.data(), nullptr);
    }

    DatasetInfo ds;
    ds.name = var_names[i];
    ds.num_components = ncomps[i];
    ds.type = dtype_name(dtype);
    ds.dims.reserve(hdims.size());
    for (hsize_t v : hdims) {
      ds.dims.push_back(static_cast<std::size_t>(v));
    }
    datasets_.push_back(ds);
    dataset_by_name_[ds.name] = ds;

    ParthenonFieldInfo finfo;
    finfo.name = ds.name;
    finfo.num_components = ds.num_components;
    finfo.type = ds.type;
    for (int c = 0; c < ds.num_components; ++c) {
      if (cname_offset < component_names.size()) {
        finfo.component_names.push_back(component_names[cname_offset]);
      } else if (ds.num_components == 1) {
        finfo.component_names.push_back(ds.name);
      } else {
        finfo.component_names.push_back(ds.name + "_" + std::to_string(c));
      }
      ++cname_offset;
    }
    meta_.fields.push_back(std::move(finfo));

    H5Sclose(dspace);
    H5Tclose(dtype);
    H5Dclose(dset);
  }
}

void ParthenonBackend::register_field(int32_t field_id, const std::string& name) {
  FieldSpec spec;
  bool found = false;

  auto by_dataset = dataset_by_name_.find(name);
  if (by_dataset != dataset_by_name_.end()) {
    spec.dataset_name = by_dataset->first;
    spec.comp_start = 0;
    spec.comp_count = by_dataset->second.num_components;
    found = true;
  } else {
    for (const auto& field : meta_.fields) {
      for (size_t c = 0; c < field.component_names.size(); ++c) {
        if (field.component_names[c] == name) {
          spec.dataset_name = field.name;
          spec.comp_start = static_cast<int32_t>(c);
          spec.comp_count = 1;
          found = true;
          break;
        }
      }
      if (found) {
        break;
      }
    }
  }

  if (!found) {
    return;
  }

  std::lock_guard<std::mutex> lock(field_mutex_);
  field_map_[field_id] = std::move(spec);
}

std::optional<HostView> ParthenonBackend::get_chunk(const ChunkRef& ref) {
  if (ref.level < 0) {
    return std::nullopt;
  }
  auto lev_it = level_to_global_blocks_.find(ref.level);
  if (lev_it == level_to_global_blocks_.end()) {
    return std::nullopt;
  }
  if (ref.block < 0 || ref.block >= static_cast<int32_t>(lev_it->second.size())) {
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

  auto dset_it = dataset_by_name_.find(spec.dataset_name);
  if (dset_it == dataset_by_name_.end()) {
    return std::nullopt;
  }
  const auto& ds = dset_it->second;
  if (ds.dims.size() < 4) {
    return std::nullopt;
  }

  const int32_t gblock = lev_it->second[static_cast<size_t>(ref.block)];
  if (gblock < 0 || static_cast<std::size_t>(gblock) >= ds.dims[0]) {
    return std::nullopt;
  }

  hid_t dset = H5Dopen2(file_id_, ds.name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    return std::nullopt;
  }
  hid_t file_space = H5Dget_space(dset);
  const int rank = H5Sget_simple_extent_ndims(file_space);
  if (rank < 4) {
    H5Sclose(file_space);
    H5Dclose(dset);
    return std::nullopt;
  }

  std::vector<hsize_t> dims(static_cast<size_t>(rank), 0);
  H5Sget_simple_extent_dims(file_space, dims.data(), nullptr);

  std::vector<hsize_t> offset(static_cast<size_t>(rank), 0);
  std::vector<hsize_t> count(static_cast<size_t>(rank), 1);
  offset[0] = static_cast<hsize_t>(gblock);
  for (int i = 1; i < rank; ++i) {
    count[static_cast<size_t>(i)] = dims[static_cast<size_t>(i)];
  }

  if (H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset.data(), nullptr, count.data(),
                          nullptr) < 0) {
    H5Sclose(file_space);
    H5Dclose(dset);
    return std::nullopt;
  }

  std::vector<hsize_t> mem_dims(count.begin() + 1, count.end());
  hid_t mem_space = H5Screate_simple(static_cast<int>(mem_dims.size()), mem_dims.data(), nullptr);

  hid_t dtype = H5Dget_type(dset);
  const std::string type = dtype_name(dtype);
  const std::size_t bytes_per = (type == "float32") ? 4 : ((type == "float64") ? 8 : 0);
  if (bytes_per == 0) {
    H5Tclose(dtype);
    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dset);
    return std::nullopt;
  }

  std::size_t block_elems = 1;
  for (auto d : mem_dims) {
    block_elems *= static_cast<std::size_t>(d);
  }

  std::vector<std::uint8_t> block_raw(block_elems * bytes_per, 0);
  hid_t mem_type = (bytes_per == 4) ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
  if (H5Dread(dset, mem_type, mem_space, file_space, H5P_DEFAULT, block_raw.data()) < 0) {
    H5Tclose(dtype);
    H5Sclose(mem_space);
    H5Sclose(file_space);
    H5Dclose(dset);
    return std::nullopt;
  }

  H5Tclose(dtype);
  H5Sclose(mem_space);
  H5Sclose(file_space);
  H5Dclose(dset);

  const std::size_t spatial_elems =
      static_cast<std::size_t>(mem_dims[mem_dims.size() - 1]) *
      static_cast<std::size_t>(mem_dims[mem_dims.size() - 2]) *
      static_cast<std::size_t>(mem_dims[mem_dims.size() - 3]);
  if (spatial_elems == 0) {
    return std::nullopt;
  }

  std::size_t total_components = 1;
  for (size_t i = 0; i + 3 < mem_dims.size(); ++i) {
    total_components *= static_cast<std::size_t>(mem_dims[i]);
  }
  if (total_components == 0) {
    total_components = 1;
  }

  if (spec.comp_start < 0 || spec.comp_count <= 0 ||
      static_cast<std::size_t>(spec.comp_start + spec.comp_count) > total_components) {
    return std::nullopt;
  }

  HostView out;
  const std::size_t comp_bytes = spatial_elems * bytes_per;
  out.data.resize(static_cast<std::size_t>(spec.comp_count) * comp_bytes);
  for (int c = 0; c < spec.comp_count; ++c) {
    const std::size_t src = static_cast<std::size_t>(spec.comp_start + c) * comp_bytes;
    const std::size_t dst = static_cast<std::size_t>(c) * comp_bytes;
    std::memcpy(out.data.data() + dst, block_raw.data() + src, comp_bytes);
  }

  return out;
}

bool ParthenonBackend::has_chunk(const ChunkRef& ref) const {
  if (ref.level < 0) {
    return false;
  }
  auto lev_it = level_to_global_blocks_.find(ref.level);
  if (lev_it == level_to_global_blocks_.end()) {
    return false;
  }
  if (ref.block < 0 || ref.block >= static_cast<int32_t>(lev_it->second.size())) {
    return false;
  }
  std::lock_guard<std::mutex> lock(field_mutex_);
  return field_map_.find(ref.field) != field_map_.end();
}

DatasetMetadata ParthenonBackend::get_metadata() const {
  DatasetMetadata out;
  out.prob_lo = meta_.prob_lo;
  out.prob_hi = meta_.prob_hi;
  out.ref_ratio = meta_.ref_ratio;
  return out;
}

ParthenonMetadata ParthenonBackend::metadata() const { return meta_; }

}  // namespace kangaroo

#endif  // KANGAROO_USE_PARTHENON_HDF5
