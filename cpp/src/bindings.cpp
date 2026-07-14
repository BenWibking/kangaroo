#include "kangaroo/runtime.hpp"
#include "kangaroo/amr_patch_codec.hpp"
#include "kangaroo/buffer_resolution.hpp"
#include "kangaroo/chunk_buffer.hpp"
#include "kangaroo/data_service_local.hpp"

#ifdef KANGAROO_USE_NANOBIND
#include <nanobind/nanobind.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <chrono>
#include <limits>

#include <hpx/serialization/serialize.hpp>
#include <hpx/version.hpp>
#include <msgpack.hpp>

#ifdef KANGAROO_USE_PARTHENON_HDF5
#include <hdf5.h>
#endif

namespace nb = nanobind;

namespace {

std::atomic<std::uint64_t> g_py_event_counter{0};

#ifdef KANGAROO_USE_PARTHENON_HDF5
void write_string_attribute(hid_t object, const char* name,
                            const std::vector<std::string>& values) {
  constexpr std::size_t width = 16;
  const hsize_t extent = values.size();
  hid_t space = H5Screate_simple(1, &extent, nullptr);
  hid_t type = H5Tcopy(H5T_C_S1);
  H5Tset_size(type, width);
  H5Tset_strpad(type, H5T_STR_NULLTERM);
  hid_t attribute = H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
  std::vector<char> data(values.size() * width, '\0');
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::copy_n(values[i].data(), std::min(values[i].size(), width - 1),
                data.data() + i * width);
  }
  H5Awrite(attribute, type, data.data());
  H5Aclose(attribute);
  H5Tclose(type);
  H5Sclose(space);
}

void write_i64_attribute(hid_t object, const char* name,
                         const std::vector<std::int64_t>& values) {
  const hsize_t extent = values.size();
  hid_t space = H5Screate_simple(1, &extent, nullptr);
  hid_t attribute = H5Acreate2(object, name, H5T_NATIVE_LLONG, space,
                               H5P_DEFAULT, H5P_DEFAULT);
  H5Awrite(attribute, H5T_NATIVE_LLONG, values.data());
  H5Aclose(attribute);
  H5Sclose(space);
}

void write_test_parthenon_file(const std::string& path) {
  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) throw std::runtime_error("failed to create test Parthenon file");
  hid_t info = H5Gcreate2(file, "/Info", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  write_string_attribute(info, "OutputDatasetNames", {"prim"});
  write_i64_attribute(info, "NumComponents", {4});
  write_string_attribute(info, "ComponentNames", {"rho", "v1", "v2", "v3"});
  write_i64_attribute(info, "MeshBlockSize", {2, 1, 1});
  write_i64_attribute(info, "RootGridSize", {2, 1, 1});
  H5Gclose(info);

  const hsize_t level_extent = 1;
  hid_t level_space = H5Screate_simple(1, &level_extent, nullptr);
  hid_t levels = H5Dcreate2(file, "/Levels", H5T_NATIVE_LLONG, level_space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  const std::int64_t level = 0;
  H5Dwrite(levels, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &level);
  H5Dclose(levels);
  H5Sclose(level_space);

  const hsize_t location_extents[2] = {1, 3};
  hid_t location_space = H5Screate_simple(2, location_extents, nullptr);
  hid_t locations = H5Dcreate2(file, "/LogicalLocations", H5T_NATIVE_LLONG,
                               location_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  const std::array<std::int64_t, 3> location{0, 0, 0};
  H5Dwrite(locations, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           location.data());
  H5Dclose(locations);
  H5Sclose(location_space);

  const hsize_t field_extents[5] = {1, 4, 1, 1, 2};
  hid_t field_space = H5Screate_simple(5, field_extents, nullptr);
  hid_t field = H5Dcreate2(file, "/prim", H5T_IEEE_F64LE, field_space,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  const std::array<double, 8> values{0.0, 1.0, 10.0, 11.0,
                                     20.0, 21.0, 30.0, 31.0};
  H5Dwrite(field, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
  H5Dclose(field);
  H5Sclose(field_space);
  H5Fclose(file);
}
#endif

nb::object require_key(const nb::dict& d, const char* key) {
  if (!d.contains(key)) {
    throw std::runtime_error(std::string("missing key: ") + key);
  }
  return d[nb::str(key)];
}

kangaroo::RunMeta parse_runmeta(const nb::object& obj) {
  kangaroo::RunMeta meta;
  nb::list steps_list;
  nb::object particle_obj = nb::none();
  if (nb::isinstance<nb::dict>(obj)) {
    auto root = nb::cast<nb::dict>(obj);
    steps_list = nb::cast<nb::list>(require_key(root, "steps"));
    if (root.contains("particle_species")) {
      particle_obj = root[nb::str("particle_species")];
    }
  } else {
    steps_list = nb::cast<nb::list>(obj);
  }
  meta.steps.reserve(steps_list.size());
  for (auto step_item : steps_list) {
    auto step_dict = nb::cast<nb::dict>(step_item);
    kangaroo::StepMeta step;
    step.step = nb::cast<int32_t>(require_key(step_dict, "step"));

    auto levels_list = nb::cast<nb::list>(require_key(step_dict, "levels"));
    step.levels.reserve(levels_list.size());
    for (auto level_item : levels_list) {
      auto level_dict = nb::cast<nb::dict>(level_item);
      kangaroo::LevelMeta level;

      auto geom_dict = nb::cast<nb::dict>(require_key(level_dict, "geom"));
      auto dx = nb::cast<nb::tuple>(require_key(geom_dict, "dx"));
      auto x0 = nb::cast<nb::tuple>(require_key(geom_dict, "x0"));
      level.geom.dx[0] = nb::cast<double>(dx[0]);
      level.geom.dx[1] = nb::cast<double>(dx[1]);
      level.geom.dx[2] = nb::cast<double>(dx[2]);
      level.geom.x0[0] = nb::cast<double>(x0[0]);
      level.geom.x0[1] = nb::cast<double>(x0[1]);
      level.geom.x0[2] = nb::cast<double>(x0[2]);
      if (geom_dict.contains("index_origin")) {
        auto origin = nb::cast<nb::tuple>(geom_dict["index_origin"]);
        level.geom.index_origin[0] = nb::cast<int32_t>(origin[0]);
        level.geom.index_origin[1] = nb::cast<int32_t>(origin[1]);
        level.geom.index_origin[2] = nb::cast<int32_t>(origin[2]);
      }
      if (geom_dict.contains("is_periodic")) {
        auto is_periodic = nb::cast<nb::tuple>(geom_dict["is_periodic"]);
        level.geom.is_periodic[0] = nb::cast<bool>(is_periodic[0]);
        level.geom.is_periodic[1] = nb::cast<bool>(is_periodic[1]);
        level.geom.is_periodic[2] = nb::cast<bool>(is_periodic[2]);
      }
      level.geom.ref_ratio = nb::cast<int>(require_key(geom_dict, "ref_ratio"));

      auto boxes_list = nb::cast<nb::list>(require_key(level_dict, "boxes"));
      level.boxes.reserve(boxes_list.size());
      for (auto box_item : boxes_list) {
        auto box_pair = nb::cast<nb::tuple>(box_item);
        auto lo = nb::cast<nb::tuple>(box_pair[0]);
        auto hi = nb::cast<nb::tuple>(box_pair[1]);
        kangaroo::BlockBox box;
        box.lo = {nb::cast<int32_t>(lo[0]), nb::cast<int32_t>(lo[1]), nb::cast<int32_t>(lo[2])};
        box.hi = {nb::cast<int32_t>(hi[0]), nb::cast<int32_t>(hi[1]), nb::cast<int32_t>(hi[2])};
        level.boxes.push_back(box);
      }

      step.levels.push_back(std::move(level));
    }

    meta.steps.push_back(std::move(step));
  }

  if (!particle_obj.is_none()) {
    if (nb::isinstance<nb::dict>(particle_obj)) {
      auto d = nb::cast<nb::dict>(particle_obj);
      for (auto item : d) {
        auto key = nb::cast<std::string>(item.first);
        auto value = nb::cast<int64_t>(item.second);
        meta.particle_species.emplace_back(std::move(key), value);
      }
    } else if (nb::isinstance<nb::list>(particle_obj)) {
      auto entries = nb::cast<nb::list>(particle_obj);
      meta.particle_species.reserve(entries.size());
      for (auto item : entries) {
        auto pair = nb::cast<nb::tuple>(item);
        if (pair.size() != 2) {
          throw std::runtime_error("particle_species list entries must be (name, count)");
        }
        meta.particle_species.emplace_back(nb::cast<std::string>(pair[0]), nb::cast<int64_t>(pair[1]));
      }
    } else {
      throw std::runtime_error("particle_species must be a dict or list");
    }
  }

  return meta;
}

double now_seconds() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

kangaroo::Runtime& binding_runtime() {
  static kangaroo::Runtime runtime;
  return runtime;
}

kangaroo::IndexBox3 parse_index_box(nb::tuple lo, nb::tuple hi) {
  kangaroo::IndexBox3 b;
  b.lo[0] = nb::cast<int32_t>(lo[0]);
  b.lo[1] = nb::cast<int32_t>(lo[1]);
  b.lo[2] = nb::cast<int32_t>(lo[2]);
  b.hi[0] = nb::cast<int32_t>(hi[0]);
  b.hi[1] = nb::cast<int32_t>(hi[1]);
  b.hi[2] = nb::cast<int32_t>(hi[2]);
  return b;
}

kangaroo::ScalarType parse_scalar_type(const std::string& dtype) {
  if (dtype == "opaque") return kangaroo::ScalarType::kOpaque;
  if (dtype == "u8") return kangaroo::ScalarType::kU8;
  if (dtype == "i64") return kangaroo::ScalarType::kI64;
  if (dtype == "f32") return kangaroo::ScalarType::kF32;
  if (dtype == "f64") return kangaroo::ScalarType::kF64;
  throw std::runtime_error("unsupported chunk dtype: " + dtype);
}

kangaroo::ChunkBuffer host_view_from_bytes(
    nb::bytes payload,
    const std::string& dtype = "opaque",
    const std::vector<std::uint64_t>& shape = {}) {
  const auto* raw = reinterpret_cast<const std::uint8_t*>(payload.data());
  std::vector<std::uint8_t> bytes(raw, raw + payload.size());
  const auto scalar = parse_scalar_type(dtype);
  if (scalar == kangaroo::ScalarType::kOpaque) {
    if (!shape.empty() && (shape.size() != 1 || shape[0] != bytes.size())) {
      throw std::runtime_error("opaque chunk shape must equal its byte count");
    }
    return kangaroo::ChunkBuffer::opaque(std::move(bytes));
  }
  if (shape.empty()) {
    throw std::runtime_error("numeric chunk writes require an explicit shape");
  }
  return kangaroo::ChunkBuffer::wrap(
      kangaroo::SharedByteBuffer(std::move(bytes)),
      kangaroo::BufferDesc::contiguous(scalar, shape));
}

nb::bytes host_view_bytes(const kangaroo::ChunkBuffer& view) {
  const auto bytes = view.byte_view();
  return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

nb::dict chunk_buffer_dict(const kangaroo::ChunkBuffer& buffer) {
  const auto& desc = buffer.desc();
  nb::dict out;
  out["data"] = nb::bytes(
      reinterpret_cast<const char*>(buffer.byte_view().data()), buffer.byte_view().size());
  out["dtype"] = kangaroo::scalar_type_name(desc.scalar);
  out["shape"] = std::vector<std::uint64_t>(
      desc.extents.begin(), desc.extents.begin() + desc.rank);
  out["strides_bytes"] = std::vector<std::int64_t>(
      desc.strides_bytes.begin(), desc.strides_bytes.begin() + desc.rank);
  return out;
}

nb::dict subbox_view_dict(const kangaroo::SubboxView& view) {
  auto to_tuple = [](const int32_t v[3]) {
    return nb::make_tuple(v[0], v[1], v[2]);
  };
  nb::dict d;
  const auto bytes = view.data.byte_view();
  d["data"] = nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  d["dtype"] = kangaroo::scalar_type_name(view.data.desc().scalar);
  d["lo"] = to_tuple(view.box.lo);
  d["hi"] = to_tuple(view.box.hi);
  return d;
}

class DynamicBoundTestData final : public kangaroo::DataService {
 public:
  DynamicBoundTestData(std::uint64_t particle_records, std::size_t chunk_bytes)
      : particle_records_(particle_records), chunk_bytes_(chunk_bytes) {}

  int home_rank(const kangaroo::ChunkRef&) const override { return 0; }
  std::size_t estimate_host_bytes(const kangaroo::ChunkRef& ref) const override {
    return ref.block == 1 ? chunk_bytes_ : sizeof(double);
  }
  std::optional<std::uint64_t> estimate_particle_chunk_records(
      const std::string&, std::int64_t) const override {
    return particle_records_;
  }
  kangaroo::ChunkBuffer alloc_host(
      const kangaroo::ChunkRef&, const kangaroo::ResolvedBufferSpec&) override {
    throw std::runtime_error("test data store does not allocate");
  }
  hpx::future<kangaroo::ChunkBuffer> get_host(const kangaroo::ChunkRef&) override {
    return hpx::make_exceptional_future<kangaroo::ChunkBuffer>(
        std::runtime_error("test data store does not load"));
  }
  hpx::future<kangaroo::SubboxView> get_subbox(const kangaroo::ChunkSubboxRef&) override {
    return hpx::make_exceptional_future<kangaroo::SubboxView>(
        std::runtime_error("test data store does not load subboxes"));
  }
  hpx::future<void> put_host(const kangaroo::ChunkRef&, kangaroo::ChunkBuffer) override {
    return hpx::make_ready_future();
  }

 private:
  std::uint64_t particle_records_ = 0;
  std::size_t chunk_bytes_ = 0;
};

std::vector<std::uint8_t> pack_particle_bound_params() {
  msgpack::sbuffer packed;
  msgpack::packer<msgpack::sbuffer> pk(&packed);
  pk.pack_map(2);
  pk.pack(std::string("particle_type"));
  pk.pack(std::string("particles"));
  pk.pack(std::string("field_name"));
  pk.pack(std::string("value"));
  return {packed.data(), packed.data() + packed.size()};
}

std::vector<std::uint8_t> pack_amr_bound_params() {
  msgpack::sbuffer packed;
  msgpack::packer<msgpack::sbuffer> pk(&packed);
  pk.pack_map(5);
  pk.pack(std::string("input_field"));
  pk.pack_int32(11);
  pk.pack(std::string("input_version"));
  pk.pack_int32(0);
  pk.pack(std::string("input_step"));
  pk.pack_int32(0);
  pk.pack(std::string("input_level"));
  pk.pack_int16(0);
  pk.pack(std::string("halo_cells"));
  pk.pack_int32(1);
  return {packed.data(), packed.data() + packed.size()};
}

}  // namespace

NB_MODULE(_core, m) {
  m.def("test_chunk_buffer_descriptor",
        [](const std::string& dtype,
           const std::vector<std::uint64_t>& extents,
           const std::string& layout) {
          kangaroo::ScalarType scalar = kangaroo::ScalarType::kOpaque;
          if (dtype == "u8") scalar = kangaroo::ScalarType::kU8;
          else if (dtype == "i64") scalar = kangaroo::ScalarType::kI64;
          else if (dtype == "f32") scalar = kangaroo::ScalarType::kF32;
          else if (dtype == "f64") scalar = kangaroo::ScalarType::kF64;
          else if (dtype != "opaque") throw std::runtime_error("unknown scalar type: " + dtype);

          kangaroo::BufferDesc desc;
          if (layout == "contiguous") {
            desc = kangaroo::BufferDesc::contiguous(scalar, extents);
          } else if (layout == "plotfile") {
            if (extents.size() != 3) throw std::runtime_error("plotfile layout requires rank 3");
            desc = kangaroo::BufferDesc::plotfile_grid(
                scalar, {extents[0], extents[1], extents[2]});
          } else {
            throw std::runtime_error("unknown layout: " + layout);
          }
          auto buffer = kangaroo::ChunkBuffer::allocate(desc, kangaroo::InitPolicy::kZero);
          nb::dict out;
          out["dtype"] = kangaroo::scalar_type_name(desc.scalar);
          out["rank"] = desc.rank;
          out["extents"] = std::vector<std::uint64_t>(
              desc.extents.begin(), desc.extents.begin() + desc.rank);
          out["strides_bytes"] = std::vector<std::int64_t>(
              desc.strides_bytes.begin(), desc.strides_bytes.begin() + desc.rank);
          out["elements"] = desc.element_count();
          out["bytes"] = buffer.bytes();
          return out;
        },
        nb::arg("dtype"), nb::arg("extents"), nb::arg("layout") = "contiguous");
  m.def("test_chunk_buffer_layout_values", []() {
    const std::array<std::uint64_t, 3> extents{2, 3, 4};
    auto runtime = kangaroo::ChunkBuffer::allocate(
        kangaroo::BufferDesc::runtime_grid(kangaroo::ScalarType::kF64, extents));
    auto plotfile = kangaroo::ChunkBuffer::allocate(
        kangaroo::BufferDesc::plotfile_grid(kangaroo::ScalarType::kF64, extents));
    auto runtime_view = runtime.mutable_view<double, 3>();
    auto plotfile_view = plotfile.mutable_view<double, 3>();
    for (std::uint64_t i = 0; i < extents[0]; ++i) {
      for (std::uint64_t j = 0; j < extents[1]; ++j) {
        for (std::uint64_t k = 0; k < extents[2]; ++k) {
          const double value = static_cast<double>(100 * i + 10 * j + k);
          runtime_view(i, j, k) = value;
          plotfile_view(i, j, k) = value;
        }
      }
    }
    auto plotfile_as_runtime = plotfile.copy_to(
        kangaroo::BufferDesc::runtime_grid(kangaroo::ScalarType::kF64, extents));
    const auto runtime_const = std::as_const(runtime).view<double, 3>();
    const auto plotfile_const = std::as_const(plotfile_as_runtime).view<double, 3>();
    nb::list values;
    for (std::uint64_t i = 0; i < extents[0]; ++i)
      for (std::uint64_t j = 0; j < extents[1]; ++j)
        for (std::uint64_t k = 0; k < extents[2]; ++k)
          values.append(nb::make_tuple(runtime_const(i, j, k), plotfile_const(i, j, k)));
    return values;
  });
  m.def("test_chunk_buffer_grid_region", []() {
    const std::array<std::uint64_t, 3> extents{4, 3, 2};
    auto source = kangaroo::ChunkBuffer::allocate(
        kangaroo::BufferDesc::plotfile_grid(kangaroo::ScalarType::kI64, extents));
    auto grid = source.mutable_view<std::int64_t, 3>();
    for (std::uint64_t i = 0; i < extents[0]; ++i)
      for (std::uint64_t j = 0; j < extents[1]; ++j)
        for (std::uint64_t k = 0; k < extents[2]; ++k)
          grid(i, j, k) = static_cast<std::int64_t>(100 * i + 10 * j + k);
    auto region = source.copy_grid_region({1, 1, 0}, {2, 2, 2});
    auto values = region.view<std::int64_t, 3>();
    nb::list out;
    for (std::uint64_t i = 0; i < 2; ++i)
      for (std::uint64_t j = 0; j < 2; ++j)
        for (std::uint64_t k = 0; k < 2; ++k) out.append(values(i, j, k));
    return out;
  });
  m.def("test_chunk_buffer_layout_copy_converts_dtype", []() {
    const std::array<std::uint64_t, 3> extents{2, 1, 2};
    auto source = kangaroo::ChunkBuffer::allocate(
        kangaroo::BufferDesc::plotfile_grid(kangaroo::ScalarType::kF32, extents));
    auto input = source.mutable_view<float, 3>();
    input(0, 0, 0) = 1.25F;
    input(0, 0, 1) = 2.5F;
    input(1, 0, 0) = 3.75F;
    input(1, 0, 1) = 5.0F;
    auto converted = source.copy_to(
        kangaroo::BufferDesc::runtime_grid(kangaroo::ScalarType::kF64, extents));
    auto output = converted.view<double, 3>();
    return nb::make_tuple(output(0, 0, 0), output(0, 0, 1),
                          output(1, 0, 0), output(1, 0, 1));
  });
  m.def("test_amr_patch_codec_roundtrip", []() {
    kangaroo::AmrPatchRecord patch;
    patch.level = 2;
    patch.box.lo[0] = 4;
    patch.box.lo[1] = 5;
    patch.box.lo[2] = 6;
    patch.box.hi[0] = 5;
    patch.box.hi[1] = 6;
    patch.box.hi[2] = 7;
    patch.geom.dx[0] = 0.5;
    patch.geom.dx[1] = 0.25;
    patch.geom.dx[2] = 0.125;
    patch.geom.is_periodic[1] = true;
    const std::array<std::uint64_t, 3> extents{2, 2, 2};
    patch.data = kangaroo::ChunkBuffer::allocate(
        kangaroo::BufferDesc::plotfile_grid(kangaroo::ScalarType::kF32, extents));
    patch.data.mutable_view<float, 3>()(1, 1, 1) = 7.5F;
    const std::array<kangaroo::AmrPatchRecord, 1> records{patch};
    auto encoded = kangaroo::encode_amr_patch_payload(records);
    auto decoded = kangaroo::decode_amr_patch_payload(encoded.byte_view());
    const auto value = decoded.at(0).data.view<float, 3>()(1, 1, 1);
    return nb::make_tuple(decoded.size(), decoded.at(0).level,
                          decoded.at(0).box.lo[0], decoded.at(0).geom.dx[2],
                          decoded.at(0).geom.is_periodic[1], value,
                          kangaroo::scalar_type_name(decoded.at(0).data.desc().scalar));
  });
  m.def("test_amr_patch_codec_rejects_malformed", []() {
    const std::array<std::uint8_t, 1> malformed{0x90};
    return kangaroo::decode_amr_patch_payload(malformed).size();
  });
  m.def("test_chunk_buffer_cow", []() {
    const std::array<std::uint64_t, 1> extents{3};
    auto original = kangaroo::ChunkBuffer::allocate(
        kangaroo::BufferDesc::contiguous(kangaroo::ScalarType::kI64, extents));
    original.mutable_array<std::int64_t>()(1) = 7;
    auto copy = original;
    copy.mutable_array<std::int64_t>()(1) = 11;
    return nb::make_tuple(original.array<std::int64_t>()(1), copy.array<std::int64_t>()(1));
  });
  m.def("test_chunk_buffer_init_policy", [](std::uint64_t elements) {
    const std::array<std::uint64_t, 1> extents{elements};
    const auto desc = kangaroo::BufferDesc::contiguous(kangaroo::ScalarType::kU8, extents);
    auto uninitialized = kangaroo::ChunkBuffer::allocate(
        desc, kangaroo::InitPolicy::kUninitialized);
    auto zeroed = kangaroo::ChunkBuffer::allocate(desc, kangaroo::InitPolicy::kZero);
    const auto zeroed_bytes = zeroed.byte_view();
    const bool all_zero = std::all_of(
        zeroed_bytes.begin(), zeroed_bytes.end(),
        [](std::uint8_t value) { return value == std::uint8_t{0}; });
    return nb::make_tuple(
        uninitialized.uses_uninitialized_storage(), zeroed.uses_uninitialized_storage(), all_zero);
  });
  m.def("test_chunk_buffer_dynamic", [](std::uint64_t capacity, std::uint64_t extent) {
    auto buffer = kangaroo::ChunkBuffer::allocate_dynamic(kangaroo::ScalarType::kF64, capacity);
    buffer.commit_dynamic_extent(extent);
    return nb::make_tuple(buffer.desc().extents[0], buffer.bytes(), buffer.capacity_bytes());
  });
  m.def("test_chunk_buffer_dynamic_write", [](std::uint64_t capacity,
                                               const std::vector<double>& values) {
    auto buffer = kangaroo::ChunkBuffer::allocate_dynamic(kangaroo::ScalarType::kF64, capacity);
    auto output = buffer.mutable_dynamic_array<double>();
    for (std::size_t i = 0; i < values.size(); ++i) output.at(i) = values[i];
    buffer.commit_dynamic_extent(values.size());
    const auto visible = buffer.array<double>();
    std::vector<double> restored;
    restored.reserve(visible.extent(0));
    for (std::size_t i = 0; i < visible.extent(0); ++i) restored.push_back(visible(i));
    return restored;
  });
  m.def("test_chunk_buffer_dynamic_cow", [](bool copy_assignment) {
    auto original = kangaroo::ChunkBuffer::allocate_dynamic(
        kangaroo::ScalarType::kI64, 3, kangaroo::InitPolicy::kZero);
    kangaroo::ChunkBuffer copy;
    if (copy_assignment) {
      copy = original;
    } else {
      auto constructed = original;
      copy = std::move(constructed);
    }

    auto copy_output = copy.mutable_dynamic_array<std::int64_t>();
    copy_output(0) = 11;
    copy_output(1) = 22;
    copy.commit_dynamic_extent(2);

    const bool original_still_uncommitted = original.awaiting_dynamic_extent_commit();
    const auto original_bytes_before_commit = original.bytes();
    auto original_output = original.mutable_dynamic_array<std::int64_t>();
    original_output(0) = 7;
    original.commit_dynamic_extent(1);

    return nb::make_tuple(
        original_still_uncommitted, original_bytes_before_commit,
        original.desc().extents[0], original.array<std::int64_t>()(0),
        copy.desc().extents[0], copy.array<std::int64_t>()(0),
        copy.array<std::int64_t>()(1));
  });
  m.def("test_chunk_buffer_async_byte_writer", []() {
    auto buffer = kangaroo::ChunkBuffer::allocate_dynamic(
        kangaroo::ScalarType::kOpaque, 8);
    auto writer = buffer.begin_async_dynamic_write();
    const std::array<std::uint8_t, 3> payload{1, 2, 3};
    writer.replace(payload);
    return std::vector<std::uint8_t>(buffer.byte_view().begin(), buffer.byte_view().end());
  });
  m.def("test_chunk_buffer_async_byte_writer_survives_move", []() {
    auto buffer = kangaroo::ChunkBuffer::allocate_dynamic(
        kangaroo::ScalarType::kOpaque, 8);
    auto writer = buffer.begin_async_dynamic_write();
    auto moved = std::move(buffer);
    const std::array<std::uint8_t, 3> payload{1, 2, 3};
    writer.replace(payload);
    return std::vector<std::uint8_t>(moved.byte_view().begin(), moved.byte_view().end());
  });
  m.def("test_chunk_buffer_async_byte_writer_preserves_cow", []() {
    auto written = kangaroo::ChunkBuffer::allocate_dynamic(
        kangaroo::ScalarType::kOpaque, 8, kangaroo::InitPolicy::kZero);
    auto untouched = written;
    auto writer = written.begin_async_dynamic_write();
    const std::array<std::uint8_t, 3> payload{1, 2, 3};
    writer.replace(payload);
    untouched.commit_dynamic_extent(payload.size());
    return nb::make_tuple(
        std::vector<std::uint8_t>(written.byte_view().begin(), written.byte_view().end()),
        std::vector<std::uint8_t>(untouched.byte_view().begin(), untouched.byte_view().end()));
  });
  m.def("test_chunk_buffer_async_byte_writer_reuse", []() {
    auto buffer = kangaroo::ChunkBuffer::allocate_dynamic(
        kangaroo::ScalarType::kOpaque, 8);
    auto writer = buffer.begin_async_dynamic_write();
    const std::array<std::uint8_t, 1> payload{1};
    writer.replace(payload);
    writer.replace(payload);
  });
#ifdef KANGAROO_USE_PARTHENON_HDF5
  m.def("test_parthenon_component_storage", [](const std::string& path) {
    write_test_parthenon_file(path);
    kangaroo::ParthenonBackend backend(path);
    constexpr std::int32_t field_id = 7;
    backend.register_field(field_id, "v2");
    const kangaroo::ChunkRef ref{0, 0, field_id, 0, 0};
    auto buffer = backend.get_chunk(ref);
    if (!buffer.has_value()) {
      throw std::runtime_error("failed to read test Parthenon component");
    }
    const auto values = buffer->view<double, 3>();
    return nb::make_tuple(
        std::vector<double>{values(0, 0, 0), values(1, 0, 0)}, buffer->bytes(),
        buffer->resident_bytes(), backend.estimate_chunk_bytes(ref));
  });
#endif
  m.def("test_chunk_buffer_dynamic_roundtrip", [](std::uint64_t capacity, std::uint64_t extent) {
    auto buffer = kangaroo::ChunkBuffer::allocate_dynamic(kangaroo::ScalarType::kF64, capacity);
    buffer.commit_dynamic_extent(extent);

    std::vector<char> archive_bytes;
    hpx::serialization::output_archive output_archive(archive_bytes);
    output_archive << buffer;

    kangaroo::ChunkBuffer restored;
    hpx::serialization::input_archive input_archive(archive_bytes, archive_bytes.size());
    input_archive >> restored;
    return nb::make_tuple(
        restored.desc().extents[0], restored.bytes(), restored.capacity_bytes());
  });
  m.def("test_backend_chunk_dynamic_capacity",
        [](const std::string& dtype, const std::string& kernel,
           std::uint64_t particle_records, const std::vector<std::uint64_t>& input_bytes) {
          DynamicBoundTestData data(particle_records, 0);
          kangaroo::RunMeta meta;
          kangaroo::LevelMeta level;
          level.geom.dx[0] = level.geom.dx[1] = level.geom.dx[2] = 1.0;
          level.boxes.push_back(kangaroo::BlockBox{{0, 0, 0}, {0, 0, 0}});
          meta.steps.push_back(kangaroo::StepMeta{0, {level}});

          kangaroo::TaskTemplateIR task;
          task.kernel = kernel;
          task.params_msgpack = pack_particle_bound_params();
          task.dynamic_output_bound =
              std::make_shared<const kangaroo::DynamicOutputBoundEvaluator>(
                  [kernel](const kangaroo::DynamicOutputBoundContext& context)
                      -> std::optional<std::uint64_t> {
                    if (kernel == "particle_value_counts_reduce") {
                      std::uint64_t bytes = 0;
                      for (const auto& input : context.inputs) {
                        bytes = kangaroo::checked_add(bytes, input.payload_bytes);
                      }
                      return bytes;
                    }
                    const auto records = context.data.estimate_particle_chunk_records(
                        "particles", context.block);
                    if (!records.has_value()) return std::nullopt;
                    if (kernel == "particle_load_field_chunk_f64") return *records;
                    return kangaroo::checked_add(
                        sizeof(std::uint64_t),
                        kangaroo::checked_multiply(
                            *records, sizeof(double) + sizeof(std::int64_t)));
                  });
          kangaroo::BufferSpecIR spec;
          spec.scalar = parse_scalar_type(dtype);
          spec.shape_kind = kangaroo::ShapeRuleKind::kDynamic;
          spec.dynamic_upper_bound.kind = kangaroo::DynamicUpperBoundKind::kBackendChunk;
          std::vector<kangaroo::ChunkBuffer> inputs;
          inputs.reserve(input_bytes.size());
          for (const auto bytes : input_bytes) {
            inputs.push_back(kangaroo::ChunkBuffer::opaque(
                std::vector<std::uint8_t>(static_cast<std::size_t>(bytes))));
          }
          const auto resolved = kangaroo::resolve_output_spec_for_task(
              spec, task, data, meta, 0, 0, 0, 0, inputs);
          return *resolved.dynamic_capacity_elements;
        });
  m.def("test_block_shape_extent", [](std::int32_t lo, std::int32_t hi) {
    DynamicBoundTestData data(0, 0);
    kangaroo::RunMeta meta;
    kangaroo::LevelMeta level;
    level.boxes.push_back(kangaroo::BlockBox{{lo, 0, 0}, {hi, 0, 0}});
    meta.steps.push_back(kangaroo::StepMeta{0, {level}});

    kangaroo::TaskTemplateIR task;
    kangaroo::BufferSpecIR spec;
    spec.scalar = kangaroo::ScalarType::kU8;
    spec.shape_kind = kangaroo::ShapeRuleKind::kBlock;
    const auto resolved = kangaroo::resolve_output_spec_for_task(
        spec, task, data, meta, 0, 0, 0, 0, {});
    return resolved.desc.extents[0];
  });
  m.def("test_amr_subbox_dynamic_capacity", [](std::uint64_t source_chunk_bytes) {
    DynamicBoundTestData data(0, source_chunk_bytes);
    kangaroo::RunMeta meta;
    kangaroo::LevelMeta level;
    level.geom.dx[0] = level.geom.dx[1] = level.geom.dx[2] = 1.0;
    level.boxes.push_back(kangaroo::BlockBox{{0, 0, 0}, {0, 0, 0}});
    level.boxes.push_back(kangaroo::BlockBox{{1, 0, 0}, {1, 0, 0}});
    meta.steps.push_back(kangaroo::StepMeta{0, {level}});

    kangaroo::TaskTemplateIR task;
    task.kernel = "amr_subbox_fetch_pack";
    task.params_msgpack = pack_amr_bound_params();
    task.dynamic_output_bound =
        std::make_shared<const kangaroo::DynamicOutputBoundEvaluator>(
            [](const kangaroo::DynamicOutputBoundContext& context) {
              return kangaroo::estimate_amr_subbox_pack_capacity(
                  context,
                  kangaroo::AmrSubboxPackParams{
                      .input_field = 11,
                      .input_version = 0,
                      .input_step = 0,
                      .input_level = 0,
                      .halo_cells = 1});
            });
    kangaroo::BufferSpecIR spec;
    spec.scalar = kangaroo::ScalarType::kOpaque;
    spec.shape_kind = kangaroo::ShapeRuleKind::kDynamic;
    spec.dynamic_upper_bound.kind = kangaroo::DynamicUpperBoundKind::kAmrSubboxPack;
    const auto resolved = kangaroo::resolve_output_spec_for_task(
        spec, task, data, meta, 0, 0, 0, 0, {});
    return *resolved.dynamic_capacity_elements;
  });
  m.def("test_amr_subbox_dynamic_capacity_wide_coordinates",
        [](std::uint64_t source_chunk_bytes, bool wide_source) {
          DynamicBoundTestData data(0, source_chunk_bytes);
          kangaroo::RunMeta meta;
          kangaroo::LevelMeta level;
          level.geom.dx[0] = level.geom.dx[1] = level.geom.dx[2] = 1.0;
          if (wide_source) {
            level.boxes.push_back(kangaroo::BlockBox{{0, 0, 0}, {0, 0, 0}});
            level.boxes.push_back(kangaroo::BlockBox{
                {std::numeric_limits<std::int32_t>::min(), 0, 0},
                {std::numeric_limits<std::int32_t>::max(), 0, 0}});
          } else {
            level.geom.index_origin[0] = std::numeric_limits<std::int32_t>::max();
            constexpr std::int32_t kTargetIndex =
                std::numeric_limits<std::int32_t>::min() + 2;
            level.boxes.push_back(
                kangaroo::BlockBox{{kTargetIndex, 0, 0}, {kTargetIndex, 0, 0}});
            level.boxes.push_back(kangaroo::BlockBox{
                {kTargetIndex + 1, 0, 0}, {kTargetIndex + 1, 0, 0}});
          }
          meta.steps.push_back(kangaroo::StepMeta{0, {level}});

          kangaroo::TaskTemplateIR task;
          task.dynamic_output_bound =
              std::make_shared<const kangaroo::DynamicOutputBoundEvaluator>(
                  [](const kangaroo::DynamicOutputBoundContext& context) {
                    return kangaroo::estimate_amr_subbox_pack_capacity(
                        context,
                        kangaroo::AmrSubboxPackParams{
                            .input_field = 11,
                            .input_version = 0,
                            .input_step = 0,
                            .input_level = 0,
                            .halo_cells = 1});
                  });
          kangaroo::BufferSpecIR spec;
          spec.scalar = kangaroo::ScalarType::kOpaque;
          spec.shape_kind = kangaroo::ShapeRuleKind::kDynamic;
          spec.dynamic_upper_bound.kind =
              kangaroo::DynamicUpperBoundKind::kAmrSubboxPack;
          const auto resolved = kangaroo::resolve_output_spec_for_task(
              spec, task, data, meta, 0, 0, 0, 0, {});
          return *resolved.dynamic_capacity_elements;
        });
  m.def("hpx_configuration_string", []() { return hpx::configuration_string(); });
  m.def("set_event_log_path", &kangaroo::set_event_log_path);
  m.def("set_perfetto_trace_path", &kangaroo::set_perfetto_trace_path);
  m.def("test_get_subbox",
        [](kangaroo::DatasetHandle& dataset,
           int32_t step,
           int16_t level,
           int32_t field,
           int32_t version,
           int32_t block,
           nb::tuple chunk_lo,
           nb::tuple chunk_hi,
           nb::tuple request_lo,
           nb::tuple request_hi) -> nb::dict {
          (void)binding_runtime().locality_id();

          kangaroo::ChunkSubboxRef ref;
          ref.chunk = kangaroo::ChunkRef{step, level, field, version, block};
          ref.chunk_box = parse_index_box(chunk_lo, chunk_hi);
          ref.request_box = parse_index_box(request_lo, request_hi);

          kangaroo::DataServiceLocal data_service(0, &dataset);
          auto out = data_service.get_subbox(ref).get();
          return subbox_view_dict(out);
        },
        nb::arg("dataset"),
        nb::arg("step"),
        nb::arg("level"),
        nb::arg("field"),
        nb::arg("version"),
        nb::arg("block"),
        nb::arg("chunk_lo"),
        nb::arg("chunk_hi"),
        nb::arg("request_lo"),
        nb::arg("request_hi"));
  m.def("test_data_service_pending_then_put",
        [](kangaroo::DatasetHandle& dataset,
           int32_t step,
           int16_t level,
           int32_t field,
           int32_t version,
           int32_t block,
           nb::bytes payload,
           int64_t timeout_ms) -> nb::dict {
          (void)binding_runtime().locality_id();

          kangaroo::DataServiceLocal data_service(0, &dataset);
          kangaroo::ChunkRef ref{step, level, field, version, block};

          (void)timeout_ms;
          auto host_future = data_service.get_host(ref);
          const bool returned_before_put = true;
          const bool future_ready_before_put = host_future.is_ready();

          kangaroo::ChunkBuffer payload_view = host_view_from_bytes(payload);
          data_service.put_host(ref, std::move(payload_view)).get();
          const bool ready_before_get = host_future.is_ready();
          kangaroo::ChunkBuffer out = host_future.get();

          nb::dict d;
          d["returned_before_put"] = returned_before_put;
          d["future_ready_before_put"] = future_ready_before_put;
          d["ready_before_get"] = ready_before_get;
          d["data"] = host_view_bytes(out);
          return d;
        },
        nb::arg("dataset"),
        nb::arg("step"),
        nb::arg("level"),
        nb::arg("field"),
        nb::arg("version"),
        nb::arg("block"),
        nb::arg("payload"),
        nb::arg("timeout_ms") = 50);
  m.def("test_data_service_put_then_get",
        [](kangaroo::DatasetHandle& dataset,
           int32_t step,
           int16_t level,
           int32_t field,
           int32_t version,
           int32_t block,
           nb::bytes payload) -> nb::dict {
          (void)binding_runtime().locality_id();

          kangaroo::DataServiceLocal data_service(0, &dataset);
          kangaroo::ChunkRef ref{step, level, field, version, block};
          data_service.put_host(ref, host_view_from_bytes(payload)).get();

          auto host_future = data_service.get_host(ref);
          const bool ready_before_get = host_future.is_ready();
          kangaroo::ChunkBuffer out = host_future.get();

          nb::dict d;
          d["ready_before_get"] = ready_before_get;
          d["data"] = host_view_bytes(out);
          return d;
        },
        nb::arg("dataset"),
        nb::arg("step"),
        nb::arg("level"),
        nb::arg("field"),
        nb::arg("version"),
        nb::arg("block"),
        nb::arg("payload"));
  m.def("test_data_service_many_pending_consumers_then_put",
        [](kangaroo::DatasetHandle& dataset,
           int32_t step,
           int16_t level,
           int32_t field,
           int32_t version,
           int32_t block,
           nb::bytes payload,
           int32_t consumers) -> nb::dict {
          (void)binding_runtime().locality_id();

          kangaroo::DataServiceLocal data_service(0, &dataset);
          kangaroo::ChunkRef ref{step, level, field, version, block};
          std::vector<hpx::future<kangaroo::ChunkBuffer>> futures;
          futures.reserve(static_cast<std::size_t>(consumers));
          nb::list ready_before_put;
          for (int32_t i = 0; i < consumers; ++i) {
            futures.push_back(data_service.get_host(ref));
            ready_before_put.append(futures.back().is_ready());
          }

          data_service.put_host(ref, host_view_from_bytes(payload)).get();

          nb::list ready_after_put;
          nb::list data;
          for (auto& future : futures) {
            ready_after_put.append(future.is_ready());
            data.append(host_view_bytes(future.get()));
          }

          nb::dict d;
          d["ready_before_put"] = ready_before_put;
          d["ready_after_put"] = ready_after_put;
          d["data"] = data;
          return d;
        },
        nb::arg("dataset"),
        nb::arg("step"),
        nb::arg("level"),
        nb::arg("field"),
        nb::arg("version"),
        nb::arg("block"),
        nb::arg("payload"),
        nb::arg("consumers"));
  m.def("test_data_service_subbox_pending_then_put",
        [](kangaroo::DatasetHandle& dataset,
           int32_t step,
           int16_t level,
           int32_t field,
           int32_t version,
           int32_t block,
           nb::tuple chunk_lo,
           nb::tuple chunk_hi,
           nb::tuple request_lo,
           nb::tuple request_hi,
           const std::string& dtype,
           nb::bytes payload) -> nb::dict {
          (void)binding_runtime().locality_id();

          kangaroo::ChunkSubboxRef subbox_ref;
          subbox_ref.chunk = kangaroo::ChunkRef{step, level, field, version, block};
          subbox_ref.chunk_box = parse_index_box(chunk_lo, chunk_hi);
          subbox_ref.request_box = parse_index_box(request_lo, request_hi);
          const auto shape = std::vector<std::uint64_t>{
              static_cast<std::uint64_t>(subbox_ref.chunk_box.hi[0] -
                                         subbox_ref.chunk_box.lo[0] + 1),
              static_cast<std::uint64_t>(subbox_ref.chunk_box.hi[1] -
                                         subbox_ref.chunk_box.lo[1] + 1),
              static_cast<std::uint64_t>(subbox_ref.chunk_box.hi[2] -
                                         subbox_ref.chunk_box.lo[2] + 1)};

          kangaroo::DataServiceLocal data_service(0, &dataset);
          auto subbox_future = data_service.get_subbox(subbox_ref);
          const bool future_ready_before_put = subbox_future.is_ready();
          data_service.put_host(
              subbox_ref.chunk, host_view_from_bytes(payload, dtype, shape)).get();
          const bool ready_before_get = subbox_future.is_ready();
          auto out = subbox_future.get();

          nb::dict d = subbox_view_dict(out);
          d["future_ready_before_put"] = future_ready_before_put;
          d["ready_before_get"] = ready_before_get;
          return d;
        },
        nb::arg("dataset"),
        nb::arg("step"),
        nb::arg("level"),
        nb::arg("field"),
        nb::arg("version"),
        nb::arg("block"),
        nb::arg("chunk_lo"),
        nb::arg("chunk_hi"),
        nb::arg("request_lo"),
        nb::arg("request_hi"),
        nb::arg("dtype"),
        nb::arg("payload"));
  m.def("log_task_event",
        [](const std::string& name,
           const std::string& status,
           nb::object start_obj,
           nb::object end_obj,
           nb::object id_obj,
           nb::object worker_label_obj) {
          kangaroo::TaskEvent event;
          event.name = name;
          event.kernel = name;
          event.plane = "python";
          event.status = status;
          event.stage = -1;
          event.template_index = -1;
          event.block = -1;
          event.step = 0;
          event.level = 0;
          event.locality = -1;
          event.worker = -1;
          if (!worker_label_obj.is_none()) {
            event.worker_label = nb::cast<std::string>(worker_label_obj);
          } else {
            event.worker_label = "python";
          }

          if (!id_obj.is_none()) {
            event.id = nb::cast<std::string>(id_obj);
          } else {
            event.id = "py:" + std::to_string(++g_py_event_counter);
          }

          double start = start_obj.is_none() ? now_seconds() : nb::cast<double>(start_obj);
          double end = end_obj.is_none() ? start : nb::cast<double>(end_obj);
          event.ts = end;
          event.start = start;
          event.end = end;
          kangaroo::log_task_event(event);
        },
        nb::arg("name"),
        nb::arg("status"),
        nb::arg("start") = nb::none(),
        nb::arg("end") = nb::none(),
        nb::arg("id") = nb::none(),
        nb::arg("worker_label") = nb::none());
  m.def("log_phase_event",
        [](const std::string& name,
           const std::string& status,
           nb::object start_obj,
           nb::object end_obj,
           nb::object category_obj,
           nb::object worker_label_obj,
           nb::object locality_obj) {
          kangaroo::PhaseEvent event;
          event.name = name;
          event.category = category_obj.is_none() ? "kangaroo.phase" : nb::cast<std::string>(category_obj);
          event.status = status;
          event.locality = locality_obj.is_none() ? 0 : nb::cast<int32_t>(locality_obj);
          event.worker = -1;
          event.worker_label = worker_label_obj.is_none() ? "python" : nb::cast<std::string>(worker_label_obj);

          double start = start_obj.is_none() ? now_seconds() : nb::cast<double>(start_obj);
          double end = end_obj.is_none() ? start : nb::cast<double>(end_obj);
          event.ts = end;
          event.start = start;
          event.end = end;
          kangaroo::log_phase_event(event);
        },
        nb::arg("name"),
        nb::arg("status"),
        nb::arg("start") = nb::none(),
        nb::arg("end") = nb::none(),
        nb::arg("category") = nb::none(),
        nb::arg("worker_label") = nb::none(),
        nb::arg("locality") = nb::none());

  nb::class_<kangaroo::Runtime>(m, "Runtime")
      .def(nb::init<>())
      .def("__init__",
           [](kangaroo::Runtime* self, nb::object config_obj, nb::object args_obj) {
             std::vector<std::string> cfg;
             std::vector<std::string> args;
             if (!config_obj.is_none()) {
               auto list = nb::cast<nb::list>(config_obj);
               cfg.reserve(list.size());
               for (auto item : list) {
                 cfg.push_back(nb::cast<std::string>(item));
               }
             }
             if (!args_obj.is_none()) {
               auto list = nb::cast<nb::list>(args_obj);
               args.reserve(list.size());
               for (auto item : list) {
                 args.push_back(nb::cast<std::string>(item));
               }
             }
             new (self) kangaroo::Runtime(cfg, args);
           },
           nb::arg("hpx_config") = nb::none(),
           nb::arg("hpx_args") = nb::none())
      .def("alloc_field_id", &kangaroo::Runtime::alloc_field_id)
      .def("mark_field_persistent", &kangaroo::Runtime::mark_field_persistent)
      .def("kernels", &kangaroo::Runtime::kernels, nb::rv_policy::reference)
      .def("set_event_log_path", &kangaroo::Runtime::set_event_log_path)
      .def("set_perfetto_trace_path", &kangaroo::Runtime::set_perfetto_trace_path)
      .def("locality_id", &kangaroo::Runtime::locality_id)
      .def("num_localities", &kangaroo::Runtime::num_localities)
      .def("chunk_home_rank", &kangaroo::Runtime::chunk_home_rank)
      .def("wait_for_console_release",
           &kangaroo::Runtime::wait_for_console_release,
           nb::call_guard<nb::gil_scoped_release>())
      .def("release_console_workers", &kangaroo::Runtime::release_console_workers)
      .def("get_task_chunk",
           [](kangaroo::Runtime& self, int32_t step, int16_t level, int32_t field,
              int32_t version, int32_t block, kangaroo::DatasetHandle* dataset) {
             auto view = self.get_task_chunk(step, level, field, version, block, dataset);
             const auto bytes = view.byte_view();
             return nb::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
           },
           nb::arg("step"),
           nb::arg("level"),
           nb::arg("field"),
           nb::arg("version") = 0,
           nb::arg("block"),
           nb::arg("dataset") = nb::none())
      .def("get_task_chunk_info",
           [](kangaroo::Runtime& self, int32_t step, int16_t level, int32_t field,
              int32_t version, int32_t block, kangaroo::DatasetHandle* dataset) {
             return chunk_buffer_dict(
                 self.get_task_chunk(step, level, field, version, block, dataset));
           },
           nb::arg("step"), nb::arg("level"), nb::arg("field"),
           nb::arg("version") = 0, nb::arg("block"), nb::arg("dataset") = nb::none())
      .def("preload_dataset", &kangaroo::Runtime::preload_dataset)
      .def("run_packed_plan",
           &kangaroo::Runtime::run_packed_plan,
           nb::call_guard<nb::gil_scoped_release>())
      .def("run_packed_plan",
           [](kangaroo::Runtime& self, nb::bytes packed, kangaroo::RunMetaHandle& runmeta,
              kangaroo::DatasetHandle& dataset) {
             // Copy Python-owned bytes while the GIL is held, then release it for the
             // long-running C++/HPX execution.
             auto* data = static_cast<const std::uint8_t*>(packed.data());
             std::vector<std::uint8_t> buffer(data, data + packed.size());
             nb::gil_scoped_release release;
             self.run_packed_plan(buffer, runmeta, dataset);
           });

  nb::class_<kangaroo::KernelRegistry>(m, "KernelRegistry")
      .def("list", &kangaroo::KernelRegistry::list_kernel_descs);

  nb::class_<kangaroo::RunMetaHandle>(m, "RunMetaHandle")
      .def("__init__", [](kangaroo::RunMetaHandle* self, nb::object payload) {
        new (self) kangaroo::RunMetaHandle();
        self->meta = parse_runmeta(payload);
      });

  nb::class_<kangaroo::DatasetHandle>(m, "DatasetHandle")
      .def("__init__", [](kangaroo::DatasetHandle* self, const std::string& uri, int32_t step,
                          int16_t level) {
        new (self) kangaroo::DatasetHandle();
        self->uri = uri;
        self->step = step;
        self->level = level;

        if (uri.rfind("amrex://", 0) == 0 || uri.rfind("file://", 0) == 0) {
            std::string path = uri;
            if (uri.rfind("amrex://", 0) == 0) {
                path = uri.substr(8);
            } else {
                path = uri.substr(7);
            }
            bool as_parthenon = false;
            if (path.size() >= 5) {
              const auto suffix = path.substr(path.size() - 5);
              as_parthenon = (suffix == ".phdf" || suffix == ".hdf5");
            }
            if (!as_parthenon && path.size() >= 3) {
              const auto suffix = path.substr(path.size() - 3);
              as_parthenon = (suffix == ".h5");
            }
            if (as_parthenon) {
#ifdef KANGAROO_USE_PARTHENON_HDF5
              self->backend = std::make_shared<kangaroo::ParthenonBackend>(path);
#else
              throw std::runtime_error("Parthenon backend not enabled in this build");
#endif
            } else {
              self->backend = std::make_shared<kangaroo::PlotfileBackend>(path);
            }
        } else if (uri.rfind("parthenon://", 0) == 0) {
#ifdef KANGAROO_USE_PARTHENON_HDF5
            std::string path = uri.substr(12);
            self->backend = std::make_shared<kangaroo::ParthenonBackend>(path);
#else
            throw std::runtime_error("Parthenon backend not enabled in this build");
#endif
        } else if (uri.rfind("openpmd://", 0) == 0) {
#ifdef KANGAROO_USE_OPENPMD
            self->backend = std::make_shared<kangaroo::OpenPMDBackend>(uri);
#else
            throw std::runtime_error("openPMD backend not enabled in this build");
#endif
        } else if (uri.rfind("memory://", 0) == 0) {
            self->backend = std::make_shared<kangaroo::MemoryBackend>();
        } else {
            throw std::runtime_error("unsupported dataset URI scheme: " + uri);
        }
      })
      .def("register_field", [](kangaroo::DatasetHandle& self, int32_t field_id, int32_t component) {
          if (auto plt = std::dynamic_pointer_cast<kangaroo::PlotfileBackend>(self.backend)) {
              plt->register_field(field_id, component);
          }
      })
      .def("register_field", [](kangaroo::DatasetHandle& self, int32_t field_id,
                                const std::string& name) {
#ifdef KANGAROO_USE_OPENPMD
          if (auto opmd = std::dynamic_pointer_cast<kangaroo::OpenPMDBackend>(self.backend)) {
              opmd->register_field(field_id, name);
              return;
          }
#else
          (void)self;
          (void)field_id;
          (void)name;
#endif
#ifdef KANGAROO_USE_PARTHENON_HDF5
          if (auto phdf = std::dynamic_pointer_cast<kangaroo::ParthenonBackend>(self.backend)) {
              phdf->register_field(field_id, name);
          }
#endif
      })
      .def("list_meshes", [](kangaroo::DatasetHandle& self) -> nb::list {
#ifdef KANGAROO_USE_OPENPMD
          if (auto opmd = std::dynamic_pointer_cast<kangaroo::OpenPMDBackend>(self.backend)) {
              nb::list out;
              for (const auto& name : opmd->list_meshes(self.step)) {
                  out.append(name);
              }
              return out;
          }
#endif
          return nb::list();
      })
      .def("select_mesh", [](kangaroo::DatasetHandle& self, const std::string& name) {
#ifdef KANGAROO_USE_OPENPMD
          if (auto opmd = std::dynamic_pointer_cast<kangaroo::OpenPMDBackend>(self.backend)) {
              opmd->select_mesh(name);
          }
#else
          (void)self;
          (void)name;
#endif
      })
      .def("list_particle_types", [](kangaroo::DatasetHandle& self) -> nb::list {
          nb::list out;
          if (!self.backend) {
            return out;
          }
          if (auto* reader = self.backend->get_plotfile_reader()) {
            for (const auto& name : reader->particle_types()) {
              out.append(name);
            }
          }
          return out;
      })
      .def("list_particle_fields", [](kangaroo::DatasetHandle& self, const std::string& particle_type) -> nb::list {
          nb::list out;
          if (!self.backend) {
            return out;
          }
          if (auto* reader = self.backend->get_plotfile_reader()) {
            for (const auto& name : reader->particle_fields(particle_type)) {
              out.append(name);
            }
          }
          return out;
      }, nb::arg("particle_type"))
      .def("particle_chunk_count",
           [](kangaroo::DatasetHandle& self, const std::string& particle_type) -> int64_t {
             if (!self.backend) {
               throw std::runtime_error("dataset backend is not initialized");
             }
             auto* reader = self.backend->get_plotfile_reader();
             if (!reader) {
               throw std::runtime_error(
                   "particle chunk metadata is only supported for AMReX plotfiles");
             }
             return reader->particle_chunk_count(particle_type);
           },
           nb::arg("particle_type"))
      .def("read_particle_field_chunk",
           [](kangaroo::DatasetHandle& self, const std::string& particle_type,
              const std::string& field_name, int64_t chunk_index) -> nb::dict {
             if (!self.backend) {
               throw std::runtime_error("dataset backend is not initialized");
             }
             auto* reader = self.backend->get_plotfile_reader();
             if (!reader) {
               throw std::runtime_error(
                   "particle chunked field access is only supported for AMReX plotfiles");
             }
             auto data = reader->read_particle_field_chunk(particle_type, field_name, chunk_index);
             nb::dict out;
             out["count"] = data.count;
             out["dtype"] = data.dtype;
             out["data"] =
                 nb::bytes(reinterpret_cast<const char*>(data.bytes.data()), data.bytes.size());
             return out;
           },
           nb::arg("particle_type"),
           nb::arg("field_name"),
           nb::arg("chunk_index"))
      .def("metadata", [](kangaroo::DatasetHandle& self) -> nb::dict {
          if (!self.backend) return nb::dict();
          auto* reader = self.backend->get_plotfile_reader();
          if (reader) {

            const auto& hdr = reader->header();
            nb::dict d;
            d["var_names"] = hdr.var_names;
            d["finest_level"] = hdr.finest_level;
            d["time"] = hdr.time;
            d["prob_lo"] = hdr.prob_lo;
            d["prob_hi"] = hdr.prob_hi;
            d["ref_ratio"] = hdr.ref_ratio;
            d["cell_size"] = hdr.cell_size;
            
            auto to_tuple = [](const kangaroo::plotfile::IntVect& iv) {
              return nb::make_tuple(iv.x, iv.y, iv.z);
            };
            auto box_to_tuple = [&](const kangaroo::plotfile::Box& box) {
              return nb::make_tuple(to_tuple(box.lo), to_tuple(box.hi));
            };

            nb::list levels;
            for (int level = 0; level <= hdr.finest_level; ++level) {
              nb::list boxes;
              const auto& vismf = reader->vismf_header(level);
              for (const auto& box : vismf.box_array.boxes) {
                boxes.append(box_to_tuple(box));
              }
              levels.append(boxes);
            }
            d["level_boxes"] = levels;

            nb::list prob_domains;
            for (const auto& box : hdr.prob_domain) {
              prob_domains.append(box_to_tuple(box));
            }
            d["prob_domain"] = prob_domains;

            return d;
          }

#ifdef KANGAROO_USE_OPENPMD
          if (auto opmd = std::dynamic_pointer_cast<kangaroo::OpenPMDBackend>(self.backend)) {
            auto meta = opmd->metadata(self.step);
            nb::dict d;
            nb::list var_names;
            for (const auto& field : meta.fields) {
              var_names.append(field.name);
            }
            d["var_names"] = var_names;
            d["mesh_names"] = meta.mesh_names;
            d["selected_mesh"] = meta.selected_mesh;
            d["finest_level"] = meta.finest_level;
            d["prob_lo"] = nb::make_tuple(meta.prob_lo[0], meta.prob_lo[1], meta.prob_lo[2]);
            d["prob_hi"] = nb::make_tuple(meta.prob_hi[0], meta.prob_hi[1], meta.prob_hi[2]);

            nb::list ref_ratio;
            for (const auto& ratio : meta.ref_ratio) {
              ref_ratio.append(ratio[0]);
            }
            d["ref_ratio"] = ref_ratio;

            nb::list cell_size;
            for (const auto& size : meta.cell_size) {
              cell_size.append(nb::make_tuple(size[0], size[1], size[2]));
            }
            d["cell_size"] = cell_size;

            nb::list levels;
            for (const auto& level_boxes : meta.level_boxes) {
              nb::list boxes;
              for (const auto& box : level_boxes) {
                auto lo = nb::make_tuple(box.first[0], box.first[1], box.first[2]);
                auto hi = nb::make_tuple(box.second[0], box.second[1], box.second[2]);
                boxes.append(nb::make_tuple(lo, hi));
              }
              levels.append(boxes);
            }
            d["level_boxes"] = levels;

            nb::list prob_domains;
            for (const auto& box : meta.prob_domain) {
              auto lo = nb::make_tuple(box.first[0], box.first[1], box.first[2]);
              auto hi = nb::make_tuple(box.second[0], box.second[1], box.second[2]);
              prob_domains.append(nb::make_tuple(lo, hi));
            }
            d["prob_domain"] = prob_domains;

            return d;
          }
#endif
#ifdef KANGAROO_USE_PARTHENON_HDF5
          if (auto phdf = std::dynamic_pointer_cast<kangaroo::ParthenonBackend>(self.backend)) {
            auto meta = phdf->metadata();
            nb::dict d;
            d["var_names"] = meta.var_names;
            d["finest_level"] = meta.finest_level;
            d["prob_lo"] = meta.prob_lo;
            d["prob_hi"] = meta.prob_hi;
            d["ref_ratio"] = meta.ref_ratio;
            d["cell_size"] = meta.cell_size;
            d["time"] = meta.time;

            nb::dict vinfo;
            for (const auto& field : meta.fields) {
              nb::dict entry;
              entry["num_components"] = field.num_components;
              entry["component_names"] = field.component_names;
              entry["type"] = field.type;
              vinfo[field.name.c_str()] = entry;
            }
            d["variable_info"] = vinfo;

            nb::list levels;
            nb::list prob_domains;
            for (size_t i = 0; i < meta.level_boxes.size(); ++i) {
              nb::list boxes;
              for (const auto& box : meta.level_boxes[i]) {
                auto lo = nb::make_tuple(box.first[0], box.first[1], box.first[2]);
                auto hi = nb::make_tuple(box.second[0], box.second[1], box.second[2]);
                boxes.append(nb::make_tuple(lo, hi));
              }
              levels.append(boxes);

              auto dom_lo = nb::make_tuple(meta.prob_domain[i].first[0], meta.prob_domain[i].first[1],
                                           meta.prob_domain[i].first[2]);
              auto dom_hi = nb::make_tuple(meta.prob_domain[i].second[0], meta.prob_domain[i].second[1],
                                           meta.prob_domain[i].second[2]);
              prob_domains.append(nb::make_tuple(dom_lo, dom_hi));
            }
            d["level_boxes"] = levels;
            d["prob_domain"] = prob_domains;
            return d;
          }
#endif

          return nb::dict();
      })
      .def("set_chunk",
           [](kangaroo::DatasetHandle& self, int32_t field, int32_t version, int32_t block,
              nb::bytes payload, const std::string& dtype,
              const std::vector<std::uint64_t>& shape) {
             auto view = host_view_from_bytes(payload, dtype, shape);
             kangaroo::ChunkRef ref{self.step, self.level, field, version, block};
             self.set_chunk(ref, std::move(view));
           },
           nb::arg("field"), nb::arg("version"), nb::arg("block"), nb::arg("payload"),
           nb::arg("dtype") = "opaque", nb::arg("shape") = std::vector<std::uint64_t>{})
      .def("set_chunk_ref",
           [](kangaroo::DatasetHandle& self,
              int32_t step,
              int16_t level,
              int32_t field,
              int32_t version,
              int32_t block,
              nb::bytes payload,
              const std::string& dtype,
              const std::vector<std::uint64_t>& shape) {
             auto view = host_view_from_bytes(payload, dtype, shape);
             kangaroo::ChunkRef ref{step, level, field, version, block};
             self.set_chunk(ref, std::move(view));
           },
           nb::arg("step"), nb::arg("level"), nb::arg("field"), nb::arg("version"),
           nb::arg("block"), nb::arg("payload"), nb::arg("dtype") = "opaque",
           nb::arg("shape") = std::vector<std::uint64_t>{})
      .def("read_chunk_ref",
           [](kangaroo::DatasetHandle& self,
              int32_t step,
              int16_t level,
              int32_t field,
              int32_t version,
              int32_t block) -> nb::object {
             kangaroo::ChunkRef ref{step, level, field, version, block};
             std::optional<kangaroo::ChunkBuffer> view;
             {
               nb::gil_scoped_release release;
               view = self.get_chunk(ref);
             }
             if (!view.has_value()) {
               return nb::none();
             }
             return nb::cast(host_view_bytes(*view));
           },
           nb::arg("step"),
           nb::arg("level"),
           nb::arg("field"),
           nb::arg("version"),
           nb::arg("block"))
      .def("read_chunks_ref",
           [](kangaroo::DatasetHandle& self, nb::list ref_items) {
             std::vector<kangaroo::ChunkRef> refs;
             refs.reserve(ref_items.size());
             for (auto item : ref_items) {
               auto tuple = nb::cast<nb::tuple>(item);
               if (tuple.size() != 5) {
                 throw std::runtime_error(
                     "read_chunks_ref expects refs as (step, level, field, version, block)");
               }
               refs.push_back(kangaroo::ChunkRef{
                   nb::cast<int32_t>(tuple[0]),
                   nb::cast<int16_t>(tuple[1]),
                   nb::cast<int32_t>(tuple[2]),
                   nb::cast<int32_t>(tuple[3]),
                   nb::cast<int32_t>(tuple[4]),
               });
             }

             std::vector<std::optional<kangaroo::ChunkBuffer>> views;
             {
               nb::gil_scoped_release release;
               views = self.get_chunks(refs);
             }

             nb::list out;
             for (const auto& view : views) {
               if (view.has_value()) {
                 out.append(host_view_bytes(*view));
               } else {
                 out.append(nb::none());
               }
             }
             return out;
           },
           nb::arg("refs"))
      .def("read_chunks_ref_sizes",
           [](kangaroo::DatasetHandle& self, nb::list ref_items) {
             std::vector<kangaroo::ChunkRef> refs;
             refs.reserve(ref_items.size());
             for (auto item : ref_items) {
               auto tuple = nb::cast<nb::tuple>(item);
               if (tuple.size() != 5) {
                 throw std::runtime_error(
                     "read_chunks_ref_sizes expects refs as (step, level, field, version, block)");
               }
               refs.push_back(kangaroo::ChunkRef{
                   nb::cast<int32_t>(tuple[0]),
                   nb::cast<int16_t>(tuple[1]),
                   nb::cast<int32_t>(tuple[2]),
                   nb::cast<int32_t>(tuple[3]),
                   nb::cast<int32_t>(tuple[4]),
               });
             }

             std::vector<std::optional<kangaroo::ChunkBuffer>> views;
             {
               nb::gil_scoped_release release;
               views = self.get_chunks(refs);
             }

             nb::list out;
             for (const auto& view : views) {
               out.append(view.has_value() ? static_cast<std::uint64_t>(view->bytes())
                                           : static_cast<std::uint64_t>(0));
             }
             return out;
           },
           nb::arg("refs"))
      .def("read_chunks_ref_sizes_data_service",
           [](kangaroo::DatasetHandle& self, nb::list ref_items) {
             std::vector<kangaroo::ChunkRef> refs;
             refs.reserve(ref_items.size());
             for (auto item : ref_items) {
               auto tuple = nb::cast<nb::tuple>(item);
               if (tuple.size() != 5) {
                 throw std::runtime_error(
                     "read_chunks_ref_sizes_data_service expects refs as "
                     "(step, level, field, version, block)");
               }
               refs.push_back(kangaroo::ChunkRef{
                   nb::cast<int32_t>(tuple[0]),
                   nb::cast<int16_t>(tuple[1]),
                   nb::cast<int32_t>(tuple[2]),
                   nb::cast<int32_t>(tuple[3]),
                   nb::cast<int32_t>(tuple[4]),
               });
             }

             std::vector<std::uint64_t> sizes;
             sizes.reserve(refs.size());
             {
               nb::gil_scoped_release release;
               (void)binding_runtime().locality_id();
               kangaroo::DataServiceLocal data_service(0, &self);
               auto futures = data_service.get_hosts_shared(refs);
               auto ready = hpx::when_all(std::move(futures)).get();
               for (auto& future : ready) {
                 auto view = future.get();
                 sizes.push_back(view ? static_cast<std::uint64_t>(view->bytes())
                                      : static_cast<std::uint64_t>(0));
               }
             }

             nb::list out;
             for (std::uint64_t size : sizes) {
               out.append(size);
             }
             return out;
           },
           nb::arg("refs"));

}
#endif
