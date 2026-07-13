#include "kangaroo/amr_patch_codec.hpp"

#include <msgpack.hpp>

#include <stdexcept>
#include <string>

namespace kangaroo {

ChunkBuffer encode_amr_patch_payload(std::span<const AmrPatchRecord> patches) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(&buffer);
  packer.pack_map(1);
  packer.pack(std::string("patches"));
  packer.pack_array(patches.size());
  for (const auto& patch : patches) {
    const auto& desc = patch.data.desc();
    desc.validate(patch.data.bytes());
    if (desc.scalar == ScalarType::kOpaque || desc.rank != 3) {
      throw std::runtime_error("AMR patch data must be a numeric Block Grid");
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (patch.box.hi[axis] < patch.box.lo[axis] ||
          desc.extents[axis] != static_cast<std::uint64_t>(
                                    static_cast<std::int64_t>(patch.box.hi[axis]) -
                                    static_cast<std::int64_t>(patch.box.lo[axis]) + 1)) {
        throw std::runtime_error("AMR patch box does not match its Block Grid extents");
      }
    }
    packer.pack_map(11);
    packer.pack(std::string("level"));
    packer.pack_int16(patch.level);
    packer.pack(std::string("lo"));
    packer.pack_array(3);
    for (int axis = 0; axis < 3; ++axis) packer.pack_int32(patch.box.lo[axis]);
    packer.pack(std::string("hi"));
    packer.pack_array(3);
    for (int axis = 0; axis < 3; ++axis) packer.pack_int32(patch.box.hi[axis]);
    packer.pack(std::string("dx"));
    packer.pack_array(3);
    for (int axis = 0; axis < 3; ++axis) packer.pack_double(patch.geom.dx[axis]);
    packer.pack(std::string("x0"));
    packer.pack_array(3);
    for (int axis = 0; axis < 3; ++axis) packer.pack_double(patch.geom.x0[axis]);
    packer.pack(std::string("index_origin"));
    packer.pack_array(3);
    for (int axis = 0; axis < 3; ++axis) packer.pack_int32(patch.geom.index_origin[axis]);
    packer.pack(std::string("is_periodic"));
    packer.pack_array(3);
    for (int axis = 0; axis < 3; ++axis) packer.pack(static_cast<bool>(patch.geom.is_periodic[axis]));
    packer.pack(std::string("scalar"));
    packer.pack_uint8(static_cast<std::uint8_t>(desc.scalar));
    packer.pack(std::string("extents"));
    packer.pack_array(desc.rank);
    for (std::size_t axis = 0; axis < desc.rank; ++axis) packer.pack_uint64(desc.extents[axis]);
    packer.pack(std::string("strides_bytes"));
    packer.pack_array(desc.rank);
    for (std::size_t axis = 0; axis < desc.rank; ++axis) packer.pack_int64(desc.strides_bytes[axis]);
    packer.pack(std::string("data"));
    const auto bytes = patch.data.byte_view();
    packer.pack_bin(bytes.size());
    if (!bytes.empty()) {
      packer.pack_bin_body(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }
  return ChunkBuffer::opaque(
      std::vector<std::uint8_t>(buffer.data(), buffer.data() + buffer.size()));
}

std::vector<AmrPatchRecord> decode_amr_patch_payload(
    std::span<const std::uint8_t> payload) {
  if (payload.empty()) return {};
  auto handle = msgpack::unpack(reinterpret_cast<const char*>(payload.data()), payload.size());
  const auto root = handle.get();
  if (root.type != msgpack::type::MAP) {
    throw std::runtime_error("AMR patch payload root must be a map");
  }
  const msgpack::object* encoded_patches = nullptr;
  for (std::uint32_t i = 0; i < root.via.map.size; ++i) {
    const auto& key = root.via.map.ptr[i].key;
    if (key.type == msgpack::type::STR && key.as<std::string>() == "patches") {
      encoded_patches = &root.via.map.ptr[i].val;
      break;
    }
  }
  if (encoded_patches == nullptr || encoded_patches->type != msgpack::type::ARRAY) {
    throw std::runtime_error("AMR patch payload is missing the patches array");
  }

  std::vector<AmrPatchRecord> patches;
  patches.reserve(encoded_patches->via.array.size);
  for (std::uint32_t i = 0; i < encoded_patches->via.array.size; ++i) {
    const auto& encoded = encoded_patches->via.array.ptr[i];
    if (encoded.type != msgpack::type::MAP) {
      throw std::runtime_error("AMR patch record must be a map");
    }
    AmrPatchRecord patch;
    BufferDesc desc;
    bool have_extents = false;
    bool have_strides = false;
    std::size_t stride_rank = 0;
    std::vector<std::uint8_t> bytes;
    for (std::uint32_t j = 0; j < encoded.via.map.size; ++j) {
      const auto& key = encoded.via.map.ptr[j].key;
      const auto& value = encoded.via.map.ptr[j].val;
      if (key.type != msgpack::type::STR) continue;
      const auto name = key.as<std::string>();
      auto read_i32_triplet = [&](int32_t (&target)[3]) {
        if (value.type != msgpack::type::ARRAY || value.via.array.size != 3) {
          throw std::runtime_error("AMR patch integer triplet has invalid shape");
        }
        for (int axis = 0; axis < 3; ++axis) target[axis] = value.via.array.ptr[axis].as<int32_t>();
      };
      auto read_f64_triplet = [&](double (&target)[3]) {
        if (value.type != msgpack::type::ARRAY || value.via.array.size != 3) {
          throw std::runtime_error("AMR patch real triplet has invalid shape");
        }
        for (int axis = 0; axis < 3; ++axis) target[axis] = value.via.array.ptr[axis].as<double>();
      };
      if (name == "level") patch.level = value.as<int16_t>();
      else if (name == "lo") read_i32_triplet(patch.box.lo);
      else if (name == "hi") read_i32_triplet(patch.box.hi);
      else if (name == "dx") read_f64_triplet(patch.geom.dx);
      else if (name == "x0") read_f64_triplet(patch.geom.x0);
      else if (name == "index_origin") read_i32_triplet(patch.geom.index_origin);
      else if (name == "is_periodic") {
        if (value.type != msgpack::type::ARRAY || value.via.array.size != 3) {
          throw std::runtime_error("AMR patch periodicity triplet has invalid shape");
        }
        for (int axis = 0; axis < 3; ++axis) patch.geom.is_periodic[axis] = value.via.array.ptr[axis].as<bool>();
      } else if (name == "scalar") desc.scalar = static_cast<ScalarType>(value.as<std::uint8_t>());
      else if (name == "extents") {
        if (value.type != msgpack::type::ARRAY || value.via.array.size < 1 ||
            value.via.array.size > kMaxBufferRank) {
          throw std::runtime_error("AMR patch descriptor extents have invalid rank");
        }
        desc.rank = static_cast<std::uint8_t>(value.via.array.size);
        for (std::size_t axis = 0; axis < desc.rank; ++axis) desc.extents[axis] = value.via.array.ptr[axis].as<std::uint64_t>();
        have_extents = true;
      } else if (name == "strides_bytes") {
        if (value.type != msgpack::type::ARRAY || value.via.array.size < 1 ||
            value.via.array.size > kMaxBufferRank) {
          throw std::runtime_error("AMR patch descriptor strides have invalid rank");
        }
        for (std::size_t axis = 0; axis < value.via.array.size; ++axis) desc.strides_bytes[axis] = value.via.array.ptr[axis].as<std::int64_t>();
        stride_rank = value.via.array.size;
        have_strides = true;
      } else if (name == "data") {
        if (value.type != msgpack::type::BIN) throw std::runtime_error("AMR patch data must be binary");
        bytes.assign(value.via.bin.ptr, value.via.bin.ptr + value.via.bin.size);
      }
    }
    if (!have_extents || !have_strides || stride_rank != desc.rank || bytes.empty()) {
      throw std::runtime_error("AMR patch record is incomplete");
    }
    if (desc.scalar != ScalarType::kU8 && desc.scalar != ScalarType::kI64 &&
        desc.scalar != ScalarType::kF32 && desc.scalar != ScalarType::kF64) {
      throw std::runtime_error("AMR patch data must have a known numeric scalar type");
    }
    if (desc.rank != 3) {
      throw std::runtime_error("AMR patch data must be a Block Grid");
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (patch.box.hi[axis] < patch.box.lo[axis] ||
          desc.extents[axis] != static_cast<std::uint64_t>(
                                    static_cast<std::int64_t>(patch.box.hi[axis]) -
                                    static_cast<std::int64_t>(patch.box.lo[axis]) + 1)) {
        throw std::runtime_error("AMR patch box does not match its Block Grid extents");
      }
    }
    patch.data = ChunkBuffer::wrap(SharedByteBuffer(std::move(bytes)), desc);
    patches.push_back(std::move(patch));
  }
  return patches;
}

}  // namespace kangaroo
