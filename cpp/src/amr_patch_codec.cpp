#include "kangaroo/amr_patch_codec.hpp"

#include "amr_patch_generated.h"

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace kangaroo {

namespace {

void validate_patch(const AmrPatchRecord &patch) {
  const auto &desc = patch.data.desc();
  desc.validate(patch.data.bytes());
  if (desc.scalar == ScalarType::kOpaque || desc.rank != 3) {
    throw std::runtime_error("AMR patch data must be a numeric Block Grid");
  }
  for (int axis = 0; axis < 3; ++axis) {
    if (patch.box.hi[axis] < patch.box.lo[axis] ||
        desc.extents[axis] !=
            static_cast<std::uint64_t>(
                static_cast<std::int64_t>(patch.box.hi[axis]) -
                static_cast<std::int64_t>(patch.box.lo[axis]) + 1)) {
      throw std::runtime_error(
          "AMR patch box does not match its Block Grid extents");
    }
  }
}

} // namespace

ChunkBuffer encode_amr_patch_payload(std::span<const AmrPatchRecord> patches) {
  amr_fb::AmrPatchPayloadT payload;
  payload.patches.reserve(patches.size());
  for (const auto &patch : patches) {
    validate_patch(patch);
    const auto &desc = patch.data.desc();
    auto encoded = std::make_unique<amr_fb::AmrPatchT>();
    encoded->level = patch.level;
    encoded->lo = std::make_unique<amr_fb::Int3>(
        patch.box.lo[0], patch.box.lo[1], patch.box.lo[2]);
    encoded->hi = std::make_unique<amr_fb::Int3>(
        patch.box.hi[0], patch.box.hi[1], patch.box.hi[2]);
    encoded->dx = std::make_unique<amr_fb::Double3>(
        patch.geom.dx[0], patch.geom.dx[1], patch.geom.dx[2]);
    encoded->x0 = std::make_unique<amr_fb::Double3>(
        patch.geom.x0[0], patch.geom.x0[1], patch.geom.x0[2]);
    encoded->index_origin = std::make_unique<amr_fb::Int3>(
        patch.geom.index_origin[0], patch.geom.index_origin[1],
        patch.geom.index_origin[2]);
    encoded->is_periodic = std::make_unique<amr_fb::Bool3>(
        patch.geom.is_periodic[0], patch.geom.is_periodic[1],
        patch.geom.is_periodic[2]);
    encoded->scalar = static_cast<std::uint8_t>(desc.scalar);
    encoded->extents.assign(desc.extents.begin(),
                            desc.extents.begin() + desc.rank);
    encoded->strides_bytes.assign(desc.strides_bytes.begin(),
                                  desc.strides_bytes.begin() + desc.rank);
    const auto bytes = patch.data.byte_view();
    encoded->data.assign(bytes.begin(), bytes.end());
    payload.patches.push_back(std::move(encoded));
  }

  flatbuffers::FlatBufferBuilder builder;
  builder.Finish(amr_fb::AmrPatchPayload::Pack(builder, &payload),
                 amr_fb::AmrPatchPayloadIdentifier());
  const auto view = builder.GetBufferSpan();
  return ChunkBuffer::opaque(
      std::vector<std::uint8_t>(view.begin(), view.end()));
}

std::vector<AmrPatchRecord>
decode_amr_patch_payload(std::span<const std::uint8_t> payload) {
  if (payload.empty()) {
    return {};
  }
  flatbuffers::Verifier verifier(payload.data(), payload.size());
  if (!amr_fb::VerifyAmrPatchPayloadBuffer(verifier)) {
    throw std::runtime_error("invalid AMR patch FlatBuffer");
  }
  auto decoded = std::unique_ptr<amr_fb::AmrPatchPayloadT>(
      amr_fb::GetAmrPatchPayload(payload.data())->UnPack());

  std::vector<AmrPatchRecord> patches;
  patches.reserve(decoded->patches.size());
  for (auto &encoded : decoded->patches) {
    if (!encoded || !encoded->lo || !encoded->hi || !encoded->dx ||
        !encoded->x0 || !encoded->index_origin || !encoded->is_periodic ||
        encoded->extents.size() != 3 || encoded->strides_bytes.size() != 3 ||
        encoded->data.empty()) {
      throw std::runtime_error("AMR patch record is incomplete");
    }

    AmrPatchRecord patch;
    patch.level = encoded->level;
    patch.box.lo[0] = encoded->lo->x();
    patch.box.lo[1] = encoded->lo->y();
    patch.box.lo[2] = encoded->lo->z();
    patch.box.hi[0] = encoded->hi->x();
    patch.box.hi[1] = encoded->hi->y();
    patch.box.hi[2] = encoded->hi->z();
    patch.geom.dx[0] = encoded->dx->x();
    patch.geom.dx[1] = encoded->dx->y();
    patch.geom.dx[2] = encoded->dx->z();
    patch.geom.x0[0] = encoded->x0->x();
    patch.geom.x0[1] = encoded->x0->y();
    patch.geom.x0[2] = encoded->x0->z();
    patch.geom.index_origin[0] = encoded->index_origin->x();
    patch.geom.index_origin[1] = encoded->index_origin->y();
    patch.geom.index_origin[2] = encoded->index_origin->z();
    patch.geom.is_periodic[0] = encoded->is_periodic->x();
    patch.geom.is_periodic[1] = encoded->is_periodic->y();
    patch.geom.is_periodic[2] = encoded->is_periodic->z();

    BufferDesc desc;
    desc.scalar = static_cast<ScalarType>(encoded->scalar);
    desc.rank = 3;
    for (int axis = 0; axis < 3; ++axis) {
      desc.extents[axis] = encoded->extents[axis];
      desc.strides_bytes[axis] = encoded->strides_bytes[axis];
    }
    if (desc.scalar != ScalarType::kU8 && desc.scalar != ScalarType::kI64 &&
        desc.scalar != ScalarType::kF32 && desc.scalar != ScalarType::kF64) {
      throw std::runtime_error(
          "AMR patch data must have a known numeric scalar type");
    }

    patch.data =
        ChunkBuffer::wrap(SharedByteBuffer(std::move(encoded->data)), desc);
    validate_patch(patch);
    patches.push_back(std::move(patch));
  }
  return patches;
}

} // namespace kangaroo
