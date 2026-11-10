#pragma once

#include "kangaroo/data_service.hpp"
#include "kangaroo/runmeta.hpp"

#include <span>
#include <vector>

namespace kangaroo {

struct AmrPatchRecord {
  int16_t level = 0;
  IndexBox3 box;
  LevelGeom geom;
  ChunkBuffer data;
};

ChunkBuffer encode_amr_patch_payload(std::span<const AmrPatchRecord> patches);
std::vector<AmrPatchRecord> decode_amr_patch_payload(
    std::span<const std::uint8_t> payload);

}  // namespace kangaroo
