#include "kernel_param_support.hpp"

namespace kangaroo {

std::shared_ptr<const CoveredBoxListIR>
parse_covered_boxes_param(const msgpack::object &root) {
  const auto *boxes = find_msgpack_map_value(root, "covered_boxes");
  if (boxes == nullptr || boxes->type != msgpack::type::ARRAY) {
    return {};
  }

  auto parsed = std::make_shared<CoveredBoxListIR>();
  parsed->reserve(boxes->via.array.size);
  for (uint32_t i = 0; i < boxes->via.array.size; ++i) {
    const auto &entry = boxes->via.array.ptr[i];
    if (entry.type != msgpack::type::ARRAY || entry.via.array.size != 2) {
      continue;
    }
    const auto &lo = entry.via.array.ptr[0];
    const auto &hi = entry.via.array.ptr[1];
    if (lo.type != msgpack::type::ARRAY || hi.type != msgpack::type::ARRAY ||
        lo.via.array.size != 3 || hi.via.array.size != 3) {
      continue;
    }
    CoveredBoxIR box;
    for (uint32_t d = 0; d < 3; ++d) {
      box.lo[d] = lo.via.array.ptr[d].as<int32_t>();
      box.hi[d] = hi.via.array.ptr[d].as<int32_t>();
    }
    parsed->push_back(box);
  }
  return parsed;
}

std::size_t
covered_box_count(const std::shared_ptr<const CoveredBoxListIR> &boxes) {
  return boxes ? boxes->size() : 0;
}

bool covered_box_contains(const CoveredBoxIR &box, int i, int j, int k) {
  return i >= box.lo[0] && i <= box.hi[0] && j >= box.lo[1] && j <= box.hi[1] &&
         k >= box.lo[2] && k <= box.hi[2];
}

} // namespace kangaroo
