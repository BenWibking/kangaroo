#include "kangaroo/plan_decode.hpp"

#include <msgpack.hpp>

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace kangaroo {

namespace {

bool debug_plan_params_enabled() {
  static const bool enabled = std::getenv("KANGAROO_DEBUG_PLAN_PARAMS") != nullptr;
  return enabled;
}

std::size_t hash_bytes(const std::vector<std::uint8_t>& bytes) {
  std::size_t hash = bytes.size();
  for (auto byte : bytes) {
    hash ^= static_cast<std::size_t>(byte) + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
  }
  return hash;
}

const msgpack::object& expect_map_value(const msgpack::object& obj, const char* key) {
  if (obj.type != msgpack::type::MAP) {
    throw std::runtime_error("expected map");
  }
  for (uint32_t i = 0; i < obj.via.map.size; ++i) {
    const auto& k = obj.via.map.ptr[i].key;
    if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
      return obj.via.map.ptr[i].val;
    }
  }
  throw std::runtime_error(std::string("missing key: ") + key);
}

bool try_get_map_value(const msgpack::object& obj, const char* key, const msgpack::object** out) {
  if (obj.type != msgpack::type::MAP) {
    return false;
  }
  for (uint32_t i = 0; i < obj.via.map.size; ++i) {
    const auto& k = obj.via.map.ptr[i].key;
    if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
      *out = &obj.via.map.ptr[i].val;
      return true;
    }
  }
  return false;
}

ExecPlane parse_plane(const msgpack::object& obj) {
  auto s = obj.as<std::string>();
  if (s == "chunk") {
    return ExecPlane::Chunk;
  }
  if (s == "graph") {
    return ExecPlane::Graph;
  }
  if (s == "mixed") {
    return ExecPlane::Mixed;
  }
  throw std::runtime_error("unknown plane: " + s);
}

std::optional<std::vector<int32_t>> parse_blocks(const msgpack::object& obj) {
  if (obj.type == msgpack::type::NIL) {
    return std::nullopt;
  }
  if (obj.type != msgpack::type::ARRAY) {
    throw std::runtime_error("blocks must be array or nil");
  }
  std::vector<int32_t> blocks;
  blocks.reserve(obj.via.array.size);
  for (uint32_t i = 0; i < obj.via.array.size; ++i) {
    blocks.push_back(obj.via.array.ptr[i].as<int32_t>());
  }
  return blocks;
}

DomainIR parse_domain(const msgpack::object& obj) {
  DomainIR dom;
  dom.step = expect_map_value(obj, "step").as<int32_t>();
  dom.level = expect_map_value(obj, "level").as<int16_t>();
  const msgpack::object* blocks_obj = nullptr;
  if (try_get_map_value(obj, "blocks", &blocks_obj)) {
    dom.blocks = parse_blocks(*blocks_obj);
  }
  return dom;
}

std::vector<FieldRefIR> parse_field_refs(const msgpack::object& obj) {
  if (obj.type != msgpack::type::ARRAY) {
    throw std::runtime_error("field list must be array");
  }
  std::vector<FieldRefIR> out;
  out.reserve(obj.via.array.size);
  for (uint32_t i = 0; i < obj.via.array.size; ++i) {
    const auto& entry = obj.via.array.ptr[i];
    FieldRefIR ref;
    ref.field = expect_map_value(entry, "field").as<int32_t>();
    ref.version = expect_map_value(entry, "version").as<int32_t>();
    const msgpack::object* domain_obj = nullptr;
    if (try_get_map_value(entry, "domain", &domain_obj)) {
      ref.domain = parse_domain(*domain_obj);
    }
    out.push_back(ref);
  }
  return out;
}

DepRuleIR parse_deps(const msgpack::object& obj) {
  DepRuleIR deps;
  if (obj.type != msgpack::type::MAP) {
    return deps;
  }
  deps.kind = expect_map_value(obj, "kind").as<std::string>();
  const msgpack::object* width_obj = nullptr;
  if (try_get_map_value(obj, "width", &width_obj)) {
    deps.width = width_obj->as<int32_t>();
  }
  const msgpack::object* faces_obj = nullptr;
  if (try_get_map_value(obj, "faces", &faces_obj)) {
    if (faces_obj->type != msgpack::type::ARRAY || faces_obj->via.array.size != 6) {
      throw std::runtime_error("faces must be array of 6");
    }
    for (uint32_t i = 0; i < 6; ++i) {
      const auto& face_obj = faces_obj->via.array.ptr[i];
      if (face_obj.type == msgpack::type::BOOLEAN) {
        deps.faces[i] = face_obj.as<bool>();
      } else if (face_obj.type == msgpack::type::POSITIVE_INTEGER ||
                 face_obj.type == msgpack::type::NEGATIVE_INTEGER) {
        deps.faces[i] = face_obj.as<int64_t>() != 0;
      } else {
        throw std::runtime_error("faces entries must be bool or int");
      }
    }
  }
  const msgpack::object* halo_obj = nullptr;
  if (try_get_map_value(obj, "halo_inputs", &halo_obj)) {
    if (halo_obj->type != msgpack::type::ARRAY) {
      throw std::runtime_error("halo_inputs must be array");
    }
    deps.halo_inputs.reserve(halo_obj->via.array.size);
    for (uint32_t i = 0; i < halo_obj->via.array.size; ++i) {
      deps.halo_inputs.push_back(halo_obj->via.array.ptr[i].as<int32_t>());
    }
  }
  return deps;
}

std::vector<std::uint8_t> repack_params(const msgpack::object& obj) {
  msgpack::sbuffer buffer;
  msgpack::pack(buffer, obj);
  return std::vector<std::uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

CoveredBoxListIR parse_covered_box_list(const msgpack::object& obj) {
  if (obj.type != msgpack::type::ARRAY) {
    throw std::runtime_error("covered box list must be array");
  }
  CoveredBoxListIR boxes;
  boxes.reserve(obj.via.array.size);
  for (uint32_t i = 0; i < obj.via.array.size; ++i) {
    const auto& entry = obj.via.array.ptr[i];
    if (entry.type != msgpack::type::ARRAY || entry.via.array.size != 2) {
      throw std::runtime_error("covered box entry must be [lo, hi]");
    }
    const auto& lo = entry.via.array.ptr[0];
    const auto& hi = entry.via.array.ptr[1];
    if (lo.type != msgpack::type::ARRAY || hi.type != msgpack::type::ARRAY ||
        lo.via.array.size != 3 || hi.via.array.size != 3) {
      throw std::runtime_error("covered box lo/hi must be length-3 arrays");
    }
    CoveredBoxIR box;
    for (uint32_t d = 0; d < 3; ++d) {
      box.lo[d] = lo.via.array.ptr[d].as<int32_t>();
      box.hi[d] = hi.via.array.ptr[d].as<int32_t>();
    }
    boxes.push_back(box);
  }
  return boxes;
}

std::vector<CoveredBoxListIR> parse_shared_covered_boxes(const msgpack::object& root) {
  const msgpack::object* shared_obj = nullptr;
  if (!try_get_map_value(root, "shared_covered_boxes", &shared_obj)) {
    return {};
  }
  if (shared_obj->type != msgpack::type::ARRAY) {
    throw std::runtime_error("shared_covered_boxes must be array");
  }
  std::vector<CoveredBoxListIR> shared;
  shared.reserve(shared_obj->via.array.size);
  for (uint32_t i = 0; i < shared_obj->via.array.size; ++i) {
    shared.push_back(parse_covered_box_list(shared_obj->via.array.ptr[i]));
  }
  return shared;
}

int32_t parse_covered_boxes_ref(const msgpack::object& obj, std::size_t shared_count) {
  if (obj.type != msgpack::type::MAP) {
    return -1;
  }

  const msgpack::object* covered_boxes_ref = nullptr;
  if (!try_get_map_value(obj, "covered_boxes_ref", &covered_boxes_ref)) {
    return -1;
  }

  int64_t ref_idx = -1;
  if (covered_boxes_ref->type == msgpack::type::POSITIVE_INTEGER ||
      covered_boxes_ref->type == msgpack::type::NEGATIVE_INTEGER) {
    ref_idx = covered_boxes_ref->as<int64_t>();
  } else {
    throw std::runtime_error("covered_boxes_ref must be integer");
  }
  if (ref_idx < 0 || ref_idx >= static_cast<int64_t>(shared_count)) {
    throw std::runtime_error("covered_boxes_ref out of range");
  }
  return static_cast<int32_t>(ref_idx);
}

std::vector<std::uint8_t> repack_params_without_covered_ref(const msgpack::object& obj) {
  if (obj.type != msgpack::type::MAP) {
    return repack_params(obj);
  }

  const msgpack::object* covered_boxes_ref = nullptr;
  if (!try_get_map_value(obj, "covered_boxes_ref", &covered_boxes_ref)) {
    return repack_params(obj);
  }

  uint32_t entry_count = 0;
  for (uint32_t i = 0; i < obj.via.map.size; ++i) {
    const auto& key = obj.via.map.ptr[i].key;
    if (key.type == msgpack::type::STR) {
      const auto key_str = key.as<std::string>();
      if (key_str == "covered_boxes_ref") {
        continue;
      }
    }
    ++entry_count;
  }

  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(buffer);
  packer.pack_map(entry_count);
  for (uint32_t i = 0; i < obj.via.map.size; ++i) {
    const auto& key = obj.via.map.ptr[i].key;
    const auto& value = obj.via.map.ptr[i].val;
    if (key.type == msgpack::type::STR) {
      const auto key_str = key.as<std::string>();
      if (key_str == "covered_boxes_ref") {
        continue;
      }
    }
    packer.pack(key);
    packer.pack(value);
  }

  return std::vector<std::uint8_t>(buffer.data(), buffer.data() + buffer.size());
}

std::size_t inline_covered_box_count(const msgpack::object& obj) {
  const msgpack::object* covered_boxes = nullptr;
  if (!try_get_map_value(obj, "covered_boxes", &covered_boxes) ||
      covered_boxes->type != msgpack::type::ARRAY) {
    return 0;
  }
  return covered_boxes->via.array.size;
}

}  // namespace

PlanIR decode_plan_msgpack(std::span<const std::uint8_t> payload) {
  msgpack::object_handle handle = msgpack::unpack(reinterpret_cast<const char*>(payload.data()), payload.size());
  msgpack::object root = handle.get();
  const bool debug_plan_params = debug_plan_params_enabled();

  const auto& stages_obj = expect_map_value(root, "stages");
  if (stages_obj.type != msgpack::type::ARRAY) {
    throw std::runtime_error("stages must be array");
  }

  PlanIR plan;
  plan.shared_covered_boxes = parse_shared_covered_boxes(root);
  plan.stages.reserve(stages_obj.via.array.size);

  for (uint32_t si = 0; si < stages_obj.via.array.size; ++si) {
    const auto& stage_obj = stages_obj.via.array.ptr[si];
    StageIR stage;
    stage.name = expect_map_value(stage_obj, "name").as<std::string>();
    stage.plane = parse_plane(expect_map_value(stage_obj, "plane"));

    const auto& after_obj = expect_map_value(stage_obj, "after");
    if (after_obj.type != msgpack::type::ARRAY) {
      throw std::runtime_error("after must be array");
    }
    stage.after.reserve(after_obj.via.array.size);
    for (uint32_t i = 0; i < after_obj.via.array.size; ++i) {
      stage.after.push_back(after_obj.via.array.ptr[i].as<int32_t>());
    }

    const auto& templates_obj = expect_map_value(stage_obj, "templates");
    if (templates_obj.type != msgpack::type::ARRAY) {
      throw std::runtime_error("templates must be array");
    }
    stage.templates.reserve(templates_obj.via.array.size);

    for (uint32_t ti = 0; ti < templates_obj.via.array.size; ++ti) {
      const auto& tmpl_obj = templates_obj.via.array.ptr[ti];
      TaskTemplateIR tmpl;
      tmpl.name = expect_map_value(tmpl_obj, "name").as<std::string>();
      tmpl.plane = parse_plane(expect_map_value(tmpl_obj, "plane"));
      tmpl.kernel = expect_map_value(tmpl_obj, "kernel").as<std::string>();

      const auto& domain_obj = expect_map_value(tmpl_obj, "domain");
      tmpl.domain = parse_domain(domain_obj);

      tmpl.inputs = parse_field_refs(expect_map_value(tmpl_obj, "inputs"));
      tmpl.outputs = parse_field_refs(expect_map_value(tmpl_obj, "outputs"));
      const msgpack::object* out_bytes_obj = nullptr;
      if (try_get_map_value(tmpl_obj, "output_bytes", &out_bytes_obj)) {
        if (out_bytes_obj->type != msgpack::type::ARRAY) {
          throw std::runtime_error("output_bytes must be array");
        }
        tmpl.output_bytes.reserve(out_bytes_obj->via.array.size);
        for (uint32_t i = 0; i < out_bytes_obj->via.array.size; ++i) {
          tmpl.output_bytes.push_back(out_bytes_obj->via.array.ptr[i].as<int32_t>());
        }
      }
      tmpl.deps = parse_deps(expect_map_value(tmpl_obj, "deps"));
      const auto& params_obj = expect_map_value(tmpl_obj, "params");
      tmpl.covered_boxes_ref = parse_covered_boxes_ref(params_obj, plan.shared_covered_boxes.size());
      tmpl.params_msgpack = repack_params_without_covered_ref(params_obj);
      if (debug_plan_params) {
        if (tmpl.covered_boxes_ref >= 0) {
          const auto ref_idx = static_cast<std::size_t>(tmpl.covered_boxes_ref);
          const std::size_t covered_boxes_count = plan.shared_covered_boxes[ref_idx].size();
          std::cout << "[kangaroo][plan-params] stage=" << stage.name
                    << " template=" << tmpl.name
                    << " kernel=" << tmpl.kernel
                    << " covered_boxes_ref=" << tmpl.covered_boxes_ref
                    << " covered_boxes_count=" << covered_boxes_count
                    << " params_hash=" << hash_bytes(tmpl.params_msgpack)
                    << std::endl;
        } else if (const auto covered_boxes_count = inline_covered_box_count(params_obj);
                   covered_boxes_count > 0) {
          std::cout << "[kangaroo][plan-params] stage=" << stage.name
                    << " template=" << tmpl.name
                    << " kernel=" << tmpl.kernel
                    << " covered_boxes_ref=-1"
                    << " covered_boxes_count=" << covered_boxes_count
                    << " params_hash=" << hash_bytes(tmpl.params_msgpack)
                    << std::endl;
        }
      }

      if (tmpl.deps.kind == "FaceNeighbors") {
        if (tmpl.inputs.empty()) {
          throw std::runtime_error("FaceNeighbors deps require at least one input field");
        }
        for (int32_t idx : tmpl.deps.halo_inputs) {
          if (idx < 0 || idx >= static_cast<int32_t>(tmpl.inputs.size())) {
            throw std::runtime_error("halo_inputs index out of range");
          }
        }
      }

      stage.templates.push_back(std::move(tmpl));
    }

    plan.stages.push_back(std::move(stage));
  }

  return plan;
}

}  // namespace kangaroo
