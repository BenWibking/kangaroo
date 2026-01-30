#include "kangaroo/plan_decode.hpp"

#include <msgpack.hpp>

#include <optional>
#include <string>
#include <vector>

namespace kangaroo {

namespace {

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

}  // namespace

PlanIR decode_plan_msgpack(std::span<const std::uint8_t> payload) {
  msgpack::object_handle handle = msgpack::unpack(reinterpret_cast<const char*>(payload.data()), payload.size());
  msgpack::object root = handle.get();

  const auto& stages_obj = expect_map_value(root, "stages");
  if (stages_obj.type != msgpack::type::ARRAY) {
    throw std::runtime_error("stages must be array");
  }

  PlanIR plan;
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
      tmpl.domain.step = expect_map_value(domain_obj, "step").as<int32_t>();
      tmpl.domain.level = expect_map_value(domain_obj, "level").as<int16_t>();
      const msgpack::object* blocks_obj = nullptr;
      if (try_get_map_value(domain_obj, "blocks", &blocks_obj)) {
        tmpl.domain.blocks = parse_blocks(*blocks_obj);
      }

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
      tmpl.params_msgpack = repack_params(expect_map_value(tmpl_obj, "params"));

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
