#pragma once

#include "kangaroo/data_service.hpp"

#include <cstddef>
#include <cstdint>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace kangaroo {

struct FieldDescriptor {
  std::string name;
  std::string type;  // "float32", "float64"
  int32_t ncomp = 1;
};

struct DatasetBox {
  std::array<int32_t, 3> lo{0, 0, 0};
  std::array<int32_t, 3> hi{0, 0, 0};
};

struct DatasetVariableInfo {
  std::string name;
  int32_t num_components = 1;
  std::vector<std::string> component_names;
  std::string type;
};

struct DatasetMetadata {
  std::vector<std::string> var_names;
  std::vector<std::string> mesh_names;
  std::string selected_mesh;
  int32_t finest_level = -1;
  double time = 0.0;
  std::vector<double> prob_lo;
  std::vector<double> prob_hi;
  std::vector<int32_t> ref_ratio;
  std::vector<std::vector<double>> cell_size;
  std::vector<std::vector<DatasetBox>> level_boxes;
  std::vector<DatasetBox> prob_domain;
  std::vector<DatasetVariableInfo> fields;
};

struct ParticleFieldChunk {
  std::vector<std::uint8_t> bytes;
  std::string dtype;
  int64_t count = 0;
};

struct DatasetBackendSnapshot {
  std::string kind;
  std::map<int32_t, int32_t> component_fields;
  std::unordered_map<ChunkRef, ChunkBuffer, ChunkRefHash, ChunkRefEq> memory_chunks;
};

class DatasetBackend {
 public:
  virtual ~DatasetBackend() = default;

  virtual std::string kind() const = 0;

  virtual std::optional<ChunkBuffer> get_chunk(const ChunkRef& ref) = 0;
  virtual std::vector<std::optional<ChunkBuffer>> get_chunks(const std::vector<ChunkRef>& refs) {
    std::vector<std::optional<ChunkBuffer>> out;
    out.reserve(refs.size());
    for (const auto& ref : refs) {
      out.push_back(get_chunk(ref));
    }
    return out;
  }
  virtual bool has_chunk(const ChunkRef& ref) const = 0;
  virtual std::optional<BufferDesc> describe_chunk(const ChunkRef& ref) const {
    (void)ref;
    return std::nullopt;
  }
  virtual std::size_t estimate_chunk_bytes(const ChunkRef& ref) const {
    (void)ref;
    return 0;
  }
  virtual std::optional<std::uint64_t> estimate_particle_chunk_records(
      const std::string& particle_type, std::int64_t chunk_index) const {
    (void)particle_type;
    (void)chunk_index;
    return std::nullopt;
  }
  virtual DatasetMetadata metadata(int32_t step) const = 0;

  virtual void set_chunk(const ChunkRef& ref, ChunkBuffer view);
  virtual void register_field(int32_t field_id, const std::string& name);
  virtual void register_field_component(int32_t field_id, int32_t component_index);
  virtual std::vector<std::string> list_meshes(int32_t step) const;
  virtual void select_mesh(const std::string& name);
  virtual std::vector<std::string> list_particle_types() const;
  virtual std::vector<std::string> list_particle_fields(const std::string& particle_type) const;
  virtual int64_t particle_chunk_count(const std::string& particle_type) const;
  virtual ParticleFieldChunk read_particle_field_chunk(
      const std::string& particle_type, const std::string& field_name,
      int64_t chunk_index) const;
  virtual ParticleFieldChunk read_particle_field_grid(
      const std::string& particle_type, const std::string& field_name,
      int level, int grid_index) const;
  virtual DatasetBackendSnapshot snapshot() const;
};

std::shared_ptr<DatasetBackend> make_dataset_backend(const std::string& uri);
std::shared_ptr<DatasetBackend> restore_dataset_backend(
    const std::string& uri, const DatasetBackendSnapshot& snapshot);

}  // namespace kangaroo
