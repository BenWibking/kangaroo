#pragma once

#include "kangaroo/dataset_backend.hpp"
#include "kangaroo/plotfile_reader.hpp"

#include <map>
#include <mutex>
#include <string>

namespace kangaroo {

class PlotfileBackend : public DatasetBackend {
 public:
  explicit PlotfileBackend(std::string plotfile_dir);

  std::string kind() const override { return "amrex"; }
  std::optional<ChunkBuffer> get_chunk(const ChunkRef& ref) override;
  std::vector<std::optional<ChunkBuffer>> get_chunks(const std::vector<ChunkRef>& refs) override;
  bool has_chunk(const ChunkRef& ref) const override;
  std::optional<BufferDesc> describe_chunk(const ChunkRef& ref) const override;
  std::size_t estimate_chunk_bytes(const ChunkRef& ref) const override;
  std::optional<std::uint64_t> estimate_particle_chunk_records(
      const std::string& particle_type, std::int64_t chunk_index) const override;
  DatasetMetadata metadata(int32_t step) const override;
  void register_field(int32_t field_id, const std::string& name) override;
  void register_field_component(int32_t field_id, int32_t component_index) override;
  std::vector<std::string> list_particle_types() const override;
  std::vector<std::string> list_particle_fields(const std::string& particle_type) const override;
  int64_t particle_chunk_count(const std::string& particle_type) const override;
  ParticleFieldChunk read_particle_field_chunk(
      const std::string& particle_type, const std::string& field_name,
      int64_t chunk_index) const override;
  ParticleFieldChunk read_particle_field_grid(
      const std::string& particle_type, const std::string& field_name,
      int level, int grid_index) const override;
  DatasetBackendSnapshot snapshot() const override;

  // Helpers for field mapping
  void register_field(int32_t field_id, int32_t component_index);
  int32_t get_component_index(int32_t field_id) const;
  std::map<int32_t, int32_t> field_map() const;
  void set_field_map(std::map<int32_t, int32_t> field_map);
  
 private:
  std::string plotfile_dir_;
  plotfile::PlotfileReader reader_;
  mutable std::mutex map_mutex_;
  std::map<int32_t, int32_t> field_map_;
};

}  // namespace kangaroo
