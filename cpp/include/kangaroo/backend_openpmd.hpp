#pragma once

#ifdef KANGAROO_USE_OPENPMD

#include "kangaroo/dataset_backend.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <openPMD/openPMD.hpp>

namespace kangaroo {

struct OpenPMDFieldInfo {
  std::string name;
  std::string mesh_name;
  std::string component_name;
  std::string type;  // "float32" or "float64"
  int32_t ncomp = 1;
};

struct OpenPMDMetadata {
  std::string selected_mesh;
  std::vector<std::string> mesh_names;
  std::vector<OpenPMDFieldInfo> fields;
  int32_t finest_level = 0;
  std::vector<std::array<int32_t, 3>> ref_ratio;
  std::vector<std::array<double, 3>> cell_size;
  std::vector<std::vector<std::pair<std::array<int32_t, 3>, std::array<int32_t, 3>>>> level_boxes;
  std::vector<std::pair<std::array<int32_t, 3>, std::array<int32_t, 3>>> prob_domain;
  std::array<double, 3> prob_lo{0.0, 0.0, 0.0};
  std::array<double, 3> prob_hi{0.0, 0.0, 0.0};
};

class OpenPMDBackend : public DatasetBackend {
 public:
  explicit OpenPMDBackend(std::string uri);

  std::optional<HostView> get_chunk(const ChunkRef& ref) override;
  bool has_chunk(const ChunkRef& ref) const override;
  DatasetMetadata get_metadata() const override;

  OpenPMDMetadata metadata(int32_t step) const;
  void register_field(int32_t field_id, const std::string& name);
  std::vector<std::string> list_meshes(int32_t step) const;
  void select_mesh(const std::string& name);

 private:
  struct Patch {
    std::string mesh_name;
    openPMD::Offset storage_offset;
    openPMD::Extent storage_extent;
    std::vector<uint64_t> storage_extent_canonical;
    std::array<int, 3> storage_to_xyz{{0, 1, 2}};
    std::array<uint64_t, 3> extent_xyz{{1, 1, 1}};
    std::array<int32_t, 3> logical_lower{{0, 0, 0}};
    std::array<int32_t, 3> logical_upper{{0, 0, 0}};
    openPMD::Mesh::DataOrder data_order{openPMD::Mesh::DataOrder::C};
    bool node_centered{false};
  };

  struct LevelInfo {
    std::string mesh_name;
    std::array<double, 3> cell_size{{1.0, 1.0, 1.0}};
    std::array<double, 3> grid_origin{{0.0, 0.0, 0.0}};
    std::array<double, 3> position{{0.0, 0.0, 0.0}};
    bool node_centered{false};
    std::vector<Patch> patches;
    std::array<int32_t, 3> domain_lo{{0, 0, 0}};
    std::array<int32_t, 3> domain_hi{{0, 0, 0}};
  };

  struct Cache {
    std::string selected_mesh;
    std::vector<std::string> mesh_names;
    std::vector<OpenPMDFieldInfo> fields;
    std::vector<LevelInfo> levels;
    int32_t finest_level{0};
    std::array<double, 3> prob_lo{{0.0, 0.0, 0.0}};
    std::array<double, 3> prob_hi{{0.0, 0.0, 0.0}};
    std::vector<std::array<int32_t, 3>> ref_ratio;
  };

  struct FieldSpec {
    std::string component_name;
  };

  struct GeometryInfo {
    std::vector<std::string> storage_axis_labels;
    std::vector<uint64_t> extent;
    std::vector<double> grid_spacing;
    std::vector<double> grid_origin;
    std::vector<double> position;
    std::array<int, 3> storage_to_xyz{{0, 1, 2}};
  };

  Cache build_cache(int32_t step) const;
  const Cache& get_cache(int32_t step) const;

  static std::pair<std::string, int> parse_mesh_level(const std::string& mesh_name);
  static GeometryInfo get_geometry_xyz(const openPMD::Mesh& mesh,
                                       const openPMD::MeshRecordComponent* rep,
                                       openPMD::Mesh::DataOrder order);
  static std::vector<int> axis_transpose(const std::vector<std::string>& src,
                                         const std::vector<std::string>& dst);
  static bool is_cell_centered(const openPMD::Mesh& mesh,
                               const openPMD::MeshRecordComponent* rep);
  static std::string datatype_string(openPMD::Datatype dtype);

  template <typename T>
  static void scale_values(T* data, size_t count, double unit_si) {
    if (data == nullptr || count == 0) {
      return;
    }
    const T scale = static_cast<T>(unit_si);
    for (size_t i = 0; i < count; ++i) {
      data[i] *= scale;
    }
  }

  template <typename T>
  static void remap_to_xyz_layout(T* data, size_t count, const Patch& patch);

  std::string uri_;
  std::string path_;
  std::string selected_mesh_;

  std::unique_ptr<openPMD::Series> series_;
  std::vector<uint64_t> iteration_indices_;

  mutable std::mutex cache_mutex_;
  mutable std::unordered_map<int32_t, Cache> cache_by_step_;

  mutable std::mutex field_mutex_;
  std::unordered_map<int32_t, FieldSpec> field_map_;
};

}  // namespace kangaroo

#endif  // KANGAROO_USE_OPENPMD
