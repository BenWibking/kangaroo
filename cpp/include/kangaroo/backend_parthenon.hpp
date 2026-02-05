#pragma once

#ifdef KANGAROO_USE_PARTHENON_HDF5

#include "kangaroo/dataset_backend.hpp"

#include <hdf5.h>

#include <cstddef>
#include <cstdint>
#include <array>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace kangaroo {

struct ParthenonFieldInfo {
  std::string name;
  int32_t num_components = 1;
  std::vector<std::string> component_names;
  std::string type;  // "float32" or "float64"
};

struct ParthenonMetadata {
  std::vector<std::string> var_names;
  std::vector<ParthenonFieldInfo> fields;
  int32_t finest_level = -1;
  std::vector<double> prob_lo;
  std::vector<double> prob_hi;
  std::vector<int32_t> ref_ratio;
  std::vector<std::vector<double>> cell_size;
  std::vector<std::vector<std::pair<std::array<int32_t, 3>, std::array<int32_t, 3>>>> level_boxes;
  std::vector<std::pair<std::array<int32_t, 3>, std::array<int32_t, 3>>> prob_domain;
  double time = 0.0;
};

class ParthenonBackend : public DatasetBackend {
 public:
  explicit ParthenonBackend(std::string path);
  ~ParthenonBackend() override;

  std::optional<HostView> get_chunk(const ChunkRef& ref) override;
  bool has_chunk(const ChunkRef& ref) const override;
  DatasetMetadata get_metadata() const override;

  ParthenonMetadata metadata() const;
  void register_field(int32_t field_id, const std::string& name);

 private:
  struct FieldSpec {
    std::string dataset_name;
    int32_t comp_start = 0;
    int32_t comp_count = 1;
  };

  struct DatasetInfo {
    std::string name;
    int32_t num_components = 1;
    std::string type;
    std::vector<std::size_t> dims;
  };

  std::vector<std::string> parse_string_attr(hid_t obj, const char* name) const;
  std::vector<int64_t> parse_i64_attr(hid_t obj, const char* name) const;
  std::vector<double> parse_f64_attr(hid_t obj, const char* name) const;
  void load_metadata();

  std::string path_;
  hid_t file_id_ = -1;

  std::vector<int64_t> levels_;
  std::vector<std::array<int64_t, 3>> logical_locations_;
  std::unordered_map<int32_t, std::vector<int32_t>> level_to_global_blocks_;
  std::vector<DatasetInfo> datasets_;
  std::unordered_map<std::string, DatasetInfo> dataset_by_name_;

  ParthenonMetadata meta_;

  mutable std::mutex field_mutex_;
  std::map<int32_t, FieldSpec> field_map_;
};

}  // namespace kangaroo

#endif  // KANGAROO_USE_PARTHENON_HDF5
