#pragma once

#include "kangaroo/data_service.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace kangaroo {

namespace plotfile {
class PlotfileReader;
}

struct FieldDescriptor {
  std::string name;
  std::string type;  // "float32", "float64"
  int32_t ncomp = 1;
};

struct DatasetMetadata {
  std::vector<double> prob_lo;
  std::vector<double> prob_hi;
  std::vector<int32_t> ref_ratio;
};

class DatasetBackend {
 public:
  virtual ~DatasetBackend() = default;

  virtual std::optional<HostView> get_chunk(const ChunkRef& ref) = 0;
  virtual bool has_chunk(const ChunkRef& ref) const = 0;
  virtual DatasetMetadata get_metadata() const = 0;

  // Pragmatic access for metadata discovery
  virtual const plotfile::PlotfileReader* get_plotfile_reader() const { return nullptr; }
};

}  // namespace kangaroo
