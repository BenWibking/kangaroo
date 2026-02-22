#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <string>
#include <string_view>
#include <vector>

namespace kangaroo::plotfile {

struct IntVect {
  int32_t x{0};
  int32_t y{0};
  int32_t z{0};
};

struct Box {
  IntVect lo;
  IntVect hi;

  int64_t num_pts() const;
};

struct BoxArray {
  std::vector<Box> boxes;
};

enum class RealType {
  kFloat32,
  kFloat64
};

struct RealDescriptor {
  RealType type{RealType::kFloat64};
  bool little_endian{true};

  static RealDescriptor parse(std::istream& is);
};

struct FabOnDisk {
  std::string file_name;
  int64_t offset{0};
};

struct VisMFHeader {
  int32_t version{0};   // Version_v1 only for DiskGalaxy
  int32_t how{0};
  int32_t ncomp{0};
  IntVect ngrow{};
  BoxArray box_array{};
  std::vector<FabOnDisk> fab_on_disk;
  std::vector<std::vector<double>> min_vals;
  std::vector<std::vector<double>> max_vals;
};

struct PlotfileHeader {
  std::string file_version;
  int32_t ncomp{0};
  std::vector<std::string> var_names;
  int32_t spacedim{0};
  double time{0.0};
  int32_t finest_level{0};
  std::vector<double> prob_lo;
  std::vector<double> prob_hi;
  std::vector<int32_t> ref_ratio;
  std::vector<Box> prob_domain;
  std::vector<int32_t> level_steps;
  std::vector<std::vector<double>> cell_size;
  int32_t coord_sys{0};
  int32_t bwidth{0};
  std::vector<std::string> mf_name;
};

struct FabHeader {
  Box box;
  int32_t ncomp{0};
  RealDescriptor real_desc{};
  bool old_format{false};
};

struct FabData {
  std::vector<std::uint8_t> bytes;
  int32_t nx{0};
  int32_t ny{0};
  int32_t nz{0};
  int32_t ncomp{0};
  RealType type{RealType::kFloat64};
};

struct ParticleHeader {
  std::string version;
  int32_t spatial_dim{0};
  int32_t num_real{0};
  int32_t num_int{0};
  int32_t finest_level{0};
  bool is_single{false};
  bool is_checkpoint{false};
  bool legacy{false};
  std::vector<std::string> real_names;
  std::vector<std::string> int_names;
  std::vector<int32_t> num_grids;
  std::vector<std::vector<int32_t>> file_nums;
  std::vector<std::vector<int32_t>> particle_counts;
  std::vector<std::vector<int64_t>> offsets;
};

struct ParticleArrayData {
  std::vector<std::uint8_t> bytes;
  std::string dtype;
  int64_t count{0};
};

PlotfileHeader parse_plotfile_header(const std::string& plotfile_dir);
VisMFHeader parse_vismf_header(const std::string& cell_h_path);
FabHeader read_fab_header(std::istream& is);
FabData read_fab_payload(std::istream& is, const FabHeader& hdr, int32_t comp_start,
                         int32_t comp_count);

class PlotfileReader {
public:
  explicit PlotfileReader(std::string plotfile_dir);

  const PlotfileHeader& header() const;
  const VisMFHeader& vismf_header(int level) const;

  int32_t num_levels() const;
  int32_t num_fabs(int level) const;

  FabData read_fab(int level, int fab_index, int comp_start, int comp_count);
  std::vector<std::string> particle_types() const;
  std::vector<std::string> particle_fields(const std::string& particle_type) const;
  int64_t particle_chunk_count(const std::string& particle_type) const;
  ParticleArrayData read_particle_field_chunk(const std::string& particle_type,
                                              const std::string& field_name,
                                              int64_t chunk_index) const;

private:
  struct ParticleSpecies {
    std::string name;
    std::string dir;
    ParticleHeader header;
  };

  void discover_particles();
  const ParticleSpecies& get_particle_species(const std::string& particle_type) const;

  std::string plotfile_dir_;
  PlotfileHeader header_;
  std::vector<VisMFHeader> vismf_headers_;
  std::vector<ParticleSpecies> particles_;
};

}  // namespace kangaroo::plotfile
