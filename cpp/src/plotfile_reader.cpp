#include "kangaroo/plotfile_reader.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace kangaroo::plotfile {

namespace {

constexpr std::streamsize kIgnoreMax = 100000;

bool is_little_endian_host() {
  const uint16_t v = 0x0102;
  const auto* p = reinterpret_cast<const uint8_t*>(&v);
  return p[0] == 0x02;
}

void expect_char(std::istream& is, char expected) {
  char c = 0;
  is >> c;
  if (c != expected) {
    throw std::runtime_error(std::string("expected '") + expected + "'");
  }
}

std::string trim_copy(const std::string& s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
    ++start;
  }
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
    --end;
  }
  return s.substr(start, end - start);
}

IntVect read_intvect(std::istream& is) {
  IntVect v{};
  expect_char(is, '(');
  is >> v.x;
  int* vals[2] = {&v.y, &v.z};
  for (int i = 0; i < 2; ++i) {
    is >> std::ws;
    if (is.peek() == ',') {
      is.ignore(kIgnoreMax, ',');
      is >> *vals[i];
    } else {
      *vals[i] = 0;
    }
  }
  is.ignore(kIgnoreMax, ')');
  return v;
}

Box read_box(std::istream& is) {
  Box box{};
  is >> std::ws;
  char c = 0;
  is >> c;
  if (c != '(') {
    throw std::runtime_error("expected '(' to start Box");
  }
  box.lo = read_intvect(is);
  box.hi = read_intvect(is);
  is >> std::ws;
  if (is.peek() == '(') {
    (void)read_intvect(is);  // optional IndexType, ignore
  }
  is.ignore(kIgnoreMax, ')');
  return box;
}

BoxArray read_box_array(std::istream& is) {
  BoxArray array;
  is.ignore(kIgnoreMax, '(');
  int count = 0;
  unsigned long long hash = 0;
  is >> count >> hash;
  if (count < 0 || count > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid BoxArray size");
  }
  array.boxes.reserve(static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    array.boxes.push_back(read_box(is));
  }
  is.ignore(kIgnoreMax, ')');
  return array;
}

template <typename T>
std::vector<T> read_descriptor_array(std::istream& is) {
  std::vector<T> out;
  expect_char(is, '(');
  int size = 0;
  is >> size;
  expect_char(is, ',');
  expect_char(is, '(');
  if (size < 0 || size > std::numeric_limits<int>::max()) {
    throw std::runtime_error("invalid descriptor array size");
  }
  out.resize(static_cast<size_t>(size));
  for (int i = 0; i < size; ++i) {
    is >> out[i];
  }
  expect_char(is, ')');
  expect_char(is, ')');
  return out;
}

std::vector<std::vector<double>> read_real_matrix(std::istream& is) {
  long long n = 0;
  long long m = 0;
  char ch = 0;
  is >> n >> ch >> m;
  if (ch != ',') {
    throw std::runtime_error("expected ',' in real matrix header");
  }
  if (n < 0 || m < 0) {
    throw std::runtime_error("invalid real matrix size");
  }
  std::vector<std::vector<double>> mat(static_cast<size_t>(n));
  for (long long i = 0; i < n; ++i) {
    mat[i].resize(static_cast<size_t>(m));
    for (long long j = 0; j < m; ++j) {
      is >> mat[i][j] >> ch;
      if (ch != ',') {
        throw std::runtime_error("expected ',' after real value");
      }
    }
  }
  return mat;
}

void swap_bytes_inplace(std::uint8_t* data, size_t count, size_t width) {
  for (size_t i = 0; i < count; ++i) {
    std::uint8_t* p = data + i * width;
    for (size_t j = 0; j < width / 2; ++j) {
      std::swap(p[j], p[width - 1 - j]);
    }
  }
}

struct ParticleBlock {
  int32_t count{0};
  int32_t num_real{0};
  int32_t num_int{0};
  bool is_single{false};
  int32_t int_bytes{0};
  std::vector<std::int64_t> int_data;
  std::vector<float> real_data_single;
  std::vector<double> real_data_double;
};

std::vector<std::string> read_particle_fields(const ParticleHeader& header) {
  std::vector<std::string> out;
  if (header.spatial_dim >= 1) {
    out.push_back("x");
  }
  if (header.spatial_dim >= 2) {
    out.push_back("y");
  }
  if (header.spatial_dim >= 3) {
    out.push_back("z");
  }
  out.insert(out.end(), header.real_names.begin(), header.real_names.end());
  out.insert(out.end(), header.int_names.begin(), header.int_names.end());
  return out;
}

bool parse_particle_header(const std::string& header_path, ParticleHeader& out) {
  out = ParticleHeader{};
  std::ifstream in(header_path);
  if (!in) {
    return false;
  }

  std::vector<std::string> tokens;
  std::string token;
  while (in >> token) {
    tokens.push_back(token);
  }
  if (tokens.size() < 3) {
    return false;
  }

  out.version = tokens[0];
  const bool is_modern = out.version.find("Version_Two_Dot_") != std::string::npos;
  if (!is_modern) {
    return false;
  }
  out.is_single = out.version.find("single") != std::string::npos;

  auto parse_i32 = [&](size_t idx, int32_t& value) -> bool {
    if (idx >= tokens.size()) {
      return false;
    }
    try {
      value = std::stoi(tokens[idx]);
      return true;
    } catch (...) {
      return false;
    }
  };
  auto parse_i64 = [&](size_t idx, int64_t& value) -> bool {
    if (idx >= tokens.size()) {
      return false;
    }
    try {
      value = std::stoll(tokens[idx]);
      return true;
    } catch (...) {
      return false;
    }
  };

  int32_t dm = 0;
  int32_t nr = 0;
  if (!parse_i32(1, dm) || !parse_i32(2, nr)) {
    return false;
  }

  auto parse_modern = [&]() -> bool {
    size_t idx = 3;
    if (nr < 0 || tokens.size() < idx + static_cast<size_t>(nr) + 1) {
      return false;
    }
    out.real_names.reserve(static_cast<size_t>(nr));
    for (int i = 0; i < nr; ++i) {
      out.real_names.push_back(tokens[idx++]);
    }

    int32_t ni = 0;
    if (!parse_i32(idx, ni)) {
      return false;
    }
    ++idx;
    if (ni < 0 || tokens.size() < idx + static_cast<size_t>(ni) + 4) {
      return false;
    }

    std::vector<std::string> int_names;
    int_names.reserve(static_cast<size_t>(ni));
    for (int i = 0; i < ni; ++i) {
      int_names.push_back(tokens[idx++]);
    }

    int32_t checkpoint_flag = 0;
    int64_t num_particles = 0;
    int64_t next_id = 0;
    int32_t finest_level = 0;
    if (!parse_i32(idx++, checkpoint_flag) || !parse_i64(idx++, num_particles) ||
        !parse_i64(idx++, next_id) || !parse_i32(idx++, finest_level)) {
      return false;
    }
    if (finest_level < 0 || tokens.size() < idx + static_cast<size_t>(finest_level + 1)) {
      return false;
    }

    out.num_grids.reserve(static_cast<size_t>(finest_level + 1));
    for (int lev = 0; lev <= finest_level; ++lev) {
      int32_t ngrids = 0;
      if (!parse_i32(idx++, ngrids) || ngrids < 0) {
        return false;
      }
      out.num_grids.push_back(ngrids);
    }

    out.file_nums.resize(static_cast<size_t>(finest_level + 1));
    out.particle_counts.resize(static_cast<size_t>(finest_level + 1));
    out.offsets.resize(static_cast<size_t>(finest_level + 1));
    for (int lev = 0; lev <= finest_level; ++lev) {
      const int32_t ngrids = out.num_grids[static_cast<size_t>(lev)];
      out.file_nums[static_cast<size_t>(lev)].reserve(static_cast<size_t>(ngrids));
      out.particle_counts[static_cast<size_t>(lev)].reserve(static_cast<size_t>(ngrids));
      out.offsets[static_cast<size_t>(lev)].reserve(static_cast<size_t>(ngrids));
      for (int g = 0; g < ngrids; ++g) {
        int32_t file_num = 0;
        int32_t count = 0;
        int64_t offset = 0;
        if (!parse_i32(idx++, file_num) || !parse_i32(idx++, count) || !parse_i64(idx++, offset)) {
          return false;
        }
        out.file_nums[static_cast<size_t>(lev)].push_back(file_num);
        out.particle_counts[static_cast<size_t>(lev)].push_back(count);
        out.offsets[static_cast<size_t>(lev)].push_back(offset);
      }
    }

    out.spatial_dim = dm;
    out.num_real = dm + nr;
    out.num_int = 2 * (checkpoint_flag != 0) + ni;
    out.is_checkpoint = checkpoint_flag != 0;
    out.finest_level = finest_level;
    out.int_names.clear();
    if (out.is_checkpoint) {
      out.int_names.push_back("id");
      out.int_names.push_back("cpu");
    }
    out.int_names.insert(out.int_names.end(), int_names.begin(), int_names.end());
    out.legacy = false;
    return true;
  };

  if (parse_modern()) {
    return true;
  }
  return false;
}

std::string particle_data_file(const std::string& species_dir, int level, int file_num) {
  {
    std::ostringstream os;
    os << species_dir << "/Level_" << level << "/DATA_" << std::setfill('0') << std::setw(5)
       << file_num;
    const std::string path = os.str();
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  {
    std::ostringstream os;
    os << species_dir << "/Level_" << level << "/DATA_" << std::setfill('0') << std::setw(4)
       << file_num;
    const std::string path = os.str();
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  return {};
}

int64_t next_offset_for_file(const ParticleHeader& header, int level, int file_num,
                             int64_t offset) {
  int64_t next_offset = -1;
  if (level < 0 || level >= static_cast<int>(header.file_nums.size()) ||
      level >= static_cast<int>(header.offsets.size())) {
    return next_offset;
  }
  const auto& file_nums = header.file_nums[static_cast<size_t>(level)];
  const auto& offsets = header.offsets[static_cast<size_t>(level)];
  for (size_t g = 0; g < file_nums.size(); ++g) {
    if (file_nums[g] != file_num) {
      continue;
    }
    const int64_t candidate = offsets[g];
    if (candidate > offset && (next_offset < 0 || candidate < next_offset)) {
      next_offset = candidate;
    }
  }
  return next_offset;
}

ParticleBlock read_particle_block(const std::string& species_dir, const ParticleHeader& header,
                                  int level, int grid_index) {
  ParticleBlock block;
  if (level < 0 || level > header.finest_level) {
    throw std::runtime_error("particle level out of range");
  }
  if (level >= static_cast<int>(header.file_nums.size()) ||
      level >= static_cast<int>(header.particle_counts.size()) ||
      level >= static_cast<int>(header.offsets.size())) {
    throw std::runtime_error("particle header missing level metadata");
  }
  if (grid_index < 0 || grid_index >= static_cast<int>(header.file_nums[static_cast<size_t>(level)].size())) {
    throw std::runtime_error("particle grid index out of range");
  }

  const int file_num = header.file_nums[static_cast<size_t>(level)][static_cast<size_t>(grid_index)];
  const int count = header.particle_counts[static_cast<size_t>(level)][static_cast<size_t>(grid_index)];
  const int64_t offset = header.offsets[static_cast<size_t>(level)][static_cast<size_t>(grid_index)];
  block.count = count;
  block.num_real = header.num_real;
  block.num_int = header.num_int;
  block.is_single = header.is_single;
  block.int_bytes = static_cast<int32_t>(sizeof(std::int64_t));

  if (count <= 0) {
    return block;
  }

  const std::string path = particle_data_file(species_dir, level, file_num);
  if (path.empty()) {
    throw std::runtime_error("particle data file missing");
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open particle data file");
  }
  in.seekg(0, std::ios::end);
  const int64_t file_size = static_cast<int64_t>(in.tellg());
  in.seekg(offset, std::ios::beg);

  const int64_t next_offset = next_offset_for_file(header, level, file_num, offset);
  int64_t available_bytes = -1;
  if (next_offset > offset) {
    available_bytes = next_offset - offset;
  } else if (file_size >= 0) {
    available_bytes = file_size - offset;
  }

  const int64_t real_bytes = (block.is_single ? 4LL : 8LL) * static_cast<int64_t>(block.num_real) *
                             static_cast<int64_t>(count);

  if (block.num_int > 0 && available_bytes >= 0) {
    const int64_t int_bytes_total = available_bytes - real_bytes;
    const int64_t denom = static_cast<int64_t>(block.num_int) * static_cast<int64_t>(count);
    if (int_bytes_total >= 0 && denom > 0 && int_bytes_total % denom == 0) {
      const int64_t candidate = int_bytes_total / denom;
      if (candidate == 4 || candidate == 8) {
        block.int_bytes = static_cast<int32_t>(candidate);
      }
    }
  }

  if (block.num_int > 0) {
    const size_t int_count = static_cast<size_t>(block.num_int) * static_cast<size_t>(count);
    block.int_data.resize(int_count);
    if (block.int_bytes == 8) {
      in.read(reinterpret_cast<char*>(block.int_data.data()),
              static_cast<std::streamsize>(int_count * sizeof(std::int64_t)));
    } else {
      std::vector<std::int32_t> tmp(int_count);
      in.read(reinterpret_cast<char*>(tmp.data()),
              static_cast<std::streamsize>(int_count * sizeof(std::int32_t)));
      if (in) {
        for (size_t i = 0; i < int_count; ++i) {
          block.int_data[i] = static_cast<std::int64_t>(tmp[i]);
        }
      }
    }
    if (!in) {
      throw std::runtime_error("failed to read particle integer data");
    }
  }

  const size_t real_count = static_cast<size_t>(block.num_real) * static_cast<size_t>(count);
  if (block.is_single) {
    block.real_data_single.resize(real_count);
    in.read(reinterpret_cast<char*>(block.real_data_single.data()),
            static_cast<std::streamsize>(real_count * sizeof(float)));
  } else {
    block.real_data_double.resize(real_count);
    in.read(reinterpret_cast<char*>(block.real_data_double.data()),
            static_cast<std::streamsize>(real_count * sizeof(double)));
  }
  if (!in) {
    throw std::runtime_error("failed to read particle real data");
  }

  return block;
}

}  // namespace

int64_t Box::num_pts() const {
  const int64_t nx = static_cast<int64_t>(hi.x) - lo.x + 1;
  const int64_t ny = static_cast<int64_t>(hi.y) - lo.y + 1;
  const int64_t nz = static_cast<int64_t>(hi.z) - lo.z + 1;
  return nx * ny * nz;
}

RealDescriptor RealDescriptor::parse(std::istream& is) {
  RealDescriptor desc;
  expect_char(is, '(');
  auto fmt = read_descriptor_array<long long>(is);
  expect_char(is, ',');
  auto ord = read_descriptor_array<int>(is);
  expect_char(is, ')');
  if (fmt.empty()) {
    throw std::runtime_error("empty RealDescriptor format array");
  }
  if (fmt[0] == 32) {
    desc.type = RealType::kFloat32;
  } else if (fmt[0] == 64) {
    desc.type = RealType::kFloat64;
  } else {
    throw std::runtime_error("unsupported RealDescriptor format");
  }
  bool ascending = true;
  bool descending = true;
  for (size_t i = 0; i < ord.size(); ++i) {
    const int expected_asc = static_cast<int>(i + 1);
    const int expected_desc = static_cast<int>(ord.size() - i);
    ascending = ascending && (ord[i] == expected_asc);
    descending = descending && (ord[i] == expected_desc);
  }
  if (ascending) {
    desc.little_endian = false;
  } else if (descending) {
    desc.little_endian = true;
  } else {
    throw std::runtime_error("unsupported RealDescriptor byte order");
  }
  return desc;
}

PlotfileHeader parse_plotfile_header(const std::string& plotfile_dir) {
  PlotfileHeader header;
  const std::string path = plotfile_dir + "/Header";
  std::ifstream is(path);
  if (!is) {
    throw std::runtime_error("failed to open plotfile Header: " + path);
  }
  std::string line;
  while (std::getline(is, line)) {
    line = trim_copy(line);
    if (!line.empty()) {
      header.file_version = line;
      break;
    }
  }
  is >> header.ncomp;
  header.var_names.resize(static_cast<size_t>(header.ncomp));
  std::getline(is, line);
  for (int i = 0; i < header.ncomp; ++i) {
    std::getline(is, line);
    header.var_names[static_cast<size_t>(i)] = trim_copy(line);
  }
  is >> header.spacedim >> header.time >> header.finest_level;
  const int nlevels = header.finest_level + 1;
  header.prob_lo.resize(static_cast<size_t>(header.spacedim));
  header.prob_hi.resize(static_cast<size_t>(header.spacedim));
  for (int i = 0; i < header.spacedim; ++i) {
    is >> header.prob_lo[static_cast<size_t>(i)];
  }
  for (int i = 0; i < header.spacedim; ++i) {
    is >> header.prob_hi[static_cast<size_t>(i)];
  }
  header.ref_ratio.resize(static_cast<size_t>(nlevels), 0);
  for (int i = 0; i < header.finest_level; ++i) {
    is >> header.ref_ratio[static_cast<size_t>(i)];
  }
  header.prob_domain.resize(static_cast<size_t>(nlevels));
  for (int i = 0; i < nlevels; ++i) {
    header.prob_domain[static_cast<size_t>(i)] = read_box(is);
  }
  header.level_steps.resize(static_cast<size_t>(nlevels));
  for (int i = 0; i < nlevels; ++i) {
    is >> header.level_steps[static_cast<size_t>(i)];
  }
  header.cell_size.resize(static_cast<size_t>(nlevels));
  for (int lev = 0; lev < nlevels; ++lev) {
    header.cell_size[static_cast<size_t>(lev)].resize(static_cast<size_t>(header.spacedim));
    for (int dim = 0; dim < header.spacedim; ++dim) {
      is >> header.cell_size[static_cast<size_t>(lev)][static_cast<size_t>(dim)];
    }
  }
  is >> header.coord_sys;
  is >> header.bwidth;

  header.mf_name.resize(static_cast<size_t>(nlevels));
  for (int lev = 0; lev < nlevels; ++lev) {
    int levtmp = 0;
    int ngrids = 0;
    double gtime = 0.0;
    int levstep = 0;
    is >> levtmp >> ngrids >> gtime;
    is >> levstep;
    for (int grid = 0; grid < ngrids; ++grid) {
      for (int dim = 0; dim < header.spacedim; ++dim) {
        double lo = 0.0;
        double hi = 0.0;
        is >> lo >> hi;
      }
    }
    std::string relname;
    is >> relname;
    header.mf_name[static_cast<size_t>(lev)] = relname;
  }
  if (!is.good()) {
    throw std::runtime_error("failed to parse plotfile Header: " + path);
  }
  return header;
}

VisMFHeader parse_vismf_header(const std::string& cell_h_path) {
  VisMFHeader header;
  std::ifstream is(cell_h_path);
  if (!is) {
    throw std::runtime_error("failed to open VisMF header: " + cell_h_path);
  }
  is >> header.version >> header.how >> header.ncomp;
  is >> std::ws;
  if (is.peek() == '(') {
    header.ngrow = read_intvect(is);
  } else {
    int ng = 0;
    is >> ng;
    header.ngrow = IntVect{ng, ng, ng};
  }
  header.box_array = read_box_array(is);

  long long n_fab = 0;
  is >> n_fab;
  if (n_fab < 0) {
    throw std::runtime_error("invalid FabOnDisk count");
  }
  header.fab_on_disk.resize(static_cast<size_t>(n_fab));
  for (long long i = 0; i < n_fab; ++i) {
    std::string tag;
    is >> tag;
    if (tag != "FabOnDisk:") {
      throw std::runtime_error("expected FabOnDisk entry");
    }
    is >> header.fab_on_disk[static_cast<size_t>(i)].file_name;
    is >> header.fab_on_disk[static_cast<size_t>(i)].offset;
  }

  if (header.version == 1) {
    header.min_vals = read_real_matrix(is);
    header.max_vals = read_real_matrix(is);
  } else {
    throw std::runtime_error("unsupported VisMF header version");
  }

  if (!is.good()) {
    throw std::runtime_error("failed to parse VisMF header: " + cell_h_path);
  }
  return header;
}

FabHeader read_fab_header(std::istream& is) {
  FabHeader header;
  char c = 0;
  is >> c;
  if (c != 'F') {
    throw std::runtime_error("expected FAB header");
  }
  is >> c;
  if (c != 'A') {
    throw std::runtime_error("expected FAB header");
  }
  is >> c;
  if (c != 'B') {
    throw std::runtime_error("expected FAB header");
  }
  is >> c;
  if (c == ':') {
    throw std::runtime_error("old FAB format not supported");
  } else {
    if (!std::isspace(static_cast<unsigned char>(c))) {
      is.putback(c);
    }
    header.real_desc = RealDescriptor::parse(is);
    header.box = read_box(is);
    is >> header.ncomp;
    // Consume the end-of-line delimiter after the ASCII header so binary data starts aligned.
    if (is.peek() == '\r') {
      is.get();
      if (is.peek() == '\n') {
        is.get();
      }
    } else if (is.peek() == '\n') {
      is.get();
    }
  }
  if (!is.good()) {
    throw std::runtime_error("failed to parse FAB header");
  }
  return header;
}

FabData read_fab_payload(std::istream& is, const FabHeader& hdr, int32_t comp_start,
                         int32_t comp_count) {
  FabData data;
  const int64_t npts = hdr.box.num_pts();
  if (npts <= 0) {
    throw std::runtime_error("invalid FAB box size");
  }
  if (comp_start < 0 || comp_count <= 0 || comp_start + comp_count > hdr.ncomp) {
    throw std::runtime_error("invalid component range");
  }
  const size_t bytes_per = (hdr.real_desc.type == RealType::kFloat32) ? 4 : 8;
  const size_t total_count = static_cast<size_t>(npts) * static_cast<size_t>(hdr.ncomp);
  const size_t total_bytes = total_count * bytes_per;
  std::vector<std::uint8_t> raw(total_bytes);
  is.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(total_bytes));
  if (!is.good()) {
    throw std::runtime_error("failed to read FAB payload");
  }
  const bool need_swap = (hdr.real_desc.little_endian != is_little_endian_host());
  if (need_swap) {
    swap_bytes_inplace(raw.data(), total_count, bytes_per);
  }
  const size_t comp_stride = static_cast<size_t>(npts) * bytes_per;
  data.bytes.resize(static_cast<size_t>(comp_count) * comp_stride);
  for (int32_t c = 0; c < comp_count; ++c) {
    const size_t src = static_cast<size_t>(comp_start + c) * comp_stride;
    const size_t dst = static_cast<size_t>(c) * comp_stride;
    std::memcpy(data.bytes.data() + dst, raw.data() + src, comp_stride);
  }
  data.nx = hdr.box.hi.x - hdr.box.lo.x + 1;
  data.ny = hdr.box.hi.y - hdr.box.lo.y + 1;
  data.nz = hdr.box.hi.z - hdr.box.lo.z + 1;
  data.ncomp = comp_count;
  data.type = hdr.real_desc.type;
  return data;
}

PlotfileReader::PlotfileReader(std::string plotfile_dir)
    : plotfile_dir_(std::move(plotfile_dir)) {
  header_ = parse_plotfile_header(plotfile_dir_);
  vismf_headers_.resize(static_cast<size_t>(header_.finest_level + 1));
  for (int level = 0; level <= header_.finest_level; ++level) {
    const std::string cell_h = plotfile_dir_ + "/Level_" + std::to_string(level) + "/Cell_H";
    vismf_headers_[static_cast<size_t>(level)] = parse_vismf_header(cell_h);
  }
  discover_particles();
}

const PlotfileHeader& PlotfileReader::header() const { return header_; }

const VisMFHeader& PlotfileReader::vismf_header(int level) const {
  return vismf_headers_.at(static_cast<size_t>(level));
}

int32_t PlotfileReader::num_levels() const { return header_.finest_level + 1; }

int32_t PlotfileReader::num_fabs(int level) const {
  return static_cast<int32_t>(vismf_headers_.at(static_cast<size_t>(level)).box_array.boxes.size());
}

FabData PlotfileReader::read_fab(int level, int fab_index, int comp_start,
                                 int comp_count) {
  const auto& v = vismf_headers_.at(static_cast<size_t>(level));
  const auto& fod = v.fab_on_disk.at(static_cast<size_t>(fab_index));
  const std::string path =
      plotfile_dir_ + "/Level_" + std::to_string(level) + "/" + fod.file_name;

  std::ifstream is(path, std::ios::binary);
  if (!is) {
    throw std::runtime_error("failed to open FAB file: " + path);
  }
  is.seekg(fod.offset, std::ios::beg);
  FabHeader fab_hdr = read_fab_header(is);
  return read_fab_payload(is, fab_hdr, comp_start, comp_count);
}

void PlotfileReader::discover_particles() {
  particles_.clear();
  namespace fs = std::filesystem;
  const fs::path root(plotfile_dir_);
  if (!fs::exists(root) || !fs::is_directory(root)) {
    return;
  }
  for (const auto& entry : fs::directory_iterator(root)) {
    if (!entry.is_directory()) {
      continue;
    }
    const std::string name = entry.path().filename().string();
    if (name.size() < 10 || name.find("_particles") == std::string::npos) {
      continue;
    }
    const fs::path header_path = entry.path() / "Header";
    if (!fs::exists(header_path)) {
      continue;
    }
    ParticleHeader pheader;
    if (!parse_particle_header(header_path.string(), pheader)) {
      continue;
    }
    ParticleSpecies species;
    species.name = name;
    species.dir = entry.path().string();
    species.header = std::move(pheader);
    particles_.push_back(std::move(species));
  }
  std::sort(particles_.begin(), particles_.end(),
            [](const ParticleSpecies& a, const ParticleSpecies& b) { return a.name < b.name; });
}

const PlotfileReader::ParticleSpecies&
PlotfileReader::get_particle_species(const std::string& particle_type) const {
  for (const auto& species : particles_) {
    if (species.name == particle_type) {
      return species;
    }
  }
  throw std::runtime_error("particle type not found: " + particle_type);
}

std::vector<std::string> PlotfileReader::particle_types() const {
  std::vector<std::string> out;
  out.reserve(particles_.size());
  for (const auto& species : particles_) {
    out.push_back(species.name);
  }
  return out;
}

std::vector<std::string> PlotfileReader::particle_fields(const std::string& particle_type) const {
  const auto& species = get_particle_species(particle_type);
  return read_particle_fields(species.header);
}

int64_t PlotfileReader::particle_chunk_count(const std::string& particle_type) const {
  const auto& species = get_particle_species(particle_type);
  int64_t chunks = 0;
  for (const auto& level_counts : species.header.particle_counts) {
    chunks += static_cast<int64_t>(level_counts.size());
  }
  return chunks;
}

ParticleArrayData PlotfileReader::read_particle_field_chunk(const std::string& particle_type,
                                                            const std::string& field_name,
                                                            int64_t chunk_index) const {
  const auto& species = get_particle_species(particle_type);
  if (chunk_index < 0) {
    throw std::runtime_error("particle chunk index out of range");
  }

  int lev = -1;
  int grid = -1;
  int64_t cursor = 0;
  for (int l = 0; l <= species.header.finest_level; ++l) {
    const int ngrids =
        static_cast<int>(species.header.particle_counts[static_cast<size_t>(l)].size());
    if (chunk_index < cursor + ngrids) {
      lev = l;
      grid = static_cast<int>(chunk_index - cursor);
      break;
    }
    cursor += ngrids;
  }
  if (lev < 0 || grid < 0) {
    throw std::runtime_error("particle chunk index out of range");
  }

  enum class Kind { X, Y, Z, Real, Int };
  auto resolve_field = [&](const std::string& fname) -> std::pair<Kind, int> {
    const auto fields = read_particle_fields(species.header);
    auto it = std::find(fields.begin(), fields.end(), fname);
    if (it == fields.end()) {
      throw std::runtime_error("particle field not found: " + particle_type + "/" + fname);
    }
    if (fname == "x") {
      return {Kind::X, 0};
    }
    if (fname == "y") {
      return {Kind::Y, 1};
    }
    if (fname == "z") {
      return {Kind::Z, 2};
    }
    auto rit = std::find(species.header.real_names.begin(), species.header.real_names.end(), fname);
    if (rit != species.header.real_names.end()) {
      return {Kind::Real, static_cast<int>(rit - species.header.real_names.begin()) +
                             species.header.spatial_dim};
    }
    auto iit = std::find(species.header.int_names.begin(), species.header.int_names.end(), fname);
    if (iit != species.header.int_names.end()) {
      return {Kind::Int, static_cast<int>(iit - species.header.int_names.begin())};
    }
    throw std::runtime_error("particle field mapping failed: " + particle_type + "/" + fname);
  };
  const auto [kind, index] = resolve_field(field_name);

  ParticleBlock block = read_particle_block(species.dir, species.header, lev, grid);
  ParticleArrayData out;
  out.count = static_cast<int64_t>(block.count);
  if (kind == Kind::Int) {
    out.dtype = "int64";
    out.bytes.resize(static_cast<size_t>(block.count) * sizeof(std::int64_t));
    auto* dst = reinterpret_cast<std::int64_t*>(out.bytes.data());
    for (int i = 0; i < block.count; ++i) {
      const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(block.num_int) +
                         static_cast<size_t>(index);
      dst[i] = block.int_data[idx];
    }
    return out;
  }

  out.dtype = block.is_single ? "float32" : "float64";
  if (block.is_single) {
    out.bytes.resize(static_cast<size_t>(block.count) * sizeof(float));
    auto* dst = reinterpret_cast<float*>(out.bytes.data());
    for (int i = 0; i < block.count; ++i) {
      const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(block.num_real) +
                         static_cast<size_t>(index);
      dst[i] = block.real_data_single[idx];
    }
  } else {
    out.bytes.resize(static_cast<size_t>(block.count) * sizeof(double));
    auto* dst = reinterpret_cast<double*>(out.bytes.data());
    for (int i = 0; i < block.count; ++i) {
      const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(block.num_real) +
                         static_cast<size_t>(index);
      dst[i] = block.real_data_double[idx];
    }
  }
  return out;
}

}  // namespace kangaroo::plotfile
