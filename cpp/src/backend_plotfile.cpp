#include "kangaroo/backend_plotfile.hpp"
#include "kangaroo/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <unordered_map>

#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_local/get_worker_thread_num.hpp>

namespace kangaroo {

namespace {

template <typename InT, typename OutT>
void transpose_plotfile_axes(const InT* in, OutT* out, int nx, int ny, int nz) {
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t in_idx = (static_cast<std::size_t>(k) * ny + j) * nx + i;
        const std::size_t out_idx = (static_cast<std::size_t>(i) * ny + j) * nz + k;
        out[out_idx] = static_cast<OutT>(in[in_idx]);
      }
    }
  }
}

struct FabGroupKey {
  int32_t level = 0;
  int32_t block = 0;

  bool operator==(const FabGroupKey& other) const {
    return level == other.level && block == other.block;
  }
};

struct FabGroupKeyHash {
  std::size_t operator()(const FabGroupKey& key) const {
    const auto level = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.level));
    const auto block = static_cast<std::uint64_t>(static_cast<std::uint32_t>(key.block));
    return static_cast<std::size_t>((level << 32U) ^ block);
  }
};

template <typename T>
HostView transposed_component_view(const plotfile::FabData& fab, int32_t component_offset) {
  const std::size_t npts =
      static_cast<std::size_t>(fab.nx) * static_cast<std::size_t>(fab.ny) *
      static_cast<std::size_t>(fab.nz);
  const auto* in = reinterpret_cast<const T*>(fab.bytes.data()) +
                   static_cast<std::size_t>(component_offset) * npts;

  HostView view;
  view.data.resize(npts * sizeof(T));
  auto* out = reinterpret_cast<T*>(view.data.data());
  transpose_plotfile_axes(in, out, fab.nx, fab.ny, fab.nz);
  return view;
}

bool plotfile_zero_copy_reads_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("KANGAROO_PLOTFILE_ZERO_COPY_READS");
    if (env == nullptr || *env == '\0') {
      return true;
    }
    const std::string value(env);
    return value != "0" && value != "false" && value != "FALSE" && value != "off" &&
           value != "OFF";
  }();
  return enabled;
}

double now_seconds() {
  const auto now = std::chrono::system_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

void log_plotfile_read_fab_event(const char* status,
                                 const ChunkRef& ref,
                                 std::size_t bytes,
                                 double start,
                                 double end,
                                 const std::string& file_name,
                                 int64_t file_offset,
                                 int32_t comp_start,
                                 int32_t comp_count) {
  if (!has_event_log()) {
    return;
  }
  DataEvent event;
  event.op = "plotfile_read_fab";
  event.mode = "plotfile";
  event.status = status;
  event.file = file_name;
  event.ref = ref;
  try {
    event.locality = hpx::get_locality_id();
  } catch (...) {
    event.locality = -1;
  }
  event.target_locality = event.locality;
  try {
    event.worker = static_cast<int32_t>(hpx::get_worker_thread_num());
  } catch (...) {
    event.worker = -1;
  }
  event.bytes = bytes;
  event.file_offset = file_offset;
  event.comp_start = comp_start;
  event.comp_count = comp_count;
  event.ts = end;
  event.start = start;
  event.end = end;
  log_data_event(event);
}

}  // namespace

PlotfileBackend::PlotfileBackend(std::string plotfile_dir)
    : plotfile_dir_(std::move(plotfile_dir)), reader_(plotfile_dir_) {}

std::optional<HostView> PlotfileBackend::get_chunk(const ChunkRef& ref) {
  auto chunks = get_chunks(std::vector<ChunkRef>{ref});
  if (chunks.empty()) {
    return std::nullopt;
  }
  return std::move(chunks.front());
}

std::vector<std::optional<HostView>> PlotfileBackend::get_chunks(const std::vector<ChunkRef>& refs) {
  std::vector<std::optional<HostView>> out(refs.size());
  if (refs.empty()) {
    return out;
  }

  std::unordered_map<FabGroupKey, std::vector<std::size_t>, FabGroupKeyHash> groups;
  std::vector<int32_t> components(refs.size(), -1);
  for (std::size_t i = 0; i < refs.size(); ++i) {
    const auto& ref = refs[i];
    if (ref.level < 0 || ref.level >= reader_.num_levels()) {
      continue;
    }
    if (ref.block < 0 || ref.block >= reader_.num_fabs(ref.level)) {
      continue;
    }

    const int32_t comp = get_component_index(ref.field);
    if (comp < 0) {
      continue;
    }
    components[i] = comp;
    groups[FabGroupKey{ref.level, ref.block}].push_back(i);
  }

  for (const auto& [key, indices] : groups) {
    if (indices.empty()) {
      continue;
    }

    std::vector<std::size_t> sorted = indices;
    std::sort(sorted.begin(), sorted.end(), [&](std::size_t lhs, std::size_t rhs) {
      return components[lhs] < components[rhs];
    });

    for (std::size_t run_begin = 0; run_begin < sorted.size();) {
      int32_t min_comp = components[sorted[run_begin]];
      int32_t max_comp = min_comp;
      std::size_t run_end = run_begin + 1;
      while (run_end < sorted.size() && components[sorted[run_end]] <= max_comp + 1) {
        max_comp = std::max(max_comp, components[sorted[run_end]]);
        ++run_end;
      }

      try {
        const auto& fod =
            reader_.vismf_header(key.level).fab_on_disk.at(static_cast<std::size_t>(key.block));
        ChunkRef read_ref = refs[sorted[run_begin]];
        const double read_start = now_seconds();
        log_plotfile_read_fab_event("start",
                                    read_ref,
                                    0,
                                    read_start,
                                    read_start,
                                    fod.file_name,
                                    fod.offset,
                                    min_comp,
                                    max_comp - min_comp + 1);
        auto fab = reader_.read_fab(key.level, key.block, min_comp, max_comp - min_comp + 1);
        const double read_end = now_seconds();
        log_plotfile_read_fab_event("end",
                                    read_ref,
                                    fab.bytes.size(),
                                    read_start,
                                    read_end,
                                    fod.file_name,
                                    fod.offset,
                                    min_comp,
                                    max_comp - min_comp + 1);

        if (plotfile_zero_copy_reads_enabled()) {
          const std::size_t npts =
              static_cast<std::size_t>(fab.nx) * static_cast<std::size_t>(fab.ny) *
              static_cast<std::size_t>(fab.nz);
          const std::size_t bytes_per =
              fab.type == plotfile::RealType::kFloat32 ? sizeof(float) : sizeof(double);
          const std::size_t component_bytes = npts * bytes_per;
          SharedByteBuffer read_buffer(std::move(fab.bytes));
          for (std::size_t run_pos = run_begin; run_pos < run_end; ++run_pos) {
            const std::size_t idx = sorted[run_pos];
            const int32_t component_offset = components[idx] - min_comp;
            HostView view;
            view.data = read_buffer.slice(static_cast<std::size_t>(component_offset) *
                                              component_bytes,
                                          component_bytes);
            view.layout = HostViewLayout::kPlotfileKJI;
            out[idx] = std::move(view);
          }
        } else {
          for (std::size_t run_pos = run_begin; run_pos < run_end; ++run_pos) {
            const std::size_t idx = sorted[run_pos];
            const int32_t component_offset = components[idx] - min_comp;
            if (fab.type == plotfile::RealType::kFloat32) {
              out[idx] = transposed_component_view<float>(fab, component_offset);
            } else {
              out[idx] = transposed_component_view<double>(fab, component_offset);
            }
          }
        }
      } catch (const std::exception& e) {
        const auto& fod =
            reader_.vismf_header(key.level).fab_on_disk.at(static_cast<std::size_t>(key.block));
        ChunkRef read_ref = refs[sorted[run_begin]];
        const double error_ts = now_seconds();
        log_plotfile_read_fab_event("error",
                                    read_ref,
                                    0,
                                    error_ts,
                                    error_ts,
                                    fod.file_name,
                                    fod.offset,
                                    min_comp,
                                    max_comp - min_comp + 1);
        std::cerr << "PlotfileBackend: failed to read coalesced component run: " << e.what()
                  << "\n";
      }

      run_begin = run_end;
    }
  }

  return out;
}

bool PlotfileBackend::has_chunk(const ChunkRef& ref) const {
  if (ref.level < 0 || ref.level >= reader_.num_levels()) {
    return false;
  }
  if (ref.block < 0 || ref.block >= reader_.num_fabs(ref.level)) {
    return false;
  }
  return get_component_index(ref.field) >= 0;
}

std::size_t PlotfileBackend::estimate_chunk_bytes(const ChunkRef& ref) const {
  if (ref.level < 0 || ref.level >= reader_.num_levels()) {
    return 0;
  }
  const auto& header = reader_.vismf_header(ref.level);
  const auto block = static_cast<std::size_t>(ref.block);
  if (ref.block < 0 || block >= header.box_array.boxes.size()) {
    return 0;
  }
  if (get_component_index(ref.field) < 0) {
    return 0;
  }
  const auto points = header.box_array.boxes[block].num_pts();
  if (points <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(points) * sizeof(double);
}

DatasetMetadata PlotfileBackend::get_metadata() const {
  DatasetMetadata meta;
  const auto& h = reader_.header();
  meta.prob_lo = h.prob_lo;
  meta.prob_hi = h.prob_hi;
  meta.ref_ratio = h.ref_ratio;
  return meta;
}

void PlotfileBackend::register_field(int32_t field_id, int32_t component_index) {
  std::lock_guard<std::mutex> lock(map_mutex_);
  field_map_[field_id] = component_index;
}

int32_t PlotfileBackend::get_component_index(int32_t field_id) const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  auto it = field_map_.find(field_id);
  if (it != field_map_.end()) {
    return it->second;
  }
  return -1;
}

std::map<int32_t, int32_t> PlotfileBackend::field_map() const {
  std::lock_guard<std::mutex> lock(map_mutex_);
  return field_map_;
}

void PlotfileBackend::set_field_map(std::map<int32_t, int32_t> field_map) {
  std::lock_guard<std::mutex> lock(map_mutex_);
  field_map_ = std::move(field_map);
}

}  // namespace kangaroo
