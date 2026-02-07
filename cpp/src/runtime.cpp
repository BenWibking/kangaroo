#include "kangaroo/runtime.hpp"

#include "kangaroo/plan_decode.hpp"
#include "kangaroo/plotfile_reader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <iterator>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <msgpack.hpp>

#include <hpx/include/actions.hpp>
#include <hpx/collectives/broadcast_direct.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime.hpp>

namespace kangaroo {

namespace {

std::mutex g_ctx_mutex;
std::once_flag g_hpx_start_once;
thread_local bool g_hpx_thread_registered = false;
bool g_hpx_started = false;
bool g_hpx_cfg_set = false;
bool g_hpx_cmdline_set = false;
std::vector<std::string> g_hpx_cfg;
std::vector<std::string> g_hpx_cmdline;
RunMeta g_runmeta;
bool g_has_runmeta = false;
DatasetHandle g_dataset;
bool g_has_dataset = false;
KernelRegistry* g_kernel_registry = nullptr;
std::unordered_map<int32_t, PlanIR> g_plans;
std::mutex g_event_log_mutex;
std::string g_event_log_path;
bool g_event_log_enabled = false;

std::string json_escape(const std::string& value);

struct EventLogWorker {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<TaskEvent> queue;
  std::thread thread;
  bool running = false;
  bool stop = false;
  std::string path;
  bool enabled = false;

  void set_path(const std::string& next_path) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      path = next_path;
      enabled = !path.empty();
    }
    start_if_needed();
    cv.notify_all();
  }

  void start_if_needed() {
    bool should_start = false;
    {
      std::lock_guard<std::mutex> lock(mutex);
      should_start = enabled && !running;
      if (should_start) {
        running = true;
      }
    }
    if (should_start) {
      thread = std::thread([this]() { run(); });
    }
  }

  void enqueue(TaskEvent event) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!enabled) {
        return;
      }
      queue.push_back(std::move(event));
    }
    cv.notify_one();
  }

  void run() {
    std::ofstream out;
    std::string active_path;
    for (;;) {
      TaskEvent event;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] {
          return stop || !queue.empty() || path != active_path;
        });
        if (stop && queue.empty()) {
          break;
        }
        if (path != active_path) {
          active_path = path;
          out.close();
          if (!active_path.empty()) {
            out.open(active_path, std::ios::app);
          }
        }
        if (queue.empty()) {
          continue;
        }
        event = std::move(queue.front());
        queue.pop_front();
      }
      if (!out) {
        continue;
      }
      auto write_string = [&](const char* key, const std::string& value) {
        out << '"' << key << "\":\"" << json_escape(value) << '"';
      };
      out << '{';
      out << "\"type\":\"task\",";
      write_string("id", event.id);
      out << ',';
      write_string("name", event.name);
      out << ',';
      write_string("kernel", event.kernel);
      out << ',';
      write_string("plane", event.plane);
      out << ',';
      write_string("status", event.status);
      out << ",\"stage\":" << event.stage;
      out << ",\"template\":" << event.template_index;
      out << ",\"block\":" << event.block;
      out << ",\"step\":" << event.step;
      out << ",\"level\":" << event.level;
      out << ",\"locality\":" << event.locality;
      if (!event.worker_label.empty()) {
        out << ',';
        write_string("worker", event.worker_label);
      } else {
        out << ",\"worker\":\"worker-" << event.worker << '"';
      }
      out << std::fixed << std::setprecision(6);
      out << ",\"ts\":" << event.ts;
      out << ",\"start\":" << event.start;
      out << ",\"end\":" << event.end;
      out << "}\n";
      out.flush();
    }
  }

  ~EventLogWorker() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    }
    cv.notify_all();
    if (thread.joinable()) {
      thread.join();
    }
  }
};

EventLogWorker g_event_log_worker;

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
        break;
    }
  }
  return out;
}

const std::string& event_log_path() {
  return g_event_log_path;
}

void init_event_log_from_env() {
  const char* env = std::getenv("KANGAROO_EVENT_LOG");
  if (env && *env != '\0') {
    {
      std::lock_guard<std::mutex> lock(g_event_log_mutex);
      g_event_log_path = env;
      g_event_log_enabled = true;
    }
    g_event_log_worker.set_path(g_event_log_path);
  }
}

std::shared_ptr<plotfile::PlotfileReader> get_plotfile_reader(const std::string& path) {
  static std::mutex reader_mutex;
  static std::unordered_map<std::string, std::weak_ptr<plotfile::PlotfileReader>> readers;

  std::lock_guard<std::mutex> lock(reader_mutex);
  auto it = readers.find(path);
  if (it != readers.end()) {
    if (auto shared = it->second.lock()) {
      return shared;
    }
  }
  auto shared = std::make_shared<plotfile::PlotfileReader>(path);
  readers[path] = shared;
  return shared;
}

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

struct SamplePatch {
  int16_t level = 0;
  IndexBox3 box;
  LevelGeom geom;
  HostView view;
  int32_t bytes_per_value = 4;
};

double cell_edge(const LevelGeom& geom, int axis, int idx) {
  return geom.x0[axis] + (idx - geom.index_origin[axis]) * geom.dx[axis];
}

double cell_center(const LevelGeom& geom, int axis, int idx) {
  return cell_edge(geom, axis, idx) + 0.5 * geom.dx[axis];
}

int32_t coord_to_index(const LevelGeom& geom, int axis, double x) {
  return static_cast<int32_t>(std::floor((x - geom.x0[axis]) / geom.dx[axis])) + geom.index_origin[axis];
}

double wrap_coord(double x, double lo, double hi) {
  const double period = hi - lo;
  if (!(period > 0.0)) {
    return x;
  }
  double y = std::fmod(x - lo, period);
  if (y < 0.0) {
    y += period;
  }
  return lo + y;
}

bool solve_3x3(double a[3][3], double b[3], double x[3]) {
  double m[3][4] = {
      {a[0][0], a[0][1], a[0][2], b[0]},
      {a[1][0], a[1][1], a[1][2], b[1]},
      {a[2][0], a[2][1], a[2][2], b[2]},
  };

  for (int col = 0; col < 3; ++col) {
    int piv = col;
    double best = std::abs(m[col][col]);
    for (int r = col + 1; r < 3; ++r) {
      const double cand = std::abs(m[r][col]);
      if (cand > best) {
        best = cand;
        piv = r;
      }
    }
    if (best < 1e-30) {
      return false;
    }
    if (piv != col) {
      for (int c = col; c < 4; ++c) {
        std::swap(m[col][c], m[piv][c]);
      }
    }
    const double inv = 1.0 / m[col][col];
    for (int c = col; c < 4; ++c) {
      m[col][c] *= inv;
    }
    for (int r = 0; r < 3; ++r) {
      if (r == col) {
        continue;
      }
      const double f = m[r][col];
      if (f == 0.0) {
        continue;
      }
      for (int c = col; c < 4; ++c) {
        m[r][c] -= f * m[col][c];
      }
    }
  }

  x[0] = m[0][3];
  x[1] = m[1][3];
  x[2] = m[2][3];
  return true;
}

std::optional<double> patch_value_at(const SamplePatch& p, int32_t i, int32_t j, int32_t k) {
  if (i < p.box.lo[0] || i > p.box.hi[0] || j < p.box.lo[1] || j > p.box.hi[1] ||
      k < p.box.lo[2] || k > p.box.hi[2]) {
    return std::nullopt;
  }
  const int32_t nx = p.box.hi[0] - p.box.lo[0] + 1;
  const int32_t ny = p.box.hi[1] - p.box.lo[1] + 1;
  const int32_t nz = p.box.hi[2] - p.box.lo[2] + 1;
  if (nx <= 0 || ny <= 0 || nz <= 0 || p.bytes_per_value <= 0) {
    return std::nullopt;
  }
  const int32_t li = i - p.box.lo[0];
  const int32_t lj = j - p.box.lo[1];
  const int32_t lk = k - p.box.lo[2];
  const std::size_t idx = (static_cast<std::size_t>(li) * static_cast<std::size_t>(ny) +
                           static_cast<std::size_t>(lj)) *
                              static_cast<std::size_t>(nz) +
                          static_cast<std::size_t>(lk);
  if (p.bytes_per_value == 4) {
    const std::size_t pos = idx * sizeof(float);
    if (pos + sizeof(float) > p.view.data.size()) {
      return std::nullopt;
    }
    return static_cast<double>(reinterpret_cast<const float*>(p.view.data.data())[idx]);
  }
  if (p.bytes_per_value == 8) {
    const std::size_t pos = idx * sizeof(double);
    if (pos + sizeof(double) > p.view.data.size()) {
      return std::nullopt;
    }
    return reinterpret_cast<const double*>(p.view.data.data())[idx];
  }
  return std::nullopt;
}

struct SamplePoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double value = 0.0;
};

std::optional<SamplePoint> composite_sample_at(
    std::span<const SamplePatch> patches,
    int finest_level,
    const int32_t domain_lo[3],
    const int32_t domain_hi[3],
    const bool is_periodic[3],
    const double domain_lo_edge[3],
    const double domain_hi_edge[3],
    double x, double y, double z) {
  double xyz[3] = {x, y, z};
  for (int ax = 0; ax < 3; ++ax) {
    if (is_periodic[ax]) {
      xyz[ax] = wrap_coord(xyz[ax], domain_lo_edge[ax], domain_hi_edge[ax]);
    } else if (xyz[ax] < domain_lo_edge[ax] || xyz[ax] >= domain_hi_edge[ax]) {
      return std::nullopt;
    }
  }

  for (int lev = finest_level; lev >= 0; --lev) {
    for (const auto& p : patches) {
      if (p.level != lev) {
        continue;
      }
      int32_t idx[3];
      for (int ax = 0; ax < 3; ++ax) {
        idx[ax] = coord_to_index(p.geom, ax, xyz[ax]);
      }
      if (idx[0] < p.box.lo[0] || idx[0] > p.box.hi[0] || idx[1] < p.box.lo[1] || idx[1] > p.box.hi[1] ||
          idx[2] < p.box.lo[2] || idx[2] > p.box.hi[2]) {
        continue;
      }
      auto value = patch_value_at(p, idx[0], idx[1], idx[2]);
      if (!value.has_value()) {
        continue;
      }
      SamplePoint out;
      out.x = cell_center(p.geom, 0, idx[0]);
      out.y = cell_center(p.geom, 1, idx[1]);
      out.z = cell_center(p.geom, 2, idx[2]);
      out.value = *value;
      return out;
    }
  }
  return std::nullopt;
}

std::vector<SamplePatch> unpack_sample_patches(std::span<const std::uint8_t> packed) {
  std::vector<SamplePatch> patches;
  if (packed.empty()) {
    return patches;
  }
  auto handle = msgpack::unpack(reinterpret_cast<const char*>(packed.data()), packed.size());
  auto root = handle.get();
  if (root.type != msgpack::type::MAP) {
    return patches;
  }
  const msgpack::object* arr_obj = nullptr;
  for (uint32_t i = 0; i < root.via.map.size; ++i) {
    const auto& k = root.via.map.ptr[i].key;
    if (k.type == msgpack::type::STR && k.as<std::string>() == "patches") {
      arr_obj = &root.via.map.ptr[i].val;
      break;
    }
  }
  if (arr_obj == nullptr || arr_obj->type != msgpack::type::ARRAY) {
    return patches;
  }
  patches.reserve(arr_obj->via.array.size);
  for (uint32_t i = 0; i < arr_obj->via.array.size; ++i) {
    const auto& p = arr_obj->via.array.ptr[i];
    if (p.type != msgpack::type::MAP) {
      continue;
    }
    SamplePatch patch;
    for (uint32_t j = 0; j < p.via.map.size; ++j) {
      const auto& key = p.via.map.ptr[j].key;
      const auto& val = p.via.map.ptr[j].val;
      if (key.type != msgpack::type::STR) {
        continue;
      }
      const auto ks = key.as<std::string>();
      if (ks == "level") {
        patch.level = val.as<int16_t>();
      } else if (ks == "lo" && val.type == msgpack::type::ARRAY && val.via.array.size == 3) {
        patch.box.lo[0] = val.via.array.ptr[0].as<int32_t>();
        patch.box.lo[1] = val.via.array.ptr[1].as<int32_t>();
        patch.box.lo[2] = val.via.array.ptr[2].as<int32_t>();
      } else if (ks == "hi" && val.type == msgpack::type::ARRAY && val.via.array.size == 3) {
        patch.box.hi[0] = val.via.array.ptr[0].as<int32_t>();
        patch.box.hi[1] = val.via.array.ptr[1].as<int32_t>();
        patch.box.hi[2] = val.via.array.ptr[2].as<int32_t>();
      } else if (ks == "dx" && val.type == msgpack::type::ARRAY && val.via.array.size == 3) {
        patch.geom.dx[0] = val.via.array.ptr[0].as<double>();
        patch.geom.dx[1] = val.via.array.ptr[1].as<double>();
        patch.geom.dx[2] = val.via.array.ptr[2].as<double>();
      } else if (ks == "x0" && val.type == msgpack::type::ARRAY && val.via.array.size == 3) {
        patch.geom.x0[0] = val.via.array.ptr[0].as<double>();
        patch.geom.x0[1] = val.via.array.ptr[1].as<double>();
        patch.geom.x0[2] = val.via.array.ptr[2].as<double>();
      } else if (ks == "index_origin" && val.type == msgpack::type::ARRAY && val.via.array.size == 3) {
        patch.geom.index_origin[0] = val.via.array.ptr[0].as<int32_t>();
        patch.geom.index_origin[1] = val.via.array.ptr[1].as<int32_t>();
        patch.geom.index_origin[2] = val.via.array.ptr[2].as<int32_t>();
      } else if (ks == "is_periodic" && val.type == msgpack::type::ARRAY && val.via.array.size == 3) {
        patch.geom.is_periodic[0] = val.via.array.ptr[0].as<bool>();
        patch.geom.is_periodic[1] = val.via.array.ptr[1].as<bool>();
        patch.geom.is_periodic[2] = val.via.array.ptr[2].as<bool>();
      } else if (ks == "bytes_per_value") {
        patch.bytes_per_value = val.as<int32_t>();
      } else if (ks == "data" && val.type == msgpack::type::BIN) {
        patch.view.data.assign(val.via.bin.ptr, val.via.bin.ptr + val.via.bin.size);
      }
    }
    if (!patch.view.data.empty()) {
      patches.push_back(std::move(patch));
    }
  }
  return patches;
}

void register_default_kernels(KernelRegistry& registry) {
  static const bool log_locality = []() {
    const char* env = std::getenv("KANGAROO_LOG_LOCALITY");
    return env != nullptr && *env != '\0' && *env != '0';
  }();
  registry.register_kernel(
      KernelDesc{.name = "amr_subbox_fetch_pack", .n_inputs = 0, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta& level, int32_t block, std::span<const HostView>, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t> params_msgpack) {
        if (outputs.empty() || block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        struct Params {
          int32_t input_field = -1;
          int32_t input_version = 0;
          int32_t input_step = 0;
          int16_t input_level = 0;
          int32_t bytes_per_value = 8;
          int32_t halo_cells = 1;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            auto get_key = [&](const char* key) -> const msgpack::object* {
              for (uint32_t i = 0; i < root.via.map.size; ++i) {
                const auto& k = root.via.map.ptr[i].key;
                if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
                  return &root.via.map.ptr[i].val;
                }
              }
              return nullptr;
            };
            if (const auto* fld = get_key("input_field")) params.input_field = fld->as<int32_t>();
            if (const auto* ver = get_key("input_version")) params.input_version = ver->as<int32_t>();
            if (const auto* stp = get_key("input_step")) params.input_step = stp->as<int32_t>();
            if (const auto* lev = get_key("input_level")) params.input_level = lev->as<int16_t>();
            if (const auto* bpv = get_key("bytes_per_value")) params.bytes_per_value = bpv->as<int32_t>();
            if (const auto* halo = get_key("halo_cells")) params.halo_cells = halo->as<int32_t>();
          }
        }

        outputs[0].data.clear();
        if (params.input_field < 0) {
          return hpx::make_ready_future();
        }
        if (params.bytes_per_value != 4 && params.bytes_per_value != 8) {
          return hpx::make_ready_future();
        }

        const RunMeta& meta = global_runmeta();
        if (params.input_step < 0 || static_cast<std::size_t>(params.input_step) >= meta.steps.size()) {
          return hpx::make_ready_future();
        }
        const auto& step_meta = meta.steps.at(static_cast<std::size_t>(params.input_step));
        if (params.input_level < 0 || static_cast<std::size_t>(params.input_level) >= step_meta.levels.size()) {
          return hpx::make_ready_future();
        }

        const auto& box = level.boxes.at(static_cast<std::size_t>(block));
        const int halo = std::max(1, params.halo_cells);
        const auto& target_geom = level.geom;
        double query_lo[3] = {0.0, 0.0, 0.0};
        double query_hi[3] = {0.0, 0.0, 0.0};
        for (int ax = 0; ax < 3; ++ax) {
          const int32_t lo = ax == 0 ? box.lo.x : (ax == 1 ? box.lo.y : box.lo.z);
          const int32_t hi = ax == 0 ? box.hi.x : (ax == 1 ? box.hi.y : box.hi.z);
          query_lo[ax] = cell_edge(target_geom, ax, lo) - static_cast<double>(halo) * target_geom.dx[ax];
          query_hi[ax] = cell_edge(target_geom, ax, hi + 1) + static_cast<double>(halo) * target_geom.dx[ax];
        }

        struct PackedPatch {
          int16_t level = 0;
          IndexBox3 box;
          LevelGeom geom;
          int32_t bytes_per_value = 4;
          HostView data;
        };
        std::vector<PackedPatch> packed_patches;

        DataServiceLocal data_service;
        for (int16_t lev = 0; lev < static_cast<int16_t>(step_meta.levels.size()); ++lev) {
          const auto& lev_meta = step_meta.levels.at(static_cast<std::size_t>(lev));
          int32_t req_lo[3];
          int32_t req_hi[3];
          for (int ax = 0; ax < 3; ++ax) {
            req_lo[ax] = coord_to_index(lev_meta.geom, ax, query_lo[ax]);
            req_hi[ax] = coord_to_index(lev_meta.geom, ax, query_hi[ax]);
          }
          for (int32_t b = 0; b < static_cast<int32_t>(lev_meta.boxes.size()); ++b) {
            if (lev == params.input_level && b == block) {
              continue;
            }
            const auto& ob = lev_meta.boxes.at(static_cast<std::size_t>(b));
            IndexBox3 request_box;
            request_box.lo[0] = std::max(ob.lo.x, req_lo[0]);
            request_box.lo[1] = std::max(ob.lo.y, req_lo[1]);
            request_box.lo[2] = std::max(ob.lo.z, req_lo[2]);
            request_box.hi[0] = std::min(ob.hi.x, req_hi[0]);
            request_box.hi[1] = std::min(ob.hi.y, req_hi[1]);
            request_box.hi[2] = std::min(ob.hi.z, req_hi[2]);
            if (request_box.hi[0] < request_box.lo[0] || request_box.hi[1] < request_box.lo[1] ||
                request_box.hi[2] < request_box.lo[2]) {
              continue;
            }

            ChunkSubboxRef ref;
            ref.chunk = ChunkRef{params.input_step, lev, params.input_field, params.input_version, b};
            ref.chunk_box.lo[0] = ob.lo.x;
            ref.chunk_box.lo[1] = ob.lo.y;
            ref.chunk_box.lo[2] = ob.lo.z;
            ref.chunk_box.hi[0] = ob.hi.x;
            ref.chunk_box.hi[1] = ob.hi.y;
            ref.chunk_box.hi[2] = ob.hi.z;
            ref.request_box = request_box;
            ref.bytes_per_value = params.bytes_per_value;
            auto sub = data_service.get_subbox(ref).get();
            if (sub.box.hi[0] < sub.box.lo[0] || sub.box.hi[1] < sub.box.lo[1] ||
                sub.box.hi[2] < sub.box.lo[2] || sub.data.data.empty()) {
              continue;
            }

            PackedPatch pp;
            pp.level = lev;
            pp.box = sub.box;
            pp.geom = lev_meta.geom;
            pp.bytes_per_value = params.bytes_per_value;
            pp.data = std::move(sub.data);
            packed_patches.push_back(std::move(pp));
          }
        }

        msgpack::sbuffer sbuf;
        msgpack::packer<msgpack::sbuffer> pk(&sbuf);
        pk.pack_map(1);
        pk.pack(std::string("patches"));
        pk.pack_array(packed_patches.size());
        for (const auto& p : packed_patches) {
          pk.pack_map(9);
          pk.pack(std::string("level"));
          pk.pack_int16(p.level);
          pk.pack(std::string("lo"));
          pk.pack_array(3);
          pk.pack_int32(p.box.lo[0]);
          pk.pack_int32(p.box.lo[1]);
          pk.pack_int32(p.box.lo[2]);
          pk.pack(std::string("hi"));
          pk.pack_array(3);
          pk.pack_int32(p.box.hi[0]);
          pk.pack_int32(p.box.hi[1]);
          pk.pack_int32(p.box.hi[2]);
          pk.pack(std::string("dx"));
          pk.pack_array(3);
          pk.pack_double(p.geom.dx[0]);
          pk.pack_double(p.geom.dx[1]);
          pk.pack_double(p.geom.dx[2]);
          pk.pack(std::string("x0"));
          pk.pack_array(3);
          pk.pack_double(p.geom.x0[0]);
          pk.pack_double(p.geom.x0[1]);
          pk.pack_double(p.geom.x0[2]);
          pk.pack(std::string("index_origin"));
          pk.pack_array(3);
          pk.pack_int32(p.geom.index_origin[0]);
          pk.pack_int32(p.geom.index_origin[1]);
          pk.pack_int32(p.geom.index_origin[2]);
          pk.pack(std::string("is_periodic"));
          pk.pack_array(3);
          pk.pack(static_cast<bool>(p.geom.is_periodic[0]));
          pk.pack(static_cast<bool>(p.geom.is_periodic[1]));
          pk.pack(static_cast<bool>(p.geom.is_periodic[2]));
          pk.pack(std::string("bytes_per_value"));
          pk.pack_int32(p.bytes_per_value);
          pk.pack(std::string("data"));
          pk.pack_bin(p.data.data.size());
          if (!p.data.data.empty()) {
            pk.pack_bin_body(reinterpret_cast<const char*>(p.data.data.data()), p.data.data.size());
          }
        }

        outputs[0].data.assign(sbuf.data(), sbuf.data() + sbuf.size());
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "gradU_stencil", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta& level, int32_t block, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t> params_msgpack) {
        if (inputs.size() < 2 || outputs.empty() || block < 0 ||
            static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        struct Params {
          int32_t input_field = -1;
          int32_t input_version = 0;
          int32_t input_step = 0;
          int16_t input_level = 0;
          int32_t bytes_per_value = 0;
          int32_t stencil_radius = 1;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            auto get_key = [&](const char* key) -> const msgpack::object* {
              for (uint32_t i = 0; i < root.via.map.size; ++i) {
                const auto& k = root.via.map.ptr[i].key;
                if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
                  return &root.via.map.ptr[i].val;
                }
              }
              return nullptr;
            };
            if (const auto* fld = get_key("input_field"); fld &&
                                                       (fld->type == msgpack::type::POSITIVE_INTEGER ||
                                                        fld->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.input_field = fld->as<int32_t>();
            }
            if (const auto* ver = get_key("input_version"); ver &&
                                                          (ver->type == msgpack::type::POSITIVE_INTEGER ||
                                                           ver->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.input_version = ver->as<int32_t>();
            }
            if (const auto* stp = get_key("input_step"); stp &&
                                                        (stp->type == msgpack::type::POSITIVE_INTEGER ||
                                                         stp->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.input_step = stp->as<int32_t>();
            }
            if (const auto* lev = get_key("input_level"); lev &&
                                                         (lev->type == msgpack::type::POSITIVE_INTEGER ||
                                                          lev->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.input_level = lev->as<int16_t>();
            }
            if (const auto* bpv = get_key("bytes_per_value"); bpv &&
                                                         (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                                                          bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.bytes_per_value = bpv->as<int32_t>();
            }
            if (const auto* sr = get_key("stencil_radius"); sr &&
                                                      (sr->type == msgpack::type::POSITIVE_INTEGER ||
                                                       sr->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.stencil_radius = sr->as<int32_t>();
            }
          }
        }

        const auto& box = level.boxes.at(static_cast<std::size_t>(block));
        const int32_t nx = box.hi.x - box.lo.x + 1;
        const int32_t ny = box.hi.y - box.lo.y + 1;
        const int32_t nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }

        const auto& in = inputs[0].data;
        int32_t bytes_per_value = params.bytes_per_value;
        if (bytes_per_value <= 0) {
          const std::size_t npts = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
          if (npts > 0) {
            const std::size_t guess = in.size() / npts;
            if (guess == 4 || guess == 8) {
              bytes_per_value = static_cast<int32_t>(guess);
            }
          }
        }
        if (bytes_per_value != 4 && bytes_per_value != 8) {
          return hpx::make_ready_future();
        }
        const int32_t stencil_radius = std::max(1, params.stencil_radius);

        const std::size_t out_bytes =
            static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz) *
            3 * sizeof(double);
        outputs[0].data.assign(out_bytes, 0);
        auto* out = reinterpret_cast<double*>(outputs[0].data.data());

        SamplePatch self;
        self.level = 0;
        self.box.lo[0] = box.lo.x;
        self.box.lo[1] = box.lo.y;
        self.box.lo[2] = box.lo.z;
        self.box.hi[0] = box.hi.x;
        self.box.hi[1] = box.hi.y;
        self.box.hi[2] = box.hi.z;
        self.geom = level.geom;
        self.view = inputs[0];
        self.bytes_per_value = bytes_per_value;

        const RunMeta& meta = global_runmeta();
        if (params.input_step < 0 || static_cast<std::size_t>(params.input_step) >= meta.steps.size()) {
          return hpx::make_ready_future();
        }
        const auto& step_meta = meta.steps.at(static_cast<std::size_t>(params.input_step));
        const int16_t target_level = params.input_level;
        if (target_level < 0 || static_cast<std::size_t>(target_level) >= step_meta.levels.size()) {
          return hpx::make_ready_future();
        }

        std::vector<SamplePatch> patches;
        patches.reserve(64);
        self.level = target_level;
        patches.push_back(std::move(self));
        if (!inputs[1].data.empty()) {
          auto prefetched = unpack_sample_patches(inputs[1].data);
          patches.insert(patches.end(),
                         std::make_move_iterator(prefetched.begin()),
                         std::make_move_iterator(prefetched.end()));
        }

        const auto& target_geom = level.geom;
        double query_lo[3] = {0.0, 0.0, 0.0};
        double query_hi[3] = {0.0, 0.0, 0.0};
        for (int ax = 0; ax < 3; ++ax) {
          const int32_t lo = ax == 0 ? box.lo.x : (ax == 1 ? box.lo.y : box.lo.z);
          const int32_t hi = ax == 0 ? box.hi.x : (ax == 1 ? box.hi.y : box.hi.z);
          query_lo[ax] = cell_edge(target_geom, ax, lo) - target_geom.dx[ax];
          query_hi[ax] = cell_edge(target_geom, ax, hi + 1) + target_geom.dx[ax];
        }

        int32_t domain_lo[3] = {std::numeric_limits<int32_t>::max(),
                                std::numeric_limits<int32_t>::max(),
                                std::numeric_limits<int32_t>::max()};
        int32_t domain_hi[3] = {std::numeric_limits<int32_t>::min(),
                                std::numeric_limits<int32_t>::min(),
                                std::numeric_limits<int32_t>::min()};
        if (step_meta.levels.empty()) {
          return hpx::make_ready_future();
        }
        if (!step_meta.levels.empty()) {
          for (const auto& b : step_meta.levels.front().boxes) {
            domain_lo[0] = std::min(domain_lo[0], b.lo.x);
            domain_lo[1] = std::min(domain_lo[1], b.lo.y);
            domain_lo[2] = std::min(domain_lo[2], b.lo.z);
            domain_hi[0] = std::max(domain_hi[0], b.hi.x);
            domain_hi[1] = std::max(domain_hi[1], b.hi.y);
            domain_hi[2] = std::max(domain_hi[2], b.hi.z);
          }
        }
        if (domain_hi[0] < domain_lo[0] || domain_hi[1] < domain_lo[1] || domain_hi[2] < domain_lo[2]) {
          return hpx::make_ready_future();
        }
        bool is_periodic[3] = {level.geom.is_periodic[0], level.geom.is_periodic[1], level.geom.is_periodic[2]};
        double domain_lo_edge[3] = {
            cell_edge(step_meta.levels.front().geom, 0, domain_lo[0]),
            cell_edge(step_meta.levels.front().geom, 1, domain_lo[1]),
            cell_edge(step_meta.levels.front().geom, 2, domain_lo[2]),
        };
        double domain_hi_edge[3] = {
            cell_edge(step_meta.levels.front().geom, 0, domain_hi[0] + 1),
            cell_edge(step_meta.levels.front().geom, 1, domain_hi[1] + 1),
            cell_edge(step_meta.levels.front().geom, 2, domain_hi[2] + 1),
        };

        auto self_index = [&](int i, int j, int k) -> std::size_t {
          return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(j)) *
                     static_cast<std::size_t>(nz) +
                 static_cast<std::size_t>(k);
        };
        auto read_self = [&](std::size_t idx) -> double {
          if (bytes_per_value == 4) {
            return static_cast<double>(reinterpret_cast<const float*>(in.data())[idx]);
          }
          return reinterpret_cast<const double*>(in.data())[idx];
        };

        for (int i = 0; i < nx; ++i) {
          const int32_t gi = box.lo.x + i;
          const double xc = cell_center(target_geom, 0, gi);
          for (int j = 0; j < ny; ++j) {
            const int32_t gj = box.lo.y + j;
            const double yc = cell_center(target_geom, 1, gj);
            for (int k = 0; k < nz; ++k) {
              const int32_t gk = box.lo.z + k;
              const double zc = cell_center(target_geom, 2, gk);
              const std::size_t idx = self_index(i, j, k);
              const double f0 = read_self(idx);

              std::vector<SamplePoint> samples;
              const int32_t width = 2 * stencil_radius + 1;
              samples.reserve(static_cast<std::size_t>(width * width * width - 1));
              for (int ox = -stencil_radius; ox <= stencil_radius; ++ox) {
                for (int oy = -stencil_radius; oy <= stencil_radius; ++oy) {
                  for (int oz = -stencil_radius; oz <= stencil_radius; ++oz) {
                    if (ox == 0 && oy == 0 && oz == 0) {
                      continue;
                    }
                    const double xp = xc + static_cast<double>(ox) * target_geom.dx[0];
                    const double yp = yc + static_cast<double>(oy) * target_geom.dx[1];
                    const double zp = zc + static_cast<double>(oz) * target_geom.dx[2];
                    auto s = composite_sample_at(patches, static_cast<int>(step_meta.levels.size()) - 1,
                                                 domain_lo, domain_hi, is_periodic,
                                                 domain_lo_edge, domain_hi_edge, xp, yp, zp);
                    if (!s.has_value()) {
                      continue;
                    }
                    bool duplicate = false;
                    for (const auto& existing : samples) {
                      if (std::abs(existing.x - s->x) < 1e-14 && std::abs(existing.y - s->y) < 1e-14 &&
                          std::abs(existing.z - s->z) < 1e-14) {
                        duplicate = true;
                        break;
                      }
                    }
                    if (!duplicate) {
                      samples.push_back(*s);
                    }
                  }
                }
              }

              double grad[3] = {0.0, 0.0, 0.0};
              if (samples.size() >= 3) {
                double a[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
                double bvec[3] = {0.0, 0.0, 0.0};
                for (const auto& s : samples) {
                  const double rx = s.x - xc;
                  const double ry = s.y - yc;
                  const double rz = s.z - zc;
                  const double df = s.value - f0;
                  const double r2 = rx * rx + ry * ry + rz * rz;
                  if (r2 <= 0.0) {
                    continue;
                  }
                  const double w = 1.0 / (r2 + 1e-30);
                  a[0][0] += w * rx * rx;
                  a[0][1] += w * rx * ry;
                  a[0][2] += w * rx * rz;
                  a[1][0] += w * ry * rx;
                  a[1][1] += w * ry * ry;
                  a[1][2] += w * ry * rz;
                  a[2][0] += w * rz * rx;
                  a[2][1] += w * rz * ry;
                  a[2][2] += w * rz * rz;
                  bvec[0] += w * rx * df;
                  bvec[1] += w * ry * df;
                  bvec[2] += w * rz * df;
                }
                solve_3x3(a, bvec, grad);
              }

              out[3 * idx + 0] = grad[0];
              out[3 * idx + 1] = grad[1];
              out[3 * idx + 2] = grad[2];
            }
          }
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "plotfile_load", .n_inputs = 0, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta& level, int32_t block, std::span<const HostView>, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t> params_msgpack) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }

        struct Params {
          std::string plotfile;
          int level = 0;
          int comp = 0;
          int bytes_per_value = 4;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            auto get_key = [&](const char* key) -> const msgpack::object* {
              for (uint32_t i = 0; i < root.via.map.size; ++i) {
                const auto& k = root.via.map.ptr[i].key;
                if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
                  return &root.via.map.ptr[i].val;
                }
              }
              return nullptr;
            };
            if (const auto* path = get_key("plotfile"); path && path->type == msgpack::type::STR) {
              params.plotfile = path->as<std::string>();
            }
            if (const auto* lvl = get_key("level"); lvl &&
                                                 (lvl->type == msgpack::type::POSITIVE_INTEGER ||
                                                  lvl->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.level = lvl->as<int>();
            }
            if (const auto* comp = get_key("comp"); comp &&
                                                 (comp->type == msgpack::type::POSITIVE_INTEGER ||
                                                  comp->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.comp = comp->as<int>();
            }
            if (const auto* bpv = get_key("bytes_per_value"); bpv &&
                                                         (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                                                          bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.bytes_per_value = bpv->as<int>();
            }
          }
        }

        if (params.plotfile.empty()) {
          return hpx::make_ready_future();
        }

        if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        const auto& box = level.boxes.at(static_cast<std::size_t>(block));
        const int nx = box.hi.x - box.lo.x + 1;
        const int ny = box.hi.y - box.lo.y + 1;
        const int nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }

        std::size_t out_bytes = 0;
        if (params.bytes_per_value > 0) {
          out_bytes = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                      static_cast<std::size_t>(nz) *
                      static_cast<std::size_t>(params.bytes_per_value);
        }
        if (out_bytes == 0) {
          return hpx::make_ready_future();
        }

        if (outputs[0].data.size() != out_bytes) {
          outputs[0].data.assign(out_bytes, 0);
        }

        auto reader = get_plotfile_reader(params.plotfile);
        if (!reader || params.level < 0 || params.level >= reader->num_levels()) {
          return hpx::make_ready_future();
        }
        if (block >= reader->num_fabs(params.level)) {
          return hpx::make_ready_future();
        }

        auto data = reader->read_fab(params.level, block, params.comp, 1);
        if (data.ncomp < 1 || data.nx != nx || data.ny != ny || data.nz != nz) {
          return hpx::make_ready_future();
        }

        if (data.type == plotfile::RealType::kFloat32) {
          const auto* in = reinterpret_cast<const float*>(data.bytes.data());
          if (params.bytes_per_value == 4) {
            auto* out = reinterpret_cast<float*>(outputs[0].data.data());
            transpose_plotfile_axes(in, out, nx, ny, nz);
          } else if (params.bytes_per_value == 8) {
            auto* out = reinterpret_cast<double*>(outputs[0].data.data());
            transpose_plotfile_axes(in, out, nx, ny, nz);
          }
        } else if (data.type == plotfile::RealType::kFloat64) {
          const auto* in = reinterpret_cast<const double*>(data.bytes.data());
          if (params.bytes_per_value == 8) {
            auto* out = reinterpret_cast<double*>(outputs[0].data.data());
            transpose_plotfile_axes(in, out, nx, ny, nz);
          } else if (params.bytes_per_value == 4) {
            auto* out = reinterpret_cast<float*>(outputs[0].data.data());
            transpose_plotfile_axes(in, out, nx, ny, nz);
          }
        }

        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "uniform_slice_cellavg_accumulate", .n_inputs = 1, .n_outputs = 2,
                 .needs_neighbors = false},
      [](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
         const NeighborViews&, std::span<HostView> outputs,
         std::span<const std::uint8_t> params_msgpack) {
        struct Box3 {
          std::array<int, 3> lo{0, 0, 0};
          std::array<int, 3> hi{0, 0, 0};
        };
        struct Params {
          int axis = 2;
          double coord = 0.0;
          int plane_index = 0;
          bool has_plane_index = false;
          std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
          std::array<int, 2> resolution{1, 1};
          int bytes_per_value = 4;
          std::vector<Box3> covered_boxes;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            auto get_key = [&](const char* key) -> const msgpack::object* {
              for (uint32_t i = 0; i < root.via.map.size; ++i) {
                const auto& k = root.via.map.ptr[i].key;
                if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
                  return &root.via.map.ptr[i].val;
                }
              }
              return nullptr;
            };
            if (const auto* axis = get_key("axis"); axis &&
                                                   (axis->type == msgpack::type::POSITIVE_INTEGER ||
                                                    axis->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.axis = axis->as<int>();
            }
            if (const auto* coord = get_key("coord"); coord &&
                                                     (coord->type == msgpack::type::FLOAT ||
                                                      coord->type == msgpack::type::POSITIVE_INTEGER ||
                                                      coord->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.coord = coord->as<double>();
            }
            if (const auto* plane_idx = get_key("plane_index"); plane_idx &&
                                                         (plane_idx->type == msgpack::type::POSITIVE_INTEGER ||
                                                          plane_idx->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.plane_index = plane_idx->as<int>();
              params.has_plane_index = true;
            }
            if (const auto* rect = get_key("rect"); rect && rect->type == msgpack::type::ARRAY &&
                                                   rect->via.array.size == 4) {
              for (uint32_t i = 0; i < 4; ++i) {
                params.rect[i] = rect->via.array.ptr[i].as<double>();
              }
            }
            if (const auto* res = get_key("resolution"); res && res->type == msgpack::type::ARRAY &&
                                                       res->via.array.size == 2) {
              params.resolution[0] = res->via.array.ptr[0].as<int>();
              params.resolution[1] = res->via.array.ptr[1].as<int>();
            }
            if (const auto* bpv = get_key("bytes_per_value"); bpv &&
                                                         (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                                                          bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.bytes_per_value = bpv->as<int>();
            }
            if (const auto* boxes = get_key("covered_boxes"); boxes &&
                                                     boxes->type == msgpack::type::ARRAY) {
              params.covered_boxes.clear();
              params.covered_boxes.reserve(boxes->via.array.size);
              for (uint32_t i = 0; i < boxes->via.array.size; ++i) {
                const auto& entry = boxes->via.array.ptr[i];
                if (entry.type != msgpack::type::ARRAY || entry.via.array.size != 2) {
                  continue;
                }
                const auto& lo = entry.via.array.ptr[0];
                const auto& hi = entry.via.array.ptr[1];
                if (lo.type != msgpack::type::ARRAY || hi.type != msgpack::type::ARRAY ||
                    lo.via.array.size != 3 || hi.via.array.size != 3) {
                  continue;
                }
                Box3 box;
                for (uint32_t d = 0; d < 3; ++d) {
                  box.lo[d] = lo.via.array.ptr[d].as<int>();
                  box.hi[d] = hi.via.array.ptr[d].as<int>();
                }
                params.covered_boxes.push_back(box);
              }
            }
          }
        }

        const auto out_nx = params.resolution[0];
        const auto out_ny = params.resolution[1];
        if (outputs.size() < 2 || inputs.empty() || out_nx <= 0 || out_ny <= 0) {
          return hpx::make_ready_future();
        }

        const std::size_t bytes = static_cast<std::size_t>(out_nx) *
                                  static_cast<std::size_t>(out_ny) * sizeof(double);
        if (outputs[0].data.size() != bytes) {
          outputs[0].data.assign(bytes, 0);
        } else {
          std::fill(outputs[0].data.begin(), outputs[0].data.end(), 0);
        }
        if (outputs[1].data.size() != bytes) {
          outputs[1].data.assign(bytes, 0);
        } else {
          std::fill(outputs[1].data.begin(), outputs[1].data.end(), 0);
        }

        if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        const auto& box = level.boxes.at(static_cast<std::size_t>(block));
        const int nx = box.hi.x - box.lo.x + 1;
        const int ny = box.hi.y - box.lo.y + 1;
        const int nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }

        const int axis = params.axis;
        int u_axis = 0;
        int v_axis = 1;
        if (axis == 0) {
          u_axis = 1;
          v_axis = 2;
        } else if (axis == 1) {
          u_axis = 0;
          v_axis = 2;
        } else {
          u_axis = 0;
          v_axis = 1;
        }

        auto cell_index = [&](int ax, double coord) -> int {
          const double x0 = level.geom.x0[ax];
          const double dx = level.geom.dx[ax];
          const int origin = level.geom.index_origin[ax];
          if (dx == 0.0) {
            return origin;
          }
          const double idx_f = (coord - x0) / dx;
          return static_cast<int>(std::floor(idx_f)) + origin;
        };

        const int k_global = params.has_plane_index ? params.plane_index : cell_index(axis, params.coord);
        const int k_local = (axis == 0 ? k_global - box.lo.x
                                       : (axis == 1 ? k_global - box.lo.y : k_global - box.lo.z));
        if ((axis == 0 && (k_local < 0 || k_local >= nx)) ||
            (axis == 1 && (k_local < 0 || k_local >= ny)) ||
            (axis == 2 && (k_local < 0 || k_local >= nz))) {
          return hpx::make_ready_future();
        }

        const double u0 = params.rect[0];
        const double v0 = params.rect[1];
        const double u1 = params.rect[2];
        const double v1 = params.rect[3];
        const double umin = std::min(u0, u1);
        const double umax = std::max(u0, u1);
        const double vmin = std::min(v0, v1);
        const double vmax = std::max(v0, v1);
        const double du = (out_nx > 0) ? (umax - umin) / static_cast<double>(out_nx) : 0.0;
        const double dv = (out_ny > 0) ? (vmax - vmin) / static_cast<double>(out_ny) : 0.0;
        if (du == 0.0 || dv == 0.0) {
          return hpx::make_ready_future();
        }

        auto in_index = [&](int i, int j, int k) -> std::size_t {
          return static_cast<std::size_t>((i * ny + j) * nz + k);
        };

        auto covered = [&](int ix, int iy, int iz) -> bool {
          for (const auto& b : params.covered_boxes) {
            if (ix >= b.lo[0] && ix <= b.hi[0] &&
                iy >= b.lo[1] && iy <= b.hi[1] &&
                iz >= b.lo[2] && iz <= b.hi[2]) {
              return true;
            }
          }
          return false;
        };

        const auto& in = inputs[0].data;
        auto* out_sum = reinterpret_cast<double*>(outputs[0].data.data());
        auto* out_area = reinterpret_cast<double*>(outputs[1].data.data());

        auto cell_edge = [&](int ax, int idx) -> double {
          return level.geom.x0[ax] + (idx - level.geom.index_origin[ax]) * level.geom.dx[ax];
        };

        for (int v_local = 0; v_local < (v_axis == 0 ? nx : (v_axis == 1 ? ny : nz)); ++v_local) {
          const int v_global = (v_axis == 0 ? v_local + box.lo.x
                                            : (v_axis == 1 ? v_local + box.lo.y
                                                           : v_local + box.lo.z));
          for (int u_local = 0; u_local < (u_axis == 0 ? nx : (u_axis == 1 ? ny : nz)); ++u_local) {
            const int u_global = (u_axis == 0 ? u_local + box.lo.x
                                              : (u_axis == 1 ? u_local + box.lo.y
                                                             : u_local + box.lo.z));
            int idx_global[3]{0, 0, 0};
            idx_global[axis] = k_global;
            idx_global[u_axis] = u_global;
            idx_global[v_axis] = v_global;
            if (covered(idx_global[0], idx_global[1], idx_global[2])) {
              continue;
            }

            int idx_local[3]{0, 0, 0};
            idx_local[axis] = k_local;
            idx_local[u_axis] = u_local;
            idx_local[v_axis] = v_local;
            const auto data_idx = in_index(idx_local[0], idx_local[1], idx_local[2]);
            double value = 0.0;
            if (params.bytes_per_value == 4) {
              if (data_idx * sizeof(float) < in.size()) {
                value = reinterpret_cast<const float*>(in.data())[data_idx];
              }
            } else if (params.bytes_per_value == 8) {
              if (data_idx * sizeof(double) < in.size()) {
                value = reinterpret_cast<const double*>(in.data())[data_idx];
              }
            }

            const double u_cell_lo = cell_edge(u_axis, u_global);
            const double u_cell_hi = u_cell_lo + level.geom.dx[u_axis];
            const double v_cell_lo = cell_edge(v_axis, v_global);
            const double v_cell_hi = v_cell_lo + level.geom.dx[v_axis];

            int i0 = static_cast<int>(std::floor((u_cell_lo - umin) / du));
            int i1 = static_cast<int>(std::floor((u_cell_hi - umin) / du));
            int j0 = static_cast<int>(std::floor((v_cell_lo - vmin) / dv));
            int j1 = static_cast<int>(std::floor((v_cell_hi - vmin) / dv));
            if (i1 < 0 || j1 < 0 || i0 >= out_nx || j0 >= out_ny) {
              continue;
            }
            i0 = std::max(i0, 0);
            j0 = std::max(j0, 0);
            i1 = std::min(i1, out_nx - 1);
            j1 = std::min(j1, out_ny - 1);

            for (int j = j0; j <= j1; ++j) {
              const double pv0 = vmin + static_cast<double>(j) * dv;
              const double pv1 = pv0 + dv;
              const double ov = std::max(0.0, std::min(v_cell_hi, pv1) - std::max(v_cell_lo, pv0));
              if (ov <= 0.0) {
                continue;
              }
              for (int i = i0; i <= i1; ++i) {
                const double pu0 = umin + static_cast<double>(i) * du;
                const double pu1 = pu0 + du;
                const double ou = std::max(0.0, std::min(u_cell_hi, pu1) - std::max(u_cell_lo, pu0));
                if (ou <= 0.0) {
                  continue;
                }
                const double area = ou * ov;
                const std::size_t out_idx = static_cast<std::size_t>(j) * out_nx + i;
                out_sum[out_idx] += value * area;
                out_area[out_idx] += area;
              }
            }
          }
        }

        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "uniform_projection_accumulate", .n_inputs = 1, .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
         const NeighborViews&, std::span<HostView> outputs,
         std::span<const std::uint8_t> params_msgpack) {
        struct Box3 {
          std::array<int, 3> lo{0, 0, 0};
          std::array<int, 3> hi{0, 0, 0};
        };
        struct Params {
          int axis = 2;
          std::array<double, 2> axis_bounds{0.0, 1.0};
          std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
          std::array<int, 2> resolution{1, 1};
          int bytes_per_value = 4;
          std::vector<Box3> covered_boxes;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            auto get_key = [&](const char* key) -> const msgpack::object* {
              for (uint32_t i = 0; i < root.via.map.size; ++i) {
                const auto& k = root.via.map.ptr[i].key;
                if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
                  return &root.via.map.ptr[i].val;
                }
              }
              return nullptr;
            };
            if (const auto* axis = get_key("axis"); axis &&
                                                   (axis->type == msgpack::type::POSITIVE_INTEGER ||
                                                    axis->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.axis = axis->as<int>();
            }
            if (const auto* bounds = get_key("axis_bounds"); bounds &&
                                                     bounds->type == msgpack::type::ARRAY &&
                                                     bounds->via.array.size == 2) {
              params.axis_bounds[0] = bounds->via.array.ptr[0].as<double>();
              params.axis_bounds[1] = bounds->via.array.ptr[1].as<double>();
            }
            if (const auto* rect = get_key("rect"); rect && rect->type == msgpack::type::ARRAY &&
                                                   rect->via.array.size == 4) {
              for (uint32_t i = 0; i < 4; ++i) {
                params.rect[i] = rect->via.array.ptr[i].as<double>();
              }
            }
            if (const auto* res = get_key("resolution"); res && res->type == msgpack::type::ARRAY &&
                                                       res->via.array.size == 2) {
              params.resolution[0] = res->via.array.ptr[0].as<int>();
              params.resolution[1] = res->via.array.ptr[1].as<int>();
            }
            if (const auto* bpv = get_key("bytes_per_value"); bpv &&
                                                         (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                                                          bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.bytes_per_value = bpv->as<int>();
            }
            if (const auto* boxes = get_key("covered_boxes"); boxes &&
                                                     boxes->type == msgpack::type::ARRAY) {
              params.covered_boxes.clear();
              params.covered_boxes.reserve(boxes->via.array.size);
              for (uint32_t i = 0; i < boxes->via.array.size; ++i) {
                const auto& entry = boxes->via.array.ptr[i];
                if (entry.type != msgpack::type::ARRAY || entry.via.array.size != 2) {
                  continue;
                }
                const auto& lo = entry.via.array.ptr[0];
                const auto& hi = entry.via.array.ptr[1];
                if (lo.type != msgpack::type::ARRAY || hi.type != msgpack::type::ARRAY ||
                    lo.via.array.size != 3 || hi.via.array.size != 3) {
                  continue;
                }
                Box3 box;
                for (uint32_t d = 0; d < 3; ++d) {
                  box.lo[d] = lo.via.array.ptr[d].as<int>();
                  box.hi[d] = hi.via.array.ptr[d].as<int>();
                }
                params.covered_boxes.push_back(box);
              }
            }
          }
        }

        const auto out_nx = params.resolution[0];
        const auto out_ny = params.resolution[1];
        if (outputs.empty() || inputs.empty() || out_nx <= 0 || out_ny <= 0) {
          return hpx::make_ready_future();
        }

        const std::size_t bytes = static_cast<std::size_t>(out_nx) *
                                  static_cast<std::size_t>(out_ny) * sizeof(double);
        if (outputs[0].data.size() != bytes) {
          outputs[0].data.assign(bytes, 0);
        } else {
          std::fill(outputs[0].data.begin(), outputs[0].data.end(), 0);
        }

        if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        const auto& box = level.boxes.at(static_cast<std::size_t>(block));
        const int nx = box.hi.x - box.lo.x + 1;
        const int ny = box.hi.y - box.lo.y + 1;
        const int nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }

        const int axis = params.axis;
        int u_axis = 0;
        int v_axis = 1;
        if (axis == 0) {
          u_axis = 1;
          v_axis = 2;
        } else if (axis == 1) {
          u_axis = 0;
          v_axis = 2;
        } else {
          u_axis = 0;
          v_axis = 1;
        }

        const double u0 = params.rect[0];
        const double v0 = params.rect[1];
        const double u1 = params.rect[2];
        const double v1 = params.rect[3];
        const double umin = std::min(u0, u1);
        const double umax = std::max(u0, u1);
        const double vmin = std::min(v0, v1);
        const double vmax = std::max(v0, v1);
        const double du = (out_nx > 0) ? (umax - umin) / static_cast<double>(out_nx) : 0.0;
        const double dv = (out_ny > 0) ? (vmax - vmin) / static_cast<double>(out_ny) : 0.0;
        if (du == 0.0 || dv == 0.0) {
          return hpx::make_ready_future();
        }

        const double a0 = params.axis_bounds[0];
        const double a1 = params.axis_bounds[1];
        const double amin = std::min(a0, a1);
        const double amax = std::max(a0, a1);
        if (!(amax > amin)) {
          return hpx::make_ready_future();
        }

        auto in_index = [&](int i, int j, int k) -> std::size_t {
          return static_cast<std::size_t>((i * ny + j) * nz + k);
        };

        auto covered = [&](int ix, int iy, int iz) -> bool {
          for (const auto& b : params.covered_boxes) {
            if (ix >= b.lo[0] && ix <= b.hi[0] &&
                iy >= b.lo[1] && iy <= b.hi[1] &&
                iz >= b.lo[2] && iz <= b.hi[2]) {
              return true;
            }
          }
          return false;
        };

        auto cell_edge = [&](int ax, int idx) -> double {
          return level.geom.x0[ax] + (idx - level.geom.index_origin[ax]) * level.geom.dx[ax];
        };

        const auto& in = inputs[0].data;
        auto* out_sum = reinterpret_cast<double*>(outputs[0].data.data());

        for (int i = 0; i < nx; ++i) {
          const int gx = box.lo.x + i;
          for (int j = 0; j < ny; ++j) {
            const int gy = box.lo.y + j;
            for (int k = 0; k < nz; ++k) {
              const int gz = box.lo.z + k;
              if (covered(gx, gy, gz)) {
                continue;
              }

              const int a_global = axis == 0 ? gx : (axis == 1 ? gy : gz);
              const double a_cell_lo = cell_edge(axis, a_global);
              const double a_cell_hi = a_cell_lo + level.geom.dx[axis];
              const double oa = std::max(0.0, std::min(a_cell_hi, amax) - std::max(a_cell_lo, amin));
              if (oa <= 0.0) {
                continue;
              }

              const int u_global = u_axis == 0 ? gx : (u_axis == 1 ? gy : gz);
              const int v_global = v_axis == 0 ? gx : (v_axis == 1 ? gy : gz);
              const double u_cell_lo = cell_edge(u_axis, u_global);
              const double u_cell_hi = u_cell_lo + level.geom.dx[u_axis];
              const double v_cell_lo = cell_edge(v_axis, v_global);
              const double v_cell_hi = v_cell_lo + level.geom.dx[v_axis];

              int i0 = static_cast<int>(std::floor((u_cell_lo - umin) / du));
              int i1 = static_cast<int>(std::floor((u_cell_hi - umin) / du));
              int j0 = static_cast<int>(std::floor((v_cell_lo - vmin) / dv));
              int j1 = static_cast<int>(std::floor((v_cell_hi - vmin) / dv));
              if (i1 < 0 || j1 < 0 || i0 >= out_nx || j0 >= out_ny) {
                continue;
              }
              i0 = std::max(i0, 0);
              j0 = std::max(j0, 0);
              i1 = std::min(i1, out_nx - 1);
              j1 = std::min(j1, out_ny - 1);

              double value = 0.0;
              const auto data_idx = in_index(i, j, k);
              if (params.bytes_per_value == 4) {
                if (data_idx * sizeof(float) < in.size()) {
                  value = reinterpret_cast<const float*>(in.data())[data_idx];
                }
              } else if (params.bytes_per_value == 8) {
                if (data_idx * sizeof(double) < in.size()) {
                  value = reinterpret_cast<const double*>(in.data())[data_idx];
                }
              }

              for (int jj = j0; jj <= j1; ++jj) {
                const double pv0 = vmin + static_cast<double>(jj) * dv;
                const double pv1 = pv0 + dv;
                const double ov = std::max(0.0, std::min(v_cell_hi, pv1) - std::max(v_cell_lo, pv0));
                if (ov <= 0.0) {
                  continue;
                }
                for (int ii = i0; ii <= i1; ++ii) {
                  const double pu0 = umin + static_cast<double>(ii) * du;
                  const double pu1 = pu0 + du;
                  const double ou = std::max(0.0, std::min(u_cell_hi, pu1) - std::max(u_cell_lo, pu0));
                  if (ou <= 0.0) {
                    continue;
                  }
                  const double volume = ou * ov * oa;
                  const std::size_t out_idx = static_cast<std::size_t>(jj) * out_nx + ii;
                  out_sum[out_idx] += value * volume;
                }
              }
            }
          }
        }

        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "vorticity_mag", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta& level, int32_t block, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (outputs.empty() || block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }
        const auto& box = level.boxes.at(static_cast<std::size_t>(block));
        const int32_t nx = box.hi.x - box.lo.x + 1;
        const int32_t ny = box.hi.y - box.lo.y + 1;
        const int32_t nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }
        const std::size_t ncell =
            static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
        outputs[0].data.assign(ncell * sizeof(double), 0);
        auto* out = reinterpret_cast<double*>(outputs[0].data.data());

        // Preferred path: three scalar gradient inputs [grad(vx), grad(vy), grad(vz)].
        if (inputs.size() >= 3) {
          const auto& gx = inputs[0].data;
          const auto& gy = inputs[1].data;
          const auto& gz = inputs[2].data;
          if (gx.size() >= ncell * 3 * sizeof(double) && gy.size() >= ncell * 3 * sizeof(double) &&
              gz.size() >= ncell * 3 * sizeof(double)) {
            const auto* du = reinterpret_cast<const double*>(gx.data());
            const auto* dv = reinterpret_cast<const double*>(gy.data());
            const auto* dw = reinterpret_cast<const double*>(gz.data());
            for (std::size_t idx = 0; idx < ncell; ++idx) {
              const double dudy = du[3 * idx + 1];
              const double dudz = du[3 * idx + 2];
              const double dvdx = dv[3 * idx + 0];
              const double dvdz = dv[3 * idx + 2];
              const double dwdx = dw[3 * idx + 0];
              const double dwdy = dw[3 * idx + 1];
              const double wx = dwdy - dvdz;
              const double wy = dudz - dwdx;
              const double wz = dvdx - dudy;
              out[idx] = std::sqrt(wx * wx + wy * wy + wz * wz);
            }
            return hpx::make_ready_future();
          }
        }

        // Backward-compatible fallback: one scalar gradient input -> |grad(S)|.
        if (!inputs.empty()) {
          const auto& g = inputs[0].data;
          if (g.size() >= ncell * 3 * sizeof(double)) {
            const auto* grad = reinterpret_cast<const double*>(g.data());
            for (std::size_t idx = 0; idx < ncell; ++idx) {
              const double gx = grad[3 * idx + 0];
              const double gy = grad[3 * idx + 1];
              const double gz = grad[3 * idx + 2];
              out[idx] = std::sqrt(gx * gx + gy * gy + gz * gz);
            }
          }
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "uniform_slice", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
         const NeighborViews&, std::span<HostView> outputs,
         std::span<const std::uint8_t> params_msgpack) {
        if (log_locality) {
          std::cout << "[kangaroo] uniform_slice block=" << block
                    << " locality=" << hpx::get_locality_id() << std::endl;
        }
        struct Params {
          int axis = 2;
          double coord = 0.0;
          std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
          std::array<int, 2> resolution{1, 1};
          int bytes_per_value = 4;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            auto get_key = [&](const char* key) -> const msgpack::object* {
              for (uint32_t i = 0; i < root.via.map.size; ++i) {
                const auto& k = root.via.map.ptr[i].key;
                if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
                  return &root.via.map.ptr[i].val;
                }
              }
              return nullptr;
            };
            if (const auto* axis = get_key("axis"); axis &&
                                                   (axis->type == msgpack::type::POSITIVE_INTEGER ||
                                                    axis->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.axis = axis->as<int>();
            }
            if (const auto* coord = get_key("coord"); coord &&
                                                     (coord->type == msgpack::type::FLOAT ||
                                                      coord->type == msgpack::type::POSITIVE_INTEGER ||
                                                      coord->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.coord = coord->as<double>();
            }
            if (const auto* rect = get_key("rect"); rect && rect->type == msgpack::type::ARRAY &&
                                                   rect->via.array.size == 4) {
              for (uint32_t i = 0; i < 4; ++i) {
                params.rect[i] = rect->via.array.ptr[i].as<double>();
              }
            }
            if (const auto* res = get_key("resolution"); res && res->type == msgpack::type::ARRAY &&
                                                       res->via.array.size == 2) {
              params.resolution[0] = res->via.array.ptr[0].as<int>();
              params.resolution[1] = res->via.array.ptr[1].as<int>();
            }
            if (const auto* bpv = get_key("bytes_per_value"); bpv &&
                                                         (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                                                          bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
              params.bytes_per_value = bpv->as<int>();
            }
          }
        }

        const auto out_nx = params.resolution[0];
        const auto out_ny = params.resolution[1];
        std::size_t bytes = 0;
        if (out_nx > 0 && out_ny > 0 && params.bytes_per_value > 0) {
          bytes = static_cast<std::size_t>(out_nx) * static_cast<std::size_t>(out_ny) *
                  static_cast<std::size_t>(params.bytes_per_value);
        }

        if (outputs.empty() || inputs.empty() || bytes == 0) {
          return hpx::make_ready_future();
        }

        if (outputs[0].data.size() != bytes) {
          outputs[0].data.assign(bytes, 0);
        } else {
          std::fill(outputs[0].data.begin(), outputs[0].data.end(), 0);
        }

        if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        const auto& box = level.boxes.at(static_cast<std::size_t>(block));
        const int nx = box.hi.x - box.lo.x + 1;
        const int ny = box.hi.y - box.lo.y + 1;
        const int nz = box.hi.z - box.lo.z + 1;
        if (nx <= 0 || ny <= 0 || nz <= 0) {
          return hpx::make_ready_future();
        }

        const int axis = params.axis;
        int u_axis = 0;
        int v_axis = 1;
        if (axis == 0) {
          u_axis = 1;
          v_axis = 2;
        } else if (axis == 1) {
          u_axis = 0;
          v_axis = 2;
        } else {
          u_axis = 0;
          v_axis = 1;
        }

        auto cell_index = [&](int ax, double coord) -> int {
          const double x0 = level.geom.x0[ax];
          const double dx = level.geom.dx[ax];
          const int origin = level.geom.index_origin[ax];
          if (dx == 0.0) {
            return origin;
          }
          const double idx_f = (coord - x0) / dx;
          return static_cast<int>(std::floor(idx_f)) + origin;
        };

        const int k_global = cell_index(axis, params.coord);
        const int k_local = (axis == 0 ? k_global - box.lo.x
                                       : (axis == 1 ? k_global - box.lo.y : k_global - box.lo.z));
        if ((axis == 0 && (k_local < 0 || k_local >= nx)) ||
            (axis == 1 && (k_local < 0 || k_local >= ny)) ||
            (axis == 2 && (k_local < 0 || k_local >= nz))) {
          return hpx::make_ready_future();
        }

        const double u0 = params.rect[0];
        const double v0 = params.rect[1];
        const double u1 = params.rect[2];
        const double v1 = params.rect[3];
        const double du = (out_nx > 0) ? (u1 - u0) / static_cast<double>(out_nx) : 0.0;
        const double dv = (out_ny > 0) ? (v1 - v0) / static_cast<double>(out_ny) : 0.0;

        auto in_index = [&](int i, int j, int k) -> std::size_t {
          return static_cast<std::size_t>((i * ny + j) * nz + k);
        };

        const auto& in = inputs[0].data;
        if (params.bytes_per_value == 4) {
          const auto* in_f = reinterpret_cast<const float*>(in.data());
          auto* out_f = reinterpret_cast<float*>(outputs[0].data.data());
          for (int j = 0; j < out_ny; ++j) {
            const double v = v0 + (static_cast<double>(j) + 0.5) * dv;
            const int v_global = cell_index(v_axis, v);
            const int v_local = v_axis == 0 ? v_global - box.lo.x
                                            : (v_axis == 1 ? v_global - box.lo.y
                                                           : v_global - box.lo.z);
            for (int i = 0; i < out_nx; ++i) {
              const double u = u0 + (static_cast<double>(i) + 0.5) * du;
              const int u_global = cell_index(u_axis, u);
              const int u_local = u_axis == 0 ? u_global - box.lo.x
                                              : (u_axis == 1 ? u_global - box.lo.y
                                                             : u_global - box.lo.z);
              float value = 0.0f;
              if (u_local >= 0 && v_local >= 0) {
                if ((u_axis == 0 && u_local < nx) || (u_axis == 1 && u_local < ny) ||
                    (u_axis == 2 && u_local < nz)) {
                  if ((v_axis == 0 && v_local < nx) || (v_axis == 1 && v_local < ny) ||
                      (v_axis == 2 && v_local < nz)) {
                    int ii = 0;
                    int jj = 0;
                    int kk = 0;
                    if (axis == 0) {
                      ii = k_local;
                      jj = u_axis == 1 ? u_local : v_local;
                      kk = u_axis == 2 ? u_local : v_local;
                    } else if (axis == 1) {
                      ii = u_axis == 0 ? u_local : v_local;
                      jj = k_local;
                      kk = u_axis == 2 ? u_local : v_local;
                    } else {
                      ii = u_axis == 0 ? u_local : v_local;
                      jj = u_axis == 1 ? u_local : v_local;
                      kk = k_local;
                    }
                    const auto idx = in_index(ii, jj, kk);
                    if (idx * sizeof(float) < in.size()) {
                      value = in_f[idx];
                    }
                  }
                }
              }
              out_f[static_cast<std::size_t>(j) * out_nx + i] = value;
            }
          }
        } else if (params.bytes_per_value == 8) {
          const auto* in_d = reinterpret_cast<const double*>(in.data());
          auto* out_d = reinterpret_cast<double*>(outputs[0].data.data());
          for (int j = 0; j < out_ny; ++j) {
            const double v = v0 + (static_cast<double>(j) + 0.5) * dv;
            const int v_global = cell_index(v_axis, v);
            const int v_local = v_axis == 0 ? v_global - box.lo.x
                                            : (v_axis == 1 ? v_global - box.lo.y
                                                           : v_global - box.lo.z);
            for (int i = 0; i < out_nx; ++i) {
              const double u = u0 + (static_cast<double>(i) + 0.5) * du;
              const int u_global = cell_index(u_axis, u);
              const int u_local = u_axis == 0 ? u_global - box.lo.x
                                              : (u_axis == 1 ? u_global - box.lo.y
                                                             : u_global - box.lo.z);
              double value = 0.0;
              if (u_local >= 0 && v_local >= 0) {
                if ((u_axis == 0 && u_local < nx) || (u_axis == 1 && u_local < ny) ||
                    (u_axis == 2 && u_local < nz)) {
                  if ((v_axis == 0 && v_local < nx) || (v_axis == 1 && v_local < ny) ||
                      (v_axis == 2 && v_local < nz)) {
                    int ii = 0;
                    int jj = 0;
                    int kk = 0;
                    if (axis == 0) {
                      ii = k_local;
                      jj = u_axis == 1 ? u_local : v_local;
                      kk = u_axis == 2 ? u_local : v_local;
                    } else if (axis == 1) {
                      ii = u_axis == 0 ? u_local : v_local;
                      jj = k_local;
                      kk = u_axis == 2 ? u_local : v_local;
                    } else {
                      ii = u_axis == 0 ? u_local : v_local;
                      jj = u_axis == 1 ? u_local : v_local;
                      kk = k_local;
                    }
                    const auto idx = in_index(ii, jj, kk);
                    if (idx * sizeof(double) < in.size()) {
                      value = in_d[idx];
                    }
                  }
                }
              }
              out_d[static_cast<std::size_t>(j) * out_nx + i] = value;
            }
          }
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "uniform_slice_add", .n_inputs = 2, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t> params_msgpack) {
        struct Params {
          int bytes_per_value = 8;
        } params;
        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            for (uint32_t i = 0; i < root.via.map.size; ++i) {
              const auto& k = root.via.map.ptr[i].key;
              if (k.type == msgpack::type::STR && k.as<std::string>() == "bytes_per_value") {
                const auto& v = root.via.map.ptr[i].val;
                if (v.type == msgpack::type::POSITIVE_INTEGER ||
                    v.type == msgpack::type::NEGATIVE_INTEGER) {
                  params.bytes_per_value = v.as<int>();
                }
              }
            }
          }
        }

        if (outputs.empty() || inputs.size() < 2) {
          return hpx::make_ready_future();
        }
        auto& out = outputs[0].data;
        if (out.empty()) {
          return hpx::make_ready_future();
        }

        if (params.bytes_per_value == 8) {
          const std::size_t n = out.size() / sizeof(double);
          auto* out_d = reinterpret_cast<double*>(out.data());
          std::fill(out_d, out_d + n, 0.0);
          for (const auto& in_view : inputs) {
            const auto& in = in_view.data;
            if (in.empty()) {
              continue;
            }
            const std::size_t n_in = std::min(n, in.size() / sizeof(double));
            const auto* in_d = reinterpret_cast<const double*>(in.data());
            for (std::size_t i = 0; i < n_in; ++i) {
              out_d[i] += in_d[i];
            }
          }
        } else if (params.bytes_per_value == 4) {
          const std::size_t n = out.size() / sizeof(float);
          auto* out_f = reinterpret_cast<float*>(out.data());
          std::fill(out_f, out_f + n, 0.0f);
          for (const auto& in_view : inputs) {
            const auto& in = in_view.data;
            if (in.empty()) {
              continue;
            }
            const std::size_t n_in = std::min(n, in.size() / sizeof(float));
            const auto* in_f = reinterpret_cast<const float*>(in.data());
            for (std::size_t i = 0; i < n_in; ++i) {
              out_f[i] += in_f[i];
            }
          }
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "uniform_slice_finalize", .n_inputs = 2, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t> params_msgpack) {
        struct Params {
          int bytes_per_value = 4;
          double pixel_area = 1.0;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            for (uint32_t i = 0; i < root.via.map.size; ++i) {
              const auto& k = root.via.map.ptr[i].key;
              if (k.type == msgpack::type::STR && k.as<std::string>() == "bytes_per_value") {
                const auto& v = root.via.map.ptr[i].val;
                if (v.type == msgpack::type::POSITIVE_INTEGER ||
                    v.type == msgpack::type::NEGATIVE_INTEGER) {
                  params.bytes_per_value = v.as<int>();
                }
              }
              if (k.type == msgpack::type::STR && k.as<std::string>() == "pixel_area") {
                const auto& v = root.via.map.ptr[i].val;
                if (v.type == msgpack::type::FLOAT ||
                    v.type == msgpack::type::POSITIVE_INTEGER ||
                    v.type == msgpack::type::NEGATIVE_INTEGER) {
                  params.pixel_area = v.as<double>();
                }
              }
            }
          }
        }

        if (outputs.empty() || inputs.size() < 2 || params.pixel_area == 0.0) {
          return hpx::make_ready_future();
        }

        const auto& sum = inputs[0].data;
        const auto& area = inputs[1].data;
        auto& out = outputs[0].data;
        if (out.empty()) {
          return hpx::make_ready_future();
        }
        const std::size_t n = std::min(sum.size(), area.size()) / sizeof(double);
        const auto* sum_d = reinterpret_cast<const double*>(sum.data());
        const auto* area_d = reinterpret_cast<const double*>(area.data());

        if (params.bytes_per_value == 8) {
          auto* out_d = reinterpret_cast<double*>(out.data());
          for (std::size_t i = 0; i < n; ++i) {
            if (area_d[i] == 0.0) {
              out_d[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
              out_d[i] = sum_d[i] / params.pixel_area;
            }
          }
        } else if (params.bytes_per_value == 4) {
          auto* out_f = reinterpret_cast<float*>(out.data());
          for (std::size_t i = 0; i < n; ++i) {
            if (area_d[i] == 0.0) {
              out_f[i] = std::numeric_limits<float>::quiet_NaN();
            } else {
              out_f[i] = static_cast<float>(sum_d[i] / params.pixel_area);
            }
          }
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "uniform_slice_reduce", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t> params_msgpack) {
        struct Params {
          int bytes_per_value = 4;
        } params;

        if (!params_msgpack.empty()) {
          auto handle = msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()),
                                        params_msgpack.size());
          auto root = handle.get();
          if (root.type == msgpack::type::MAP) {
            for (uint32_t i = 0; i < root.via.map.size; ++i) {
              const auto& k = root.via.map.ptr[i].key;
              if (k.type == msgpack::type::STR && k.as<std::string>() == "bytes_per_value") {
                const auto& v = root.via.map.ptr[i].val;
                if (v.type == msgpack::type::POSITIVE_INTEGER ||
                    v.type == msgpack::type::NEGATIVE_INTEGER) {
                  params.bytes_per_value = v.as<int>();
                }
              }
            }
          }
        }

        if (outputs.empty() || inputs.empty()) {
          return hpx::make_ready_future();
        }

        auto& out = outputs[0].data;
        if (out.empty()) {
          return hpx::make_ready_future();
        }
        std::fill(out.begin(), out.end(), 0);

        if (params.bytes_per_value == 4) {
          const std::size_t n = out.size() / sizeof(float);
          auto* out_f = reinterpret_cast<float*>(out.data());

          for (const auto& in_view : inputs) {
            const auto& in = in_view.data;
            if (in.empty()) {
              continue;
            }
            const std::size_t n_in = std::min(n, in.size() / sizeof(float));
            const auto* in_f = reinterpret_cast<const float*>(in.data());
            for (std::size_t i = 0; i < n_in; ++i) {
              out_f[i] += in_f[i];
            }
          }
        } else if (params.bytes_per_value == 8) {
          const std::size_t n = out.size() / sizeof(double);
          auto* out_d = reinterpret_cast<double*>(out.data());

          for (const auto& in_view : inputs) {
            const auto& in = in_view.data;
            if (in.empty()) {
              continue;
            }
            const std::size_t n_in = std::min(n, in.size() / sizeof(double));
            const auto* in_d = reinterpret_cast<const double*>(in.data());
            for (std::size_t i = 0; i < n_in; ++i) {
              out_d[i] += in_d[i];
            }
          }
        }
        return hpx::make_ready_future();
      });
}

void set_runmeta_impl(const RunMeta& meta) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_runmeta = meta;
  g_has_runmeta = true;
}

void set_dataset_impl(const DatasetHandle& dataset) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_dataset = dataset;
  g_has_dataset = true;
}

void ensure_hpx_started() {
  std::call_once(g_hpx_start_once, []() {
    std::vector<std::string> argv_storage;
    if (g_hpx_cmdline_set && !g_hpx_cmdline.empty()) {
      argv_storage = g_hpx_cmdline;
    } else {
      argv_storage.emplace_back("kangaroo");
    }
    std::vector<char*> argv;
    argv.reserve(argv_storage.size());
    for (auto& arg : argv_storage) {
      argv.push_back(const_cast<char*>(arg.c_str()));
    }
    int argc = static_cast<int>(argv.size());
    hpx::init_params params;
    if (g_hpx_cfg_set && !g_hpx_cfg.empty()) {
      params.cfg = g_hpx_cfg;
    }
    hpx::start(nullptr, argc, argv.data(), params);
    g_hpx_started = true;
  });
  if (auto* rt = hpx::get_runtime_ptr(); rt != nullptr) {
    if (!g_hpx_thread_registered) {
      try {
        g_hpx_thread_registered = hpx::register_thread(rt, "kangaroo_python");
      } catch (...) {
        g_hpx_thread_registered = true;
      }
    }
  }
}

}  // namespace

void set_global_runmeta(const RunMeta& meta) {
  set_runmeta_impl(meta);
}

const RunMeta& global_runmeta() {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  if (!g_has_runmeta) {
    throw std::runtime_error("global RunMeta not initialized");
  }
  return g_runmeta;
}

void set_global_dataset(const DatasetHandle& dataset) {
  set_dataset_impl(dataset);
  DataServiceLocal::set_dataset(&g_dataset);
}

const DatasetHandle& global_dataset() {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  if (!g_has_dataset) {
    throw std::runtime_error("global DatasetHandle not initialized");
  }
  return g_dataset;
}

void set_global_kernel_registry(KernelRegistry* registry) {
  g_kernel_registry = registry;
}

KernelRegistry& global_kernels() {
  if (!g_kernel_registry) {
    throw std::runtime_error("global KernelRegistry not initialized");
  }
  return *g_kernel_registry;
}

void set_global_plan(int32_t plan_id, const PlanIR& plan) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_plans[plan_id] = plan;
}

const PlanIR& global_plan(int32_t plan_id) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  auto it = g_plans.find(plan_id);
  if (it == g_plans.end()) {
    throw std::runtime_error("global PlanIR not initialized for plan id");
  }
  return it->second;
}

void erase_global_plan(int32_t plan_id) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_plans.erase(plan_id);
}

void set_runmeta_action(const RunMeta& meta) {
  set_runmeta_impl(meta);
}

void set_dataset_action(const DatasetHandle& dataset) {
  set_dataset_impl(dataset);
  DataServiceLocal::set_dataset(&g_dataset);
}

void set_plan_action(int32_t plan_id, const PlanIR& plan) {
  set_global_plan(plan_id, plan);
}

void erase_plan_action(int32_t plan_id) {
  erase_global_plan(plan_id);
}

void set_event_log_action(const std::string& path) {
  set_event_log_path(path);
}

void preload_action(const RunMeta& meta,
                    const DatasetHandle& dataset,
                    const std::vector<int32_t>& fields) {
  DataServiceLocal::preload(meta, dataset, fields);
}

}  // namespace kangaroo

HPX_PLAIN_ACTION(kangaroo::set_runmeta_action, kangaroo_set_runmeta_action)
HPX_PLAIN_ACTION(kangaroo::set_dataset_action, kangaroo_set_dataset_action)
HPX_PLAIN_ACTION(kangaroo::set_plan_action, kangaroo_set_plan_action)
HPX_PLAIN_ACTION(kangaroo::erase_plan_action, kangaroo_erase_plan_action)
HPX_PLAIN_ACTION(kangaroo::preload_action, kangaroo_preload_action)
HPX_PLAIN_ACTION(kangaroo::set_event_log_action, kangaroo_set_event_log_action)

namespace kangaroo {

Runtime::Runtime() {
  init_event_log_from_env();
  set_global_kernel_registry(&kernel_registry_);
  register_default_kernels(kernel_registry_);
}

Runtime::Runtime(const std::vector<std::string>& hpx_config,
                 const std::vector<std::string>& hpx_cmdline) {
  init_event_log_from_env();
  if (g_hpx_started) {
    throw std::runtime_error("HPX already started; cannot change config/args");
  }
  if (g_hpx_cfg_set || g_hpx_cmdline_set) {
    throw std::runtime_error("HPX config/args already set; use the default constructor");
  }
  if (!hpx_config.empty()) {
    g_hpx_cfg = hpx_config;
    g_hpx_cfg_set = true;
  }
  if (!hpx_cmdline.empty()) {
    g_hpx_cmdline = hpx_cmdline;
    g_hpx_cmdline_set = true;
  }
  set_global_kernel_registry(&kernel_registry_);
  register_default_kernels(kernel_registry_);
}

void DatasetHandle::set_chunk(const ChunkRef& ref, HostView view) {
  if (!backend) {
    backend = std::make_shared<MemoryBackend>();
  }
  if (auto mem = std::dynamic_pointer_cast<MemoryBackend>(backend)) {
    mem->set_chunk(ref, std::move(view));
  } else {
    throw std::runtime_error("Cannot set_chunk on read-only backend");
  }
}

std::optional<HostView> DatasetHandle::get_chunk(const ChunkRef& ref) const {
  if (backend) {
    return backend->get_chunk(ref);
  }
  return std::nullopt;
}

bool DatasetHandle::has_chunk(const ChunkRef& ref) const {
  if (backend) {
    return backend->has_chunk(ref);
  }
  return false;
}

int32_t Runtime::alloc_field_id(const std::string&) {
  return next_field_id_++;
}

void Runtime::mark_field_persistent(int32_t fid, const std::string& name) {
  persistent_fields_[fid] = name;
}

KernelRegistry& Runtime::kernels() {
  return kernel_registry_;
}

void Runtime::set_event_log_path(const std::string& path) {
  set_event_log_path(path);
}

void set_event_log_path(const std::string& path) {
  {
    std::lock_guard<std::mutex> lock(g_event_log_mutex);
    g_event_log_path = path;
    g_event_log_enabled = !path.empty();
  }
  g_event_log_worker.set_path(g_event_log_path);
}

bool has_event_log() {
  return g_event_log_enabled && !g_event_log_path.empty();
}

void log_task_event(const TaskEvent& event) {
  if (!has_event_log()) {
    return;
  }
  g_event_log_worker.enqueue(event);
}

void Runtime::run_packed_plan(const std::vector<std::uint8_t>& packed,
                              const RunMetaHandle& runmeta,
                              const DatasetHandle& dataset) {
  ensure_hpx_started();
  PlanIR plan = decode_plan_msgpack(std::span<const std::uint8_t>(packed.data(), packed.size()));

  auto localities = hpx::find_all_localities();
  if (has_event_log()) {
    hpx::lcos::broadcast<::kangaroo_set_event_log_action>(localities, event_log_path()).get();
  }
  hpx::lcos::broadcast<::kangaroo_set_runmeta_action>(localities, runmeta.meta).get();
  hpx::lcos::broadcast<::kangaroo_set_dataset_action>(localities, dataset).get();
  int32_t plan_id = next_plan_id_++;
  hpx::lcos::broadcast<::kangaroo_set_plan_action>(localities, plan_id, plan).get();

  DataServiceLocal data;
  AdjacencyServiceLocal adjacency(runmeta.meta);
  Executor executor(plan_id, runmeta.meta, data, adjacency, kernel_registry_);

  executor.run(plan).get();
  hpx::lcos::broadcast<::kangaroo_erase_plan_action>(localities, plan_id).get();
}

void Runtime::preload_dataset(const RunMetaHandle& runmeta,
                              const DatasetHandle& dataset,
                              const std::vector<int32_t>& fields) {
  ensure_hpx_started();
  auto localities = hpx::find_all_localities();
  hpx::lcos::broadcast<::kangaroo_set_runmeta_action>(localities, runmeta.meta).get();
  hpx::lcos::broadcast<::kangaroo_set_dataset_action>(localities, dataset).get();
  hpx::lcos::broadcast<::kangaroo_preload_action>(localities, runmeta.meta, dataset, fields).get();
}

HostView Runtime::get_task_chunk(int32_t step,
                                 int16_t level,
                                 int32_t field,
                                 int32_t version,
                                 int32_t block) {
  ensure_hpx_started();
  DataServiceLocal data;
  ChunkRef ref{step, level, field, version, block};
  return data.get_host(ref).get();
}

}  // namespace kangaroo
