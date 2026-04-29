#include "kangaroo/runtime.hpp"

#include "kangaroo/param_decode.hpp"
#include "kangaroo/plan_decode.hpp"
#include "kangaroo/plotfile_reader.hpp"
#include "perfetto_trace_minimal.pb.h"
#include "amrexpr.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cstring>
#include <memory>
#include <mutex>
#include <iterator>
#include <thread>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <unordered_map>
#include <vector>
#include <msgpack.hpp>
#include <unistd.h>
#include <sys/resource.h>

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
KernelRegistry* g_kernel_registry = nullptr;
std::unordered_map<int32_t, std::shared_ptr<ExecutionContext>> g_execution_contexts;
thread_local int32_t g_current_execution_context_id = 0;
std::mutex g_event_log_mutex;
std::string g_event_log_path;
bool g_event_log_enabled = false;
std::mutex g_perfetto_trace_mutex;
std::string g_perfetto_trace_path;
bool g_perfetto_trace_enabled = false;
std::atomic<int32_t> g_active_plan_runs{0};
std::atomic<int32_t> g_locality_hint{-1};
std::atomic<int32_t> g_perfetto_metrics_active_runs{0};
std::mutex g_console_release_mutex;
std::shared_ptr<hpx::promise<void>> g_console_release_promise;
hpx::shared_future<void> g_console_release_future;
bool g_console_release_requested = false;

hpx::shared_future<void> ensure_console_release_future_locked() {
  if (!g_console_release_promise || !g_console_release_future.valid()) {
    auto promise = std::make_shared<hpx::promise<void>>();
    g_console_release_future = promise->get_future().share();
    g_console_release_promise = std::move(promise);
  }
  return g_console_release_future;
}

struct ActivePlanRunGuard {
  ActivePlanRunGuard() { g_active_plan_runs.fetch_add(1, std::memory_order_acq_rel); }
  ~ActivePlanRunGuard() { g_active_plan_runs.fetch_sub(1, std::memory_order_acq_rel); }
};

bool debug_projection_enabled() {
  static const bool enabled = std::getenv("KANGAROO_DEBUG_PROJECTION") != nullptr;
  return enabled;
}

void log_projection_kernel_summary(const char* kernel,
                                   int32_t level_index,
                                   int32_t block,
                                   std::size_t covered_boxes_count,
                                   std::size_t candidates,
                                   std::size_t covered_skips,
                                   std::size_t bounds_skips,
                                   std::size_t deposited,
                                   double out_sum) {
  if (!debug_projection_enabled()) {
    return;
  }
  std::cout << "[kangaroo][projection] kernel=" << kernel
            << " locality=" << hpx::get_locality_id()
            << " level=" << level_index
            << " block=" << block
            << " covered_boxes=" << covered_boxes_count
            << " candidates=" << candidates
            << " covered_skips=" << covered_skips
            << " bounds_skips=" << bounds_skips
            << " deposited=" << deposited
            << " output_sum=" << out_sum
            << std::endl;
}

void append_particle_values_as_f64(const plotfile::ParticleArrayData& data, const std::string& name,
                                   const std::string& context, std::vector<double>& out_vals) {
  const std::size_t n = static_cast<std::size_t>(std::max<int64_t>(0, data.count));
  const std::size_t start = out_vals.size();
  out_vals.resize(start + n, 0.0);
  if (data.dtype == "float64") {
    if (data.bytes.size() < n * sizeof(double)) {
      throw std::runtime_error(context + ": short float64 payload for " + name);
    }
    const auto* in = reinterpret_cast<const double*>(data.bytes.data());
    std::copy(in, in + n, out_vals.begin() + static_cast<std::ptrdiff_t>(start));
  } else if (data.dtype == "float32") {
    if (data.bytes.size() < n * sizeof(float)) {
      throw std::runtime_error(context + ": short float32 payload for " + name);
    }
    const auto* in = reinterpret_cast<const float*>(data.bytes.data());
    for (std::size_t i = 0; i < n; ++i) {
      out_vals[start + i] = static_cast<double>(in[i]);
    }
  } else if (data.dtype == "int64") {
    if (data.bytes.size() < n * sizeof(int64_t)) {
      throw std::runtime_error(context + ": short int64 payload for " + name);
    }
    const auto* in = reinterpret_cast<const int64_t*>(data.bytes.data());
    for (std::size_t i = 0; i < n; ++i) {
      out_vals[start + i] = static_cast<double>(in[i]);
    }
  } else {
    throw std::runtime_error(context + ": unsupported dtype '" + data.dtype + "' for " + name);
  }
}

std::unordered_map<double, int64_t> decode_particle_value_counts(std::span<const std::uint8_t> bytes) {
  std::unordered_map<double, int64_t> counts;
  if (bytes.size() < sizeof(uint64_t)) {
    return counts;
  }
  uint64_t n = 0;
  std::memcpy(&n, bytes.data(), sizeof(uint64_t));
  const std::size_t expected =
      sizeof(uint64_t) + static_cast<std::size_t>(n) * (sizeof(double) + sizeof(int64_t));
  if (bytes.size() < expected) {
    return counts;
  }
  counts.reserve(static_cast<std::size_t>(n));
  const auto* ptr = bytes.data() + sizeof(uint64_t);
  for (uint64_t i = 0; i < n; ++i) {
    double value = 0.0;
    int64_t count = 0;
    std::memcpy(&value, ptr, sizeof(double));
    ptr += sizeof(double);
    std::memcpy(&count, ptr, sizeof(int64_t));
    ptr += sizeof(int64_t);
    counts[value] += count;
  }
  return counts;
}

void encode_particle_value_counts(const std::unordered_map<double, int64_t>& counts,
                                  std::vector<std::uint8_t>& out) {
  const uint64_t n = static_cast<uint64_t>(counts.size());
  out.resize(sizeof(uint64_t) + static_cast<std::size_t>(n) * (sizeof(double) + sizeof(int64_t)));
  auto* ptr = out.data();
  std::memcpy(ptr, &n, sizeof(uint64_t));
  ptr += sizeof(uint64_t);
  for (const auto& [value, count] : counts) {
    std::memcpy(ptr, &value, sizeof(double));
    ptr += sizeof(double);
    std::memcpy(ptr, &count, sizeof(int64_t));
    ptr += sizeof(int64_t);
  }
}

std::string json_escape(const std::string& value);

std::string event_log_path_for_locality(const std::string& path) {
  if (path.empty()) {
    return path;
  }
  int32_t locality = 0;
  try {
    locality = static_cast<int32_t>(hpx::get_locality_id());
  } catch (...) {
    locality = 0;
  }
  if (locality <= 0) {
    return path;
  }

  std::filesystem::path base(path);
  const std::string stem = base.stem().string();
  const std::string ext = base.extension().string();
  std::string filename;
  if (stem.empty()) {
    filename = base.filename().string() + ".locality" + std::to_string(locality);
  } else {
    filename = stem + ".locality" + std::to_string(locality) + ext;
  }
  return (base.parent_path() / filename).string();
}

struct EventLogWorker {
  std::mutex mutex;
  std::condition_variable cv;
  std::condition_variable idle_cv;
  std::deque<std::variant<TaskEvent, PhaseEvent, DataEvent>> queue;
  std::thread thread;
  bool running = false;
  bool stop = false;
  bool writing = false;
  std::string path;
  bool enabled = false;

  void set_path(const std::string& next_path) {
    std::unique_lock<std::mutex> lock(mutex);
    idle_cv.wait(lock, [&] { return queue.empty() && !writing; });
    path = next_path;
    enabled = !path.empty();
    lock.unlock();
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

  void enqueue(PhaseEvent event) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!enabled) {
        return;
      }
      queue.push_back(std::variant<TaskEvent, PhaseEvent, DataEvent>{std::move(event)});
    }
    cv.notify_one();
  }

  void enqueue(DataEvent event) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!enabled) {
        return;
      }
      queue.push_back(std::variant<TaskEvent, PhaseEvent, DataEvent>{std::move(event)});
    }
    cv.notify_one();
  }

  void run() {
    std::ofstream out;
    std::string active_path;
    for (;;) {
      std::variant<TaskEvent, PhaseEvent, DataEvent> item;
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
        item = std::move(queue.front());
        queue.pop_front();
        writing = true;
      }
      if (!out) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          writing = false;
          if (queue.empty()) {
            idle_cv.notify_all();
          }
        }
        continue;
      }
      auto write_string = [&](const char* key, const std::string& value) {
        out << '"' << key << "\":\"" << json_escape(value) << '"';
      };
      out << std::fixed << std::setprecision(6);
      if (std::holds_alternative<TaskEvent>(item)) {
        const auto& event = std::get<TaskEvent>(item);
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
        out << ",\"ts\":" << event.ts;
        out << ",\"start\":" << event.start;
        out << ",\"end\":" << event.end;
        out << "}\n";
      } else if (std::holds_alternative<PhaseEvent>(item)) {
        const auto& event = std::get<PhaseEvent>(item);
        out << '{';
        out << "\"type\":\"phase\",";
        write_string("name", event.name);
        out << ',';
        write_string("category", event.category);
        out << ',';
        write_string("status", event.status);
        out << ",\"locality\":" << event.locality;
        if (!event.worker_label.empty()) {
          out << ',';
          write_string("worker", event.worker_label);
        } else {
          out << ",\"worker\":\"worker-" << event.worker << '"';
        }
        out << ",\"ts\":" << event.ts;
        out << ",\"start\":" << event.start;
        out << ",\"end\":" << event.end;
        out << "}\n";
      } else {
        const auto& event = std::get<DataEvent>(item);
        out << '{';
        out << "\"type\":\"dataflow\",";
        write_string("op", event.op);
        out << ',';
        write_string("mode", event.mode);
        out << ',';
        write_string("status", event.status);
        out << ",\"step\":" << event.ref.step;
        out << ",\"level\":" << event.ref.level;
        out << ",\"field\":" << event.ref.field;
        out << ",\"version\":" << event.ref.version;
        out << ",\"block\":" << event.ref.block;
        out << ",\"locality\":" << event.locality;
        out << ",\"target_locality\":" << event.target_locality;
        if (!event.worker_label.empty()) {
          out << ',';
          write_string("worker", event.worker_label);
        } else {
          out << ",\"worker\":\"worker-" << event.worker << '"';
        }
        out << ",\"bytes\":" << event.bytes;
        if (event.queue_depth >= 0) {
          out << ",\"queue_depth\":" << event.queue_depth;
        }
        if (event.in_flight >= 0) {
          out << ",\"in_flight\":" << event.in_flight;
        }
        if (event.concurrency >= 0) {
          out << ",\"concurrency\":" << event.concurrency;
        }
        out << ",\"elapsed\":" << (event.end - event.start);
        out << ",\"ts\":" << event.ts;
        out << ",\"start\":" << event.start;
        out << ",\"end\":" << event.end;
        out << "}\n";
      }
      out.flush();
      {
        std::lock_guard<std::mutex> lock(mutex);
        writing = false;
        if (queue.empty()) {
          idle_cv.notify_all();
        }
      }
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

struct MetricEvent {
  std::string name;
  double value = 0.0;
  double ts = 0.0;
  int32_t locality = -1;
};

using PerfettoEvent = std::variant<TaskEvent, PhaseEvent, MetricEvent>;

uint64_t hash_u64(std::string_view s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= static_cast<uint64_t>(c);
    h *= 1099511628211ull;
  }
  return h;
}

uint64_t mix_u64(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

uint64_t to_ns(double seconds) {
  if (!std::isfinite(seconds)) {
    return 0;
  }
  if (seconds <= 0.0) {
    return 0;
  }
  const long double ns = static_cast<long double>(seconds) * 1.0e9L;
  if (ns >= static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
    return std::numeric_limits<uint64_t>::max();
  }
  return static_cast<uint64_t>(ns);
}

void append_varint(std::string& out, uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<char>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<char>(value & 0x7f));
}

bool write_trace_packet_record(std::ofstream& out, const perfetto::protos::TracePacket& packet) {
  std::string payload;
  if (!packet.SerializeToString(&payload)) {
    return false;
  }
  std::string framed;
  framed.reserve(payload.size() + 16);
  append_varint(framed, (1u << 3) | 2u);  // Trace.packet (field #1, length-delimited)
  append_varint(framed, static_cast<uint64_t>(payload.size()));
  framed += payload;
  out.write(framed.data(), static_cast<std::streamsize>(framed.size()));
  return static_cast<bool>(out);
}

std::string task_worker_name(const TaskEvent& event) {
  if (!event.worker_label.empty()) {
    return event.worker_label;
  }
  if (event.worker >= 0) {
    return "worker-" + std::to_string(event.worker);
  }
  return "worker-0";
}

std::string phase_worker_name(const PhaseEvent& event) {
  if (!event.worker_label.empty()) {
    return event.worker_label;
  }
  if (event.worker >= 0) {
    return "worker-" + std::to_string(event.worker);
  }
  return "runtime";
}

std::filesystem::path perfetto_locality_path(const std::string& base_path, int32_t locality) {
  std::filesystem::path p(base_path);
  const std::string suffix = ".loc" + [&] {
    std::ostringstream os;
    os << std::setw(3) << std::setfill('0') << std::max(0, locality);
    return os.str();
  }();
  if (!p.has_extension()) {
    p += suffix + ".pftrace";
    return p;
  }
  std::filesystem::path out = p.parent_path();
  out /= p.stem().string() + suffix + p.extension().string();
  return out;
}

struct PerfettoTraceWorker {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<PerfettoEvent> queue;
  std::thread thread;
  bool running = false;
  bool stop = false;
  std::string base_path;
  bool enabled = false;
  uint64_t config_generation = 0;

  void set_path(const std::string& next_path) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      base_path = next_path;
      enabled = !base_path.empty();
      config_generation++;
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

  void enqueue_task(TaskEvent event) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!enabled) {
        return;
      }
      queue.push_back(PerfettoEvent{std::move(event)});
    }
    cv.notify_one();
  }

  void enqueue_metric(MetricEvent event) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!enabled) {
        return;
      }
      queue.push_back(PerfettoEvent{std::move(event)});
    }
    cv.notify_one();
  }

  void enqueue_phase(PhaseEvent event) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!enabled) {
        return;
      }
      queue.push_back(PerfettoEvent{std::move(event)});
    }
    cv.notify_one();
  }

  static uint32_t sequence_id_for_locality(int32_t locality) {
    return static_cast<uint32_t>(std::max(0, locality) + 1);
  }

  static uint64_t process_track_uuid_for_locality(int32_t locality) {
    return 0x1000000000000000ULL | static_cast<uint64_t>(static_cast<uint32_t>(std::max(0, locality)));
  }

  static uint64_t thread_track_uuid_for_event(const TaskEvent& event) {
    uint64_t seed = 0x2000000000000000ULL;
    seed ^= static_cast<uint64_t>(static_cast<uint32_t>(std::max(0, event.locality))) << 32;
    uint64_t worker_bits = 0;
    if (event.worker >= 0) {
      worker_bits = static_cast<uint64_t>(static_cast<uint32_t>(event.worker));
    } else {
      worker_bits = hash_u64(task_worker_name(event)) & 0xffffffffULL;
    }
    return mix_u64(seed ^ worker_bits);
  }

  static uint64_t thread_track_uuid_for_phase(const PhaseEvent& event) {
    uint64_t seed = 0x2000000000000000ULL;
    seed ^= static_cast<uint64_t>(static_cast<uint32_t>(std::max(0, event.locality))) << 32;
    uint64_t worker_bits = 0;
    if (event.worker >= 0) {
      worker_bits = static_cast<uint64_t>(static_cast<uint32_t>(event.worker));
    } else {
      worker_bits = hash_u64(phase_worker_name(event)) & 0xffffffffULL;
    }
    return mix_u64(seed ^ worker_bits);
  }

  static uint64_t counter_track_uuid_for_metric(int32_t locality, const std::string& name) {
    uint64_t seed = 0x3000000000000000ULL;
    seed ^= static_cast<uint64_t>(static_cast<uint32_t>(std::max(0, locality))) << 32;
    seed ^= hash_u64(name);
    return mix_u64(seed);
  }

  void add_string_arg(perfetto::protos::TrackEvent* te, const std::string& key, const std::string& value) {
    auto* ann = te->add_debug_annotations();
    ann->set_name(key);
    ann->set_string_value(value);
  }

  void add_int_arg(perfetto::protos::TrackEvent* te, const std::string& key, int64_t value) {
    auto* ann = te->add_debug_annotations();
    ann->set_name(key);
    ann->set_int_value(value);
  }

  void emit_packet(std::ofstream& out,
                   const perfetto::protos::TracePacket& packet,
                   bool flush = false) {
    if (!write_trace_packet_record(out, packet)) {
      return;
    }
    if (flush) {
      out.flush();
    }
  }

  void emit_process_descriptor(std::ofstream& out,
                               uint32_t seq_id,
                               uint64_t process_track_uuid,
                               int32_t locality,
                               bool first_packet) {
    perfetto::protos::TracePacket packet;
    packet.set_trusted_packet_sequence_id(seq_id);
    if (first_packet) {
      packet.set_sequence_flags(perfetto::protos::TracePacket::SEQ_INCREMENTAL_STATE_CLEARED);
      packet.set_first_packet_on_sequence(true);
    }
    auto* td = packet.mutable_track_descriptor();
    td->set_uuid(process_track_uuid);
    td->set_name("kangaroo locality " + std::to_string(std::max(0, locality)));
    auto* pd = td->mutable_process();
    pd->set_pid(std::max(1, static_cast<int>(::getpid())));
    pd->set_process_name("kangaroo-locality-" + std::to_string(std::max(0, locality)));
    emit_packet(out, packet, true);
  }

  void emit_thread_descriptor(std::ofstream& out,
                              uint32_t seq_id,
                              uint64_t process_track_uuid,
                              uint64_t thread_track_uuid,
                              const TaskEvent& event) {
    perfetto::protos::TracePacket packet;
    packet.set_trusted_packet_sequence_id(seq_id);
    auto* td = packet.mutable_track_descriptor();
    td->set_uuid(thread_track_uuid);
    td->set_parent_uuid(process_track_uuid);
    td->set_name(task_worker_name(event));
    auto* th = td->mutable_thread();
    const int32_t locality = std::max(0, event.locality);
    const int32_t worker = std::max(0, event.worker);
    th->set_pid(std::max(1, static_cast<int>(::getpid())));
    th->set_tid(locality * 10000 + worker + 1);
    th->set_thread_name(task_worker_name(event));
    emit_packet(out, packet, false);
  }

  void emit_thread_descriptor(std::ofstream& out,
                              uint32_t seq_id,
                              uint64_t process_track_uuid,
                              uint64_t thread_track_uuid,
                              const PhaseEvent& event) {
    perfetto::protos::TracePacket packet;
    packet.set_trusted_packet_sequence_id(seq_id);
    auto* td = packet.mutable_track_descriptor();
    td->set_uuid(thread_track_uuid);
    td->set_parent_uuid(process_track_uuid);
    td->set_name(phase_worker_name(event));
    auto* th = td->mutable_thread();
    const int32_t locality = std::max(0, event.locality);
    th->set_pid(std::max(1, static_cast<int>(::getpid())));
    th->set_tid(locality * 10000 +
                static_cast<int32_t>(hash_u64(phase_worker_name(event)) % 9000ULL) + 1);
    th->set_thread_name(phase_worker_name(event));
    emit_packet(out, packet, false);
  }

  void emit_counter_descriptor(std::ofstream& out,
                               uint32_t seq_id,
                               uint64_t process_track_uuid,
                               uint64_t counter_track_uuid,
                               const std::string& counter_name) {
    perfetto::protos::TracePacket packet;
    packet.set_trusted_packet_sequence_id(seq_id);
    auto* td = packet.mutable_track_descriptor();
    td->set_uuid(counter_track_uuid);
    td->set_parent_uuid(process_track_uuid);
    td->set_name(counter_name);
    td->mutable_counter();
    emit_packet(out, packet, false);
  }

  void emit_task_event(std::ofstream& out,
                       uint32_t seq_id,
                       uint64_t track_uuid,
                       const TaskEvent& event) {
    perfetto::protos::TracePacket packet;
    packet.set_trusted_packet_sequence_id(seq_id);
    const bool is_start = event.status == "start";
    uint64_t ts_ns = is_start ? to_ns(event.start > 0.0 ? event.start : event.ts)
                              : to_ns(event.end > 0.0 ? event.end : event.ts);
    packet.set_timestamp(ts_ns);
    auto* te = packet.mutable_track_event();
    te->set_track_uuid(track_uuid);
    if (is_start) {
      te->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
      te->add_categories("kangaroo");
      te->set_name(event.name);
      add_string_arg(te, "id", event.id);
      add_string_arg(te, "kernel", event.kernel);
      add_string_arg(te, "plane", event.plane);
      add_string_arg(te, "status", event.status);
      add_string_arg(te, "worker", task_worker_name(event));
      add_int_arg(te, "stage", event.stage);
      add_int_arg(te, "template", event.template_index);
      add_int_arg(te, "block", event.block);
      add_int_arg(te, "step", event.step);
      add_int_arg(te, "level", event.level);
      add_int_arg(te, "locality", event.locality);
      add_int_arg(te, "worker_index", event.worker);
    } else {
      te->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
      if (event.status == "error") {
        add_string_arg(te, "status", "error");
      }
    }
    emit_packet(out, packet, true);
  }

  void emit_metric_event(std::ofstream& out,
                         uint32_t seq_id,
                         uint64_t counter_track_uuid,
                         const MetricEvent& event) {
    perfetto::protos::TracePacket packet;
    packet.set_trusted_packet_sequence_id(seq_id);
    packet.set_timestamp(to_ns(event.ts));
    auto* te = packet.mutable_track_event();
    te->set_type(perfetto::protos::TrackEvent::TYPE_COUNTER);
    te->set_track_uuid(counter_track_uuid);
    te->set_double_counter_value(event.value);
    emit_packet(out, packet, true);
  }

  void emit_phase_event(std::ofstream& out,
                        uint32_t seq_id,
                        uint64_t track_uuid,
                        const PhaseEvent& event) {
    perfetto::protos::TracePacket packet;
    packet.set_trusted_packet_sequence_id(seq_id);
    const bool is_start = event.status == "start";
    uint64_t ts_ns = is_start ? to_ns(event.start > 0.0 ? event.start : event.ts)
                              : to_ns(event.end > 0.0 ? event.end : event.ts);
    packet.set_timestamp(ts_ns);
    auto* te = packet.mutable_track_event();
    te->set_track_uuid(track_uuid);
    if (is_start) {
      te->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_BEGIN);
      te->add_categories(event.category.empty() ? "kangaroo.phase" : event.category);
      te->set_name(event.name);
      add_string_arg(te, "category", event.category);
      add_string_arg(te, "status", event.status);
      add_string_arg(te, "worker", phase_worker_name(event));
      add_int_arg(te, "locality", event.locality);
      add_int_arg(te, "worker_index", event.worker);
    } else {
      te->set_type(perfetto::protos::TrackEvent::TYPE_SLICE_END);
      if (event.status == "error") {
        add_string_arg(te, "status", "error");
      }
    }
    emit_packet(out, packet, true);
  }

  void run() {
    std::ofstream out;
    uint64_t active_generation = std::numeric_limits<uint64_t>::max();
    std::string active_base_path;
    bool descriptors_initialized = false;
    int32_t active_locality = -1;
    uint32_t seq_id = 1;
    uint64_t process_track_uuid = 0;
    std::unordered_map<uint64_t, bool> emitted_thread_tracks;
    std::unordered_map<uint64_t, bool> emitted_counter_tracks;

    auto reset_output = [&]() {
      out.close();
      descriptors_initialized = false;
      active_locality = -1;
      seq_id = 1;
      process_track_uuid = 0;
      emitted_thread_tracks.clear();
      emitted_counter_tracks.clear();
    };

    for (;;) {
      PerfettoEvent item;
      uint64_t generation = 0;
      std::string configured_base;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] {
          return stop || !queue.empty() || config_generation != active_generation;
        });
        if (stop && queue.empty()) {
          break;
        }
        generation = config_generation;
        configured_base = base_path;
        if (generation != active_generation) {
          active_generation = generation;
          active_base_path = configured_base;
          reset_output();
        }
        if (queue.empty()) {
          continue;
        }
        item = std::move(queue.front());
        queue.pop_front();
      }

      if (active_base_path.empty()) {
        continue;
      }

      const int32_t item_locality = std::visit(
          [](const auto& ev) -> int32_t { return std::max(0, ev.locality); }, item);

      if (active_locality < 0) {
        active_locality = item_locality;
        seq_id = sequence_id_for_locality(active_locality);
        process_track_uuid = process_track_uuid_for_locality(active_locality);
        std::filesystem::path loc_path = perfetto_locality_path(active_base_path, active_locality);
        std::error_code ec;
        if (loc_path.has_parent_path()) {
          std::filesystem::create_directories(loc_path.parent_path(), ec);
        }
        out.open(loc_path, std::ios::binary | std::ios::trunc);
        if (!out) {
          continue;
        }
      }

      if (!descriptors_initialized && out) {
        emit_process_descriptor(out, seq_id, process_track_uuid, active_locality, true);
        descriptors_initialized = true;
      }

      if (!out) {
        continue;
      }

      if (std::holds_alternative<TaskEvent>(item)) {
        const TaskEvent& event = std::get<TaskEvent>(item);
        const uint64_t thread_track_uuid = thread_track_uuid_for_event(event);
        if (emitted_thread_tracks.find(thread_track_uuid) == emitted_thread_tracks.end()) {
          emit_thread_descriptor(out, seq_id, process_track_uuid, thread_track_uuid, event);
          emitted_thread_tracks.emplace(thread_track_uuid, true);
        }
        emit_task_event(out, seq_id, thread_track_uuid, event);
      } else if (std::holds_alternative<PhaseEvent>(item)) {
        const PhaseEvent& event = std::get<PhaseEvent>(item);
        const uint64_t thread_track_uuid = thread_track_uuid_for_phase(event);
        if (emitted_thread_tracks.find(thread_track_uuid) == emitted_thread_tracks.end()) {
          emit_thread_descriptor(out, seq_id, process_track_uuid, thread_track_uuid, event);
          emitted_thread_tracks.emplace(thread_track_uuid, true);
        }
        emit_phase_event(out, seq_id, thread_track_uuid, event);
      } else {
        const MetricEvent& event = std::get<MetricEvent>(item);
        const uint64_t counter_track_uuid = counter_track_uuid_for_metric(active_locality, event.name);
        if (emitted_counter_tracks.find(counter_track_uuid) == emitted_counter_tracks.end()) {
          emit_counter_descriptor(out, seq_id, process_track_uuid, counter_track_uuid, event.name);
          emitted_counter_tracks.emplace(counter_track_uuid, true);
        }
        emit_metric_event(out, seq_id, counter_track_uuid, event);
      }
    }
    out.flush();
  }

  ~PerfettoTraceWorker() {
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

PerfettoTraceWorker g_perfetto_trace_worker;

void log_metric_event(const MetricEvent& event) {
  if (!g_perfetto_trace_enabled || g_perfetto_trace_path.empty()) {
    return;
  }
  g_perfetto_trace_worker.enqueue_metric(event);
}

struct ProcessIoCounters {
  uint64_t read_bytes = 0;
  uint64_t write_bytes = 0;
  bool valid = false;
};

ProcessIoCounters read_process_io_counters() {
  ProcessIoCounters out;
#if defined(__linux__)
  std::ifstream in("/proc/self/io");
  if (!in) {
    return out;
  }
  std::string key;
  uint64_t value = 0;
  while (in >> key >> value) {
    if (key == "read_bytes:") {
      out.read_bytes = value;
    } else if (key == "write_bytes:") {
      out.write_bytes = value;
    }
  }
  out.valid = true;
#endif
  return out;
}

std::optional<uint64_t> current_rss_bytes() {
#if defined(__linux__)
  std::ifstream in("/proc/self/statm");
  if (in) {
    uint64_t size_pages = 0;
    uint64_t resident_pages = 0;
    if (in >> size_pages >> resident_pages) {
      const long page_size = ::sysconf(_SC_PAGESIZE);
      if (page_size > 0) {
        return resident_pages * static_cast<uint64_t>(page_size);
      }
    }
  }
#endif
  return std::nullopt;
}

double process_cpu_seconds() {
  struct rusage usage {};
  if (::getrusage(RUSAGE_SELF, &usage) != 0) {
    return 0.0;
  }
  const double user_s =
      static_cast<double>(usage.ru_utime.tv_sec) + static_cast<double>(usage.ru_utime.tv_usec) / 1.0e6;
  const double sys_s =
      static_cast<double>(usage.ru_stime.tv_sec) + static_cast<double>(usage.ru_stime.tv_usec) / 1.0e6;
  return user_s + sys_s;
}

struct PerfettoMetricsSampler {
  std::mutex mutex;
  std::condition_variable cv;
  std::thread thread;
  bool running = false;
  bool stop = false;
  bool enabled = false;

  void set_enabled(bool next_enabled) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      enabled = next_enabled;
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

  void emit_sample(double now_epoch_s,
                   double dt_s,
                   double cpu_now_s,
                   double cpu_prev_s,
                   const ProcessIoCounters& io_now,
                   const ProcessIoCounters& io_prev) {
    const int32_t locality = std::max(0, g_locality_hint.load(std::memory_order_acquire));

    MetricEvent cpu_ev;
    cpu_ev.name = "cpu_percent";
    cpu_ev.ts = now_epoch_s;
    cpu_ev.locality = locality;
    if (dt_s > 1.0e-6) {
      cpu_ev.value = std::max(0.0, std::min(1000.0, 100.0 * (cpu_now_s - cpu_prev_s) / dt_s));
    } else {
      cpu_ev.value = 0.0;
    }
    log_metric_event(cpu_ev);

    if (auto rss_bytes = current_rss_bytes()) {
      MetricEvent rss_ev;
      rss_ev.name = "rss_current_bytes";
      rss_ev.ts = now_epoch_s;
      rss_ev.locality = locality;
      rss_ev.value = static_cast<double>(*rss_bytes);
      log_metric_event(rss_ev);
    }

#if defined(__linux__)
    if (io_now.valid && io_prev.valid && dt_s > 1.0e-6) {
      MetricEvent read_ev;
      read_ev.name = "io_read_bytes_per_s";
      read_ev.ts = now_epoch_s;
      read_ev.locality = locality;
      read_ev.value = static_cast<double>(io_now.read_bytes - io_prev.read_bytes) / dt_s;
      log_metric_event(read_ev);

      MetricEvent write_ev;
      write_ev.name = "io_write_bytes_per_s";
      write_ev.ts = now_epoch_s;
      write_ev.locality = locality;
      write_ev.value = static_cast<double>(io_now.write_bytes - io_prev.write_bytes) / dt_s;
      log_metric_event(write_ev);
    }
#else
    (void)io_now;
    (void)io_prev;
#endif
  }

  void run() {
    bool sampling_active = false;
    auto prev_wall = std::chrono::steady_clock::now();
    double prev_cpu_s = 0.0;
    ProcessIoCounters prev_io{};

    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::milliseconds(500), [&] {
          return stop || enabled;
        });
        if (stop) {
          break;
        }
      }

      const bool trace_enabled = g_perfetto_trace_enabled && !g_perfetto_trace_path.empty();
      const bool plan_running = g_perfetto_metrics_active_runs.load(std::memory_order_acquire) > 0;
      if (!(trace_enabled && plan_running)) {
        sampling_active = false;
        continue;
      }

      const auto now_wall = std::chrono::steady_clock::now();
      const double now_epoch_s = std::chrono::duration<double>(
          std::chrono::system_clock::now().time_since_epoch()).count();
      const double cpu_now_s = process_cpu_seconds();
      const ProcessIoCounters io_now = read_process_io_counters();

      if (!sampling_active) {
        sampling_active = true;
        prev_wall = now_wall;
        prev_cpu_s = cpu_now_s;
        prev_io = io_now;
        continue;
      }

      const double dt_s = std::chrono::duration<double>(now_wall - prev_wall).count();
      emit_sample(now_epoch_s, dt_s, cpu_now_s, prev_cpu_s, io_now, prev_io);
      prev_wall = now_wall;
      prev_cpu_s = cpu_now_s;
      prev_io = io_now;
    }
  }

  ~PerfettoMetricsSampler() {
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

PerfettoMetricsSampler g_perfetto_metrics_sampler;

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

const std::string& perfetto_trace_path() {
  return g_perfetto_trace_path;
}

void init_event_log_from_env() {
  const char* env = std::getenv("KANGAROO_EVENT_LOG");
  if (env && *env != '\0') {
    {
      std::lock_guard<std::mutex> lock(g_event_log_mutex);
      g_event_log_path = env;
      g_event_log_enabled = true;
    }
    g_event_log_worker.set_path(event_log_path_for_locality(g_event_log_path));
  }
}

void init_perfetto_trace_from_env() {
  const char* env = std::getenv("KANGAROO_PERFETTO_TRACE");
  if (env && *env != '\0') {
    {
      std::lock_guard<std::mutex> lock(g_perfetto_trace_mutex);
      g_perfetto_trace_path = env;
      g_perfetto_trace_enabled = true;
    }
    g_perfetto_trace_worker.set_path(g_perfetto_trace_path);
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

template <typename Params, typename DecodeFn>
KernelRegistry::KernelParamsPrepareFn make_kernel_params_preparer(DecodeFn decode_fn) {
  return [decode_fn = std::move(decode_fn)](
             const KernelParamContext& context) -> KernelRegistry::PreparedParams {
    if (context.params_msgpack.empty()) {
      return {};
    }
    const auto& decoded = decode_params_cached<Params>(context.params_msgpack, decode_fn);
    auto prepared = std::shared_ptr<const void>(
        new Params(decoded),
        [](const void* ptr) { delete static_cast<const Params*>(ptr); });
    return KernelRegistry::PreparedParams{std::type_index(typeid(Params)), std::move(prepared)};
  };
}

template <typename Params, typename DecodeFn>
KernelRegistry::KernelParamsPrepareFn make_covered_box_params_preparer(DecodeFn decode_fn) {
  return [decode_fn = std::move(decode_fn)](
             const KernelParamContext& context) -> KernelRegistry::PreparedParams {
    if (context.params_msgpack.empty() && !context.covered_boxes) {
      return {};
    }
    Params decoded;
    if (!context.params_msgpack.empty()) {
      decoded = decode_params_cached<Params>(context.params_msgpack, decode_fn);
    }
    if (context.covered_boxes) {
      decoded.covered_boxes = context.covered_boxes;
    }
    auto prepared = std::shared_ptr<const void>(
        new Params(std::move(decoded)),
        [](const void* ptr) { delete static_cast<const Params*>(ptr); });
    return KernelRegistry::PreparedParams{std::type_index(typeid(Params)), std::move(prepared)};
  };
}

std::shared_ptr<const CoveredBoxListIR> parse_covered_boxes_param(const msgpack::object& root) {
  const auto* boxes = find_msgpack_map_value(root, "covered_boxes");
  if (boxes == nullptr || boxes->type != msgpack::type::ARRAY) {
    return {};
  }

  auto parsed = std::make_shared<CoveredBoxListIR>();
  parsed->reserve(boxes->via.array.size);
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
    CoveredBoxIR box;
    for (uint32_t d = 0; d < 3; ++d) {
      box.lo[d] = lo.via.array.ptr[d].as<int32_t>();
      box.hi[d] = hi.via.array.ptr[d].as<int32_t>();
    }
    parsed->push_back(box);
  }
  return parsed;
}

std::size_t covered_box_count(const std::shared_ptr<const CoveredBoxListIR>& boxes) {
  return boxes ? boxes->size() : 0;
}

bool covered_box_contains(const CoveredBoxIR& box, int i, int j, int k) {
  return i >= box.lo[0] && i <= box.hi[0] &&
         j >= box.lo[1] && j <= box.hi[1] &&
         k >= box.lo[2] && k <= box.hi[2];
}

double min_dist_sq_to_interval(double a0, double a1) {
  if (a1 < 0.0) {
    return a1 * a1;
  }
  if (a0 > 0.0) {
    return a0 * a0;
  }
  return 0.0;
}

double max_dist_sq_to_interval(double a0, double a1) {
  const double aa0 = std::abs(a0);
  const double aa1 = std::abs(a1);
  const double amax = (aa0 > aa1) ? aa0 : aa1;
  return amax * amax;
}

bool sphere_may_intersect_cell(double radius2,
                               double x0,
                               double x1,
                               double y0,
                               double y1,
                               double z0,
                               double z1) {
  const double r2_min = min_dist_sq_to_interval(x0, x1) +
                        min_dist_sq_to_interval(y0, y1) +
                        min_dist_sq_to_interval(z0, z1);
  const double r2_max = max_dist_sq_to_interval(x0, x1) +
                        max_dist_sq_to_interval(y0, y1) +
                        max_dist_sq_to_interval(z0, z1);
  return radius2 >= r2_min && radius2 <= r2_max;
}

struct FluxPoint {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

void add_point_unique(std::array<FluxPoint, 16>& pts,
                      int& npts,
                      double x,
                      double y,
                      double z,
                      double tol) {
  const double tol2 = tol * tol;
  for (int i = 0; i < npts; ++i) {
    const double dx = pts[static_cast<std::size_t>(i)].x - x;
    const double dy = pts[static_cast<std::size_t>(i)].y - y;
    const double dz = pts[static_cast<std::size_t>(i)].z - z;
    if ((dx * dx + dy * dy + dz * dz) <= tol2) {
      return;
    }
  }
  if (npts < static_cast<int>(pts.size())) {
    pts[static_cast<std::size_t>(npts)] = FluxPoint{x, y, z};
    ++npts;
  }
}

double plane_box_section_area(double x0,
                              double x1,
                              double y0,
                              double y1,
                              double z0,
                              double z1,
                              double nx,
                              double ny,
                              double nz,
                              double d) {
  const double scale = std::abs(x0) + std::abs(x1) + std::abs(y0) + std::abs(y1) +
                       std::abs(z0) + std::abs(z1) + std::abs(d) + 1.0;
  const double tol = 1.0e-12 * scale;

  const std::array<FluxPoint, 8> verts{
      FluxPoint{x0, y0, z0}, FluxPoint{x1, y0, z0}, FluxPoint{x0, y1, z0},
      FluxPoint{x1, y1, z0}, FluxPoint{x0, y0, z1}, FluxPoint{x1, y0, z1},
      FluxPoint{x0, y1, z1}, FluxPoint{x1, y1, z1}};
  const std::array<std::array<int, 2>, 12> edges{
      std::array<int, 2>{0, 1}, std::array<int, 2>{2, 3},
      std::array<int, 2>{4, 5}, std::array<int, 2>{6, 7},
      std::array<int, 2>{0, 2}, std::array<int, 2>{1, 3},
      std::array<int, 2>{4, 6}, std::array<int, 2>{5, 7},
      std::array<int, 2>{0, 4}, std::array<int, 2>{1, 5},
      std::array<int, 2>{2, 6}, std::array<int, 2>{3, 7}};

  std::array<FluxPoint, 16> pts{};
  int npts = 0;

  for (const auto& edge : edges) {
    const int i0 = edge[0];
    const int i1 = edge[1];
    const auto& p0 = verts[static_cast<std::size_t>(i0)];
    const auto& p1 = verts[static_cast<std::size_t>(i1)];

    double f0 = nx * p0.x + ny * p0.y + nz * p0.z - d;
    double f1 = nx * p1.x + ny * p1.y + nz * p1.z - d;
    if (std::abs(f0) <= tol) {
      f0 = 0.0;
    }
    if (std::abs(f1) <= tol) {
      f1 = 0.0;
    }

    if (f0 == 0.0 && f1 == 0.0) {
      add_point_unique(pts, npts, p0.x, p0.y, p0.z, tol);
      add_point_unique(pts, npts, p1.x, p1.y, p1.z, tol);
      continue;
    }
    if (f0 == 0.0) {
      add_point_unique(pts, npts, p0.x, p0.y, p0.z, tol);
      continue;
    }
    if (f1 == 0.0) {
      add_point_unique(pts, npts, p1.x, p1.y, p1.z, tol);
      continue;
    }
    if ((f0 < 0.0 && f1 > 0.0) || (f0 > 0.0 && f1 < 0.0)) {
      const double t = f0 / (f0 - f1);
      const double x = p0.x + t * (p1.x - p0.x);
      const double y = p0.y + t * (p1.y - p0.y);
      const double z = p0.z + t * (p1.z - p0.z);
      add_point_unique(pts, npts, x, y, z, tol);
    }
  }

  if (npts < 3) {
    return 0.0;
  }

  double cx = 0.0;
  double cy = 0.0;
  double cz = 0.0;
  for (int i = 0; i < npts; ++i) {
    cx += pts[static_cast<std::size_t>(i)].x;
    cy += pts[static_cast<std::size_t>(i)].y;
    cz += pts[static_cast<std::size_t>(i)].z;
  }
  cx /= static_cast<double>(npts);
  cy /= static_cast<double>(npts);
  cz /= static_cast<double>(npts);

  double ax = 1.0;
  double ay = 0.0;
  double az = 0.0;
  if (std::abs(nx) > 0.9) {
    ax = 0.0;
    ay = 1.0;
    az = 0.0;
  }
  double e1x = ny * az - nz * ay;
  double e1y = nz * ax - nx * az;
  double e1z = nx * ay - ny * ax;
  const double e1norm = std::sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
  if (e1norm <= 0.0) {
    return 0.0;
  }
  e1x /= e1norm;
  e1y /= e1norm;
  e1z /= e1norm;

  const double e2x = ny * e1z - nz * e1y;
  const double e2y = nz * e1x - nx * e1z;
  const double e2z = nx * e1y - ny * e1x;

  std::array<double, 16> u{};
  std::array<double, 16> v{};
  std::array<double, 16> ang{};
  for (int i = 0; i < npts; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double rx = pts[idx].x - cx;
    const double ry = pts[idx].y - cy;
    const double rz = pts[idx].z - cz;
    u[idx] = rx * e1x + ry * e1y + rz * e1z;
    v[idx] = rx * e2x + ry * e2y + rz * e2z;
    ang[idx] = std::atan2(v[idx], u[idx]);
  }

  for (int i = 1; i < npts; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double key_ang = ang[idx];
    const double key_u = u[idx];
    const double key_v = v[idx];
    int j = i - 1;
    while (j >= 0 && ang[static_cast<std::size_t>(j)] > key_ang) {
      const std::size_t dst = static_cast<std::size_t>(j + 1);
      const std::size_t src = static_cast<std::size_t>(j);
      ang[dst] = ang[src];
      u[dst] = u[src];
      v[dst] = v[src];
      --j;
    }
    const std::size_t dst = static_cast<std::size_t>(j + 1);
    ang[dst] = key_ang;
    u[dst] = key_u;
    v[dst] = key_v;
  }

  double area2 = 0.0;
  for (int i = 0; i < npts; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const std::size_t next = static_cast<std::size_t>((i + 1 < npts) ? (i + 1) : 0);
    area2 += u[idx] * v[next] - v[idx] * u[next];
  }
  return 0.5 * std::abs(area2);
}

double spherical_section_area_in_intersecting_cell(double radius,
                                                   double x0,
                                                   double x1,
                                                   double y0,
                                                   double y1,
                                                   double z0,
                                                   double z1) {
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const double dz = z1 - z0;
  const double vol = dx * dy * dz;
  if (vol <= 0.0) {
    return 0.0;
  }

  const double xc = 0.5 * (x0 + x1);
  const double yc = 0.5 * (y0 + y1);
  const double zc = 0.5 * (z0 + z1);
  const double rc = std::sqrt(xc * xc + yc * yc + zc * zc);
  if (rc <= 0.0) {
    return 0.0;
  }

  return plane_box_section_area(x0, x1, y0, y1, z0, z1, xc / rc, yc / rc, zc / rc, radius);
}

double spherical_section_area_in_cell(double radius,
                                      double x0,
                                      double x1,
                                      double y0,
                                      double y1,
                                      double z0,
                                      double z1) {
  const double radius2 = radius * radius;
  if (!sphere_may_intersect_cell(radius2, x0, x1, y0, y1, z0, z1)) {
    return 0.0;
  }
  return spherical_section_area_in_intersecting_cell(radius, x0, x1, y0, y1, z0, z1);
}

void register_default_kernels(KernelRegistry& registry) {
  static const bool log_locality = []() {
    const char* env = std::getenv("KANGAROO_LOG_LOCALITY");
    return env != nullptr && *env != '\0' && *env != '0';
  }();
  {
    struct Params {
      int32_t input_field = -1;
      int32_t input_version = 0;
      int32_t input_step = 0;
      int16_t input_level = 0;
      int32_t bytes_per_value = 8;
      int32_t halo_cells = 1;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (root.type == msgpack::type::MAP) {
        if (const auto* fld = find_msgpack_map_value(root, "input_field")) {
          params.input_field = fld->as<int32_t>();
        }
        if (const auto* ver = find_msgpack_map_value(root, "input_version")) {
          params.input_version = ver->as<int32_t>();
        }
        if (const auto* stp = find_msgpack_map_value(root, "input_step")) {
          params.input_step = stp->as<int32_t>();
        }
        if (const auto* lev = find_msgpack_map_value(root, "input_level")) {
          params.input_level = lev->as<int16_t>();
        }
        if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value")) {
          params.bytes_per_value = bpv->as<int32_t>();
        }
        if (const auto* halo = find_msgpack_map_value(root, "halo_cells")) {
          params.halo_cells = halo->as<int32_t>();
        }
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "amr_subbox_fetch_pack", .n_inputs = 0, .n_outputs = 1, .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView>,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        if (outputs.empty() || block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

        outputs[0].data.clear();
        if (params.input_field < 0) {
          return hpx::make_ready_future();
        }
        if (params.bytes_per_value != 4 && params.bytes_per_value != 8) {
          return hpx::make_ready_future();
        }

        const RunMeta& meta = current_runmeta();
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
        auto pack_patches = [](const std::vector<PackedPatch>& packed_patches) {
          HostView packed;
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
          packed.data.assign(sbuf.data(), sbuf.data() + sbuf.size());
          return packed;
        };

        struct PendingPatch {
          int16_t level = 0;
          LevelGeom geom;
        };
        std::vector<PendingPatch> pending_patches;
        std::vector<hpx::future<SubboxView>> pending_subboxes;

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
            pending_patches.push_back(PendingPatch{.level = lev, .geom = lev_meta.geom});
            pending_subboxes.push_back(data_service.get_subbox(ref));
          }
        }

        return hpx::when_all(std::move(pending_subboxes))
            .then([pending_patches = std::move(pending_patches),
                   pack_patches = std::move(pack_patches),
                   outputs,
                   bytes_per_value = params.bytes_per_value](auto&& all) mutable {
              auto ready_subboxes = all.get();
              std::vector<PackedPatch> packed_patches;
              packed_patches.reserve(ready_subboxes.size());
              for (std::size_t i = 0; i < ready_subboxes.size(); ++i) {
                auto sub = ready_subboxes[i].get();
                if (sub.box.hi[0] < sub.box.lo[0] || sub.box.hi[1] < sub.box.lo[1] ||
                    sub.box.hi[2] < sub.box.lo[2] || sub.data.data.empty()) {
                  continue;
                }

                PackedPatch pp;
                pp.level = pending_patches[i].level;
                pp.box = sub.box;
                pp.geom = pending_patches[i].geom;
                pp.bytes_per_value = bytes_per_value;
                pp.data = std::move(sub.data);
                packed_patches.push_back(std::move(pp));
              }

              outputs[0] = pack_patches(packed_patches);
              return;
            });
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      int32_t input_field = -1;
      int32_t input_version = 0;
      int32_t input_step = 0;
      int16_t input_level = 0;
      int32_t bytes_per_value = 0;
      int32_t stencil_radius = 1;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (root.type == msgpack::type::MAP) {
        if (const auto* fld = find_msgpack_map_value(root, "input_field");
            fld && (fld->type == msgpack::type::POSITIVE_INTEGER ||
                    fld->type == msgpack::type::NEGATIVE_INTEGER)) {
          params.input_field = fld->as<int32_t>();
        }
        if (const auto* ver = find_msgpack_map_value(root, "input_version");
            ver && (ver->type == msgpack::type::POSITIVE_INTEGER ||
                    ver->type == msgpack::type::NEGATIVE_INTEGER)) {
          params.input_version = ver->as<int32_t>();
        }
        if (const auto* stp = find_msgpack_map_value(root, "input_step");
            stp && (stp->type == msgpack::type::POSITIVE_INTEGER ||
                    stp->type == msgpack::type::NEGATIVE_INTEGER)) {
          params.input_step = stp->as<int32_t>();
        }
        if (const auto* lev = find_msgpack_map_value(root, "input_level");
            lev && (lev->type == msgpack::type::POSITIVE_INTEGER ||
                    lev->type == msgpack::type::NEGATIVE_INTEGER)) {
          params.input_level = lev->as<int16_t>();
        }
        if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
            bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                    bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
          params.bytes_per_value = bpv->as<int32_t>();
        }
        if (const auto* sr = find_msgpack_map_value(root, "stencil_radius");
            sr && (sr->type == msgpack::type::POSITIVE_INTEGER ||
                   sr->type == msgpack::type::NEGATIVE_INTEGER)) {
          params.stencil_radius = sr->as<int32_t>();
        }
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "gradU_stencil", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        if (inputs.size() < 2 || outputs.empty() || block < 0 ||
            static_cast<std::size_t>(block) >= level.boxes.size()) {
          return hpx::make_ready_future();
        }

        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

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

        const RunMeta& meta = current_runmeta();
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
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::string plotfile;
      int level = 0;
      int comp = 0;
      int bytes_per_value = 4;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* path = find_msgpack_map_value(root, "plotfile");
          path && path->type == msgpack::type::STR) {
        params.plotfile = path->as<std::string>();
      }
      if (const auto* lvl = find_msgpack_map_value(root, "level");
          lvl && (lvl->type == msgpack::type::POSITIVE_INTEGER ||
                  lvl->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.level = lvl->as<int>();
      }
      if (const auto* comp = find_msgpack_map_value(root, "comp");
          comp && (comp->type == msgpack::type::POSITIVE_INTEGER ||
                   comp->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.comp = comp->as<int>();
      }
      if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
          bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                  bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bytes_per_value = bpv->as<int>();
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "plotfile_load", .n_inputs = 0, .n_outputs = 1, .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView>,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }

        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

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
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      int axis = 2;
      double coord = 0.0;
      int plane_index = 0;
      bool has_plane_index = false;
      std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
      std::array<int, 2> resolution{1, 1};
      int bytes_per_value = 4;
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* axis = find_msgpack_map_value(root, "axis");
          axis && (axis->type == msgpack::type::POSITIVE_INTEGER ||
                   axis->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.axis = axis->as<int>();
      }
      if (const auto* coord = find_msgpack_map_value(root, "coord");
          coord && (coord->type == msgpack::type::FLOAT ||
                    coord->type == msgpack::type::POSITIVE_INTEGER ||
                    coord->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.coord = coord->as<double>();
      }
      if (const auto* plane_idx = find_msgpack_map_value(root, "plane_index");
          plane_idx && (plane_idx->type == msgpack::type::POSITIVE_INTEGER ||
                        plane_idx->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.plane_index = plane_idx->as<int>();
        params.has_plane_index = true;
      }
      if (const auto* rect = find_msgpack_map_value(root, "rect");
          rect && rect->type == msgpack::type::ARRAY && rect->via.array.size == 4) {
        for (uint32_t i = 0; i < 4; ++i) {
          params.rect[i] = rect->via.array.ptr[i].as<double>();
        }
      }
      if (const auto* res = find_msgpack_map_value(root, "resolution");
          res && res->type == msgpack::type::ARRAY && res->via.array.size == 2) {
        params.resolution[0] = res->via.array.ptr[0].as<int>();
        params.resolution[1] = res->via.array.ptr[1].as<int>();
      }
      if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
          bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                  bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bytes_per_value = bpv->as<int>();
      }
      params.covered_boxes = parse_covered_boxes_param(root);
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_cellavg_accumulate", .n_inputs = 1, .n_outputs = 2,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

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

          const int k_global =
              params.has_plane_index ? params.plane_index : cell_index(axis, params.coord);
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
            if (!params.covered_boxes) {
              return false;
            }
            for (const auto& b : *params.covered_boxes) {
              if (covered_box_contains(b, ix, iy, iz)) {
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
                const double ov =
                    std::max(0.0, std::min(v_cell_hi, pv1) - std::max(v_cell_lo, pv0));
                if (ov <= 0.0) {
                  continue;
                }
                for (int i = i0; i <= i1; ++i) {
                  const double pu0 = umin + static_cast<double>(i) * du;
                  const double pu1 = pu0 + du;
                  const double ou =
                      std::max(0.0, std::min(u_cell_hi, pu1) - std::max(u_cell_lo, pu0));
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
        },
        make_covered_box_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      int axis = 2;
      std::array<double, 2> axis_bounds{0.0, 1.0};
      std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
      std::array<int, 2> resolution{1, 1};
      int bytes_per_value = 4;
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* axis = find_msgpack_map_value(root, "axis");
          axis && (axis->type == msgpack::type::POSITIVE_INTEGER ||
                   axis->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.axis = axis->as<int>();
      }
      if (const auto* bounds = find_msgpack_map_value(root, "axis_bounds");
          bounds && bounds->type == msgpack::type::ARRAY && bounds->via.array.size == 2) {
        params.axis_bounds[0] = bounds->via.array.ptr[0].as<double>();
        params.axis_bounds[1] = bounds->via.array.ptr[1].as<double>();
      }
      if (const auto* rect = find_msgpack_map_value(root, "rect");
          rect && rect->type == msgpack::type::ARRAY && rect->via.array.size == 4) {
        for (uint32_t i = 0; i < 4; ++i) {
          params.rect[i] = rect->via.array.ptr[i].as<double>();
        }
      }
      if (const auto* res = find_msgpack_map_value(root, "resolution");
          res && res->type == msgpack::type::ARRAY && res->via.array.size == 2) {
        params.resolution[0] = res->via.array.ptr[0].as<int>();
        params.resolution[1] = res->via.array.ptr[1].as<int>();
      }
      if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
          bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                  bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bytes_per_value = bpv->as<int>();
      }
      params.covered_boxes = parse_covered_boxes_param(root);
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "uniform_projection_accumulate", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

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

          std::vector<std::uint8_t> covered_mask;
          if (params.covered_boxes && !params.covered_boxes->empty()) {
            const std::size_t block_cells = static_cast<std::size_t>(nx) *
                                            static_cast<std::size_t>(ny) *
                                            static_cast<std::size_t>(nz);
            covered_mask.assign(block_cells, 0);
            for (const auto& b : *params.covered_boxes) {
              const int gx0 = std::max(box.lo.x, b.lo[0]);
              const int gy0 = std::max(box.lo.y, b.lo[1]);
              const int gz0 = std::max(box.lo.z, b.lo[2]);
              const int gx1 = std::min(box.hi.x, b.hi[0]);
              const int gy1 = std::min(box.hi.y, b.hi[1]);
              const int gz1 = std::min(box.hi.z, b.hi[2]);
              if (gx0 > gx1 || gy0 > gy1 || gz0 > gz1) {
                continue;
              }
              for (int gx = gx0; gx <= gx1; ++gx) {
                const int i = gx - box.lo.x;
                for (int gy = gy0; gy <= gy1; ++gy) {
                  const int j = gy - box.lo.y;
                  for (int gz = gz0; gz <= gz1; ++gz) {
                    const int k = gz - box.lo.z;
                    covered_mask[in_index(i, j, k)] = 1;
                  }
                }
              }
            }
          }

          auto cell_edge = [&](int ax, int idx) -> double {
            return level.geom.x0[ax] + (idx - level.geom.index_origin[ax]) * level.geom.dx[ax];
          };

          const auto& in = inputs[0].data;
          auto* out_sum = reinterpret_cast<double*>(outputs[0].data.data());
          std::size_t candidate_cells = 0;
          std::size_t covered_skips = 0;
          std::size_t bounds_skips = 0;
          std::size_t deposited_cells = 0;

          for (int i = 0; i < nx; ++i) {
            const int gx = box.lo.x + i;
            for (int j = 0; j < ny; ++j) {
              const int gy = box.lo.y + j;
              for (int k = 0; k < nz; ++k) {
                ++candidate_cells;
                const auto data_idx = in_index(i, j, k);
                if (!covered_mask.empty() && covered_mask[data_idx] != 0) {
                  ++covered_skips;
                  continue;
                }
                const int gz = box.lo.z + k;

                const int a_global = axis == 0 ? gx : (axis == 1 ? gy : gz);
                const double a_cell_lo = cell_edge(axis, a_global);
                const double a_cell_hi = a_cell_lo + level.geom.dx[axis];
                const double oa =
                    std::max(0.0, std::min(a_cell_hi, amax) - std::max(a_cell_lo, amin));
                if (oa <= 0.0) {
                  ++bounds_skips;
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
                  ++bounds_skips;
                  continue;
                }
                i0 = std::max(i0, 0);
                j0 = std::max(j0, 0);
                i1 = std::min(i1, out_nx - 1);
                j1 = std::min(j1, out_ny - 1);

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

                for (int jj = j0; jj <= j1; ++jj) {
                  const double pv0 = vmin + static_cast<double>(jj) * dv;
                  const double pv1 = pv0 + dv;
                  const double ov =
                      std::max(0.0, std::min(v_cell_hi, pv1) - std::max(v_cell_lo, pv0));
                  if (ov <= 0.0) {
                    continue;
                  }
                  for (int ii = i0; ii <= i1; ++ii) {
                    const double pu0 = umin + static_cast<double>(ii) * du;
                    const double pu1 = pu0 + du;
                    const double ou =
                        std::max(0.0, std::min(u_cell_hi, pu1) - std::max(u_cell_lo, pu0));
                    if (ou <= 0.0) {
                      continue;
                    }
                    const double volume = ou * ov * oa;
                    const std::size_t out_idx = static_cast<std::size_t>(jj) * out_nx + ii;
                    out_sum[out_idx] += value * volume;
                    ++deposited_cells;
                  }
                }
              }
            }
          }

          double total_sum = 0.0;
          for (std::size_t idx = 0;
               idx < static_cast<std::size_t>(out_nx) * static_cast<std::size_t>(out_ny); ++idx) {
            total_sum += out_sum[idx];
          }
          log_projection_kernel_summary("uniform_projection_accumulate",
                                        -1,
                                        block,
                                        covered_box_count(params.covered_boxes),
                                        candidate_cells,
                                        covered_skips,
                                        bounds_skips,
                                        deposited_cells,
                                        total_sum);

          return hpx::make_ready_future();
        },
        make_covered_box_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::string expression;
      std::vector<std::string> variables;
      std::vector<int> input_bytes_per_value;
      int out_bytes_per_value = 8;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* expr = find_msgpack_map_value(root, "expression");
          expr && expr->type == msgpack::type::STR) {
        params.expression = expr->as<std::string>();
      }
      if (const auto* vars = find_msgpack_map_value(root, "variables");
          vars && vars->type == msgpack::type::ARRAY) {
        params.variables.reserve(vars->via.array.size);
        for (uint32_t i = 0; i < vars->via.array.size; ++i) {
          const auto& v = vars->via.array.ptr[i];
          if (v.type == msgpack::type::STR) {
            params.variables.push_back(v.as<std::string>());
          }
        }
      }
      if (const auto* in_bpv = find_msgpack_map_value(root, "input_bytes_per_value");
          in_bpv && in_bpv->type == msgpack::type::ARRAY) {
        params.input_bytes_per_value.reserve(in_bpv->via.array.size);
        for (uint32_t i = 0; i < in_bpv->via.array.size; ++i) {
          const auto& v = in_bpv->via.array.ptr[i];
          if (v.type == msgpack::type::POSITIVE_INTEGER ||
              v.type == msgpack::type::NEGATIVE_INTEGER) {
            params.input_bytes_per_value.push_back(v.as<int>());
          }
        }
      }
      if (const auto* out_bpv = find_msgpack_map_value(root, "out_bytes_per_value");
          out_bpv && (out_bpv->type == msgpack::type::POSITIVE_INTEGER ||
                      out_bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.out_bytes_per_value = out_bpv->as<int>();
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "field_expr", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        if (params.expression.empty()) {
          throw std::runtime_error("field_expr requires a non-empty expression");
        }
        if (params.variables.empty()) {
          throw std::runtime_error("field_expr requires at least one variable");
        }
        if (params.variables.size() != inputs.size()) {
          throw std::runtime_error("field_expr variables/input size mismatch");
        }
        if (params.variables.size() > 8) {
          throw std::runtime_error("field_expr currently supports at most 8 variables");
        }
        if (params.out_bytes_per_value != 4 && params.out_bytes_per_value != 8) {
          throw std::runtime_error("field_expr out_bytes_per_value must be 4 or 8");
        }

        auto input_bytes_per_value = params.input_bytes_per_value;
        if (input_bytes_per_value.size() < inputs.size()) {
          input_bytes_per_value.resize(inputs.size(), 8);
        }

        auto read_value = [&](int iv, std::size_t idx) -> double {
          const auto& data = inputs[static_cast<std::size_t>(iv)].data;
          const int bpv = input_bytes_per_value[static_cast<std::size_t>(iv)];
          if (bpv == 4) {
            const std::size_t pos = idx * sizeof(float);
            if (pos < data.size()) {
              return static_cast<double>(reinterpret_cast<const float*>(data.data())[idx]);
            }
          } else if (bpv == 8) {
            const std::size_t pos = idx * sizeof(double);
            if (pos < data.size()) {
              return reinterpret_cast<const double*>(data.data())[idx];
            }
          }
          return 0.0;
        };

        std::size_t n = std::numeric_limits<std::size_t>::max();
        for (std::size_t iv = 0; iv < inputs.size(); ++iv) {
          const auto& in = inputs[iv].data;
          const int bpv = input_bytes_per_value[iv];
          if (bpv != 4 && bpv != 8) {
            throw std::runtime_error("field_expr input_bytes_per_value must be 4 or 8");
          }
          const std::size_t count = in.size() / static_cast<std::size_t>(bpv);
          n = std::min(n, count);
        }
        if (n == std::numeric_limits<std::size_t>::max()) {
          n = 0;
        }

        const std::size_t out_bytes = n * static_cast<std::size_t>(params.out_bytes_per_value);
        outputs[0].data.assign(out_bytes, 0);
        auto write_value = [&](std::size_t idx, double value) {
          if (params.out_bytes_per_value == 8) {
            reinterpret_cast<double*>(outputs[0].data.data())[idx] = value;
          } else {
            reinterpret_cast<float*>(outputs[0].data.data())[idx] = static_cast<float>(value);
          }
        };

        amrexpr::Parser parser;
        try {
          parser.define(params.expression);
          parser.registerVariables(params.variables);
        } catch (const std::runtime_error& e) {
          throw std::runtime_error(std::string("field_expr parse failed: ") + e.what());
        }

#define KANGAROO_FIELD_EXPR_CASE(N)                                                      \
        case N: {                                                                        \
          auto exe = parser.compileHost<N>();                                            \
          for (std::size_t idx = 0; idx < n; ++idx) {                                   \
            double vars[N];                                                              \
            for (int iv = 0; iv < N; ++iv) {                                             \
              vars[iv] = read_value(iv, idx);                                            \
            }                                                                            \
            write_value(idx, exe(vars));                                                 \
          }                                                                              \
          break;                                                                         \
        }

        switch (static_cast<int>(params.variables.size())) {
          KANGAROO_FIELD_EXPR_CASE(1)
          KANGAROO_FIELD_EXPR_CASE(2)
          KANGAROO_FIELD_EXPR_CASE(3)
          KANGAROO_FIELD_EXPR_CASE(4)
          KANGAROO_FIELD_EXPR_CASE(5)
          KANGAROO_FIELD_EXPR_CASE(6)
          KANGAROO_FIELD_EXPR_CASE(7)
          KANGAROO_FIELD_EXPR_CASE(8)
          default:
            throw std::runtime_error("field_expr variable count is out of range");
        }

#undef KANGAROO_FIELD_EXPR_CASE
        return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
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
  {
    struct Params {
      int axis = 2;
      double coord = 0.0;
      std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
      std::array<int, 2> resolution{1, 1};
      int bytes_per_value = 4;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* axis = find_msgpack_map_value(root, "axis");
          axis && (axis->type == msgpack::type::POSITIVE_INTEGER ||
                   axis->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.axis = axis->as<int>();
      }
      if (const auto* coord = find_msgpack_map_value(root, "coord");
          coord && (coord->type == msgpack::type::FLOAT ||
                    coord->type == msgpack::type::POSITIVE_INTEGER ||
                    coord->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.coord = coord->as<double>();
      }
      if (const auto* rect = find_msgpack_map_value(root, "rect");
          rect && rect->type == msgpack::type::ARRAY && rect->via.array.size == 4) {
        for (uint32_t i = 0; i < 4; ++i) {
          params.rect[i] = rect->via.array.ptr[i].as<double>();
        }
      }
      if (const auto* res = find_msgpack_map_value(root, "resolution");
          res && res->type == msgpack::type::ARRAY && res->via.array.size == 2) {
        params.resolution[0] = res->via.array.ptr[0].as<int>();
        params.resolution[1] = res->via.array.ptr[1].as<int>();
      }
      if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
          bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                  bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bytes_per_value = bpv->as<int>();
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "uniform_slice", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          if (log_locality) {
            std::cout << "[kangaroo] uniform_slice block=" << block
                      << " locality=" << hpx::get_locality_id() << std::endl;
          }
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

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
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::string particle_type;
      std::string field_name;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* particle_type = find_msgpack_map_value(root, "particle_type");
          particle_type && particle_type->type == msgpack::type::STR) {
        params.particle_type = particle_type->as<std::string>();
      }
      if (const auto* field_name = find_msgpack_map_value(root, "field_name");
          field_name && field_name->type == msgpack::type::STR) {
        params.field_name = field_name->as<std::string>();
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "particle_load_field_chunk_f64", .n_inputs = 0, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t block, std::span<const HostView>,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (params.particle_type.empty() || params.field_name.empty()) {
            throw std::runtime_error(
                "particle_load_field_chunk_f64 requires particle_type and field_name");
          }

          const auto& dataset = current_dataset();
          if (!dataset.backend) {
            throw std::runtime_error("particle_load_field_chunk_f64: missing dataset backend");
          }
          const auto* reader = dataset.backend->get_plotfile_reader();
          if (reader == nullptr) {
            throw std::runtime_error(
                "particle_load_field_chunk_f64 requires an AMReX plotfile-backed dataset");
          }
          auto data =
              reader->read_particle_field_chunk(params.particle_type, params.field_name, block);
          const std::size_t n = static_cast<std::size_t>(std::max<int64_t>(0, data.count));
          outputs[0].data.resize(n * sizeof(double));
          auto* out = reinterpret_cast<double*>(outputs[0].data.data());

          if (data.dtype == "float64") {
            if (data.bytes.size() < n * sizeof(double)) {
              throw std::runtime_error("particle_load_field_chunk_f64: short float64 payload");
            }
            std::memcpy(out, data.bytes.data(), n * sizeof(double));
          } else if (data.dtype == "float32") {
            if (data.bytes.size() < n * sizeof(float)) {
              throw std::runtime_error("particle_load_field_chunk_f64: short float32 payload");
            }
            const auto* in = reinterpret_cast<const float*>(data.bytes.data());
            for (std::size_t i = 0; i < n; ++i) {
              out[i] = static_cast<double>(in[i]);
            }
          } else if (data.dtype == "int64") {
            if (data.bytes.size() < n * sizeof(int64_t)) {
              throw std::runtime_error("particle_load_field_chunk_f64: short int64 payload");
            }
            const auto* in = reinterpret_cast<const int64_t*>(data.bytes.data());
            for (std::size_t i = 0; i < n; ++i) {
              out[i] = static_cast<double>(in[i]);
            }
          } else {
            throw std::runtime_error("particle_load_field_chunk_f64: unsupported particle dtype '" +
                                     data.dtype + "'");
          }
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::string particle_type;
      int level_index = -1;
      int axis = 2;
      std::array<double, 2> axis_bounds{0.0, 0.0};
      double mass_max = std::numeric_limits<double>::quiet_NaN();
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* particle_type = find_msgpack_map_value(root, "particle_type");
          particle_type && particle_type->type == msgpack::type::STR) {
        params.particle_type = particle_type->as<std::string>();
      }
      if (const auto* level_index = find_msgpack_map_value(root, "level_index");
          level_index && (level_index->type == msgpack::type::POSITIVE_INTEGER ||
                          level_index->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.level_index = level_index->as<int>();
      }
      if (const auto* axis = find_msgpack_map_value(root, "axis");
          axis && (axis->type == msgpack::type::POSITIVE_INTEGER ||
                   axis->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.axis = axis->as<int>();
      }
      if (const auto* axis_bounds = find_msgpack_map_value(root, "axis_bounds");
          axis_bounds && axis_bounds->type == msgpack::type::ARRAY &&
          axis_bounds->via.array.size == 2) {
        params.axis_bounds[0] = axis_bounds->via.array.ptr[0].as<double>();
        params.axis_bounds[1] = axis_bounds->via.array.ptr[1].as<double>();
      }
      if (const auto* mass_max = find_msgpack_map_value(root, "mass_max");
          mass_max && (mass_max->type == msgpack::type::POSITIVE_INTEGER ||
                       mass_max->type == msgpack::type::NEGATIVE_INTEGER ||
                       mass_max->type == msgpack::type::FLOAT)) {
        params.mass_max = mass_max->as<double>();
      }
      params.covered_boxes = parse_covered_boxes_param(root);
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "particle_cic_grid_accumulate", .n_inputs = 0, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView>,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          if (params.particle_type.empty()) {
            throw std::runtime_error("particle_cic_grid_accumulate: missing particle_type");
          }
          if (params.axis < 0 || params.axis > 2) {
            throw std::runtime_error("particle_cic_grid_accumulate: axis must be 0, 1, or 2");
          }
          if (params.level_index < 0) {
            throw std::runtime_error("particle_cic_grid_accumulate: missing level_index");
          }
          if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
            throw std::runtime_error("particle_cic_grid_accumulate: block index out of range");
          }

          const auto& dataset = current_dataset();
          if (!dataset.backend) {
            throw std::runtime_error("particle_cic_grid_accumulate: missing dataset backend");
          }
          const auto* reader = dataset.backend->get_plotfile_reader();
          if (reader == nullptr) {
            throw std::runtime_error(
                "particle_cic_grid_accumulate requires an AMReX plotfile-backed dataset");
          }

          auto px =
              reader->read_particle_field_grid(params.particle_type, "x", params.level_index, block);
          auto py =
              reader->read_particle_field_grid(params.particle_type, "y", params.level_index, block);
          auto pz =
              reader->read_particle_field_grid(params.particle_type, "z", params.level_index, block);
          auto pm = reader->read_particle_field_grid(params.particle_type, "mass",
                                                     params.level_index, block);

          std::vector<double> px_vals;
          std::vector<double> py_vals;
          std::vector<double> pz_vals;
          std::vector<double> pm_vals;
          append_particle_values_as_f64(px, "x", "particle_cic_grid_accumulate", px_vals);
          append_particle_values_as_f64(py, "y", "particle_cic_grid_accumulate", py_vals);
          append_particle_values_as_f64(pz, "z", "particle_cic_grid_accumulate", pz_vals);
          append_particle_values_as_f64(pm, "mass", "particle_cic_grid_accumulate", pm_vals);

          const std::size_t n = std::min(std::min(px_vals.size(), py_vals.size()),
                                         std::min(pz_vals.size(), pm_vals.size()));
          px_vals.resize(n);
          py_vals.resize(n);
          pz_vals.resize(n);
          pm_vals.resize(n);

          const int axis = params.axis;
          const int u_axis = (axis == 0) ? 1 : 0;
          const int v_axis = (axis == 2) ? 1 : 2;

          const auto& box = level.boxes[static_cast<std::size_t>(block)];
          const int box_lo[3] = {box.lo.x, box.lo.y, box.lo.z};
          const int box_hi[3] = {box.hi.x, box.hi.y, box.hi.z};
          const int nx = box_hi[0] - box_lo[0] + 1;
          const int ny = box_hi[1] - box_lo[1] + 1;
          const int nz = box_hi[2] - box_lo[2] + 1;
          if (nx <= 0 || ny <= 0 || nz <= 0) {
            return hpx::make_ready_future();
          }

          const double x0[3] = {level.geom.x0[0], level.geom.x0[1], level.geom.x0[2]};
          const double dx[3] = {level.geom.dx[0], level.geom.dx[1], level.geom.dx[2]};
          const int origin[3] = {level.geom.index_origin[0], level.geom.index_origin[1],
                                 level.geom.index_origin[2]};
          if (dx[0] == 0.0 || dx[1] == 0.0 || dx[2] == 0.0) {
            return hpx::make_ready_future();
          }

          const std::size_t out_elems = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                                        static_cast<std::size_t>(nz);
          outputs[0].data.resize(out_elems * sizeof(double));
          auto* out = reinterpret_cast<double*>(outputs[0].data.data());
          std::fill(out, out + out_elems, 0.0);

          const double cell_volume = dx[0] * dx[1] * dx[2];
          if (!(cell_volume > 0.0)) {
            return hpx::make_ready_future();
          }

          const double a_lo = std::min(params.axis_bounds[0], params.axis_bounds[1]);
          const double a_hi = std::max(params.axis_bounds[0], params.axis_bounds[1]);

          auto covered = [&](int i, int j, int k) -> bool {
            if (!params.covered_boxes) {
              return false;
            }
            for (const auto& b : *params.covered_boxes) {
              if (covered_box_contains(b, i, j, k)) {
                return true;
              }
            }
            return false;
          };

          auto out_index = [ny, nz](int i, int j, int k) -> std::size_t {
            return static_cast<std::size_t>((i * ny + j) * nz + k);
          };

          const double* coord[3] = {px_vals.data(), py_vals.data(), pz_vals.data()};
          const double du = dx[u_axis];
          const double dv = dx[v_axis];
          const double da = dx[axis];

          const double u_center0 =
              x0[u_axis] + (0.5 - static_cast<double>(origin[u_axis])) * du;
          const double v_center0 =
              x0[v_axis] + (0.5 - static_cast<double>(origin[v_axis])) * dv;
          const double a_cell_lo =
              x0[axis] + (static_cast<double>(box_lo[axis] - origin[axis])) * da;
          const double a_cell_hi =
              x0[axis] + (static_cast<double>(box_hi[axis] + 1 - origin[axis])) * da;

          auto add_density = [&](int iu, int iv, int ia, double wmass) {
            int ii = 0;
            int jj = 0;
            int kk = 0;
            if (axis == 0) {
              ii = ia;
              jj = iu;
              kk = iv;
            } else if (axis == 1) {
              ii = iu;
              jj = ia;
              kk = iv;
            } else {
              ii = iu;
              jj = iv;
              kk = ia;
            }
            if (ii < box_lo[0] || ii > box_hi[0] || jj < box_lo[1] || jj > box_hi[1] ||
                kk < box_lo[2] || kk > box_hi[2] || covered(ii, jj, kk)) {
              return;
            }
            const int i_local = ii - box_lo[0];
            const int j_local = jj - box_lo[1];
            const int k_local = kk - box_lo[2];
            if (i_local < 0 || i_local >= nx || j_local < 0 || j_local >= ny ||
                k_local < 0 || k_local >= nz) {
              return;
            }
            out[out_index(i_local, j_local, k_local)] += wmass / cell_volume;
          };

          for (std::size_t p = 0; p < n; ++p) {
            const double u = coord[u_axis][p];
            const double v = coord[v_axis][p];
            const double a = coord[axis][p];
            const double m = pm_vals[p];
            if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(a) || !std::isfinite(m)) {
              continue;
            }
            if (m <= 0.0) {
              continue;
            }
            if (std::isfinite(params.mass_max) && m > params.mass_max) {
              continue;
            }
            if (a < a_lo || a > a_hi) {
              continue;
            }
            if (a < a_cell_lo || a >= a_cell_hi) {
              continue;
            }

            const double su = (u - u_center0) / du;
            const double sv = (v - v_center0) / dv;
            const int iu0 = static_cast<int>(std::floor(su));
            const int iv0 = static_cast<int>(std::floor(sv));
            const double tu = su - static_cast<double>(iu0);
            const double tv = sv - static_cast<double>(iv0);
            const int ia = static_cast<int>(std::floor((a - x0[axis]) / da)) + origin[axis];

            add_density(iu0, iv0, ia, m * (1.0 - tu) * (1.0 - tv));
            add_density(iu0 + 1, iv0, ia, m * tu * (1.0 - tv));
            add_density(iu0, iv0 + 1, ia, m * (1.0 - tu) * tv);
            add_density(iu0 + 1, iv0 + 1, ia, m * tu * tv);
          }

          return hpx::make_ready_future();
        },
        make_covered_box_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::string particle_type;
      int level_index = -1;
      int axis = 2;
      std::array<double, 2> axis_bounds{0.0, 0.0};
      std::array<double, 4> rect{0.0, 0.0, 1.0, 1.0};
      std::array<int, 2> resolution{1, 1};
      double mass_max = std::numeric_limits<double>::quiet_NaN();
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* particle_type = find_msgpack_map_value(root, "particle_type");
          particle_type && particle_type->type == msgpack::type::STR) {
        params.particle_type = particle_type->as<std::string>();
      }
      if (const auto* level_index = find_msgpack_map_value(root, "level_index");
          level_index && (level_index->type == msgpack::type::POSITIVE_INTEGER ||
                          level_index->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.level_index = level_index->as<int>();
      }
      if (const auto* axis = find_msgpack_map_value(root, "axis");
          axis && (axis->type == msgpack::type::POSITIVE_INTEGER ||
                   axis->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.axis = axis->as<int>();
      }
      if (const auto* axis_bounds = find_msgpack_map_value(root, "axis_bounds");
          axis_bounds && axis_bounds->type == msgpack::type::ARRAY &&
          axis_bounds->via.array.size == 2) {
        params.axis_bounds[0] = axis_bounds->via.array.ptr[0].as<double>();
        params.axis_bounds[1] = axis_bounds->via.array.ptr[1].as<double>();
      }
      if (const auto* rect = find_msgpack_map_value(root, "rect");
          rect && rect->type == msgpack::type::ARRAY && rect->via.array.size == 4) {
        for (uint32_t j = 0; j < 4; ++j) {
          params.rect[j] = rect->via.array.ptr[j].as<double>();
        }
      }
      if (const auto* resolution = find_msgpack_map_value(root, "resolution");
          resolution && resolution->type == msgpack::type::ARRAY &&
          resolution->via.array.size == 2) {
        params.resolution[0] = resolution->via.array.ptr[0].as<int>();
        params.resolution[1] = resolution->via.array.ptr[1].as<int>();
      }
      if (const auto* mass_max = find_msgpack_map_value(root, "mass_max");
          mass_max && (mass_max->type == msgpack::type::POSITIVE_INTEGER ||
                       mass_max->type == msgpack::type::NEGATIVE_INTEGER ||
                       mass_max->type == msgpack::type::FLOAT)) {
        params.mass_max = mass_max->as<double>();
      }
      params.covered_boxes = parse_covered_boxes_param(root);
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "particle_cic_projection_accumulate", .n_inputs = 0, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView>,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          const int out_nx = params.resolution[0];
          const int out_ny = params.resolution[1];
          if (out_nx <= 0 || out_ny <= 0) {
            throw std::runtime_error("particle_cic_projection_accumulate: invalid resolution");
          }
          outputs[0].data.resize(
              static_cast<std::size_t>(out_nx) * static_cast<std::size_t>(out_ny) * sizeof(double));
          auto* out = reinterpret_cast<double*>(outputs[0].data.data());
          std::fill(out, out + static_cast<std::size_t>(out_nx) * static_cast<std::size_t>(out_ny),
                    0.0);

          if (params.particle_type.empty()) {
            throw std::runtime_error("particle_cic_projection_accumulate: missing particle_type");
          }
          if (params.axis < 0 || params.axis > 2) {
            throw std::runtime_error(
                "particle_cic_projection_accumulate: axis must be 0, 1, or 2");
          }
          if (params.level_index < 0) {
            throw std::runtime_error("particle_cic_projection_accumulate: missing level_index");
          }
          if (block < 0 || static_cast<std::size_t>(block) >= level.boxes.size()) {
            throw std::runtime_error(
                "particle_cic_projection_accumulate: block index out of range");
          }

          const auto& dataset = current_dataset();
          if (!dataset.backend) {
            throw std::runtime_error("particle_cic_projection_accumulate: missing dataset backend");
          }
          const auto* reader = dataset.backend->get_plotfile_reader();
          if (reader == nullptr) {
            throw std::runtime_error(
                "particle_cic_projection_accumulate requires an AMReX plotfile-backed dataset");
          }

          auto px =
              reader->read_particle_field_grid(params.particle_type, "x", params.level_index, block);
          auto py =
              reader->read_particle_field_grid(params.particle_type, "y", params.level_index, block);
          auto pz =
              reader->read_particle_field_grid(params.particle_type, "z", params.level_index, block);
          auto pm = reader->read_particle_field_grid(params.particle_type, "mass",
                                                     params.level_index, block);

          std::vector<double> px_vals;
          std::vector<double> py_vals;
          std::vector<double> pz_vals;
          std::vector<double> pm_vals;
          append_particle_values_as_f64(px, "x", "particle_cic_projection_accumulate", px_vals);
          append_particle_values_as_f64(py, "y", "particle_cic_projection_accumulate", py_vals);
          append_particle_values_as_f64(pz, "z", "particle_cic_projection_accumulate", pz_vals);
          append_particle_values_as_f64(pm, "mass", "particle_cic_projection_accumulate", pm_vals);

          const std::size_t n = std::min(std::min(px_vals.size(), py_vals.size()),
                                         std::min(pz_vals.size(), pm_vals.size()));
          px_vals.resize(n);
          py_vals.resize(n);
          pz_vals.resize(n);
          pm_vals.resize(n);

          const int axis = params.axis;
          const int u_axis = (axis == 0) ? 1 : 0;
          const int v_axis = (axis == 2) ? 1 : 2;

          const auto& box = level.boxes[static_cast<std::size_t>(block)];
          const int box_lo[3] = {box.lo.x, box.lo.y, box.lo.z};
          const int box_hi[3] = {box.hi.x, box.hi.y, box.hi.z};

          const double x0[3] = {level.geom.x0[0], level.geom.x0[1], level.geom.x0[2]};
          const double dx[3] = {level.geom.dx[0], level.geom.dx[1], level.geom.dx[2]};
          const int origin[3] = {level.geom.index_origin[0], level.geom.index_origin[1],
                                 level.geom.index_origin[2]};
          if (dx[0] == 0.0 || dx[1] == 0.0 || dx[2] == 0.0) {
            return hpx::make_ready_future();
          }

          const int nu = box_hi[u_axis] - box_lo[u_axis] + 1;
          const int nv = box_hi[v_axis] - box_lo[v_axis] + 1;
          if (nu <= 0 || nv <= 0) {
            return hpx::make_ready_future();
          }

          const double a_lo = std::min(params.axis_bounds[0], params.axis_bounds[1]);
          const double a_hi = std::max(params.axis_bounds[0], params.axis_bounds[1]);
          const double rect_u_lo = std::min(params.rect[0], params.rect[2]);
          const double rect_u_hi = std::max(params.rect[0], params.rect[2]);
          const double rect_v_lo = std::min(params.rect[1], params.rect[3]);
          const double rect_v_hi = std::max(params.rect[1], params.rect[3]);
          const double out_du = (rect_u_hi - rect_u_lo) / static_cast<double>(out_nx);
          const double out_dv = (rect_v_hi - rect_v_lo) / static_cast<double>(out_ny);
          if (out_du <= 0.0 || out_dv <= 0.0) {
            return hpx::make_ready_future();
          }
          const double inv_out_du = 1.0 / out_du;
          const double inv_out_dv = 1.0 / out_dv;

          auto covered = [&](int i, int j, int k) -> bool {
            if (!params.covered_boxes) {
              return false;
            }
            for (const auto& b : *params.covered_boxes) {
              if (covered_box_contains(b, i, j, k)) {
                return true;
              }
            }
            return false;
          };

          std::vector<double> native_mass(static_cast<std::size_t>(nu) * static_cast<std::size_t>(nv),
                                          0.0);
          auto native_index = [nu](int iu_local, int iv_local) -> std::size_t {
            return static_cast<std::size_t>(iv_local) * static_cast<std::size_t>(nu) +
                   static_cast<std::size_t>(iu_local);
          };
          std::size_t covered_skips = 0;
          std::size_t bounds_skips = 0;
          std::size_t deposited_cells = 0;
          auto native_add = [&](int iu, int iv, int ia, double wmass) {
            int ii = 0;
            int jj = 0;
            int kk = 0;
            if (axis == 0) {
              ii = ia;
              jj = iu;
              kk = iv;
            } else if (axis == 1) {
              ii = iu;
              jj = ia;
              kk = iv;
            } else {
              ii = iu;
              jj = iv;
              kk = ia;
            }
            if (ii < box_lo[0] || ii > box_hi[0] || jj < box_lo[1] || jj > box_hi[1] ||
                kk < box_lo[2] || kk > box_hi[2]) {
              ++bounds_skips;
              return;
            }
            if (covered(ii, jj, kk)) {
              ++covered_skips;
              return;
            }
            const int iu_local = iu - box_lo[u_axis];
            const int iv_local = iv - box_lo[v_axis];
            if (iu_local < 0 || iu_local >= nu || iv_local < 0 || iv_local >= nv) {
              ++bounds_skips;
              return;
            }
            native_mass[native_index(iu_local, iv_local)] += wmass;
            ++deposited_cells;
          };

          const double* coord[3] = {px_vals.data(), py_vals.data(), pz_vals.data()};
          const double du = dx[u_axis];
          const double dv = dx[v_axis];
          const double da = dx[axis];

          const double u_center0 =
              x0[u_axis] + (0.5 - static_cast<double>(origin[u_axis])) * du;
          const double v_center0 =
              x0[v_axis] + (0.5 - static_cast<double>(origin[v_axis])) * dv;
          const double a_cell_lo =
              x0[axis] + (static_cast<double>(box_lo[axis] - origin[axis])) * da;
          const double a_cell_hi =
              x0[axis] + (static_cast<double>(box_hi[axis] + 1 - origin[axis])) * da;
          std::size_t candidates = 0;

          for (std::size_t p = 0; p < n; ++p) {
            ++candidates;
            const double u = coord[u_axis][p];
            const double v = coord[v_axis][p];
            const double a = coord[axis][p];
            const double m = pm_vals[p];
            if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(a) || !std::isfinite(m)) {
              ++bounds_skips;
              continue;
            }
            if (m <= 0.0) {
              ++bounds_skips;
              continue;
            }
            if (std::isfinite(params.mass_max) && m > params.mass_max) {
              ++bounds_skips;
              continue;
            }
            if (a < a_lo || a > a_hi) {
              ++bounds_skips;
              continue;
            }
            if (a < a_cell_lo || a >= a_cell_hi) {
              ++bounds_skips;
              continue;
            }

            const double su = (u - u_center0) / du;
            const double sv = (v - v_center0) / dv;
            const int iu0 = static_cast<int>(std::floor(su));
            const int iv0 = static_cast<int>(std::floor(sv));
            const double tu = su - static_cast<double>(iu0);
            const double tv = sv - static_cast<double>(iv0);
            const int ia = static_cast<int>(std::floor((a - x0[axis]) / da)) + origin[axis];

            native_add(iu0, iv0, ia, m * (1.0 - tu) * (1.0 - tv));
            native_add(iu0 + 1, iv0, ia, m * tu * (1.0 - tv));
            native_add(iu0, iv0 + 1, ia, m * (1.0 - tu) * tv);
            native_add(iu0 + 1, iv0 + 1, ia, m * tu * tv);
          }

          const double cell_area = du * dv;
          if (cell_area <= 0.0) {
            return hpx::make_ready_future();
          }

          for (int iv_local = 0; iv_local < nv; ++iv_local) {
            const int iv = box_lo[v_axis] + iv_local;
            const double v_center =
                x0[v_axis] + (static_cast<double>(iv - origin[v_axis]) + 0.5) * dv;
            const double v_lo = v_center - 0.5 * dv;
            const double v_hi = v_center + 0.5 * dv;
            if (v_hi <= rect_v_lo || v_lo >= rect_v_hi) {
              continue;
            }

            int iy_lo = static_cast<int>(std::floor((v_lo - rect_v_lo) * inv_out_dv));
            int iy_hi = static_cast<int>(std::floor((v_hi - rect_v_lo) * inv_out_dv));
            if (iy_hi < 0 || iy_lo >= out_ny) {
              continue;
            }
            iy_lo = std::max(0, iy_lo);
            iy_hi = std::min(out_ny - 1, iy_hi);

            for (int iu_local = 0; iu_local < nu; ++iu_local) {
              const double mass = native_mass[native_index(iu_local, iv_local)];
              if (mass == 0.0) {
                continue;
              }
              const int iu = box_lo[u_axis] + iu_local;
              const double u_center =
                  x0[u_axis] + (static_cast<double>(iu - origin[u_axis]) + 0.5) * du;
              const double u_lo = u_center - 0.5 * du;
              const double u_hi = u_center + 0.5 * du;
              if (u_hi <= rect_u_lo || u_lo >= rect_u_hi) {
                continue;
              }

              int ix_lo = static_cast<int>(std::floor((u_lo - rect_u_lo) * inv_out_du));
              int ix_hi = static_cast<int>(std::floor((u_hi - rect_u_lo) * inv_out_du));
              if (ix_hi < 0 || ix_lo >= out_nx) {
                continue;
              }
              ix_lo = std::max(0, ix_lo);
              ix_hi = std::min(out_nx - 1, ix_hi);

              for (int iy = iy_lo; iy <= iy_hi; ++iy) {
                const double pv_lo = rect_v_lo + static_cast<double>(iy) * out_dv;
                const double pv_hi = pv_lo + out_dv;
                const double ov = std::max(0.0, std::min(v_hi, pv_hi) - std::max(v_lo, pv_lo));
                if (ov <= 0.0) {
                  continue;
                }
                const std::size_t row = static_cast<std::size_t>(iy) *
                                        static_cast<std::size_t>(out_nx);
                for (int ix = ix_lo; ix <= ix_hi; ++ix) {
                  const double pu_lo = rect_u_lo + static_cast<double>(ix) * out_du;
                  const double pu_hi = pu_lo + out_du;
                  const double ou = std::max(0.0, std::min(u_hi, pu_hi) - std::max(u_lo, pu_lo));
                  if (ou <= 0.0) {
                    continue;
                  }
                  const double w = (ou * ov) / cell_area;
                  out[row + static_cast<std::size_t>(ix)] += mass * w;
                }
              }
            }
          }
          double total_sum = 0.0;
          for (std::size_t idx = 0;
               idx < static_cast<std::size_t>(out_nx) * static_cast<std::size_t>(out_ny); ++idx) {
            total_sum += out[idx];
          }
          log_projection_kernel_summary("particle_cic_projection_accumulate",
                                        params.level_index,
                                        block,
                                        covered_box_count(params.covered_boxes),
                                        candidates,
                                        covered_skips,
                                        bounds_skips,
                                        deposited_cells,
                                        total_sum);
          return hpx::make_ready_future();
        },
        make_covered_box_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      double scalar = 0.0;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* scalar = find_msgpack_map_value(root, "scalar")) {
        params.scalar = scalar->as<double>();
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "particle_eq_mask", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& in = inputs[0].data;
          const std::size_t n = in.size() / sizeof(double);
          outputs[0].data.resize(n);
          const auto* in_d = reinterpret_cast<const double*>(in.data());
          auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
          for (std::size_t i = 0; i < n; ++i) {
            out[i] = (in_d[i] == params.scalar) ? 1 : 0;
          }
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));

    registry.register_kernel(
        KernelDesc{.name = "particle_abs_lt_mask", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& in = inputs[0].data;
          const std::size_t n = in.size() / sizeof(double);
          outputs[0].data.resize(n);
          const auto* in_d = reinterpret_cast<const double*>(in.data());
          auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
          for (std::size_t i = 0; i < n; ++i) {
            out[i] = (std::abs(in_d[i]) < params.scalar) ? 1 : 0;
          }
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));

    registry.register_kernel(
        KernelDesc{.name = "particle_le_mask", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& in = inputs[0].data;
          const std::size_t n = in.size() / sizeof(double);
          outputs[0].data.resize(n);
          const auto* in_d = reinterpret_cast<const double*>(in.data());
          auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
          for (std::size_t i = 0; i < n; ++i) {
            out[i] = (in_d[i] <= params.scalar) ? 1 : 0;
          }
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));

    registry.register_kernel(
        KernelDesc{.name = "particle_gt_mask", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& in = inputs[0].data;
          const std::size_t n = in.size() / sizeof(double);
          outputs[0].data.resize(n);
          const auto* in_d = reinterpret_cast<const double*>(in.data());
          auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
          for (std::size_t i = 0; i < n; ++i) {
            out[i] = (in_d[i] > params.scalar) ? 1 : 0;
          }
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::vector<double> values;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* values = find_msgpack_map_value(root, "values");
          values && values->type == msgpack::type::ARRAY) {
        params.values.reserve(values->via.array.size);
        for (uint32_t j = 0; j < values->via.array.size; ++j) {
          params.values.push_back(values->via.array.ptr[j].as<double>());
        }
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "particle_isin_mask", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& in = inputs[0].data;
          const std::size_t n = in.size() / sizeof(double);
          outputs[0].data.resize(n);
          const auto* in_d = reinterpret_cast<const double*>(in.data());
          auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
          for (std::size_t i = 0; i < n; ++i) {
            bool found = false;
            for (double x : params.values) {
              if (in_d[i] == x) {
                found = true;
                break;
              }
            }
            out[i] = found ? 1 : 0;
          }
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  registry.register_kernel(
      KernelDesc{.name = "particle_isfinite_mask", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (inputs.empty() || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto& in = inputs[0].data;
        const std::size_t n = in.size() / sizeof(double);
        outputs[0].data.resize(n);
        const auto* in_d = reinterpret_cast<const double*>(in.data());
        auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
        for (std::size_t i = 0; i < n; ++i) {
          out[i] = std::isfinite(in_d[i]) ? 1 : 0;
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "particle_and_mask", .n_inputs = 2, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (inputs.size() < 2 || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto& a = inputs[0].data;
        const auto& b = inputs[1].data;
        const std::size_t n = std::min(a.size(), b.size());
        outputs[0].data.resize(n);
        auto* out = reinterpret_cast<std::uint8_t*>(outputs[0].data.data());
        const auto* a_u = reinterpret_cast<const std::uint8_t*>(a.data());
        const auto* b_u = reinterpret_cast<const std::uint8_t*>(b.data());
        for (std::size_t i = 0; i < n; ++i) {
          out[i] = (a_u[i] != 0 && b_u[i] != 0) ? 1 : 0;
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "particle_filter", .n_inputs = 2, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (inputs.size() < 2 || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto& values = inputs[0].data;
        const auto& mask = inputs[1].data;
        const std::size_t n = std::min(values.size() / sizeof(double), mask.size());
        const auto* in_d = reinterpret_cast<const double*>(values.data());
        const auto* m_u = reinterpret_cast<const std::uint8_t*>(mask.data());
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
          if (m_u[i] != 0) {
            ++count;
          }
        }
        outputs[0].data.resize(count * sizeof(double));
        auto* out_d = reinterpret_cast<double*>(outputs[0].data.data());
        std::size_t out_idx = 0;
        for (std::size_t i = 0; i < n; ++i) {
          if (m_u[i] != 0) {
            out_d[out_idx++] = in_d[i];
          }
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "particle_subtract", .n_inputs = 2, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (inputs.size() < 2 || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto& a = inputs[0].data;
        const auto& b = inputs[1].data;
        const std::size_t n = std::min(a.size(), b.size()) / sizeof(double);
        outputs[0].data.resize(n * sizeof(double));
        const auto* a_d = reinterpret_cast<const double*>(a.data());
        const auto* b_d = reinterpret_cast<const double*>(b.data());
        auto* out_d = reinterpret_cast<double*>(outputs[0].data.data());
        for (std::size_t i = 0; i < n; ++i) {
          out_d[i] = a_d[i] - b_d[i];
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "particle_distance3", .n_inputs = 6, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (inputs.size() < 6 || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const std::size_t n = std::min(
            {inputs[0].data.size(), inputs[1].data.size(), inputs[2].data.size(),
             inputs[3].data.size(), inputs[4].data.size(), inputs[5].data.size()}) /
            sizeof(double);
        outputs[0].data.resize(n * sizeof(double));
        const auto* ax = reinterpret_cast<const double*>(inputs[0].data.data());
        const auto* ay = reinterpret_cast<const double*>(inputs[1].data.data());
        const auto* az = reinterpret_cast<const double*>(inputs[2].data.data());
        const auto* bx = reinterpret_cast<const double*>(inputs[3].data.data());
        const auto* by = reinterpret_cast<const double*>(inputs[4].data.data());
        const auto* bz = reinterpret_cast<const double*>(inputs[5].data.data());
        auto* out = reinterpret_cast<double*>(outputs[0].data.data());
        for (std::size_t i = 0; i < n; ++i) {
          const double dx = ax[i] - bx[i];
          const double dy = ay[i] - by[i];
          const double dz = az[i] - bz[i];
          out[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "particle_sum", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (inputs.empty() || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto& in = inputs[0].data;
        const std::size_t n = in.size() / sizeof(double);
        const auto* in_d = reinterpret_cast<const double*>(in.data());
        double sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          sum += in_d[i];
        }
        outputs[0].data.resize(sizeof(double));
        *reinterpret_cast<double*>(outputs[0].data.data()) = sum;
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "particle_count", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (inputs.empty() || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const auto& in = inputs[0].data;
        const std::size_t n = in.size();
        const auto* in_u = reinterpret_cast<const std::uint8_t*>(in.data());
        int64_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
          if (in_u[i] != 0) {
            ++count;
          }
        }
        outputs[0].data.resize(sizeof(int64_t));
        *reinterpret_cast<int64_t*>(outputs[0].data.data()) = count;
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "particle_len_f64", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (inputs.empty() || outputs.empty()) {
          return hpx::make_ready_future();
        }
        const int64_t n = static_cast<int64_t>(inputs[0].data.size() / sizeof(double));
        outputs[0].data.resize(sizeof(int64_t));
        *reinterpret_cast<int64_t*>(outputs[0].data.data()) = n;
        return hpx::make_ready_future();
      });
  {
    struct Params {
      bool finite_only = true;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* finite_only = find_msgpack_map_value(root, "finite_only")) {
        params.finite_only = finite_only->as<bool>();
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "particle_min", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& in = inputs[0].data;
          const std::size_t n = in.size() / sizeof(double);
          const auto* in_d = reinterpret_cast<const double*>(in.data());
          double out_v = std::numeric_limits<double>::infinity();
          bool any = false;
          for (std::size_t i = 0; i < n; ++i) {
            const double v = in_d[i];
            if (params.finite_only && !std::isfinite(v)) {
              continue;
            }
            if (!any || v < out_v) {
              out_v = v;
              any = true;
            }
          }
          outputs[0].data.resize(sizeof(double));
          *reinterpret_cast<double*>(outputs[0].data.data()) =
              any ? out_v : std::numeric_limits<double>::infinity();
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));

    registry.register_kernel(
        KernelDesc{.name = "particle_max", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (inputs.empty() || outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& in = inputs[0].data;
          const std::size_t n = in.size() / sizeof(double);
          const auto* in_d = reinterpret_cast<const double*>(in.data());
          double out_v = -std::numeric_limits<double>::infinity();
          bool any = false;
          for (std::size_t i = 0; i < n; ++i) {
            const double v = in_d[i];
            if (params.finite_only && !std::isfinite(v)) {
              continue;
            }
            if (!any || v > out_v) {
              out_v = v;
              any = true;
            }
          }
          outputs[0].data.resize(sizeof(double));
          *reinterpret_cast<double*>(outputs[0].data.data()) =
              any ? out_v : -std::numeric_limits<double>::infinity();
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::vector<double> edges;
      bool density = false;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* edges = find_msgpack_map_value(root, "edges");
          edges && edges->type == msgpack::type::ARRAY) {
        params.edges.reserve(edges->via.array.size);
        for (uint32_t j = 0; j < edges->via.array.size; ++j) {
          params.edges.push_back(edges->via.array.ptr[j].as<double>());
        }
      }
      if (const auto* density = find_msgpack_map_value(root, "density")) {
        params.density = density->as<bool>();
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "particle_histogram1d", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (inputs.empty() || outputs.empty() || params.edges.size() < 2) {
            return hpx::make_ready_future();
          }
          const std::size_t bins = params.edges.size() - 1;
          outputs[0].data.resize(bins * sizeof(double));
          auto* out = reinterpret_cast<double*>(outputs[0].data.data());
          std::fill(out, out + bins, 0.0);
          const auto& values = inputs[0].data;
          const std::size_t n = values.size() / sizeof(double);
          const auto* in_d = reinterpret_cast<const double*>(values.data());
          const bool weighted = inputs.size() >= 2;
          const auto* w_d =
              weighted ? reinterpret_cast<const double*>(inputs[1].data.data()) : nullptr;
          const std::size_t nw = weighted ? (inputs[1].data.size() / sizeof(double)) : 0;
          for (std::size_t i = 0; i < n; ++i) {
            const double x = in_d[i];
            if (!std::isfinite(x) || x < params.edges.front() || x > params.edges.back()) {
              continue;
            }
            std::size_t idx = bins - 1;
            if (x != params.edges.back()) {
              auto it = std::upper_bound(params.edges.begin(), params.edges.end(), x);
              idx = static_cast<std::size_t>(std::distance(params.edges.begin(), it) - 1);
            }
            if (idx >= bins) {
              continue;
            }
            double w = 1.0;
            if (weighted && i < nw) {
              w = w_d[i];
              if (!std::isfinite(w)) {
                continue;
              }
            }
            out[idx] += w;
          }
          if (params.density) {
            double total = 0.0;
            for (std::size_t i = 0; i < bins; ++i) {
              total += out[i];
            }
            if (total > 0.0) {
              for (std::size_t i = 0; i < bins; ++i) {
                const double width = params.edges[i + 1] - params.edges[i];
                if (width > 0.0) {
                  out[i] /= (total * width);
                }
              }
            }
          }
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::string particle_type;
      std::string field_name;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* particle_type = find_msgpack_map_value(root, "particle_type");
          particle_type && particle_type->type == msgpack::type::STR) {
        params.particle_type = particle_type->as<std::string>();
      }
      if (const auto* field_name = find_msgpack_map_value(root, "field_name");
          field_name && field_name->type == msgpack::type::STR) {
        params.field_name = field_name->as<std::string>();
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "particle_topk_modes_map", .n_inputs = 0, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t block, std::span<const HostView>,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (params.particle_type.empty() || params.field_name.empty()) {
            outputs[0].data.clear();
            return hpx::make_ready_future();
          }

          const auto& dataset = current_dataset();
          if (!dataset.backend) {
            throw std::runtime_error("particle_topk_modes_map: missing dataset backend");
          }
          const auto* reader = dataset.backend->get_plotfile_reader();
          if (reader == nullptr) {
            throw std::runtime_error(
                "particle_topk_modes_map requires an AMReX plotfile-backed dataset");
          }
          std::unordered_map<double, int64_t> counts;
          const auto data =
              reader->read_particle_field_chunk(params.particle_type, params.field_name, block);
          std::vector<double> values;
          append_particle_values_as_f64(data, params.field_name, "particle_topk_modes_map", values);
          for (double v : values) {
            if (!std::isfinite(v)) {
              continue;
            }
            counts[v] += 1;
          }
          encode_particle_value_counts(counts, outputs[0].data.mutable_vector());
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  registry.register_kernel(
      KernelDesc{.name = "particle_value_counts_reduce", .n_inputs = 1, .n_outputs = 1,
                 .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        std::unordered_map<double, int64_t> merged;
        for (const auto& in_view : inputs) {
          auto counts = decode_particle_value_counts(in_view.data);
          if (merged.empty()) {
            merged = std::move(counts);
            continue;
          }
          for (const auto& [value, count] : counts) {
            merged[value] += count;
          }
        }
        encode_particle_value_counts(merged, outputs[0].data.mutable_vector());
        return hpx::make_ready_future();
      });
  {
    struct Params {
      int64_t k = 0;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* k = find_msgpack_map_value(root, "k");
          k && (k->type == msgpack::type::POSITIVE_INTEGER ||
                k->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.k = k->as<int64_t>();
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "particle_topk_modes_finalize", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
          if (outputs.empty()) {
            return hpx::make_ready_future();
          }
          const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);
          if (inputs.empty() || params.k <= 0) {
            outputs[0].data.clear();
            return hpx::make_ready_future();
          }
          auto counts = decode_particle_value_counts(inputs[0].data);
          std::vector<std::pair<double, int64_t>> modes;
          modes.reserve(counts.size());
          for (const auto& it : counts) {
            modes.emplace_back(it.first, it.second);
          }
          std::sort(modes.begin(), modes.end(),
                    [](const auto& a, const auto& b) {
                      if (a.second != b.second) {
                        return a.second > b.second;
                      }
                      return a.first > b.first;
                    });

          const std::size_t out_len = static_cast<std::size_t>(params.k);
          outputs[0].data.resize(out_len * 2 * sizeof(double));
          auto* out = reinterpret_cast<double*>(outputs[0].data.data());
          for (std::size_t i = 0; i < out_len; ++i) {
            if (i < modes.size()) {
              out[i] = modes[i].first;
              out[out_len + i] = static_cast<double>(modes[i].second);
            } else {
              out[i] = std::numeric_limits<double>::quiet_NaN();
              out[out_len + i] = 0.0;
            }
          }
          return hpx::make_ready_future();
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  registry.register_kernel(
      KernelDesc{.name = "particle_int64_sum_reduce", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        int64_t total = 0;
        for (const auto& in_view : inputs) {
          const auto& in = in_view.data;
          if (in.size() < sizeof(int64_t)) {
            continue;
          }
          total += *reinterpret_cast<const int64_t*>(in.data());
        }
        outputs[0].data.resize(sizeof(int64_t));
        *reinterpret_cast<int64_t*>(outputs[0].data.data()) = total;
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "particle_scalar_min_reduce", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        double out_v = std::numeric_limits<double>::infinity();
        bool any = false;
        for (const auto& in_view : inputs) {
          const auto& in = in_view.data;
          if (in.size() < sizeof(double)) {
            continue;
          }
          const double v = *reinterpret_cast<const double*>(in.data());
          if (!std::isfinite(v)) {
            continue;
          }
          if (!any || v < out_v) {
            out_v = v;
            any = true;
          }
        }
        outputs[0].data.resize(sizeof(double));
        *reinterpret_cast<double*>(outputs[0].data.data()) =
            any ? out_v : std::numeric_limits<double>::infinity();
        return hpx::make_ready_future();
      });
  registry.register_kernel(
      KernelDesc{.name = "particle_scalar_max_reduce", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
      [](const LevelMeta&, int32_t, std::span<const HostView> inputs, const NeighborViews&,
         std::span<HostView> outputs, std::span<const std::uint8_t>) {
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        double out_v = -std::numeric_limits<double>::infinity();
        bool any = false;
        for (const auto& in_view : inputs) {
          const auto& in = in_view.data;
          if (in.size() < sizeof(double)) {
            continue;
          }
          const double v = *reinterpret_cast<const double*>(in.data());
          if (!std::isfinite(v)) {
            continue;
          }
          if (!any || v > out_v) {
            out_v = v;
            any = true;
          }
        }
        outputs[0].data.resize(sizeof(double));
        *reinterpret_cast<double*>(outputs[0].data.data()) =
            any ? out_v : -std::numeric_limits<double>::infinity();
        return hpx::make_ready_future();
      });
  {
    struct Params {
      std::array<double, 2> range{0.0, 1.0};
      int bins = 1;
      int bytes_per_value = 4;
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* range = find_msgpack_map_value(root, "range");
          range && range->type == msgpack::type::ARRAY && range->via.array.size == 2) {
        params.range[0] = range->via.array.ptr[0].as<double>();
        params.range[1] = range->via.array.ptr[1].as<double>();
      }
      if (const auto* bins = find_msgpack_map_value(root, "bins");
          bins && (bins->type == msgpack::type::POSITIVE_INTEGER ||
                   bins->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bins = bins->as<int>();
      }
      if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
          bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                  bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bytes_per_value = bpv->as<int>();
      }
      params.covered_boxes = parse_covered_boxes_param(root);
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "histogram1d_accumulate", .n_inputs = 1, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

        if (outputs.empty() || inputs.empty() || params.bins <= 0) {
          return hpx::make_ready_future();
        }
        const double lo = params.range[0];
        const double hi = params.range[1];
        if (!std::isfinite(lo) || !std::isfinite(hi) || hi <= lo) {
          return hpx::make_ready_future();
        }

        const std::size_t out_bytes = static_cast<std::size_t>(params.bins) * sizeof(double);
        if (outputs[0].data.size() != out_bytes) {
          outputs[0].data.assign(out_bytes, 0);
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

        auto covered = [&](int ix, int iy, int iz) -> bool {
          if (!params.covered_boxes) {
            return false;
          }
          for (const auto& b : *params.covered_boxes) {
            if (covered_box_contains(b, ix, iy, iz)) {
              return true;
            }
          }
          return false;
        };
        auto in_index = [&](int i, int j, int k) -> std::size_t {
          return static_cast<std::size_t>((i * ny + j) * nz + k);
        };
        auto read_value = [&](const HostView& view, std::size_t idx) -> double {
          const auto& data = view.data;
          if (params.bytes_per_value == 4) {
            if (idx * sizeof(float) < data.size()) {
              return static_cast<double>(reinterpret_cast<const float*>(data.data())[idx]);
            }
          } else if (params.bytes_per_value == 8) {
            if (idx * sizeof(double) < data.size()) {
              return reinterpret_cast<const double*>(data.data())[idx];
            }
          }
          return 0.0;
        };

        const bool weighted = inputs.size() >= 2;
        const double inv_dx = static_cast<double>(params.bins) / (hi - lo);
        auto* out = reinterpret_cast<double*>(outputs[0].data.data());

        for (int i = 0; i < nx; ++i) {
          const int gi = box.lo.x + i;
          for (int j = 0; j < ny; ++j) {
            const int gj = box.lo.y + j;
            for (int k = 0; k < nz; ++k) {
              const int gk = box.lo.z + k;
              if (covered(gi, gj, gk)) {
                continue;
              }
              const auto idx = in_index(i, j, k);
              const double value = read_value(inputs[0], idx);
              if (!std::isfinite(value) || value < lo || value > hi) {
                continue;
              }
              int bin = 0;
              if (value == hi) {
                bin = params.bins - 1;
              } else {
                bin = static_cast<int>(std::floor((value - lo) * inv_dx));
              }
              if (bin < 0 || bin >= params.bins) {
                continue;
              }
              double weight = 1.0;
              if (weighted) {
                weight = read_value(inputs[1], idx);
                if (!std::isfinite(weight)) {
                  continue;
                }
              }
              out[static_cast<std::size_t>(bin)] += weight;
            }
          }
        }
        return hpx::make_ready_future();
        },
        make_covered_box_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      std::array<double, 2> x_range{0.0, 1.0};
      std::array<double, 2> y_range{0.0, 1.0};
      std::array<int, 2> bins{1, 1};
      int bytes_per_value = 4;
      std::string weight_mode{"input"};
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (root.type == msgpack::type::MAP) {
        if (const auto* x_range = find_msgpack_map_value(root, "x_range");
            x_range && x_range->type == msgpack::type::ARRAY && x_range->via.array.size == 2) {
          params.x_range[0] = x_range->via.array.ptr[0].as<double>();
          params.x_range[1] = x_range->via.array.ptr[1].as<double>();
        }
        if (const auto* y_range = find_msgpack_map_value(root, "y_range");
            y_range && y_range->type == msgpack::type::ARRAY && y_range->via.array.size == 2) {
          params.y_range[0] = y_range->via.array.ptr[0].as<double>();
          params.y_range[1] = y_range->via.array.ptr[1].as<double>();
        }
        if (const auto* bins = find_msgpack_map_value(root, "bins");
            bins && bins->type == msgpack::type::ARRAY && bins->via.array.size == 2) {
          params.bins[0] = bins->via.array.ptr[0].as<int>();
          params.bins[1] = bins->via.array.ptr[1].as<int>();
        }
        if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
            bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                    bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
          params.bytes_per_value = bpv->as<int>();
        }
        if (const auto* mode = find_msgpack_map_value(root, "weight_mode");
            mode && mode->type == msgpack::type::STR) {
          params.weight_mode = mode->as<std::string>();
        }
        params.covered_boxes = parse_covered_boxes_param(root);
      }
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "histogram2d_accumulate", .n_inputs = 2, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

        const int nx_bins = params.bins[0];
        const int ny_bins = params.bins[1];
        if (outputs.empty() || inputs.size() < 2 || nx_bins <= 0 || ny_bins <= 0) {
          return hpx::make_ready_future();
        }
        const double xlo = params.x_range[0];
        const double xhi = params.x_range[1];
        const double ylo = params.y_range[0];
        const double yhi = params.y_range[1];
        if (!std::isfinite(xlo) || !std::isfinite(xhi) || !std::isfinite(ylo) || !std::isfinite(yhi) ||
            xhi <= xlo || yhi <= ylo) {
          return hpx::make_ready_future();
        }

        const std::size_t out_bytes = static_cast<std::size_t>(nx_bins) *
                                      static_cast<std::size_t>(ny_bins) * sizeof(double);
        if (outputs[0].data.size() != out_bytes) {
          outputs[0].data.assign(out_bytes, 0);
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

        auto covered = [&](int ix, int iy, int iz) -> bool {
          if (!params.covered_boxes) {
            return false;
          }
          for (const auto& b : *params.covered_boxes) {
            if (covered_box_contains(b, ix, iy, iz)) {
              return true;
            }
          }
          return false;
        };
        auto in_index = [&](int i, int j, int k) -> std::size_t {
          return static_cast<std::size_t>((i * ny + j) * nz + k);
        };
        auto read_value = [&](const HostView& view, std::size_t idx) -> double {
          const auto& data = view.data;
          if (params.bytes_per_value == 4) {
            if (idx * sizeof(float) < data.size()) {
              return static_cast<double>(reinterpret_cast<const float*>(data.data())[idx]);
            }
          } else if (params.bytes_per_value == 8) {
            if (idx * sizeof(double) < data.size()) {
              return reinterpret_cast<const double*>(data.data())[idx];
            }
          }
          return 0.0;
        };

        const bool weighted = inputs.size() >= 3;
        const double cell_volume = level.geom.dx[0] * level.geom.dx[1] * level.geom.dx[2];
        const double inv_dx = static_cast<double>(nx_bins) / (xhi - xlo);
        const double inv_dy = static_cast<double>(ny_bins) / (yhi - ylo);
        auto* out = reinterpret_cast<double*>(outputs[0].data.data());

        for (int i = 0; i < nx; ++i) {
          const int gi = box.lo.x + i;
          for (int j = 0; j < ny; ++j) {
            const int gj = box.lo.y + j;
            for (int k = 0; k < nz; ++k) {
              const int gk = box.lo.z + k;
              if (covered(gi, gj, gk)) {
                continue;
              }
              const auto idx = in_index(i, j, k);
              const double x = read_value(inputs[0], idx);
              const double y = read_value(inputs[1], idx);
              if (!std::isfinite(x) || !std::isfinite(y) || x < xlo || x > xhi || y < ylo || y > yhi) {
                continue;
              }
              int ix = (x == xhi) ? (nx_bins - 1) : static_cast<int>(std::floor((x - xlo) * inv_dx));
              int iy = (y == yhi) ? (ny_bins - 1) : static_cast<int>(std::floor((y - ylo) * inv_dy));
              if (ix < 0 || ix >= nx_bins || iy < 0 || iy >= ny_bins) {
                continue;
              }
              double weight = 1.0;
              if (weighted) {
                weight = read_value(inputs[2], idx);
                if (!std::isfinite(weight)) {
                  continue;
                }
              } else if (params.weight_mode == "cell_mass") {
                weight = x * cell_volume;
                if (!std::isfinite(weight)) {
                  continue;
                }
              } else if (params.weight_mode == "cell_volume") {
                weight = cell_volume;
              }
              out[static_cast<std::size_t>(iy) * static_cast<std::size_t>(nx_bins) +
                  static_cast<std::size_t>(ix)] += weight;
            }
          }
        }
        return hpx::make_ready_future();
        },
        make_covered_box_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      double radius = 0.0;
      double gamma = 5.0 / 3.0;
      int bytes_per_value = 8;
      std::shared_ptr<const CoveredBoxListIR> covered_boxes;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* radius = find_msgpack_map_value(root, "radius");
          radius && (radius->type == msgpack::type::FLOAT ||
                     radius->type == msgpack::type::POSITIVE_INTEGER ||
                     radius->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.radius = radius->as<double>();
      }
      if (const auto* gamma = find_msgpack_map_value(root, "gamma");
          gamma && (gamma->type == msgpack::type::FLOAT ||
                    gamma->type == msgpack::type::POSITIVE_INTEGER ||
                    gamma->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.gamma = gamma->as<double>();
      }
      if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
          bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                  bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bytes_per_value = bpv->as<int>();
      }
      params.covered_boxes = parse_covered_boxes_param(root);
      return params;
    };

    registry.register_kernel(
        KernelDesc{.name = "flux_surface_integral_accumulate", .n_inputs = 9, .n_outputs = 1,
                   .needs_neighbors = false},
        [decode_params](const LevelMeta& level, int32_t block, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

        const std::size_t out_bytes = 4 * sizeof(double);
        if (outputs.empty()) {
          return hpx::make_ready_future();
        }
        if (outputs[0].data.size() != out_bytes) {
          outputs[0].data.assign(out_bytes, 0);
        } else {
          std::fill(outputs[0].data.begin(), outputs[0].data.end(), 0);
        }
        if (inputs.size() < 9 || !std::isfinite(params.radius) || params.radius <= 0.0 ||
            !std::isfinite(params.gamma) || params.gamma <= 1.0) {
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

        auto in_index = [&](int i, int j, int k) -> std::size_t {
          return static_cast<std::size_t>((i * ny + j) * nz + k);
        };
        std::vector<std::uint8_t> covered_mask;
        if (params.covered_boxes && !params.covered_boxes->empty()) {
          const std::size_t block_cells = static_cast<std::size_t>(nx) *
                                          static_cast<std::size_t>(ny) *
                                          static_cast<std::size_t>(nz);
          for (const auto& b : *params.covered_boxes) {
            const int gx0 = std::max(box.lo.x, b.lo[0]);
            const int gy0 = std::max(box.lo.y, b.lo[1]);
            const int gz0 = std::max(box.lo.z, b.lo[2]);
            const int gx1 = std::min(box.hi.x, b.hi[0]);
            const int gy1 = std::min(box.hi.y, b.hi[1]);
            const int gz1 = std::min(box.hi.z, b.hi[2]);
            if (gx0 > gx1 || gy0 > gy1 || gz0 > gz1) {
              continue;
            }
            if (covered_mask.empty()) {
              covered_mask.assign(block_cells, 0);
            }
            for (int gx = gx0; gx <= gx1; ++gx) {
              const int i = gx - box.lo.x;
              for (int gy = gy0; gy <= gy1; ++gy) {
                const int j = gy - box.lo.y;
                for (int gz = gz0; gz <= gz1; ++gz) {
                  const int k = gz - box.lo.z;
                  covered_mask[in_index(i, j, k)] = 1;
                }
              }
            }
          }
        }
        auto read_value = [&](const HostView& view, std::size_t idx) -> double {
          const auto& data = view.data;
          if (params.bytes_per_value == 4) {
            if ((idx + 1) * sizeof(float) <= data.size()) {
              return static_cast<double>(reinterpret_cast<const float*>(data.data())[idx]);
            }
          } else if (params.bytes_per_value == 8) {
            if ((idx + 1) * sizeof(double) <= data.size()) {
              return reinterpret_cast<const double*>(data.data())[idx];
            }
          }
          return 0.0;
        };
        auto cell_edge = [&](int axis, int global_idx) -> double {
          return level.geom.x0[axis] +
                 (static_cast<double>(global_idx - level.geom.index_origin[axis]) *
                  level.geom.dx[axis]);
        };

        auto* out = reinterpret_cast<double*>(outputs[0].data.data());
        const double gamma_minus_one = params.gamma - 1.0;
        const double radius2 = params.radius * params.radius;

        for (int i = 0; i < nx; ++i) {
          const int gi = box.lo.x + i;
          const double x0 = cell_edge(0, gi);
          const double x1 = x0 + level.geom.dx[0];
          const double x = 0.5 * (x0 + x1);
          for (int j = 0; j < ny; ++j) {
            const int gj = box.lo.y + j;
            const double y0 = cell_edge(1, gj);
            const double y1 = y0 + level.geom.dx[1];
            const double y = 0.5 * (y0 + y1);
            for (int k = 0; k < nz; ++k) {
              const int gk = box.lo.z + k;
              const double z0 = cell_edge(2, gk);
              const double z1 = z0 + level.geom.dx[2];
              if (!sphere_may_intersect_cell(radius2, x0, x1, y0, y1, z0, z1)) {
                continue;
              }

              const auto idx = in_index(i, j, k);
              if (!covered_mask.empty() && covered_mask[idx] != 0) {
                continue;
              }

              const double area = spherical_section_area_in_intersecting_cell(
                  params.radius, x0, x1, y0, y1, z0, z1);
              if (area <= 0.0) {
                continue;
              }

              const double z = 0.5 * (z0 + z1);
              const double r = std::sqrt(x * x + y * y + z * z);

              const double rho = read_value(inputs[0], idx);
              if (r <= 0.0 || rho <= 0.0 || !std::isfinite(rho)) {
                continue;
              }

              const double momx = read_value(inputs[1], idx);
              const double momy = read_value(inputs[2], idx);
              const double momz = read_value(inputs[3], idx);
              const double energy_density = read_value(inputs[4], idx);
              const double scalar_density = read_value(inputs[5], idx);
              const double bx = read_value(inputs[6], idx);
              const double by = read_value(inputs[7], idx);
              const double bz = read_value(inputs[8], idx);
              if (!std::isfinite(momx) || !std::isfinite(momy) || !std::isfinite(momz) ||
                  !std::isfinite(energy_density) || !std::isfinite(scalar_density) ||
                  !std::isfinite(bx) || !std::isfinite(by) || !std::isfinite(bz)) {
                continue;
              }

              const double vx = momx / rho;
              const double vy = momy / rho;
              const double vz = momz / rho;
              const double vr = (x * momx + y * momy + z * momz) / (rho * r);
              const double rhat_x = x / r;
              const double rhat_y = y / r;
              const double rhat_z = z / r;

              const double kinetic =
                  0.5 * (momx * momx + momy * momy + momz * momz) / rho;
              const double emag = 0.5 * (bx * bx + by * by + bz * bz);
              const double ehydro = energy_density - emag;
              const double pgas = gamma_minus_one * (ehydro - kinetic);
              if (!std::isfinite(vr) || !std::isfinite(pgas)) {
                continue;
              }

              const double bdotv = vx * bx + vy * by + vz * bz;
              const double br = rhat_x * bx + rhat_y * by + rhat_z * bz;
              const double mass_flux_density = rho * vr;
              const double hydro_energy_flux_density = (ehydro + pgas) * vr;
              const double mhd_energy_flux_density =
                  (energy_density + pgas + emag) * vr - bdotv * br;

              out[0] += mass_flux_density * area;
              out[1] += hydro_energy_flux_density * area;
              out[2] += mhd_energy_flux_density * area;
              out[3] += (scalar_density * vr) * area;
            }
          }
        }
        return hpx::make_ready_future();
        },
        make_covered_box_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      int bytes_per_value = 8;
    };
    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
          bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                  bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bytes_per_value = bpv->as<int>();
      }
      return params;
    };
    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_add", .n_inputs = 2, .n_outputs = 1, .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

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
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      int bytes_per_value = 4;
      double pixel_area = 1.0;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
          bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                  bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bytes_per_value = bpv->as<int>();
      }
      if (const auto* area = find_msgpack_map_value(root, "pixel_area");
          area && (area->type == msgpack::type::FLOAT ||
                   area->type == msgpack::type::POSITIVE_INTEGER ||
                   area->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.pixel_area = area->as<double>();
      }
      return params;
    };
    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_finalize", .n_inputs = 2, .n_outputs = 1, .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

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
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
  {
    struct Params {
      int bytes_per_value = 4;
    };

    auto decode_params = [](const msgpack::object& root) {
      Params params;
      if (const auto* bpv = find_msgpack_map_value(root, "bytes_per_value");
          bpv && (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                  bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
        params.bytes_per_value = bpv->as<int>();
      }
      return params;
    };
    registry.register_kernel(
        KernelDesc{.name = "uniform_slice_reduce", .n_inputs = 1, .n_outputs = 1, .needs_neighbors = false},
        [decode_params](const LevelMeta&, int32_t, std::span<const HostView> inputs,
                        const NeighborViews&, std::span<HostView> outputs,
                        std::span<const std::uint8_t> params_msgpack) {
        const auto& params = decode_params_cached<Params>(params_msgpack, decode_params);

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
        },
        make_kernel_params_preparer<Params>(decode_params));
  }
}

std::shared_ptr<ExecutionContext> build_execution_context_impl(int32_t run_id,
                                                               const RunMeta& meta,
                                                               const DatasetHandle& dataset,
                                                               const PlanIR& plan,
                                                               std::shared_ptr<ChunkStore> chunk_store) {
  PlanIR prepared = plan;
  prepare_plan(prepared, global_kernels());

  bool needs_adjacency = false;
  for (const auto& stage : prepared.stages) {
    for (const auto& tmpl : stage.templates) {
      if (tmpl.deps.kind == "FaceNeighbors") {
        needs_adjacency = true;
        break;
      }
    }
    if (needs_adjacency) {
      break;
    }
  }

  auto ctx = std::make_shared<ExecutionContext>();
  ctx->run_id = run_id;
  ctx->meta = meta;
  ctx->dataset = dataset;
  ctx->plan = std::move(prepared);
  if (needs_adjacency) {
    ctx->adjacency = std::make_shared<AdjacencyServiceLocal>(ctx->meta);
  } else {
    ctx->adjacency = std::make_shared<EmptyAdjacencyService>();
  }
  ctx->chunk_store = std::move(chunk_store);
  if (ctx->chunk_store == nullptr) {
    ctx->chunk_store = std::make_shared<ChunkStore>();
  }
  return ctx;
}

bool domain_contains_block(const DomainIR& domain, const RunMeta& meta, const ChunkRef& ref) {
  if (domain.step != ref.step || domain.level != ref.level) {
    return false;
  }
  if (domain.blocks.has_value()) {
    const auto& blocks = domain.blocks.value();
    return std::find(blocks.begin(), blocks.end(), ref.block) != blocks.end();
  }
  if (ref.step < 0 || static_cast<std::size_t>(ref.step) >= meta.steps.size()) {
    return false;
  }
  const auto& step = meta.steps.at(static_cast<std::size_t>(ref.step));
  if (ref.level < 0 || static_cast<std::size_t>(ref.level) >= step.levels.size()) {
    return false;
  }
  const auto& level = step.levels.at(static_cast<std::size_t>(ref.level));
  return ref.block >= 0 && static_cast<std::size_t>(ref.block) < level.boxes.size();
}

bool graph_template_contains_output_block(const TaskTemplateIR& tmpl, const ChunkRef& ref) {
  if (!tmpl.graph_reduce.has_value()) {
    return false;
  }
  const auto& params = tmpl.graph_reduce.value();
  if (!params.output_blocks.empty()) {
    return std::find(params.output_blocks.begin(), params.output_blocks.end(), ref.block) !=
           params.output_blocks.end();
  }
  const int32_t fan_in = std::max(1, params.fan_in);
  const int32_t num_inputs = std::max(0, params.num_inputs);
  const int32_t n_groups = (num_inputs + fan_in - 1) / fan_in;
  return ref.block >= params.output_base && ref.block < params.output_base + n_groups;
}

bool template_may_produce_chunk(const TaskTemplateIR& tmpl,
                                const RunMeta& meta,
                                const ChunkRef& ref) {
  if (tmpl.domain.step != ref.step || tmpl.domain.level != ref.level) {
    return false;
  }
  const bool block_matches =
      tmpl.plane == ExecPlane::Graph ? graph_template_contains_output_block(tmpl, ref)
                                     : domain_contains_block(tmpl.domain, meta, ref);
  if (!block_matches) {
    return false;
  }
  for (const auto& output : tmpl.outputs) {
    if (output.field == ref.field && output.version == ref.version) {
      return true;
    }
  }
  return false;
}

bool context_may_produce_chunk(const ExecutionContext& ctx, const ChunkRef& ref) {
  for (const auto& stage : ctx.plan.stages) {
    for (const auto& tmpl : stage.templates) {
      if (template_may_produce_chunk(tmpl, ctx.meta, ref)) {
        return true;
      }
    }
  }
  return false;
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

void set_global_kernel_registry(KernelRegistry* registry) {
  g_kernel_registry = registry;
}

KernelRegistry& global_kernels() {
  if (!g_kernel_registry) {
    throw std::runtime_error("global KernelRegistry not initialized");
  }
  return *g_kernel_registry;
}

void set_execution_context(int32_t run_id,
                           const RunMeta& meta,
                           const DatasetHandle& dataset,
                           const PlanIR& plan) {
  std::shared_ptr<ChunkStore> chunk_store;
  {
    std::lock_guard<std::mutex> lock(g_ctx_mutex);
    auto it = g_execution_contexts.find(run_id);
    if (it != g_execution_contexts.end()) {
      chunk_store = it->second->chunk_store;
    }
  }
  auto ctx = build_execution_context_impl(run_id, meta, dataset, plan, std::move(chunk_store));
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_execution_contexts[run_id] = std::move(ctx);
}

std::shared_ptr<ExecutionContext> execution_context_shared(int32_t run_id) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  auto it = g_execution_contexts.find(run_id);
  if (it == g_execution_contexts.end()) {
    throw std::runtime_error("execution context not initialized for run id");
  }
  return it->second;
}

const ExecutionContext& execution_context(int32_t run_id) {
  return *execution_context_shared(run_id);
}

bool execution_context_may_produce_chunk(int32_t run_id, const ChunkRef& ref) {
  return context_may_produce_chunk(*execution_context_shared(run_id), ref);
}

void erase_execution_context(int32_t run_id) {
  std::lock_guard<std::mutex> lock(g_ctx_mutex);
  g_execution_contexts.erase(run_id);
}

ScopedExecutionContext::ScopedExecutionContext(int32_t run_id)
    : previous_run_id_(g_current_execution_context_id) {
  g_current_execution_context_id = run_id;
}

ScopedExecutionContext::~ScopedExecutionContext() {
  g_current_execution_context_id = previous_run_id_;
}

const RunMeta& current_runmeta() {
  if (g_current_execution_context_id != 0) {
    return execution_context(g_current_execution_context_id).meta;
  }
  throw std::runtime_error("current RunMeta requires an active execution context");
}

const DatasetHandle& current_dataset() {
  if (g_current_execution_context_id != 0) {
    return execution_context(g_current_execution_context_id).dataset;
  }
  throw std::runtime_error("current DatasetHandle requires an active execution context");
}

void set_execution_context_action(int32_t run_id,
                                  const RunMeta& meta,
                                  const DatasetHandle& dataset,
                                  const PlanIR& plan) {
  set_execution_context(run_id, meta, dataset, plan);
}

void erase_execution_context_action(int32_t run_id) {
  erase_execution_context(run_id);
}

void set_event_log_action(const std::string& path) {
  set_event_log_path(path);
}

void set_perfetto_trace_action(const std::string& path) {
  set_perfetto_trace_path(path);
}

void set_perfetto_metrics_sampling_active_action(bool active) {
  g_locality_hint.store(static_cast<int32_t>(hpx::get_locality_id()), std::memory_order_release);
  if (active) {
    g_perfetto_metrics_active_runs.fetch_add(1, std::memory_order_acq_rel);
  } else {
    const int32_t prev = g_perfetto_metrics_active_runs.fetch_sub(1, std::memory_order_acq_rel);
    if (prev <= 0) {
      g_perfetto_metrics_active_runs.store(0, std::memory_order_release);
    }
  }
}

void preload_action(int32_t run_id,
                    const std::vector<int32_t>& fields) {
  auto ctx = execution_context_shared(run_id);
  DataServiceLocal::preload(ctx->meta, ctx->dataset, ctx->chunk_store, fields);
}

void release_console_worker_action() {
  std::shared_ptr<hpx::promise<void>> promise;
  {
    std::lock_guard<std::mutex> lock(g_console_release_mutex);
    ensure_console_release_future_locked();
    if (g_console_release_requested) {
      return;
    }
    g_console_release_requested = true;
    promise = g_console_release_promise;
  }
  promise->set_value();
}

}  // namespace kangaroo

HPX_PLAIN_ACTION(kangaroo::set_execution_context_action, kangaroo_set_execution_context_action)
HPX_PLAIN_ACTION(kangaroo::erase_execution_context_action, kangaroo_erase_execution_context_action)
HPX_PLAIN_ACTION(kangaroo::preload_action, kangaroo_preload_action)
HPX_PLAIN_ACTION(kangaroo::set_event_log_action, kangaroo_set_event_log_action)
HPX_PLAIN_ACTION(kangaroo::set_perfetto_trace_action, kangaroo_set_perfetto_trace_action)
HPX_PLAIN_ACTION(kangaroo::set_perfetto_metrics_sampling_active_action,
                 kangaroo_set_perfetto_metrics_sampling_active_action)
HPX_PLAIN_ACTION(kangaroo::release_console_worker_action, kangaroo_release_console_worker_action)

namespace kangaroo {

Runtime::Runtime() {
  init_event_log_from_env();
  init_perfetto_trace_from_env();
  set_global_kernel_registry(&kernel_registry_);
  register_default_kernels(kernel_registry_);
}

Runtime::Runtime(const std::vector<std::string>& hpx_config,
                 const std::vector<std::string>& hpx_cmdline) {
  init_event_log_from_env();
  init_perfetto_trace_from_env();
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
  ::kangaroo::set_event_log_path(path);
}

void Runtime::set_perfetto_trace_path(const std::string& path) {
  ::kangaroo::set_perfetto_trace_path(path);
}

void set_event_log_path(const std::string& path) {
  {
    std::lock_guard<std::mutex> lock(g_event_log_mutex);
    g_event_log_path = path;
    g_event_log_enabled = !path.empty();
  }
  g_event_log_worker.set_path(event_log_path_for_locality(g_event_log_path));
}

bool has_event_log() {
  return g_event_log_enabled && !g_event_log_path.empty();
}

void set_perfetto_trace_path(const std::string& path) {
  {
    std::lock_guard<std::mutex> lock(g_perfetto_trace_mutex);
    g_perfetto_trace_path = path;
    g_perfetto_trace_enabled = !path.empty();
  }
  g_perfetto_trace_worker.set_path(g_perfetto_trace_path);
  g_perfetto_metrics_sampler.set_enabled(!path.empty());
}

bool has_perfetto_trace() {
  return g_perfetto_trace_enabled && !g_perfetto_trace_path.empty();
}

void log_task_event(const TaskEvent& event) {
  const bool json_enabled = has_event_log();
  const bool perfetto_enabled = has_perfetto_trace();
  if (!json_enabled && !perfetto_enabled) {
    return;
  }
  if (json_enabled) {
    g_event_log_worker.enqueue(event);
  }
  if (perfetto_enabled) {
    g_perfetto_trace_worker.enqueue_task(event);
  }
}

void log_phase_event(const PhaseEvent& event) {
  const bool json_enabled = has_event_log();
  const bool perfetto_enabled = has_perfetto_trace();
  if (!json_enabled && !perfetto_enabled) {
    return;
  }
  if (json_enabled) {
    g_event_log_worker.enqueue(event);
  }
  if (perfetto_enabled) {
    g_perfetto_trace_worker.enqueue_phase(event);
  }
}

void log_data_event(const DataEvent& event) {
  if (!has_event_log()) {
    return;
  }
  g_event_log_worker.enqueue(event);
}

void Runtime::run_packed_plan(const std::vector<std::uint8_t>& packed,
                              const RunMetaHandle& runmeta,
                              const DatasetHandle& dataset) {
  ensure_hpx_started();
  g_locality_hint.store(static_cast<int32_t>(hpx::get_locality_id()), std::memory_order_release);
  ActivePlanRunGuard active_run_guard;

  auto now_seconds = []() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
  };
  auto emit_phase_span = [&](std::string_view name,
                             std::string_view category,
                             double start,
                             double end,
                             const std::string& worker_label = "runtime-main") {
    const int32_t locality = std::max(0, g_locality_hint.load(std::memory_order_acquire));
    PhaseEvent start_event;
    start_event.name = std::string(name);
    start_event.category = std::string(category);
    start_event.status = "start";
    start_event.locality = locality;
    start_event.worker = -1;
    start_event.worker_label = worker_label;
    start_event.ts = start;
    start_event.start = start;
    start_event.end = start;
    log_phase_event(start_event);

    PhaseEvent end_event = start_event;
    end_event.status = "end";
    end_event.ts = end;
    end_event.start = start;
    end_event.end = end;
    log_phase_event(end_event);
  };
  auto timed_phase = [&](std::string_view name, std::string_view category, auto&& fn) {
    const double start = now_seconds();
    decltype(auto) result = fn();
    const double end = now_seconds();
    emit_phase_span(name, category, start, end);
    return result;
  };

  PlanIR plan = timed_phase("runtime_decode_plan_msgpack", "kangaroo.runtime.setup", [&]() {
    return decode_plan_msgpack(std::span<const std::uint8_t>(packed.data(), packed.size()));
  });

  auto localities = hpx::find_all_localities();
  if (has_event_log()) {
    timed_phase("runtime_broadcast_event_log", "kangaroo.runtime.setup", [&]() {
      hpx::lcos::broadcast<::kangaroo_set_event_log_action>(localities, event_log_path()).get();
      return 0;
    });
  }
  if (has_perfetto_trace()) {
    timed_phase("runtime_broadcast_perfetto_trace", "kangaroo.runtime.setup", [&]() {
      hpx::lcos::broadcast<::kangaroo_set_perfetto_trace_action>(localities, perfetto_trace_path()).get();
      return 0;
    });
    timed_phase("runtime_enable_perfetto_metrics", "kangaroo.runtime.setup", [&]() {
      hpx::lcos::broadcast<::kangaroo_set_perfetto_metrics_sampling_active_action>(localities, true).get();
      return 0;
    });
  }
  const bool reused_preload_context = preload_run_id_ != 0;
  int32_t plan_id = reused_preload_context ? preload_run_id_ : next_plan_id_++;
  preload_run_id_ = 0;
  timed_phase("runtime_broadcast_execution_context", "kangaroo.runtime.setup", [&]() {
    hpx::lcos::broadcast<::kangaroo_set_execution_context_action>(
        localities, plan_id, runmeta.meta, dataset, plan)
        .get();
    return 0;
  });
  auto ctx = execution_context_shared(plan_id);

  auto erase_context = [&](int32_t run_id) {
    timed_phase("runtime_erase_execution_context", "kangaroo.runtime.cleanup", [&]() {
      hpx::lcos::broadcast<::kangaroo_erase_execution_context_action>(localities, run_id).get();
      return 0;
    });
  };
  auto stop_perfetto_sampling = [&]() {
    if (has_perfetto_trace()) {
      timed_phase("runtime_disable_perfetto_metrics", "kangaroo.runtime.cleanup", [&]() {
        hpx::lcos::broadcast<::kangaroo_set_perfetto_metrics_sampling_active_action>(localities, false)
            .get();
        return 0;
      });
    }
  };

  try {
    DataServiceLocal data(plan_id, &dataset);
    Executor executor(plan_id, ctx->meta, data, *ctx->adjacency);
    timed_phase("runtime_execute_plan", "kangaroo.runtime.execute", [&]() {
      executor.run(ctx->plan).get();
      return 0;
    });
  } catch (...) {
    try {
      erase_context(plan_id);
    } catch (...) {
    }
    try {
      stop_perfetto_sampling();
    } catch (...) {
    }
    throw;
  }

  retained_output_run_id_ = plan_id;
  if (std::find(retained_output_run_ids_.begin(), retained_output_run_ids_.end(), plan_id) ==
      retained_output_run_ids_.end()) {
    retained_output_run_ids_.push_back(plan_id);
  }
  stop_perfetto_sampling();
}

void Runtime::preload_dataset(const RunMetaHandle& runmeta,
                              const DatasetHandle& dataset,
                              const std::vector<int32_t>& fields) {
  ensure_hpx_started();
  auto localities = hpx::find_all_localities();
  auto erase_context = [&](int32_t run_id) {
    hpx::lcos::broadcast<::kangaroo_erase_execution_context_action>(localities, run_id).get();
  };
  if (preload_run_id_ != 0) {
    const int32_t old_preload_run_id = preload_run_id_;
    erase_context(preload_run_id_);
    retained_output_run_ids_.erase(
        std::remove(retained_output_run_ids_.begin(), retained_output_run_ids_.end(),
                    old_preload_run_id),
        retained_output_run_ids_.end());
    preload_run_id_ = 0;
  }
  preload_run_id_ = next_plan_id_++;
  hpx::lcos::broadcast<::kangaroo_set_execution_context_action>(
      localities, preload_run_id_, runmeta.meta, dataset, PlanIR{})
      .get();
  hpx::lcos::broadcast<::kangaroo_preload_action>(localities, preload_run_id_, fields).get();
}

HostView Runtime::get_task_chunk(int32_t step,
                                 int16_t level,
                                 int32_t field,
                                 int32_t version,
                                 int32_t block,
                                 const DatasetHandle* dataset) {
  ensure_hpx_started();
  if (g_active_plan_runs.load(std::memory_order_acquire) > 0) {
    throw std::runtime_error(
        "output retrieval is not allowed while a plan run is in progress");
  }
  if (retained_output_run_id_ == 0 && preload_run_id_ == 0 && dataset == nullptr) {
    throw std::runtime_error("no retained execution context or dataset available for get_task_chunk");
  }
  ChunkRef ref{step, level, field, version, block};

  std::vector<int32_t> candidate_run_ids;
  candidate_run_ids.reserve(retained_output_run_ids_.size() + 2);
  if (retained_output_run_id_ != 0) {
    candidate_run_ids.push_back(retained_output_run_id_);
  }
  for (auto it = retained_output_run_ids_.rbegin(); it != retained_output_run_ids_.rend(); ++it) {
    if (*it != retained_output_run_id_) {
      candidate_run_ids.push_back(*it);
    }
  }
  if (preload_run_id_ != 0) {
    candidate_run_ids.push_back(preload_run_id_);
  }

  for (int32_t run_id : candidate_run_ids) {
    std::shared_ptr<ExecutionContext> ctx;
    try {
      ctx = execution_context_shared(run_id);
    } catch (...) {
      continue;
    }
    if (context_may_produce_chunk(*ctx, ref) || ctx->dataset.has_chunk(ref)) {
      DataServiceLocal data(run_id, dataset);
      return data.get_host(ref).get();
    }
  }

  if (dataset != nullptr && dataset->has_chunk(ref)) {
    DataServiceLocal data(0, dataset);
    return data.get_host(ref).get();
  }

  throw std::runtime_error("chunk is not available from retained outputs or dataset");
}

int32_t Runtime::locality_id() {
  ensure_hpx_started();
  return static_cast<int32_t>(hpx::get_locality_id());
}

int32_t Runtime::num_localities() {
  ensure_hpx_started();
  return static_cast<int32_t>(hpx::find_all_localities().size());
}

int32_t Runtime::chunk_home_rank(int32_t step, int16_t level, int32_t block) {
  ensure_hpx_started();
  DataServiceLocal data;
  return static_cast<int32_t>(data.home_rank(ChunkRef{step, level, 0, 0, block}));
}

void Runtime::wait_for_console_release() {
  ensure_hpx_started();
  if (hpx::get_locality_id() == 0) {
    return;
  }
  hpx::shared_future<void> release_future;
  {
    std::lock_guard<std::mutex> lock(g_console_release_mutex);
    release_future = ensure_console_release_future_locked();
    if (g_console_release_requested) {
      return;
    }
  }
  release_future.wait();
}

void Runtime::release_console_workers() {
  ensure_hpx_started();
  std::shared_ptr<hpx::promise<void>> promise;
  {
    std::lock_guard<std::mutex> lock(g_console_release_mutex);
    ensure_console_release_future_locked();
    if (!g_console_release_requested) {
      g_console_release_requested = true;
      promise = g_console_release_promise;
    }
  }
  if (promise) {
    promise->set_value();
  }

  auto localities = hpx::find_all_localities();
  if (localities.size() <= 1) {
    return;
  }
  hpx::lcos::broadcast<::kangaroo_release_console_worker_action>(localities).get();
}

}  // namespace kangaroo
