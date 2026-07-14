#include "kangaroo/runtime.hpp"

#include "kangaroo/default_kernels.hpp"
#include "kangaroo/plan_decode.hpp"
#include "perfetto_trace_minimal.pb.h"

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
#include <tuple>
#include <variant>
#include <unordered_map>
#include <vector>
#include <unistd.h>
#include <sys/resource.h>

#include <hpx/include/actions.hpp>
#include <hpx/collectives/broadcast_direct.hpp>
#include <hpx/hpx_start.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/resource_partitioner.hpp>
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
std::atomic<int32_t> g_next_runtime_run_id{1};
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

int32_t allocate_runtime_run_id() {
  return g_next_runtime_run_id.fetch_add(1, std::memory_order_acq_rel);
}


int positive_env_int(const char* name) {
  const char* env = std::getenv(name);
  if (env == nullptr || *env == '\0') {
    return 0;
  }
  try {
    const int parsed = std::stoi(env);
    if (parsed > 0) {
      return parsed;
    }
  } catch (...) {
  }
  return 0;
}

std::string plotfile_io_pool_name() {
  const char* env = std::getenv("KANGAROO_PLOTFILE_IO_POOL");
  if (env != nullptr && *env != '\0') {
    return env;
  }
  return "plotfile_io";
}

int plotfile_io_pool_threads() {
  return positive_env_int("KANGAROO_PLOTFILE_IO_POOL_THREADS");
}


void configure_plotfile_io_pool(hpx::resource::partitioner& rp) {
  const std::string pool_name = plotfile_io_pool_name();
  const int requested_threads = plotfile_io_pool_threads();
  if (pool_name.empty() || requested_threads <= 0) {
    return;
  }

  std::vector<hpx::resource::pu> pus;
  for (const auto& domain : rp.numa_domains()) {
    for (const auto& core : domain.cores()) {
      for (const auto& pu : core.pus()) {
        pus.push_back(pu);
      }
    }
  }
  if (pus.empty()) {
    return;
  }

  try {
    rp.create_thread_pool(pool_name);
  } catch (...) {
  }
  for (int i = 0; i < requested_threads; ++i) {
    rp.add_resource(pus[static_cast<std::size_t>(i) % pus.size()], pool_name, false);
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
        if (!event.file.empty()) {
          out << ',';
          write_string("file", event.file);
        }
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
        if (event.estimated_bytes > 0) {
          out << ",\"estimated_bytes\":" << event.estimated_bytes;
        }
        if (event.file_offset >= 0) {
          out << ",\"file_offset\":" << event.file_offset;
        }
        if (event.comp_start >= 0) {
          out << ",\"comp_start\":" << event.comp_start;
        }
        if (event.comp_count >= 0) {
          out << ",\"comp_count\":" << event.comp_count;
        }
        if (event.queue_depth >= 0) {
          out << ",\"queue_depth\":" << event.queue_depth;
        }
        if (event.in_flight >= 0) {
          out << ",\"in_flight\":" << event.in_flight;
        }
        if (event.concurrency >= 0) {
          out << ",\"concurrency\":" << event.concurrency;
        }
        if (event.in_flight_bytes >= 0) {
          out << ",\"in_flight_bytes\":" << event.in_flight_bytes;
        }
        if (event.byte_limit >= 0) {
          out << ",\"byte_limit\":" << event.byte_limit;
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
  } else {
    std::lock_guard<ChunkStore::Mutex> lock(ctx->chunk_store->mutex);
    ctx->chunk_store->consumer_counts.clear();
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
    if (output.field.field == ref.field && output.field.version == ref.version) {
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
    if (plotfile_io_pool_threads() > 0) {
      params.rp_mode = static_cast<hpx::resource::partitioner_mode>(
          static_cast<int>(hpx::resource::partitioner_mode::allow_dynamic_pools) |
          static_cast<int>(hpx::resource::partitioner_mode::allow_oversubscription));
      params.rp_callback =
          [](hpx::resource::partitioner& rp,
             hpx::program_options::variables_map const&) { configure_plotfile_io_pool(rp); };
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

void DatasetHandle::set_chunk(const ChunkRef& ref, ChunkBuffer view) {
  if (!backend) {
    uri = "memory://runtime";
    backend = make_dataset_backend(uri);
  }
  backend->set_chunk(ref, std::move(view));
}

std::optional<ChunkBuffer> DatasetHandle::get_chunk(const ChunkRef& ref) const {
  if (backend) {
    return backend->get_chunk(ref);
  }
  return std::nullopt;
}

std::vector<std::optional<ChunkBuffer>> DatasetHandle::get_chunks(
    const std::vector<ChunkRef>& refs) const {
  if (backend) {
    return backend->get_chunks(refs);
  }
  return std::vector<std::optional<ChunkBuffer>>(refs.size());
}

bool DatasetHandle::has_chunk(const ChunkRef& ref) const {
  if (backend) {
    return backend->has_chunk(ref);
  }
  return false;
}

std::size_t DatasetHandle::estimate_chunk_bytes(const ChunkRef& ref) const {
  if (backend) {
    return backend->estimate_chunk_bytes(ref);
  }
  return 0;
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
  timed_phase("runtime_validate_plan_output_bounds", "kangaroo.runtime.setup", [&]() {
    validate_plan_output_bounds(plan, global_kernels());
    return 0;
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
  int32_t plan_id = reused_preload_context ? preload_run_id_ : allocate_runtime_run_id();
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
  preload_run_id_ = allocate_runtime_run_id();
  hpx::lcos::broadcast<::kangaroo_set_execution_context_action>(
      localities, preload_run_id_, runmeta.meta, dataset, PlanIR{})
      .get();
  hpx::lcos::broadcast<::kangaroo_preload_action>(localities, preload_run_id_, fields).get();
}

ChunkBuffer Runtime::get_task_chunk(int32_t step,
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
