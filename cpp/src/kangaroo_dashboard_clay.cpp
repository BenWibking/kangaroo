#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_ttf/SDL_ttf.h>

#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#define CLAY_IMPLEMENTATION
#include "../../extern/clay/clay.h"

using json = nlohmann::json;

namespace {

constexpr Uint32 kFontId = 0;

const Clay_Color kBg = Clay_Color{14, 18, 24, 255};
const Clay_Color kPanel = Clay_Color{25, 32, 42, 255};
const Clay_Color kPanelAlt = Clay_Color{30, 38, 50, 255};
const Clay_Color kText = Clay_Color{233, 237, 242, 255};
const Clay_Color kMuted = Clay_Color{146, 158, 176, 255};
const Clay_Color kAccent = Clay_Color{56, 160, 255, 255};
const Clay_Color kGood = Clay_Color{58, 194, 121, 255};
const Clay_Color kWarn = Clay_Color{255, 176, 59, 255};
const Clay_Color kBad = Clay_Color{235, 87, 87, 255};

constexpr float kTimelinePixelsPerSecond = 140.0f;
constexpr float kLaneLabelWidthPx = 150.0f;
constexpr float kLaneGapPx = 8.0f;
constexpr float kBarHeightPx = 12.0f;
constexpr float kTaskScrollPaddingPx = 8.0f;

struct DashboardConfig {
  std::optional<std::filesystem::path> metrics_path;
  std::optional<std::filesystem::path> plan_path;
  std::vector<std::string> run_command;
  int update_interval_ms = 500;
  int history_seconds = 300;
  int max_tasks = 2000;
  int threads_per_locality = 1;
  std::string title = "Kangaroo Dashboard (Clay)";
};

struct MetricPoint {
  double runtime_s = 0.0;
  double mem_used_gb = 0.0;
  double mem_total_gb = 0.0;
  double cpu_percent = 0.0;
  double io_read_mbps = 0.0;
  double io_write_mbps = 0.0;
};

struct TaskData {
  std::string id;
  std::string name;
  std::string worker;
  std::string status;
  double start = 0.0;
  double end = 0.0;
};

struct DagNode {
  std::string label;
  int depth = 0;
};

struct DagEdge {
  int src = 0;
  int dst = 0;
};

class EventLogReader {
 public:
  explicit EventLogReader(std::filesystem::path path) : path_(std::move(path)) {}

  std::vector<json> ReadEvents() {
    std::vector<json> events;
    struct stat st {};
    if (stat(path_.c_str(), &st) != 0) {
      return events;
    }
    if (!inode_ || *inode_ != st.st_ino) {
      inode_ = st.st_ino;
      offset_ = 0;
    }

    std::ifstream in(path_);
    if (!in) {
      return events;
    }
    in.seekg(offset_);
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) {
        continue;
      }
      try {
        auto payload = json::parse(line);
        if (payload.is_object()) {
          events.push_back(std::move(payload));
        }
      } catch (...) {
      }
    }
    offset_ = in.tellg();
    if (offset_ < 0) {
      offset_ = std::filesystem::file_size(path_);
    }
    return events;
  }

 private:
  std::filesystem::path path_;
  std::optional<ino_t> inode_;
  std::streamoff offset_ = 0;
};

class WorkflowProcess {
 public:
  void Start(std::vector<std::string> command, std::function<void(const std::string&)> on_line) {
    Stop();
    running_.store(true);
    exit_code_.store(-1);
    thread_ = std::thread([this, cmd = std::move(command), on_line = std::move(on_line)]() {
      const std::string cmdline = JoinShell(cmd) + " 2>&1";
      FILE* pipe = popen(cmdline.c_str(), "r");
      if (!pipe) {
        running_.store(false);
        exit_code_.store(127);
        return;
      }
      std::array<char, 4096> buf{};
      while (fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
        on_line(std::string(buf.data()));
      }
      int code = pclose(pipe);
      running_.store(false);
      exit_code_.store(code);
    });
  }

  void Stop() {
    if (thread_.joinable()) {
      thread_.join();
    }
    running_.store(false);
  }

  [[nodiscard]] bool running() const { return running_.load(); }
  [[nodiscard]] int exit_code() const { return exit_code_.load(); }

  ~WorkflowProcess() { Stop(); }

 private:
  static std::string ShellEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (char c : s) {
      if (c == '\'') {
        out += "'\\''";
      } else {
        out.push_back(c);
      }
    }
    out.push_back('\'');
    return out;
  }

  static std::string JoinShell(const std::vector<std::string>& argv) {
    std::ostringstream os;
    for (size_t i = 0; i < argv.size(); ++i) {
      if (i > 0) {
        os << ' ';
      }
      os << ShellEscape(argv[i]);
    }
    return os.str();
  }

  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<int> exit_code_{-1};
};

class MetricSampler {
 public:
  MetricPoint Sample(double runtime_s) {
    const auto now = std::chrono::steady_clock::now();
    const double dt = std::max(
        std::chrono::duration<double>(now - prev_time_).count(), 1e-6);
    prev_time_ = now;

    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);

    const double cpu_now = static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
                           static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6;
    const double cpu_percent = std::max(0.0, std::min(1000.0, 100.0 * (cpu_now - prev_cpu_s_) / dt));
    prev_cpu_s_ = cpu_now;

#if defined(__APPLE__)
    const double rss_gb = static_cast<double>(usage.ru_maxrss) / 1e9;
#else
    const double rss_gb = static_cast<double>(usage.ru_maxrss) * 1024.0 / 1e9;
#endif

    MetricPoint m;
    m.runtime_s = runtime_s;
    m.mem_used_gb = rss_gb;
    m.mem_total_gb = std::max(rss_gb, 1.0);
    m.cpu_percent = cpu_percent;
    m.io_read_mbps = 0.0;
    m.io_write_mbps = 0.0;
    return m;
  }

 private:
  std::chrono::steady_clock::time_point prev_time_ = std::chrono::steady_clock::now();
  double prev_cpu_s_ = 0.0;
};

class DashboardState {
 public:
  explicit DashboardState(DashboardConfig cfg)
      : cfg_(std::move(cfg)),
        start_wall_epoch_s_(NowEpoch()),
        start_steady_(std::chrono::steady_clock::now()) {
    if (cfg_.metrics_path) {
      reader_.emplace(*cfg_.metrics_path);
    }
  }

  void Update() {
    std::lock_guard<std::mutex> guard(mu_);
    EnsurePlanLoaded();

    bool saw_metrics = false;
    if (reader_) {
      for (const auto& ev : reader_->ReadEvents()) {
        const std::string type = ev.value("type", "");
        if (type == "task") {
          HandleTaskEvent(ev);
        } else if (type == "metrics") {
          HandleMetricEvent(ev);
          saw_metrics = true;
        }
      }
    }

    if (workflow_started_) {
      if (workflow_.running()) {
        workflow_status_ = "running";
      } else if (workflow_status_ == "running") {
        workflow_status_ = "finished";
        workflow_end_steady_ = std::chrono::steady_clock::now();
      }
    }

    if (!saw_metrics && workflow_status_ == "running") {
      PushMetric(sample_.Sample(RuntimeSecondsLocked()));
    }

    PruneHistory();
  }

  void LaunchWorkflow() {
    if (cfg_.run_command.empty() || workflow_.running()) {
      return;
    }
    workflow_started_ = true;
    workflow_status_ = "running";
    workflow_start_steady_ = std::chrono::steady_clock::now();
    workflow_end_steady_.reset();
    task_zero_epoch_s_ = NowEpoch();
    tasks_.clear();
    task_order_.clear();
    task_end_cache_.clear();
    worker_lanes_.clear();
    metrics_.clear();

    std::vector<std::string> cmd = cfg_.run_command;
    if (cfg_.threads_per_locality > 0 && !ThreadsConfigured(cmd)) {
      InjectThreads(cmd, cfg_.threads_per_locality);
    }

    if (cfg_.metrics_path) {
      setenv("KANGAROO_EVENT_LOG", cfg_.metrics_path->c_str(), 1);
    }
    if (cfg_.plan_path) {
      setenv("KANGAROO_DASHBOARD_PLAN", cfg_.plan_path->c_str(), 1);
    }

    workflow_.Start(std::move(cmd), [](const std::string& line) {
      std::cerr << "[workflow] " << line;
    });
  }

  [[nodiscard]] std::string status() const {
    std::lock_guard<std::mutex> guard(mu_);
    return workflow_status_;
  }

  void threads_plus() {
    std::lock_guard<std::mutex> guard(mu_);
    cfg_.threads_per_locality += 1;
  }

  void threads_minus() {
    std::lock_guard<std::mutex> guard(mu_);
    cfg_.threads_per_locality = std::max(1, cfg_.threads_per_locality - 1);
  }

  [[nodiscard]] int threads() const {
    std::lock_guard<std::mutex> guard(mu_);
    return cfg_.threads_per_locality;
  }

  [[nodiscard]] std::string title() const { return cfg_.title; }
  [[nodiscard]] int update_interval_ms() const { return cfg_.update_interval_ms; }

  [[nodiscard]] double runtime_s() const {
    std::lock_guard<std::mutex> guard(mu_);
    if (metrics_.empty()) {
      return RuntimeSecondsLocked();
    }
    return metrics_.back().runtime_s;
  }

  [[nodiscard]] double task_zero_epoch_s() const {
    std::lock_guard<std::mutex> guard(mu_);
    return task_zero_epoch_s_.value_or(start_wall_epoch_s_);
  }

  [[nodiscard]] std::optional<std::pair<double, double>> timeline_zoom() const {
    std::lock_guard<std::mutex> guard(mu_);
    return timeline_zoom_;
  }

  void set_timeline_zoom(double start_runtime, double end_runtime) {
    std::lock_guard<std::mutex> guard(mu_);
    const double lo = std::min(start_runtime, end_runtime);
    const double hi = std::max(start_runtime, end_runtime);
    if (hi - lo < 1e-6) {
      return;
    }
    timeline_zoom_ = std::make_pair(lo, hi);
  }

  void clear_timeline_zoom() {
    std::lock_guard<std::mutex> guard(mu_);
    timeline_zoom_.reset();
  }

  [[nodiscard]] bool timeline_show_full_runtime() const {
    std::lock_guard<std::mutex> guard(mu_);
    return timeline_show_full_runtime_;
  }

  void toggle_timeline_show_full_runtime() {
    std::lock_guard<std::mutex> guard(mu_);
    timeline_show_full_runtime_ = !timeline_show_full_runtime_;
  }

  [[nodiscard]] std::vector<MetricPoint> recent_metrics(size_t max_points = 60) const {
    std::lock_guard<std::mutex> guard(mu_);
    std::vector<MetricPoint> out;
    const size_t n = std::min(max_points, metrics_.size());
    out.reserve(n);
    for (size_t i = metrics_.size() - n; i < metrics_.size(); ++i) {
      out.push_back(metrics_[i]);
    }
    return out;
  }

  [[nodiscard]] std::vector<TaskData> recent_tasks(size_t max_tasks = 120) const {
    std::lock_guard<std::mutex> guard(mu_);
    std::vector<TaskData> out;
    const size_t n = std::min(max_tasks, task_order_.size());
    out.reserve(n);
    for (size_t i = task_order_.size() - n; i < task_order_.size(); ++i) {
      auto it = tasks_.find(task_order_[i]);
      if (it != tasks_.end()) {
        out.push_back(it->second);
      }
    }
    return out;
  }

  [[nodiscard]] std::vector<std::pair<std::string, double>> flame_totals(size_t max_rows = 30) const {
    std::lock_guard<std::mutex> guard(mu_);
    std::map<std::string, double> totals;
    const double now = NowEpoch();
    for (const auto& id : task_order_) {
      auto it = tasks_.find(id);
      if (it == tasks_.end()) {
        continue;
      }
      const auto& t = it->second;
      double end = t.end;
      if (t.status == "start" && end <= t.start) {
        end = now;
      }
      totals[t.name] += std::max(0.0, end - t.start);
    }
    std::vector<std::pair<std::string, double>> out(totals.begin(), totals.end());
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    if (out.size() > max_rows) {
      out.resize(max_rows);
    }
    return out;
  }

  [[nodiscard]] std::vector<DagNode> dag_nodes() const {
    std::lock_guard<std::mutex> guard(mu_);
    return dag_nodes_;
  }

  [[nodiscard]] std::vector<DagEdge> dag_edges() const {
    std::lock_guard<std::mutex> guard(mu_);
    return dag_edges_;
  }

 private:
  static double NowEpoch() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
  }

  double RuntimeSeconds() const {
    std::lock_guard<std::mutex> guard(mu_);
    return RuntimeSecondsLocked();
  }

  double RuntimeSecondsLocked() const {
    if (!workflow_started_ || !workflow_start_steady_) {
      return 0.0;
    }
    const auto end = workflow_end_steady_.value_or(std::chrono::steady_clock::now());
    return std::chrono::duration<double>(end - *workflow_start_steady_).count();
  }

  void PushMetric(const MetricPoint& m) {
    metrics_.push_back(m);
    const size_t max_points = std::max(
        1,
        static_cast<int>(cfg_.history_seconds * 1000 / std::max(1, cfg_.update_interval_ms)));
    if (metrics_.size() > static_cast<size_t>(max_points)) {
      metrics_.erase(metrics_.begin(), metrics_.begin() + static_cast<long>(metrics_.size() - max_points));
    }
  }

  static double NormalizeEventTime(double ts, double zero_epoch) {
    if (ts < 1.0e9) {
      return zero_epoch + ts;
    }
    return ts;
  }

  void HandleMetricEvent(const json& ev) {
    if (workflow_status_ != "running") {
      return;
    }
    MetricPoint m;
    m.runtime_s = ev.value("runtime_s", RuntimeSecondsLocked());
    m.mem_used_gb = ev.value("mem_used_gb", 0.0);
    m.mem_total_gb = ev.value("mem_total_gb", 0.0);
    m.cpu_percent = ev.value("cpu_percent", 0.0);
    m.io_read_mbps = ev.value("io_read_mbps", 0.0);
    m.io_write_mbps = ev.value("io_write_mbps", 0.0);
    PushMetric(m);
  }

  std::string FallbackTaskId(const json& ev) const {
    std::ostringstream os;
    os << ev.value("stage", "?") << ":" << ev.value("template", "?") << ":"
       << ev.value("block", "?") << ":" << ev.value("start", 0.0);
    return os.str();
  }

  void HandleTaskEvent(const json& ev) {
    std::string id = ev.value("id", "");
    if (id.empty()) {
      id = FallbackTaskId(ev);
    }
    TaskData data;
    auto it = tasks_.find(id);
    if (it != tasks_.end()) {
      data = it->second;
    }

    data.id = id;
    data.name = ev.value("name", data.name.empty() ? std::string("task") : data.name);
    data.status = ev.value("status", std::string("complete"));

    std::string worker = ev.value("worker", "");
    if (worker.empty()) {
      if (ev.contains("locality")) {
        worker = "loc-" + std::to_string(ev.value("locality", 0));
      } else {
        worker = "worker-0";
      }
    }
    data.worker = worker;

    const double zero = task_zero_epoch_s_.value_or(start_wall_epoch_s_);
    const double start = NormalizeEventTime(ev.value("start", ev.value("ts", 0.0)), zero);
    const double end = NormalizeEventTime(ev.value("end", start), zero);

    if (data.status == "start") {
      data.start = start;
      data.end = std::max(end, start);
    } else if (data.status == "end" || data.status == "error") {
      data.start = start;
      data.end = std::max(end, start);
      task_end_cache_[id] = data.end;
    } else {
      data.start = start;
      data.end = std::max(end, start);
    }

    if (tasks_.find(id) == tasks_.end()) {
      task_order_.push_back(id);
    }
    tasks_[id] = data;

    while (task_order_.size() > static_cast<size_t>(cfg_.max_tasks)) {
      tasks_.erase(task_order_.front());
      task_order_.pop_front();
    }
  }

  void PruneHistory() {
    const double window = NowEpoch() - static_cast<double>(cfg_.history_seconds);
    while (!task_order_.empty()) {
      auto it = tasks_.find(task_order_.front());
      if (it == tasks_.end()) {
        task_order_.pop_front();
        continue;
      }
      if (workflow_end_steady_) {
        break;
      }
      if (it->second.end >= window) {
        break;
      }
      tasks_.erase(task_order_.front());
      task_order_.pop_front();
    }
  }

  void EnsurePlanLoaded() {
    if (plan_loaded_ || !cfg_.plan_path || !std::filesystem::exists(*cfg_.plan_path)) {
      return;
    }
    std::ifstream in(*cfg_.plan_path);
    if (!in) {
      return;
    }
    json payload;
    try {
      in >> payload;
    } catch (...) {
      return;
    }

    if (!payload.contains("stages") || !payload["stages"].is_array()) {
      return;
    }

    const auto& stages = payload["stages"];
    std::vector<int> depth(stages.size(), 0);
    std::vector<std::vector<int>> parents(stages.size());

    for (size_t i = 0; i < stages.size(); ++i) {
      if (!stages[i].is_object()) {
        continue;
      }
      if (!stages[i].contains("after") || !stages[i]["after"].is_array()) {
        continue;
      }
      for (const auto& parent : stages[i]["after"]) {
        if (!parent.is_number_integer()) {
          continue;
        }
        const int p = parent.get<int>();
        if (p < 0 || p >= static_cast<int>(i)) {
          continue;
        }
        parents[i].push_back(p);
        depth[i] = std::max(depth[i], depth[p] + 1);
      }
    }

    dag_nodes_.clear();
    dag_edges_.clear();
    std::vector<int> stage_to_node(stages.size(), -1);

    for (size_t i = 0; i < stages.size(); ++i) {
      const auto& stage = stages[i];
      const std::string name = stage.value("name", std::string("stage-" + std::to_string(i)));
      const std::string plane = stage.value("plane", std::string(""));

      bool emitted_any = false;
      if (stage.contains("templates") && stage["templates"].is_array() && !stage["templates"].empty()) {
        for (const auto& tmpl : stage["templates"]) {
          std::string tname = tmpl.value("name", "tmpl");
          if (tmpl.contains("domain") && tmpl["domain"].is_object() &&
              tmpl["domain"].contains("blocks") && tmpl["domain"]["blocks"].is_array() &&
              !tmpl["domain"]["blocks"].empty()) {
            for (const auto& blk : tmpl["domain"]["blocks"]) {
              DagNode n;
              n.depth = depth[i];
              n.label = name + " :: " + tname + " [" + plane + "] b" + std::to_string(blk.get<int>());
              dag_nodes_.push_back(std::move(n));
              emitted_any = true;
            }
          } else {
            DagNode n;
            n.depth = depth[i];
            n.label = name + " :: " + tname + " [" + plane + "]";
            stage_to_node[i] = static_cast<int>(dag_nodes_.size());
            dag_nodes_.push_back(std::move(n));
            emitted_any = true;
          }
        }
      }

      if (!emitted_any) {
        DagNode n;
        n.depth = depth[i];
        n.label = name + " [" + plane + "]";
        stage_to_node[i] = static_cast<int>(dag_nodes_.size());
        dag_nodes_.push_back(std::move(n));
      }
    }

    for (size_t i = 0; i < parents.size(); ++i) {
      if (stage_to_node[i] < 0) {
        continue;
      }
      for (int p : parents[i]) {
        if (p >= 0 && p < static_cast<int>(stage_to_node.size()) && stage_to_node[p] >= 0) {
          dag_edges_.push_back(DagEdge{stage_to_node[p], stage_to_node[i]});
        }
      }
    }

    plan_loaded_ = true;
  }

  static bool ThreadsConfigured(const std::vector<std::string>& argv) {
    for (size_t i = 0; i < argv.size(); ++i) {
      if (argv[i].find("hpx.os_threads") != std::string::npos) {
        return true;
      }
      if (argv[i] == "--hpx-arg" && i + 1 < argv.size() &&
          argv[i + 1].find("--hpx:threads") != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  static void InjectThreads(std::vector<std::string>& argv, int threads) {
    auto it = std::find(argv.begin(), argv.end(), "--");
    const size_t pos = (it == argv.end()) ? argv.size() : static_cast<size_t>(std::distance(argv.begin(), std::next(it)));
    argv.insert(argv.begin() + static_cast<long>(pos), "--hpx-config");
    argv.insert(argv.begin() + static_cast<long>(pos + 1), "hpx.os_threads=" + std::to_string(threads));
  }

  mutable std::mutex mu_;
  DashboardConfig cfg_;
  std::optional<EventLogReader> reader_;
  MetricSampler sample_;
  WorkflowProcess workflow_;

  std::deque<MetricPoint> metrics_;
  std::unordered_map<std::string, TaskData> tasks_;
  std::deque<std::string> task_order_;
  std::unordered_map<std::string, double> task_end_cache_;
  std::unordered_map<std::string, int> worker_lanes_;

  std::vector<DagNode> dag_nodes_;
  std::vector<DagEdge> dag_edges_;

  const double start_wall_epoch_s_;
  const std::chrono::steady_clock::time_point start_steady_;
  std::optional<double> task_zero_epoch_s_;

  bool plan_loaded_ = false;
  bool workflow_started_ = false;
  std::string workflow_status_ = "idle";
  std::optional<std::chrono::steady_clock::time_point> workflow_start_steady_;
  std::optional<std::chrono::steady_clock::time_point> workflow_end_steady_;
  std::optional<std::pair<double, double>> timeline_zoom_;
  bool timeline_show_full_runtime_ = false;
};

struct UiActions {
  bool start = false;
  bool threads_plus = false;
  bool threads_minus = false;
  bool timeline_toggle_range = false;
};

typedef struct {
  SDL_Renderer* renderer;
  TTF_TextEngine* textEngine;
  TTF_Font** fonts;
} Clay_SDL3RendererData;

std::vector<std::string> g_frame_text_storage;
bool g_left_click_released_this_frame = false;
float g_render_scale_x = 1.0f;
float g_render_scale_y = 1.0f;
struct TaskTimelineViewState {
  bool valid = false;
  double view_start_runtime = 0.0;
  double view_end_runtime = 0.0;
  double pixels_per_second = kTimelinePixelsPerSecond;
} g_task_timeline_view;
float g_task_timeline_min_width_px = 0.0f;

Clay_String ToClay(const std::string& s) {
  return Clay_String{.length = static_cast<int32_t>(s.size()), .chars = s.c_str()};
}

Clay_String ToClayPersistent(std::string s) {
  g_frame_text_storage.push_back(std::move(s));
  const std::string& stored = g_frame_text_storage.back();
  return Clay_String{.length = static_cast<int32_t>(stored.size()), .chars = stored.c_str()};
}

std::string FormatDouble(double v, int prec = 1) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(prec) << v;
  return os.str();
}

double NiceTickStep(double span_s, int target_ticks = 8) {
  if (span_s <= 0.0) {
    return 1.0;
  }
  const double raw = span_s / static_cast<double>(std::max(2, target_ticks));
  const double exponent = std::floor(std::log10(raw));
  const double base = std::pow(10.0, exponent);
  const double scaled = raw / base;
  double nice = 1.0;
  if (scaled <= 1.0) {
    nice = 1.0;
  } else if (scaled <= 2.0) {
    nice = 2.0;
  } else if (scaled <= 5.0) {
    nice = 5.0;
  } else {
    nice = 10.0;
  }
  return nice * base;
}

std::string FormatRuntimeTick(double runtime_s) {
  if (runtime_s < 1.0) {
    return FormatDouble(runtime_s * 1000.0, 1) + " ms";
  }
  if (runtime_s < 10.0) {
    return FormatDouble(runtime_s, 3) + " s";
  }
  if (runtime_s < 100.0) {
    return FormatDouble(runtime_s, 2) + " s";
  }
  return FormatDouble(runtime_s, 1) + " s";
}

void HandleClayErrors(Clay_ErrorData errorData) {
  std::cerr << "[clay] " << errorData.errorText.chars << "\n";
}

void RenderClayCommands(Clay_SDL3RendererData* renderer_data, Clay_RenderCommandArray* commands) {
  const float sx = g_render_scale_x;
  const float sy = g_render_scale_y;
  for (int32_t i = 0; i < commands->length; ++i) {
    Clay_RenderCommand* cmd = Clay_RenderCommandArray_Get(commands, i);
    if (cmd == nullptr) {
      continue;
    }

    const Clay_BoundingBox bb = cmd->boundingBox;
    const SDL_FRect rect{
        .x = bb.x * sx,
        .y = bb.y * sy,
        .w = bb.width * sx,
        .h = bb.height * sy,
    };

    switch (cmd->commandType) {
      case CLAY_RENDER_COMMAND_TYPE_RECTANGLE: {
        const auto& c = cmd->renderData.rectangle.backgroundColor;
        SDL_SetRenderDrawBlendMode(renderer_data->renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer_data->renderer, static_cast<Uint8>(c.r), static_cast<Uint8>(c.g),
                               static_cast<Uint8>(c.b), static_cast<Uint8>(c.a));
        SDL_RenderFillRect(renderer_data->renderer, &rect);
        break;
      }
      case CLAY_RENDER_COMMAND_TYPE_BORDER: {
        const auto& b = cmd->renderData.border;
        SDL_SetRenderDrawBlendMode(renderer_data->renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer_data->renderer, static_cast<Uint8>(b.color.r), static_cast<Uint8>(b.color.g),
                               static_cast<Uint8>(b.color.b), static_cast<Uint8>(b.color.a));
        if (b.width.left > 0) {
          SDL_FRect line{.x = rect.x, .y = rect.y, .w = static_cast<float>(b.width.left) * sx, .h = rect.h};
          SDL_RenderFillRect(renderer_data->renderer, &line);
        }
        if (b.width.right > 0) {
          SDL_FRect line{.x = rect.x + rect.w - static_cast<float>(b.width.right) * sx,
                         .y = rect.y,
                         .w = static_cast<float>(b.width.right) * sx,
                         .h = rect.h};
          SDL_RenderFillRect(renderer_data->renderer, &line);
        }
        if (b.width.top > 0) {
          SDL_FRect line{.x = rect.x, .y = rect.y, .w = rect.w, .h = static_cast<float>(b.width.top) * sy};
          SDL_RenderFillRect(renderer_data->renderer, &line);
        }
        if (b.width.bottom > 0) {
          SDL_FRect line{.x = rect.x,
                         .y = rect.y + rect.h - static_cast<float>(b.width.bottom) * sy,
                         .w = rect.w,
                         .h = static_cast<float>(b.width.bottom) * sy};
          SDL_RenderFillRect(renderer_data->renderer, &line);
        }
        break;
      }
      case CLAY_RENDER_COMMAND_TYPE_TEXT: {
        const auto& t = cmd->renderData.text;
        TTF_Font* font = renderer_data->fonts[t.fontId];
        if (font == nullptr) {
          break;
        }
        const float font_scale = std::max(sx, sy);
        const int scaled_font_size =
            std::max(1, static_cast<int>(std::lround(static_cast<float>(t.fontSize) * font_scale)));
        TTF_SetFontSize(font, scaled_font_size);
        TTF_Text* text = TTF_CreateText(renderer_data->textEngine, font, t.stringContents.chars, t.stringContents.length);
        if (text == nullptr) {
          break;
        }
        TTF_SetTextColor(text, static_cast<Uint8>(t.textColor.r), static_cast<Uint8>(t.textColor.g),
                         static_cast<Uint8>(t.textColor.b), static_cast<Uint8>(t.textColor.a));
        TTF_DrawRendererText(text, rect.x, rect.y);
        TTF_DestroyText(text);
        break;
      }
      case CLAY_RENDER_COMMAND_TYPE_SCISSOR_START: {
        const SDL_Rect clip{
            .x = static_cast<int>(std::lround(bb.x * sx)),
            .y = static_cast<int>(std::lround(bb.y * sy)),
            .w = static_cast<int>(std::lround(bb.width * sx)),
            .h = static_cast<int>(std::lround(bb.height * sy)),
        };
        SDL_SetRenderClipRect(renderer_data->renderer, &clip);
        break;
      }
      case CLAY_RENDER_COMMAND_TYPE_SCISSOR_END: {
        SDL_SetRenderClipRect(renderer_data->renderer, nullptr);
        break;
      }
      case CLAY_RENDER_COMMAND_TYPE_IMAGE:
      case CLAY_RENDER_COMMAND_TYPE_CUSTOM:
      case CLAY_RENDER_COMMAND_TYPE_NONE:
      default:
        break;
    }
  }
}

static inline Clay_Dimensions SDLMeasureText(Clay_StringSlice text, Clay_TextElementConfig* config, void* userData) {
  auto** fonts = static_cast<TTF_Font**>(userData);
  TTF_Font* font = fonts[config->fontId];
  if (font == nullptr) {
    return Clay_Dimensions{0, 0};
  }
  int width = 0;
  int height = 0;
  const float sx = g_render_scale_x;
  const float sy = g_render_scale_y;
  const float font_scale = std::max(sx, sy);
  const int scaled_font_size =
      std::max(1, static_cast<int>(std::lround(static_cast<float>(config->fontSize) * font_scale)));
  TTF_SetFontSize(font, scaled_font_size);
  TTF_GetStringSize(font, text.chars, text.length, &width, &height);
  return Clay_Dimensions{
      static_cast<float>(width) / std::max(1.0f, sx),
      static_cast<float>(height) / std::max(1.0f, sy)};
}

void DrawTextLine(const std::string& text, Clay_Color color = kText, int size = 16) {
  CLAY_TEXT(ToClayPersistent(text), CLAY_TEXT_CONFIG({.textColor = color, .fontId = kFontId, .fontSize = static_cast<uint16_t>(size)}));
}

Clay_Color RandomTaskColor(const std::string& id) {
  const size_t h = std::hash<std::string>{}(id);
  return Clay_Color{
      .r = static_cast<float>(80 + (h & 0x7F)),
      .g = static_cast<float>(80 + ((h >> 8) & 0x7F)),
      .b = static_cast<float>(80 + ((h >> 16) & 0x7F)),
      .a = 255.0f,
  };
}

void DrawButton(const char* id, const std::string& label, Clay_Color color) {
  Clay_ElementId button_id = Clay_GetElementId(
      Clay_String{.length = static_cast<int32_t>(std::strlen(id)), .chars = id});
  Clay_Color hover_color{
      .r = std::fmin(255.0f, color.r + 20.0f),
      .g = std::fmin(255.0f, color.g + 20.0f),
      .b = std::fmin(255.0f, color.b + 20.0f),
      .a = color.a,
  };
  CLAY(button_id, {
    .layout = {.padding = CLAY_PADDING_ALL(10)},
    .backgroundColor = Clay_Hovered() ? hover_color : color,
    .cornerRadius = CLAY_CORNER_RADIUS(8)
  }) {
    DrawTextLine(label, kText, 15);
  }
}

void DrawMetricsPanel(const DashboardState& state) {
  const auto metrics = state.recent_metrics(80);
  MetricPoint latest;
  if (!metrics.empty()) {
    latest = metrics.back();
  }

  CLAY(CLAY_ID("metrics_panel"), {
    .layout = {.sizing = {.width = CLAY_SIZING_GROW(0)}, .padding = CLAY_PADDING_ALL(12), .childGap = 10, .layoutDirection = CLAY_TOP_TO_BOTTOM},
    .backgroundColor = kPanel,
    .cornerRadius = CLAY_CORNER_RADIUS(10)
  }) {
    DrawTextLine("Runtime Metrics", kText, 18);
    DrawTextLine("Runtime: " + FormatDouble(latest.runtime_s, 2) + " s", kMuted, 14);
    DrawTextLine("Memory: " + FormatDouble(latest.mem_used_gb, 3) + " GB", kMuted, 14);
    DrawTextLine("CPU: " + FormatDouble(latest.cpu_percent, 1) + " %", kMuted, 14);
    DrawTextLine("I/O R/W: " + FormatDouble(latest.io_read_mbps, 1) + " / " + FormatDouble(latest.io_write_mbps, 1) + " MB/s", kMuted, 14);

    CLAY(CLAY_ID("cpu_bars"), {
      .layout = {
        .sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(64)},
        .padding = CLAY_PADDING_ALL(6),
        .childGap = 2,
        .childAlignment = {.y = CLAY_ALIGN_Y_BOTTOM},
        .layoutDirection = CLAY_LEFT_TO_RIGHT
      },
      .backgroundColor = kPanelAlt,
      .cornerRadius = CLAY_CORNER_RADIUS(6)
    }) {
      for (size_t i = 0; i < metrics.size(); ++i) {
        const auto& m = metrics[i];
        const float h = static_cast<float>(std::clamp(m.cpu_percent / 100.0, 0.02, 1.0) * 52.0);
        Clay_Color c = m.cpu_percent > 90.0 ? kBad : (m.cpu_percent > 70.0 ? kWarn : kGood);
        CLAY(CLAY_IDI("cpu_bar", static_cast<int32_t>(i)), {
          .layout = {.sizing = {.width = CLAY_SIZING_FIXED(3), .height = CLAY_SIZING_FIXED(h)}},
          .backgroundColor = c
        }) {}
      }
    }
  }
}

void DrawTaskStreamPanel(const DashboardState& state) {
  const auto tasks = state.recent_tasks(320);
  const auto now_epoch = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
  const bool show_full_runtime = state.timeline_show_full_runtime();
  double zero_epoch = state.task_zero_epoch_s();
  for (const auto& t : tasks) {
    zero_epoch = std::min(zero_epoch, t.start);
  }

  struct RenderTask {
    std::string id;
    std::string name;
    std::string worker;
    double start_runtime = 0.0;
    double end_runtime = 0.0;
  };

  std::vector<RenderTask> render_tasks;
  render_tasks.reserve(tasks.size());
  double min_start_runtime = std::numeric_limits<double>::infinity();
  double max_end_runtime = 0.0;
  for (const auto& t : tasks) {
    const double end_epoch = (t.status == "start" && t.end <= t.start) ? now_epoch : std::max(t.end, t.start);
    const double start_runtime = std::max(0.0, t.start - zero_epoch);
    const double end_runtime = std::max(start_runtime, end_epoch - zero_epoch);
    min_start_runtime = std::min(min_start_runtime, start_runtime);
    max_end_runtime = std::max(max_end_runtime, end_runtime);
    render_tasks.push_back(RenderTask{
        .id = t.id.empty() ? t.name : t.id,
        .name = t.name,
        .worker = t.worker.empty() ? "worker-0" : t.worker,
        .start_runtime = start_runtime,
        .end_runtime = end_runtime,
    });
  }
  if (!std::isfinite(min_start_runtime)) {
    min_start_runtime = 0.0;
    max_end_runtime = std::max(1.0, state.runtime_s());
  } else {
    if (show_full_runtime) {
      max_end_runtime = std::max(max_end_runtime, std::max(0.0, state.runtime_s()));
    }
    max_end_runtime = std::max(max_end_runtime, min_start_runtime + 1e-6);
  }
  double view_start_runtime = min_start_runtime;
  double view_end_runtime = max_end_runtime;
  if (auto zoom = state.timeline_zoom(); zoom.has_value()) {
    view_start_runtime = std::max(0.0, zoom->first);
    view_end_runtime = std::max(view_start_runtime + 1e-6, zoom->second);
  }
  const double span_s = std::max(1e-6, view_end_runtime - view_start_runtime);
  g_task_timeline_view = TaskTimelineViewState{
      .valid = true,
      .view_start_runtime = view_start_runtime,
      .view_end_runtime = view_end_runtime,
  };

  std::map<std::string, std::vector<RenderTask>> lanes;
  for (auto& rt : render_tasks) {
    lanes[rt.worker].push_back(rt);
  }
  for (auto& [_, lane_tasks] : lanes) {
    std::sort(lane_tasks.begin(), lane_tasks.end(), [](const RenderTask& a, const RenderTask& b) {
      if (a.start_runtime == b.start_runtime) {
        return a.end_runtime < b.end_runtime;
      }
      return a.start_runtime < b.start_runtime;
    });
  }

  const float content_span_px = static_cast<float>(span_s * kTimelinePixelsPerSecond);
  const float min_lane_width_px = std::max(1.0f, g_task_timeline_min_width_px);
  const float timeline_draw_width_px = std::max(content_span_px, min_lane_width_px);
  const double timeline_pixels_per_second = timeline_draw_width_px / span_s;
  g_task_timeline_view.pixels_per_second = timeline_pixels_per_second;

  CLAY(CLAY_ID("tasks_panel"), {
    .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(360)}, .padding = CLAY_PADDING_ALL(12), .childGap = 10, .layoutDirection = CLAY_TOP_TO_BOTTOM},
    .backgroundColor = kPanel,
    .cornerRadius = CLAY_CORNER_RADIUS(10)
  }) {
    DrawTextLine("Task Stream", kText, 18);
    CLAY(CLAY_ID("tasks_panel_controls"), {
      .layout = {.sizing = {.width = CLAY_SIZING_GROW(0)}, .childGap = 8, .layoutDirection = CLAY_LEFT_TO_RIGHT}
    }) {
      DrawTextLine("Drag to zoom timeline, right-click to reset zoom", kMuted, 12);
      DrawButton(
          "timeline_mode_toggle",
          show_full_runtime ? "Range: Full runtime" : "Range: Task span",
          kPanelAlt);
    }
    CLAY(CLAY_ID("tasks_scroll"), {
      .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0)}, .padding = CLAY_PADDING_ALL(8), .childGap = 6, .layoutDirection = CLAY_TOP_TO_BOTTOM},
      .backgroundColor = kPanelAlt,
      .cornerRadius = CLAY_CORNER_RADIUS(6),
      .clip = {.horizontal = true, .vertical = true, .childOffset = Clay_GetScrollOffset()}
    }) {
      CLAY(CLAY_ID("task_time_axis_row"), {
        .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(28)}, .childGap = 8, .layoutDirection = CLAY_LEFT_TO_RIGHT}
      }) {
        CLAY(CLAY_ID("task_time_axis_label_spacer"), {
          .layout = {.sizing = {.width = CLAY_SIZING_FIXED(kLaneLabelWidthPx), .height = CLAY_SIZING_GROW(0)}}
        }) {}
        CLAY(CLAY_ID("task_time_axis"), {
          .layout = {.sizing = {.width = CLAY_SIZING_FIXED(timeline_draw_width_px), .height = CLAY_SIZING_GROW(0)}, .layoutDirection = CLAY_LEFT_TO_RIGHT}
        }) {
          const double tick_step = NiceTickStep(span_s, 8);
          const double start_tick = std::ceil(view_start_runtime / tick_step) * tick_step;
          double cursor_runtime = view_start_runtime;
          int32_t tick_idx = 0;
          for (double tick_runtime = start_tick; tick_runtime <= view_end_runtime + 1e-9;
               tick_runtime += tick_step) {
            const double gap_runtime = std::max(0.0, tick_runtime - cursor_runtime);
            const float gap_px = static_cast<float>(gap_runtime * timeline_pixels_per_second);
            if (gap_px > 0.0f) {
              CLAY(CLAY_IDI("task_tick_gap", tick_idx), {
                .layout = {.sizing = {.width = CLAY_SIZING_FIXED(gap_px), .height = CLAY_SIZING_GROW(0)}}
              }) {}
            }
            CLAY(CLAY_IDI("task_tick", tick_idx), {
              .layout = {.sizing = {.width = CLAY_SIZING_FIXED(1), .height = CLAY_SIZING_FIXED(8)}},
              .backgroundColor = kMuted
            }) {
              CLAY(CLAY_IDI("task_tick_label", tick_idx), {
                .layout = {.sizing = {.width = CLAY_SIZING_FIT(0), .height = CLAY_SIZING_FIT(0)}, .padding = CLAY_PADDING_ALL(2)},
                .backgroundColor = Clay_Color{20, 24, 30, 220},
                .cornerRadius = CLAY_CORNER_RADIUS(3),
                .floating = {
                  .offset = {0, -4},
                  .zIndex = 30000,
                  .attachPoints = {.element = CLAY_ATTACH_POINT_LEFT_BOTTOM, .parent = CLAY_ATTACH_POINT_LEFT_TOP},
                  .pointerCaptureMode = CLAY_POINTER_CAPTURE_MODE_PASSTHROUGH,
                  .attachTo = CLAY_ATTACH_TO_PARENT
                }
              }) {
                DrawTextLine(FormatRuntimeTick(tick_runtime), kMuted, 11);
              }
            }
            cursor_runtime = tick_runtime;
            tick_idx += 1;
          }
          const double tail_runtime = std::max(0.0, view_end_runtime - cursor_runtime);
          const float tail_px = static_cast<float>(tail_runtime * timeline_pixels_per_second);
          if (tail_px > 0.0f) {
            CLAY(CLAY_ID("task_tick_tail"), {
              .layout = {.sizing = {.width = CLAY_SIZING_FIXED(tail_px), .height = CLAY_SIZING_GROW(0)}}
            }) {}
          }
        }
      }

      int32_t row_idx = 0;
      for (const auto& [worker, lane_tasks] : lanes) {
        CLAY(CLAY_IDI("task_lane_row", row_idx), {
          .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(kBarHeightPx)}, .childGap = 8, .layoutDirection = CLAY_LEFT_TO_RIGHT}
        }) {
          CLAY(CLAY_IDI("task_lane_label", row_idx), {
            .layout = {.sizing = {.width = CLAY_SIZING_FIXED(kLaneLabelWidthPx), .height = CLAY_SIZING_FIXED(kBarHeightPx)}}
          }) {
            DrawTextLine(worker, kMuted, 12);
          }
          CLAY(CLAY_IDI("task_lane_timeline", row_idx), {
            .layout = {.sizing = {.width = CLAY_SIZING_FIXED(timeline_draw_width_px), .height = CLAY_SIZING_FIXED(kBarHeightPx)}, .layoutDirection = CLAY_LEFT_TO_RIGHT}
          }) {
            int32_t lane_task_idx = 0;
            double cursor_runtime = view_start_runtime;
            for (const auto& rt : lane_tasks) {
              const double draw_start_runtime = std::max(rt.start_runtime, view_start_runtime);
              const double draw_end_runtime = std::min(rt.end_runtime, view_end_runtime);
              if (draw_end_runtime <= draw_start_runtime) {
                lane_task_idx += 1;
                continue;
              }
              const double gap_runtime = std::max(0.0, draw_start_runtime - cursor_runtime);
              const double draw_dur_runtime = std::max(0.0, draw_end_runtime - draw_start_runtime);
              const double full_dur_runtime = std::max(0.0, rt.end_runtime - rt.start_runtime);
              const float gap_px = static_cast<float>(gap_runtime * timeline_pixels_per_second);
              const float width_px = static_cast<float>(draw_dur_runtime * timeline_pixels_per_second);
              if (gap_px > 0.0f) {
                CLAY(CLAY_IDI("task_gap", row_idx * 10000 + lane_task_idx), {
                  .layout = {.sizing = {.width = CLAY_SIZING_FIXED(gap_px), .height = CLAY_SIZING_FIXED(kBarHeightPx)}}
                }) {}
              }
              if (width_px <= 0.0f) {
                cursor_runtime = std::max(cursor_runtime, draw_end_runtime);
                lane_task_idx += 1;
                continue;
              }
              const Clay_Color bar_color = RandomTaskColor(rt.id);
              CLAY(CLAY_IDI("task_bar", row_idx * 10000 + lane_task_idx), {
                .layout = {.sizing = {.width = CLAY_SIZING_FIXED(width_px), .height = CLAY_SIZING_FIXED(kBarHeightPx)}},
                .backgroundColor = bar_color,
                .cornerRadius = CLAY_CORNER_RADIUS(4)
              }) {
                if (Clay_Hovered()) {
                  CLAY(CLAY_IDI("task_tooltip", row_idx * 10000 + lane_task_idx), {
                    .layout = {.sizing = {.width = CLAY_SIZING_FIT(0), .height = CLAY_SIZING_FIT(0)}, .padding = CLAY_PADDING_ALL(6)},
                    .backgroundColor = kBg,
                    .cornerRadius = CLAY_CORNER_RADIUS(5),
                    .floating = {
                      .offset = {0, -6},
                      .zIndex = 32000,
                      .attachPoints = {.element = CLAY_ATTACH_POINT_LEFT_BOTTOM, .parent = CLAY_ATTACH_POINT_LEFT_TOP},
                      .pointerCaptureMode = CLAY_POINTER_CAPTURE_MODE_PASSTHROUGH,
                      .attachTo = CLAY_ATTACH_TO_PARENT
                    },
                    .border = {.color = kMuted, .width = {1, 1, 1, 1, 0}}
                  }) {
                    DrawTextLine(rt.name + "  [" + rt.worker + "]  " + FormatDouble(full_dur_runtime * 1000.0, 1) + " ms", kText, 12);
                  }
                }
              }
              cursor_runtime = std::max(cursor_runtime, draw_end_runtime);
              lane_task_idx += 1;
            }
            const double tail_runtime = std::max(0.0, view_end_runtime - cursor_runtime);
            const float tail_px = static_cast<float>(tail_runtime * timeline_pixels_per_second);
            if (tail_px > 0.0f) {
              CLAY(CLAY_IDI("task_tail", row_idx), {
                .layout = {.sizing = {.width = CLAY_SIZING_FIXED(tail_px), .height = CLAY_SIZING_FIXED(kBarHeightPx)}}
              }) {}
            }
          }
        }
        row_idx += 1;
      }
    }
  }
}

void DrawFlamePanel(const DashboardState& state) {
  const auto totals = state.flame_totals(20);
  double sum = 0.0;
  for (const auto& [_, d] : totals) {
    sum += d;
  }

  CLAY(CLAY_ID("flame_panel"), {
    .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(240)}, .padding = CLAY_PADDING_ALL(12), .childGap = 8, .layoutDirection = CLAY_TOP_TO_BOTTOM},
    .backgroundColor = kPanel,
    .cornerRadius = CLAY_CORNER_RADIUS(10)
  }) {
    DrawTextLine("Flamegraph (total durations)", kText, 18);
    CLAY(CLAY_ID("flame_bar"), {
      .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(18)}, .padding = CLAY_PADDING_ALL(2), .childGap = 1, .layoutDirection = CLAY_LEFT_TO_RIGHT},
      .backgroundColor = kPanelAlt,
      .cornerRadius = CLAY_CORNER_RADIUS(6)
    }) {
      for (size_t i = 0; i < totals.size(); ++i) {
        const double frac = (sum > 0.0) ? (totals[i].second / sum) : 0.0;
        const float w = static_cast<float>(std::max(2.0, frac * 480.0));
        Clay_Color c = Clay_Color{
            .r = static_cast<float>(60 + (i * 31) % 170),
            .g = static_cast<float>(90 + (i * 47) % 140),
            .b = static_cast<float>(120 + (i * 29) % 120),
            .a = 255.0f,
        };
        CLAY(CLAY_IDI("flame_slice", static_cast<int32_t>(i)), {.layout = {.sizing = {.width = CLAY_SIZING_FIXED(w), .height = CLAY_SIZING_GROW(0)}}, .backgroundColor = c}) {}
      }
    }

    CLAY(CLAY_ID("flame_list"), {
      .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0)}, .childGap = 4, .layoutDirection = CLAY_TOP_TO_BOTTOM},
      .clip = {.vertical = true, .childOffset = Clay_GetScrollOffset()}
    }) {
      for (const auto& [name, duration] : totals) {
        DrawTextLine(name + ": " + FormatDouble(duration, 3) + " s", kMuted, 13);
      }
    }
  }
}

void DrawDagPanel(const DashboardState& state) {
  const auto nodes = state.dag_nodes();
  const auto edges = state.dag_edges();

  CLAY(CLAY_ID("dag_panel"), {
    .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_FIXED(260)}, .padding = CLAY_PADDING_ALL(12), .childGap = 10, .layoutDirection = CLAY_TOP_TO_BOTTOM},
    .backgroundColor = kPanel,
    .cornerRadius = CLAY_CORNER_RADIUS(10)
  }) {
    DrawTextLine("Task Graph (DAG)", kText, 18);
    DrawTextLine("Nodes: " + std::to_string(nodes.size()) + "  Edges: " + std::to_string(edges.size()), kMuted, 13);

    CLAY(CLAY_ID("dag_scroll"), {
      .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0)}, .padding = CLAY_PADDING_ALL(8), .childGap = 5, .layoutDirection = CLAY_TOP_TO_BOTTOM},
      .backgroundColor = kPanelAlt,
      .cornerRadius = CLAY_CORNER_RADIUS(6),
      .clip = {.vertical = true, .childOffset = Clay_GetScrollOffset()}
    }) {
      for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        CLAY(CLAY_IDI("dag_row", static_cast<int32_t>(i)), {.layout = {.sizing = {.width = CLAY_SIZING_GROW(0)}, .childGap = 6, .layoutDirection = CLAY_LEFT_TO_RIGHT}}) {
          CLAY(CLAY_IDI("dag_indent", static_cast<int32_t>(i)), {.layout = {.sizing = {.width = CLAY_SIZING_FIXED(static_cast<float>(n.depth * 16)), .height = CLAY_SIZING_FIXED(1)}}}) {}
          DrawTextLine(n.label, kMuted, 13);
        }
      }
    }
  }
}

Clay_RenderCommandArray BuildLayout(DashboardState& state, UiActions* actions) {
  g_frame_text_storage.clear();
  g_frame_text_storage.reserve(1024);

  Clay_BeginLayout();

  CLAY(CLAY_ID("root"), {
    .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0)}, .padding = CLAY_PADDING_ALL(12), .childGap = 10, .layoutDirection = CLAY_TOP_TO_BOTTOM},
    .backgroundColor = kBg
  }) {
    CLAY(CLAY_ID("header"), {
      .layout = {.sizing = {.width = CLAY_SIZING_GROW(0)}, .padding = CLAY_PADDING_ALL(10), .childGap = 12, .childAlignment = {.y = CLAY_ALIGN_Y_CENTER}, .layoutDirection = CLAY_LEFT_TO_RIGHT},
      .backgroundColor = kPanel,
      .cornerRadius = CLAY_CORNER_RADIUS(10)
    }) {
      DrawTextLine(state.title(), kText, 22);
      DrawTextLine("Runtime: " + FormatDouble(state.runtime_s(), 2) + " s", kMuted, 14);
      DrawTextLine("Workflow: " + state.status(), kMuted, 14);

      DrawButton("threads_minus", "-", kPanelAlt);
      DrawTextLine("Threads/locality: " + std::to_string(state.threads()), kText, 14);
      DrawButton("threads_plus", "+", kPanelAlt);
      DrawButton("start_workflow", "Start Workflow", kAccent);
    }

    CLAY(CLAY_ID("top_row"), {
      .layout = {.sizing = {.width = CLAY_SIZING_GROW(0)}, .childGap = 10, .layoutDirection = CLAY_LEFT_TO_RIGHT}
    }) {
      DrawMetricsPanel(state);
      DrawDagPanel(state);
    }

    CLAY(CLAY_ID("bottom_row"), {
      .layout = {.sizing = {.width = CLAY_SIZING_GROW(0), .height = CLAY_SIZING_GROW(0)}, .childGap = 10, .layoutDirection = CLAY_LEFT_TO_RIGHT}
    }) {
      DrawTaskStreamPanel(state);
      DrawFlamePanel(state);
    }
  }

  auto commands = Clay_EndLayout();

  const Clay_ElementId start_id = Clay_GetElementId(CLAY_STRING("start_workflow"));
  const Clay_ElementId minus_id = Clay_GetElementId(CLAY_STRING("threads_minus"));
  const Clay_ElementId plus_id = Clay_GetElementId(CLAY_STRING("threads_plus"));
  const Clay_ElementId timeline_toggle_id = Clay_GetElementId(CLAY_STRING("timeline_mode_toggle"));
  actions->start = g_left_click_released_this_frame && Clay_PointerOver(start_id);
  actions->threads_minus = g_left_click_released_this_frame && Clay_PointerOver(minus_id);
  actions->threads_plus = g_left_click_released_this_frame && Clay_PointerOver(plus_id);
  actions->timeline_toggle_range =
      g_left_click_released_this_frame && Clay_PointerOver(timeline_toggle_id);

  return commands;
}

std::optional<std::filesystem::path> FindFontPath() {
  const std::array<std::filesystem::path, 3> candidates = {
      std::filesystem::path("resources/Roboto-Regular.ttf"),
      std::filesystem::path("extern/clay/examples/SDL3-simple-demo/resources/Roboto-Regular.ttf"),
      std::filesystem::path("../extern/clay/examples/SDL3-simple-demo/resources/Roboto-Regular.ttf")};
  for (const auto& p : candidates) {
    if (std::filesystem::exists(p)) {
      return p;
    }
  }
  return std::nullopt;
}

void PrintUsage() {
  std::cout
      << "Usage: kangaroo_dashboard_clay [--metrics <events.jsonl>] [--plan <plan.json>] [--run <script.py> [-- args...]]\n"
      << "                              [--interval-ms N] [--history-seconds N] [--threads-per-locality N] [--title TEXT]\n";
}

bool ParseArgs(int argc, char** argv, DashboardConfig* cfg) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 1; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }

  for (size_t i = 0; i < args.size(); ++i) {
    const auto& a = args[i];
    auto require_value = [&](const char* flag) -> std::optional<std::string> {
      if (i + 1 >= args.size()) {
        std::cerr << "Missing value for " << flag << "\n";
        return std::nullopt;
      }
      return args[++i];
    };

    if (a == "-h" || a == "--help") {
      PrintUsage();
      return false;
    }
    if (a == "--metrics") {
      auto v = require_value("--metrics");
      if (!v) return false;
      cfg->metrics_path = *v;
      continue;
    }
    if (a == "--plan") {
      auto v = require_value("--plan");
      if (!v) return false;
      cfg->plan_path = *v;
      continue;
    }
    if (a == "--interval-ms") {
      auto v = require_value("--interval-ms");
      if (!v) return false;
      cfg->update_interval_ms = std::max(50, std::stoi(*v));
      continue;
    }
    if (a == "--history-seconds") {
      auto v = require_value("--history-seconds");
      if (!v) return false;
      cfg->history_seconds = std::max(10, std::stoi(*v));
      continue;
    }
    if (a == "--threads-per-locality") {
      auto v = require_value("--threads-per-locality");
      if (!v) return false;
      cfg->threads_per_locality = std::max(1, std::stoi(*v));
      continue;
    }
    if (a == "--title") {
      auto v = require_value("--title");
      if (!v) return false;
      cfg->title = *v;
      continue;
    }
    if (a == "--run") {
      auto v = require_value("--run");
      if (!v) return false;
      const char* py = std::getenv("PYTHON");
      cfg->run_command = {py ? py : "python3", *v};
      if (i + 1 < args.size() && args[i + 1] == "--") {
        for (size_t j = i + 2; j < args.size(); ++j) {
          cfg->run_command.push_back(args[j]);
        }
        break;
      }
      continue;
    }

    std::cerr << "Unknown argument: " << a << "\n";
    PrintUsage();
    return false;
  }

  if (!cfg->run_command.empty()) {
    const auto temp_dir = std::filesystem::temp_directory_path() /
                          ("kangaroo-dashboard-clay-" + std::to_string(getpid()));
    std::filesystem::create_directories(temp_dir);
    if (!cfg->metrics_path) {
      cfg->metrics_path = temp_dir / "events.jsonl";
    }
    if (!cfg->plan_path) {
      cfg->plan_path = temp_dir / "plan.json";
    }

    std::cout << "[dashboard] workflow log: " << cfg->metrics_path->string() << "\n";
    std::cout << "[dashboard] workflow plan: " << cfg->plan_path->string() << "\n";
  }

  return true;
}

}  // namespace

int main(int argc, char** argv) {
  DashboardConfig cfg;
  if (!ParseArgs(argc, argv, &cfg)) {
    return 0;
  }
  const int update_interval_ms = cfg.update_interval_ms;

  if (!TTF_Init()) {
    std::cerr << "Failed to init SDL_ttf: " << SDL_GetError() << "\n";
    return 1;
  }

  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;
  if (!SDL_CreateWindowAndRenderer(
          cfg.title.c_str(), 1500, 900, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY, &window, &renderer)) {
    std::cerr << "Failed to create window/renderer: " << SDL_GetError() << "\n";
    return 1;
  }

  Clay_SDL3RendererData renderer_data{};
  renderer_data.renderer = renderer;
  renderer_data.textEngine = TTF_CreateRendererTextEngine(renderer);
  renderer_data.fonts = static_cast<TTF_Font**>(SDL_calloc(1, sizeof(TTF_Font*)));

  auto font_path = FindFontPath();
  if (!font_path) {
    std::cerr << "Unable to locate Roboto-Regular.ttf (expected under extern/clay examples resources).\n";
    return 1;
  }
  renderer_data.fonts[kFontId] = TTF_OpenFont(font_path->c_str(), 16);
  if (!renderer_data.fonts[kFontId]) {
    std::cerr << "Failed to load font: " << SDL_GetError() << "\n";
    return 1;
  }

  const uint64_t arena_size = Clay_MinMemorySize();
  Clay_Arena arena = Clay_CreateArenaWithCapacityAndMemory(arena_size, static_cast<char*>(SDL_malloc(arena_size)));
  int width = 1500;
  int height = 900;
  SDL_GetWindowSize(window, &width, &height);
  int output_width = width;
  int output_height = height;
  if (!SDL_GetRenderOutputSize(renderer, &output_width, &output_height)) {
    output_width = width;
    output_height = height;
  }
  g_render_scale_x =
      (width > 0) ? (static_cast<float>(output_width) / static_cast<float>(width)) : 1.0f;
  g_render_scale_y =
      (height > 0) ? (static_cast<float>(output_height) / static_cast<float>(height)) : 1.0f;
  Clay_Initialize(
      arena, Clay_Dimensions{static_cast<float>(width), static_cast<float>(height)}, Clay_ErrorHandler{HandleClayErrors});
  Clay_SetMeasureTextFunction(SDLMeasureText, renderer_data.fonts);

  DashboardState state(std::move(cfg));

  auto last_update = std::chrono::steady_clock::now();
  auto last_frame = std::chrono::steady_clock::now();
  auto point_in = [](Clay_Vector2 p, const Clay_BoundingBox& bb) {
    return p.x >= bb.x && p.x <= bb.x + bb.width && p.y >= bb.y && p.y <= bb.y + bb.height;
  };
  auto update_render_metrics = [&](int* window_w, int* window_h, int* output_w, int* output_h) {
    SDL_GetWindowSize(window, window_w, window_h);
    if (!SDL_GetRenderOutputSize(renderer, output_w, output_h)) {
      *output_w = *window_w;
      *output_h = *window_h;
    }
    g_render_scale_x =
        (*window_w > 0) ? (static_cast<float>(*output_w) / static_cast<float>(*window_w)) : 1.0f;
    g_render_scale_y =
        (*window_h > 0) ? (static_cast<float>(*output_h) / static_cast<float>(*window_h)) : 1.0f;
  };
  Clay_Vector2 mouse_pos{0, 0};
  bool mouse_down = false;
  bool mouse_release_pending = false;
  bool timeline_drag_active = false;
  float timeline_drag_anchor_x = 0.0f;
  float timeline_drag_current_x = 0.0f;
  bool tasks_panel_bbox_valid = false;
  Clay_BoundingBox tasks_panel_bbox{};
  bool tasks_scroll_bbox_valid = false;
  Clay_BoundingBox tasks_scroll_bbox{};
  float tasks_scroll_x = 0.0f;
  bool timeline_lane_bbox_valid = false;
  Clay_BoundingBox timeline_lane_bbox{};
  Clay_Vector2 wheel_delta{0, 0};
  bool running = true;
  update_render_metrics(&width, &height, &output_width, &output_height);

  while (running) {
    wheel_delta = Clay_Vector2{0, 0};
    g_left_click_released_this_frame = false;

    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      switch (e.type) {
        case SDL_EVENT_QUIT:
          running = false;
          break;
        case SDL_EVENT_WINDOW_RESIZED:
        case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
        case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
          update_render_metrics(&width, &height, &output_width, &output_height);
          Clay_SetLayoutDimensions(Clay_Dimensions{static_cast<float>(width), static_cast<float>(height)});
          break;
        case SDL_EVENT_MOUSE_MOTION:
          mouse_pos = Clay_Vector2{static_cast<float>(e.motion.x), static_cast<float>(e.motion.y)};
          break;
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (e.button.button == SDL_BUTTON_LEFT) {
              mouse_pos = Clay_Vector2{static_cast<float>(e.button.x), static_cast<float>(e.button.y)};
            if (tasks_panel_bbox_valid && point_in(mouse_pos, tasks_panel_bbox)) {
              timeline_drag_active = true;
              timeline_drag_anchor_x = mouse_pos.x;
              timeline_drag_current_x = mouse_pos.x;
            } else {
              timeline_drag_active = false;
            }
            mouse_down = true;
            mouse_release_pending = false;
          }
          break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
          if (e.button.button == SDL_BUTTON_LEFT) {
            mouse_pos = Clay_Vector2{static_cast<float>(e.button.x), static_cast<float>(e.button.y)};
            // Keep the press active for this frame so quick taps still produce
            // CLAY_POINTER_DATA_PRESSED_THIS_FRAME.
            mouse_release_pending = true;
            g_left_click_released_this_frame = true;
            if (timeline_drag_active && g_task_timeline_view.valid && tasks_scroll_bbox_valid) {
              timeline_drag_current_x = mouse_pos.x;
              const float drag_px = std::fabs(timeline_drag_current_x - timeline_drag_anchor_x);
              if (drag_px >= 4.0f) {
                const float timeline_origin_x =
                    tasks_scroll_bbox.x + kTaskScrollPaddingPx + kLaneLabelWidthPx + kLaneGapPx;
                const double content_span_px =
                    (g_task_timeline_view.view_end_runtime - g_task_timeline_view.view_start_runtime) *
                    g_task_timeline_view.pixels_per_second;
                auto x_to_runtime = [&](float mouse_x) {
                  const double raw_content_x =
                      static_cast<double>(mouse_x - timeline_origin_x + tasks_scroll_x);
                  const double content_x = std::clamp(raw_content_x, 0.0, content_span_px);
                  return g_task_timeline_view.view_start_runtime +
                         content_x / g_task_timeline_view.pixels_per_second;
                };
                const double unclamped_a = x_to_runtime(timeline_drag_anchor_x);
                const double unclamped_b = x_to_runtime(timeline_drag_current_x);
                const double lo = g_task_timeline_view.view_start_runtime;
                const double hi = g_task_timeline_view.view_end_runtime;
                const double a = std::clamp(unclamped_a, lo, hi);
                const double b = std::clamp(unclamped_b, lo, hi);
                if (std::fabs(b - a) > 1e-6) {
                  state.set_timeline_zoom(a, b);
                }
              }
            }
            timeline_drag_active = false;
          }
          if (e.button.button == SDL_BUTTON_RIGHT) {
            Clay_Vector2 pos{
                static_cast<float>(e.button.x), static_cast<float>(e.button.y)};
            if (tasks_panel_bbox_valid && point_in(pos, tasks_panel_bbox)) {
              state.clear_timeline_zoom();
            }
          }
          break;
        case SDL_EVENT_MOUSE_WHEEL:
          wheel_delta = Clay_Vector2{static_cast<float>(e.wheel.x), static_cast<float>(e.wheel.y)};
          if (wheel_delta.x == 0.0f && (SDL_GetModState() & SDL_KMOD_SHIFT)) {
            wheel_delta = Clay_Vector2{wheel_delta.y, 0.0f};
          }
          break;
        case SDL_EVENT_KEY_UP:
          if (e.key.scancode == SDL_SCANCODE_R) {
            state.LaunchWorkflow();
          }
          break;
        default:
          break;
      }
    }

    const auto now = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double>(now - last_frame).count();
    last_frame = now;

    Clay_SetPointerState(mouse_pos, mouse_down);
    Clay_UpdateScrollContainers(false, wheel_delta, static_cast<float>(dt));

    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count() >= update_interval_ms) {
      state.Update();
      last_update = now;
    }

    UiActions actions;
    auto commands = BuildLayout(state, &actions);

    if (timeline_drag_active && mouse_down) {
      timeline_drag_current_x = mouse_pos.x;
    }

    const Clay_ElementId tasks_scroll_id = Clay_GetElementId(CLAY_STRING("tasks_scroll"));
    const Clay_ElementData tasks_scroll_elem = Clay_GetElementData(tasks_scroll_id);
    const Clay_ScrollContainerData tasks_scroll_data = Clay_GetScrollContainerData(tasks_scroll_id);
    const Clay_ElementId first_timeline_lane_id =
        Clay_GetElementIdWithIndex(CLAY_STRING("task_lane_timeline"), 0);
    const Clay_ElementData first_timeline_lane_elem = Clay_GetElementData(first_timeline_lane_id);
    const Clay_ElementId tasks_panel_id = Clay_GetElementId(CLAY_STRING("tasks_panel"));
    const Clay_ElementData tasks_panel_elem = Clay_GetElementData(tasks_panel_id);
    tasks_panel_bbox_valid = tasks_panel_elem.found;
    if (tasks_panel_bbox_valid) {
      tasks_panel_bbox = tasks_panel_elem.boundingBox;
    }
    tasks_scroll_bbox_valid = tasks_scroll_elem.found;
    if (tasks_scroll_bbox_valid) {
      tasks_scroll_bbox = tasks_scroll_elem.boundingBox;
      tasks_scroll_x =
          (tasks_scroll_data.found && tasks_scroll_data.scrollPosition != nullptr)
              ? tasks_scroll_data.scrollPosition->x
              : 0.0f;
      g_task_timeline_min_width_px =
          std::max(0.0f, tasks_scroll_bbox.width - 2.0f * kTaskScrollPaddingPx - kLaneLabelWidthPx - kLaneGapPx);
    }
    timeline_lane_bbox_valid = first_timeline_lane_elem.found;
    if (timeline_lane_bbox_valid) {
      timeline_lane_bbox = first_timeline_lane_elem.boundingBox;
    }

    if (actions.threads_minus) {
      state.threads_minus();
    }
    if (actions.threads_plus) {
      state.threads_plus();
    }
    if (actions.start) {
      state.LaunchWorkflow();
    }
    if (actions.timeline_toggle_range) {
      state.toggle_timeline_show_full_runtime();
    }

    if (mouse_release_pending) {
      mouse_down = false;
      mouse_release_pending = false;
    }

    SDL_SetRenderDrawColor(renderer, 10, 12, 16, 255);
    SDL_RenderClear(renderer);
    RenderClayCommands(&renderer_data, &commands);

    if (timeline_drag_active) {
      float x0 = timeline_drag_anchor_x;
      float x1 = timeline_drag_current_x;
      if (x0 > x1) {
        std::swap(x0, x1);
      }
      float clip_left = tasks_panel_bbox_valid ? tasks_panel_bbox.x : 0.0f;
      float clip_right =
          tasks_panel_bbox_valid ? (tasks_panel_bbox.x + tasks_panel_bbox.width) : static_cast<float>(width);
      float clip_top = tasks_scroll_bbox_valid ? tasks_scroll_bbox.y : 0.0f;
      float clip_bottom =
          tasks_scroll_bbox_valid ? (tasks_scroll_bbox.y + tasks_scroll_bbox.height) : static_cast<float>(height);
      if (timeline_lane_bbox_valid) {
        clip_left = timeline_lane_bbox.x;
        clip_right = timeline_lane_bbox.x + timeline_lane_bbox.width;
      }
      x0 = std::clamp(x0, clip_left, clip_right);
      x1 = std::clamp(x1, clip_left, clip_right);
      if (x1 > x0 + 1.0f && clip_bottom > clip_top + 1.0f) {
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_FRect rect{
            .x = x0 * g_render_scale_x,
            .y = clip_top * g_render_scale_y,
            .w = (x1 - x0) * g_render_scale_x,
            .h = (clip_bottom - clip_top) * g_render_scale_y,
        };
        SDL_SetRenderDrawColor(renderer, 56, 160, 255, 56);
        SDL_RenderFillRect(renderer, &rect);
        SDL_SetRenderDrawColor(renderer, 56, 160, 255, 220);
        SDL_RenderRect(renderer, &rect);
      }
    }
    SDL_RenderPresent(renderer);

    SDL_Delay(16);
  }

  if (renderer_data.fonts) {
    if (renderer_data.fonts[kFontId]) {
      TTF_CloseFont(renderer_data.fonts[kFontId]);
    }
    SDL_free(renderer_data.fonts);
  }
  if (renderer_data.textEngine) {
    TTF_DestroyRendererTextEngine(renderer_data.textEngine);
  }
  if (renderer) {
    SDL_DestroyRenderer(renderer);
  }
  if (window) {
    SDL_DestroyWindow(window);
  }

  TTF_Quit();
  return 0;
}
