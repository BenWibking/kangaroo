#include "kangaroo/data_service_local.hpp"

#include "kangaroo/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hpx/executors/parallel_executor.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/post.hpp>
#include <hpx/runtime_local/thread_pool_helpers.hpp>
#include <hpx/runtime_local/get_worker_thread_num.hpp>

HPX_PLAIN_ACTION(kangaroo::data_get_local_impl, kangaroo_data_get_local_action)
HPX_PLAIN_ACTION(kangaroo::data_put_local_impl, kangaroo_data_put_local_action)

namespace kangaroo {

namespace {

bool debug_dataflow_enabled() {
  static const bool enabled = std::getenv("KANGAROO_DEBUG_DATAFLOW") != nullptr;
  return enabled;
}

void log_dataflow_fetch(const char* op,
                        const ChunkRef& ref,
                        int here,
                        int target,
                        std::size_t bytes) {
  if (!debug_dataflow_enabled()) {
    return;
  }
  std::cout << "[kangaroo][dataflow] op=" << op
            << " requester=" << here
            << " target=" << target
            << " step=" << ref.step
            << " level=" << ref.level
            << " field=" << ref.field
            << " version=" << ref.version
            << " block=" << ref.block
            << " bytes=" << bytes
            << " empty=" << (bytes == 0 ? 1 : 0)
            << std::endl;
}

double now_seconds() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

int dataset_load_concurrency() {
  static const int value = []() {
    auto parse_env = [](const char* name) -> int {
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
    };

    if (const int parsed = parse_env("KANGAROO_PLOTFILE_READ_CONCURRENCY"); parsed > 0) {
      return parsed;
    }
    if (const int parsed = parse_env("KANGAROO_DATASET_LOAD_CONCURRENCY"); parsed > 0) {
      return parsed;
    }
    return 16;
  }();
  return value;
}

void log_dataflow_event(const char* op,
                        const char* status,
                        const ChunkRef& ref,
                        int here,
                        int target,
                        std::size_t bytes,
                        double start) {
  if (!has_event_log()) {
    return;
  }
  const double end = now_seconds();
  DataEvent event;
  event.op = op;
  event.mode = here == target ? "local" : "remote";
  event.status = status;
  event.ref = ref;
  event.locality = here;
  event.target_locality = target;
  event.worker = static_cast<int32_t>(hpx::get_worker_thread_num());
  event.bytes = bytes;
  event.ts = end;
  event.start = start;
  event.end = end;
  log_data_event(event);
}

void log_dataflow_marker(const char* op,
                         const ChunkRef& ref,
                         int here,
                         int target,
                         std::size_t bytes) {
  const double start = now_seconds();
  log_dataflow_event(op, "end", ref, here, target, bytes, start);
}

void log_dataset_load_event(const char* op,
                            const char* status,
                            const ChunkRef& ref,
                            std::size_t bytes,
                            double start,
                            double end,
                            int32_t queue_depth,
                            int32_t in_flight) {
  if (!has_event_log()) {
    return;
  }
  const int here = hpx::get_locality_id();
  DataEvent event;
  event.op = op;
  event.mode = "local";
  event.status = status;
  event.ref = ref;
  event.locality = here;
  event.target_locality = here;
  event.worker = static_cast<int32_t>(hpx::get_worker_thread_num());
  event.bytes = bytes;
  event.ts = end;
  event.start = start;
  event.end = end;
  event.queue_depth = queue_depth;
  event.in_flight = in_flight;
  event.concurrency = dataset_load_concurrency();
  log_data_event(event);
}

SubboxView build_subbox_view(const HostView& chunk, const ChunkSubboxRef& ref) {
  SubboxView out;
  out.bytes_per_value = ref.bytes_per_value;
  out.box = ref.request_box;

  if (chunk.data.empty() || ref.bytes_per_value <= 0) {
    return out;
  }

  const int32_t nx = ref.chunk_box.hi[0] - ref.chunk_box.lo[0] + 1;
  const int32_t ny = ref.chunk_box.hi[1] - ref.chunk_box.lo[1] + 1;
  const int32_t nz = ref.chunk_box.hi[2] - ref.chunk_box.lo[2] + 1;
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return out;
  }

  const int32_t ox0 = std::max(ref.chunk_box.lo[0], ref.request_box.lo[0]);
  const int32_t oy0 = std::max(ref.chunk_box.lo[1], ref.request_box.lo[1]);
  const int32_t oz0 = std::max(ref.chunk_box.lo[2], ref.request_box.lo[2]);
  const int32_t ox1 = std::min(ref.chunk_box.hi[0], ref.request_box.hi[0]);
  const int32_t oy1 = std::min(ref.chunk_box.hi[1], ref.request_box.hi[1]);
  const int32_t oz1 = std::min(ref.chunk_box.hi[2], ref.request_box.hi[2]);

  if (ox1 < ox0 || oy1 < oy0 || oz1 < oz0) {
    out.box.hi[0] = out.box.lo[0] - 1;
    out.box.hi[1] = out.box.lo[1] - 1;
    out.box.hi[2] = out.box.lo[2] - 1;
    return out;
  }

  out.box.lo[0] = ox0;
  out.box.lo[1] = oy0;
  out.box.lo[2] = oz0;
  out.box.hi[0] = ox1;
  out.box.hi[1] = oy1;
  out.box.hi[2] = oz1;

  const std::size_t bytes_per = static_cast<std::size_t>(ref.bytes_per_value);
  const std::size_t elems_total =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
  const std::size_t needed = elems_total * bytes_per;
  if (chunk.data.size() < needed) {
    return SubboxView{};
  }

  const int32_t onx = ox1 - ox0 + 1;
  const int32_t ony = oy1 - oy0 + 1;
  const int32_t onz = oz1 - oz0 + 1;
  const std::size_t out_bytes = static_cast<std::size_t>(onx) * static_cast<std::size_t>(ony) *
                                static_cast<std::size_t>(onz) * bytes_per;
  out.data.data.resize(out_bytes, 0);

  auto out_index = [&](int32_t i, int32_t j, int32_t k) -> std::size_t {
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ony) +
            static_cast<std::size_t>(j)) *
               static_cast<std::size_t>(onz) +
           static_cast<std::size_t>(k);
  };

  HostGridView3D input(chunk, nx, ny, nz, ref.bytes_per_value);
  auto* dst = out.data.data.data();
  for (int32_t i = 0; i < onx; ++i) {
    const int32_t gi = ox0 + i;
    const int32_t li = gi - ref.chunk_box.lo[0];
    for (int32_t j = 0; j < ony; ++j) {
      const int32_t gj = oy0 + j;
      const int32_t lj = gj - ref.chunk_box.lo[1];
      for (int32_t k = 0; k < onz; ++k) {
        const int32_t gk = oz0 + k;
        const int32_t lk = gk - ref.chunk_box.lo[2];
        const std::size_t dst_byte = out_index(i, j, k) * bytes_per;
        (void)input.copy_value_bytes(li, lj, lk, dst + dst_byte);
      }
    }
  }

  return out;
}

ChunkSlot make_pending_slot() {
  ChunkSlot slot;
  slot.promise = std::make_shared<hpx::promise<std::shared_ptr<HostView>>>();
  slot.future = slot.promise->get_future().share();
  return slot;
}

void fulfill_dataset_load(const std::shared_ptr<ChunkStore>& chunk_store,
                          const ChunkRef& ref,
                          HostView view) {
  auto shared_view = std::make_shared<HostView>(std::move(view));
  std::shared_ptr<hpx::promise<std::shared_ptr<HostView>>> promise;
  {
    std::lock_guard<ChunkStore::Mutex> lock(chunk_store->mutex);
    auto [it, inserted] = chunk_store->data.try_emplace(ref, make_pending_slot());
    (void)inserted;
    if (it->second.ready) {
      return;
    }
    it->second.ready = true;
    it->second.dataset_load_started = false;
    it->second.value = shared_view;
    promise = it->second.promise;
  }

  hpx::post([promise = std::move(promise), shared_view = std::move(shared_view)]() mutable {
    promise->set_value(std::move(shared_view));
  });
}

void fail_dataset_load(const std::shared_ptr<ChunkStore>& chunk_store,
                       const ChunkRef& ref,
                       std::exception_ptr error) {
  std::shared_ptr<hpx::promise<std::shared_ptr<HostView>>> promise;
  {
    std::lock_guard<ChunkStore::Mutex> lock(chunk_store->mutex);
    auto [it, inserted] = chunk_store->data.try_emplace(ref, make_pending_slot());
    (void)inserted;
    if (it->second.ready) {
      return;
    }
    it->second.ready = true;
    it->second.dataset_load_started = false;
    promise = it->second.promise;
  }

  hpx::post([promise = std::move(promise), error = std::move(error)]() mutable {
    promise->set_exception(error);
  });
}

struct DatasetLoadRequest {
  std::shared_ptr<const DatasetHandle> dataset;
  std::shared_ptr<ChunkStore> chunk_store;
  ChunkRef ref;
  double enqueue_time = 0.0;
  int32_t queue_depth_at_start = 0;
  int32_t in_flight_at_start = 0;
};

using DatasetLoadBatch = std::vector<DatasetLoadRequest>;

class DatasetLoadQueue;
DatasetLoadQueue& dataset_load_queue();
void post_dataset_load(DatasetLoadBatch batch);

bool same_load_batch(const DatasetLoadRequest& a, const DatasetLoadRequest& b) {
  return a.dataset.get() == b.dataset.get() &&
         a.chunk_store.get() == b.chunk_store.get() &&
         a.ref.step == b.ref.step &&
         a.ref.level == b.ref.level &&
         a.ref.version == b.ref.version &&
         a.ref.block == b.ref.block;
}

bool same_load_batch(const DatasetLoadBatch& a, const DatasetLoadBatch& b) {
  return !a.empty() && !b.empty() && same_load_batch(a.front(), b.front());
}

bool same_load_batch(const DatasetLoadBatch& batch, const DatasetLoadRequest& request) {
  return !batch.empty() && same_load_batch(batch.front(), request);
}

class DatasetLoadQueue {
 public:
  void enqueue(DatasetLoadRequest request) {
    DatasetLoadBatch requests;
    requests.push_back(std::move(request));
    enqueue_many(std::move(requests));
  }

  void enqueue_many(DatasetLoadBatch requests) {
    if (requests.empty()) {
      return;
    }
    const double enqueue_time = now_seconds();
    std::vector<DatasetLoadBatch> batches;
    for (auto& request : requests) {
      request.enqueue_time = enqueue_time;
      auto it = std::find_if(batches.begin(), batches.end(), [&](const auto& batch) {
        return same_load_batch(batch, request);
      });
      if (it == batches.end()) {
        DatasetLoadBatch batch;
        batch.push_back(std::move(request));
        batches.push_back(std::move(batch));
      } else {
        it->push_back(std::move(request));
      }
    }

    {
      std::lock_guard<hpx::mutex> lock(mutex_);
      for (auto& batch : batches) {
        if (batch.empty()) {
          continue;
        }

        auto pending_it = std::find_if(pending_.begin(), pending_.end(), [&](const auto& pending) {
          return same_load_batch(pending, batch);
        });

        if (pending_it == pending_.end()) {
          pending_.push_back(std::move(batch));
          const auto& queued = pending_.back();
          const int32_t queue_depth = static_cast<int32_t>(pending_.size());
          for (const auto& request : queued) {
            log_dataset_load_event("dataset_load_queue",
                                   "end",
                                   request.ref,
                                   0,
                                   request.enqueue_time,
                                   request.enqueue_time,
                                   queue_depth,
                                   in_flight_);
          }
          continue;
        }

        const int32_t queue_depth = static_cast<int32_t>(pending_.size());
        for (auto& request : batch) {
          log_dataset_load_event("dataset_load_queue",
                                 "end",
                                 request.ref,
                                 0,
                                 request.enqueue_time,
                                 request.enqueue_time,
                                 queue_depth,
                                 in_flight_);
          pending_it->push_back(std::move(request));
        }
      }
      schedule_pump_locked();
    }
  }

  void run(DatasetLoadBatch requests) {
    struct FinishGuard {
      DatasetLoadQueue& queue;
      ~FinishGuard() { queue.finish_one(); }
    } finish_guard{*this};

    if (requests.empty()) {
      return;
    }

    const double read_start = now_seconds();
    for (const auto& request : requests) {
      log_dataset_load_event("dataset_load_start",
                             "end",
                             request.ref,
                             0,
                             request.enqueue_time,
                             read_start,
                             request.queue_depth_at_start,
                             request.in_flight_at_start);
    }

    try {
      if (!requests.front().dataset) {
        throw std::runtime_error("dataset not initialized");
      }
      std::vector<ChunkRef> refs;
      refs.reserve(requests.size());
      for (const auto& request : requests) {
        refs.push_back(request.ref);
      }
      auto views = requests.front().dataset->get_chunks(refs);
      if (views.size() != requests.size()) {
        throw std::runtime_error("dataset backend returned wrong number of chunks");
      }
      const double read_end = now_seconds();
      for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        if (!views[i].has_value()) {
          log_dataset_load_event("dataset_load_read",
                                 "error",
                                 request.ref,
                                 0,
                                 read_start,
                                 read_end,
                                 request.queue_depth_at_start,
                                 request.in_flight_at_start);
          fail_dataset_load(request.chunk_store,
                            request.ref,
                            std::make_exception_ptr(
                                std::runtime_error("dataset chunk disappeared during load")));
          continue;
        }
        const std::size_t bytes = views[i]->data.size();
        fulfill_dataset_load(request.chunk_store, request.ref, std::move(*views[i]));
        log_dataset_load_event("dataset_load_read",
                               "end",
                               request.ref,
                               bytes,
                               read_start,
                               read_end,
                               request.queue_depth_at_start,
                               request.in_flight_at_start);
      }
    } catch (...) {
      const double read_end = now_seconds();
      for (const auto& request : requests) {
        log_dataset_load_event("dataset_load_read",
                               "error",
                               request.ref,
                               0,
                               read_start,
                               read_end,
                               request.queue_depth_at_start,
                               request.in_flight_at_start);
        fail_dataset_load(request.chunk_store, request.ref, std::current_exception());
      }
    }
  }

 private:
  void schedule_pump_locked() {
    if (pump_scheduled_) {
      return;
    }
    pump_scheduled_ = true;
    hpx::post([] { dataset_load_queue().pump(); });
  }

  void pump() {
    std::vector<DatasetLoadBatch> ready;
    {
      std::lock_guard<hpx::mutex> lock(mutex_);
      pump_scheduled_ = false;
      pump_locked(ready);
      if (!pending_.empty() && in_flight_ < dataset_load_concurrency()) {
        schedule_pump_locked();
      }
    }
    post_ready(std::move(ready));
  }

  void pump_locked(std::vector<DatasetLoadBatch>& ready) {
    const int32_t max_in_flight = dataset_load_concurrency();
    while (in_flight_ < max_in_flight && !pending_.empty()) {
      DatasetLoadBatch batch = std::move(pending_.front());
      pending_.pop_front();
      ++in_flight_;
      for (auto& request : batch) {
        request.queue_depth_at_start = static_cast<int32_t>(pending_.size());
        request.in_flight_at_start = in_flight_;
      }
      ready.push_back(std::move(batch));
    }
  }

  void finish_one() {
    std::vector<DatasetLoadBatch> ready;
    {
      std::lock_guard<hpx::mutex> lock(mutex_);
      if (in_flight_ > 0) {
        --in_flight_;
      }
      pump_locked(ready);
      if (!pending_.empty() && in_flight_ < dataset_load_concurrency()) {
        schedule_pump_locked();
      }
    }
    post_ready(std::move(ready));
  }

  static void post_ready(std::vector<DatasetLoadBatch> ready) {
    for (auto& batch : ready) {
      post_dataset_load(std::move(batch));
    }
  }

  hpx::mutex mutex_;
  std::deque<DatasetLoadBatch> pending_;
  int32_t in_flight_ = 0;
  bool pump_scheduled_ = false;
};

std::string dataset_load_pool_name() {
  static const std::string value = []() -> std::string {
    const char* env = std::getenv("KANGAROO_PLOTFILE_IO_POOL");
    if (env != nullptr && *env != '\0') {
      return env;
    }
    return "plotfile_io";
  }();
  return value;
}

void post_dataset_load(DatasetLoadBatch batch) {
  auto batch_ptr = std::make_shared<DatasetLoadBatch>(std::move(batch));
  const auto task = [batch_ptr]() {
    dataset_load_queue().run(std::move(*batch_ptr));
  };

  const std::string pool_name = dataset_load_pool_name();
  if (!pool_name.empty()) {
    try {
      if (hpx::resource::pool_exists(pool_name)) {
        auto& pool = hpx::resource::get_thread_pool(pool_name);
        hpx::execution::parallel_executor exec(&pool);
        hpx::post(exec, std::move(task));
        return;
      }
    } catch (...) {
    }
  }

  hpx::post(std::move(task));
}

DatasetLoadQueue& dataset_load_queue() {
  static DatasetLoadQueue queue;
  return queue;
}

}  // namespace

DataServiceLocal::DataServiceLocal(int32_t run_id,
                                   const DatasetHandle* dataset,
                                   std::shared_ptr<ChunkStore> chunk_store)
    : run_id_(run_id), dataset_(dataset), chunk_store_(std::move(chunk_store)) {
  if (run_id_ == 0 && chunk_store_ == nullptr) {
    chunk_store_ = std::make_shared<ChunkStore>();
  }
}

const DatasetHandle* DataServiceLocal::resolve_dataset() const {
  if (dataset_ != nullptr) {
    return dataset_;
  }
  if (run_id_ != 0) {
    return &execution_context(run_id_).dataset;
  }
  return nullptr;
}

std::shared_ptr<ChunkStore> DataServiceLocal::resolve_chunk_store() const {
  if (chunk_store_ != nullptr) {
    return chunk_store_;
  }
  if (run_id_ != 0) {
    return execution_context_shared(run_id_)->chunk_store;
  }
  return nullptr;
}

int DataServiceLocal::home_rank(const ChunkRef& ref) const {
  std::size_t h = 0xcbf29ce484222325ull;
  auto mix = [&](auto v) {
    h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  };
  mix(ref.step);
  mix(ref.level);
  mix(ref.block);
  auto localities = hpx::find_all_localities();
  return static_cast<int>(h % localities.size());
}

HostView DataServiceLocal::alloc_host(const ChunkRef&, std::size_t bytes) {
  HostView view;
  view.data.resize(bytes);
  return view;
}

hpx::shared_future<std::shared_ptr<HostView>> DataServiceLocal::get_local_shared_impl(
    const ChunkRef& ref) {
  std::shared_ptr<ExecutionContext> ctx;
  const DatasetHandle* dataset = dataset_;
  auto chunk_store = chunk_store_;
  if (run_id_ != 0) {
    ctx = execution_context_shared(run_id_);
    if (dataset == nullptr) {
      dataset = &ctx->dataset;
    }
    if (chunk_store == nullptr) {
      chunk_store = ctx->chunk_store;
    }
  }
  if (chunk_store == nullptr) {
    throw std::runtime_error("chunk store not initialized");
  }

  hpx::shared_future<std::shared_ptr<HostView>> future;
  {
    std::lock_guard<ChunkStore::Mutex> lock(chunk_store->mutex);
    auto it = chunk_store->data.find(ref);
    if (it != chunk_store->data.end() &&
        (it->second.ready || it->second.dataset_load_started || dataset == nullptr)) {
      return it->second.future;
    }
  }

  const bool dataset_has_chunk = dataset != nullptr && dataset->has_chunk(ref);
  if (!dataset_has_chunk && run_id_ != 0 && !execution_context_may_produce_chunk(run_id_, ref)) {
    return hpx::make_exceptional_future<std::shared_ptr<HostView>>(
               std::runtime_error("chunk is not available from run dataset or planned outputs"))
        .share();
  }

  bool can_probe_dataset = false;
  {
    std::lock_guard<ChunkStore::Mutex> lock(chunk_store->mutex);
    auto [it, inserted] = chunk_store->data.try_emplace(ref, make_pending_slot());
    (void)inserted;
    future = it->second.future;
    can_probe_dataset = !it->second.ready && !it->second.dataset_load_started && dataset_has_chunk;
  }

  if (!can_probe_dataset) {
    return future;
  }

  std::shared_ptr<const DatasetHandle> dataset_for_load;
  if (ctx) {
    dataset_for_load = std::shared_ptr<const DatasetHandle>(ctx, &ctx->dataset);
  } else if (dataset != nullptr) {
    dataset_for_load = std::make_shared<DatasetHandle>(*dataset);
  }

  bool start_load = false;
  {
    std::lock_guard<ChunkStore::Mutex> lock(chunk_store->mutex);
    auto it = chunk_store->data.find(ref);
    if (it != chunk_store->data.end() && !it->second.ready && !it->second.dataset_load_started &&
        dataset_has_chunk) {
      it->second.dataset_load_started = true;
      future = it->second.future;
      start_load = true;
    }
  }

  if (start_load) {
    DatasetLoadRequest request;
    request.dataset = std::move(dataset_for_load);
    request.chunk_store = std::move(chunk_store);
    request.ref = ref;
    dataset_load_queue().enqueue(std::move(request));
  }

  return future;
}

std::vector<hpx::shared_future<std::shared_ptr<HostView>>>
DataServiceLocal::get_local_shared_batch_impl(const std::vector<ChunkRef>& refs) {
  std::shared_ptr<ExecutionContext> ctx;
  const DatasetHandle* dataset = dataset_;
  auto chunk_store = chunk_store_;
  if (run_id_ != 0) {
    ctx = execution_context_shared(run_id_);
    if (dataset == nullptr) {
      dataset = &ctx->dataset;
    }
    if (chunk_store == nullptr) {
      chunk_store = ctx->chunk_store;
    }
  }
  if (chunk_store == nullptr) {
    throw std::runtime_error("chunk store not initialized");
  }

  std::vector<hpx::shared_future<std::shared_ptr<HostView>>> futures(refs.size());
  std::vector<std::size_t> unresolved;
  unresolved.reserve(refs.size());
  {
    std::lock_guard<ChunkStore::Mutex> lock(chunk_store->mutex);
    for (std::size_t i = 0; i < refs.size(); ++i) {
      auto it = chunk_store->data.find(refs[i]);
      if (it != chunk_store->data.end() &&
          (it->second.ready || it->second.dataset_load_started || dataset == nullptr)) {
        futures[i] = it->second.future;
      } else {
        unresolved.push_back(i);
      }
    }
  }

  struct PendingRef {
    std::size_t index = 0;
    ChunkRef ref;
    bool dataset_has_chunk = false;
  };

  std::vector<PendingRef> pending_refs;
  pending_refs.reserve(unresolved.size());
  for (std::size_t idx : unresolved) {
    const auto& ref = refs[idx];
    const bool dataset_has_chunk = dataset != nullptr && dataset->has_chunk(ref);
    if (!dataset_has_chunk && run_id_ != 0 && !execution_context_may_produce_chunk(run_id_, ref)) {
      futures[idx] = hpx::make_exceptional_future<std::shared_ptr<HostView>>(
                         std::runtime_error(
                             "chunk is not available from run dataset or planned outputs"))
                         .share();
      continue;
    }
    pending_refs.push_back(PendingRef{idx, ref, dataset_has_chunk});
  }

  std::shared_ptr<const DatasetHandle> dataset_for_load;
  if (ctx) {
    dataset_for_load = std::shared_ptr<const DatasetHandle>(ctx, &ctx->dataset);
  } else if (dataset != nullptr) {
    dataset_for_load = std::make_shared<DatasetHandle>(*dataset);
  }

  std::vector<ChunkRef> load_refs;
  load_refs.reserve(pending_refs.size());
  {
    std::lock_guard<ChunkStore::Mutex> lock(chunk_store->mutex);
    for (const auto& pending : pending_refs) {
      auto [it, inserted] = chunk_store->data.try_emplace(pending.ref, make_pending_slot());
      (void)inserted;
      futures[pending.index] = it->second.future;
      if (!it->second.ready && !it->second.dataset_load_started && pending.dataset_has_chunk) {
        it->second.dataset_load_started = true;
        load_refs.push_back(pending.ref);
      }
    }
  }

  DatasetLoadBatch load_requests;
  load_requests.reserve(load_refs.size());
  for (const auto& ref : load_refs) {
    DatasetLoadRequest request;
    request.dataset = dataset_for_load;
    request.chunk_store = chunk_store;
    request.ref = ref;
    load_requests.push_back(std::move(request));
  }

  if (!load_requests.empty()) {
    dataset_load_queue().enqueue_many(std::move(load_requests));
  }

  return futures;
}

hpx::future<HostView> DataServiceLocal::get_local_impl(const ChunkRef& ref) {
  auto ready = get_local_shared_impl(ref);
  if (ready.is_ready()) {
    auto view = ready.get();
    return hpx::make_ready_future(*view);
  }
  return ready.then([](auto&& ready_ref) {
    auto view = ready_ref.get();
    return *view;
  });
}

hpx::shared_future<std::shared_ptr<HostView>> DataServiceLocal::get_host_shared(
    const ChunkRef& ref) {
  const int target = home_rank(ref);
  const int here = hpx::get_locality_id();
  if (target == here) {
    return get_local_shared_impl(ref);
  }

  return get_host(ref).then([](auto&& result) {
    return std::make_shared<HostView>(result.get());
  }).share();
}

std::vector<hpx::shared_future<std::shared_ptr<HostView>>> DataServiceLocal::get_hosts_shared(
    const std::vector<ChunkRef>& refs) {
  std::vector<hpx::shared_future<std::shared_ptr<HostView>>> out(refs.size());
  if (refs.empty()) {
    return out;
  }

  const int here = hpx::get_locality_id();
  std::vector<std::size_t> local_indices;
  std::vector<ChunkRef> local_refs;
  local_indices.reserve(refs.size());
  local_refs.reserve(refs.size());

  for (std::size_t i = 0; i < refs.size(); ++i) {
    if (home_rank(refs[i]) == here) {
      local_indices.push_back(i);
      local_refs.push_back(refs[i]);
    } else {
      out[i] = get_host_shared(refs[i]);
    }
  }

  if (!local_refs.empty()) {
    auto local = get_local_shared_batch_impl(local_refs);
    for (std::size_t j = 0; j < local_refs.size(); ++j) {
      out[local_indices[j]] = std::move(local[j]);
    }
  }

  return out;
}

hpx::future<HostView> data_get_local_impl(int32_t run_id, const ChunkRef& ref) {
  DataServiceLocal data_service(run_id);
  return data_service.get_local_impl(ref);
}

void data_put_local_impl(int32_t run_id, const ChunkRef& ref, HostView view) {
  DataServiceLocal data_service(run_id);
  data_service.put_local_impl(ref, std::move(view));
}

void DataServiceLocal::put_local_impl(const ChunkRef& ref, HostView view) {
  const int here = hpx::get_locality_id();
  const std::size_t bytes = view.data.size();
  log_dataflow_marker("put_local_enter", ref, here, here, bytes);

  auto chunk_store = resolve_chunk_store();
  if (chunk_store == nullptr) {
    throw std::runtime_error("chunk store not initialized");
  }
  auto shared_view = std::make_shared<HostView>(std::move(view));
  std::shared_ptr<hpx::promise<std::shared_ptr<HostView>>> promise;
  log_dataflow_marker("put_local_before_lock", ref, here, here, bytes);
  const double lock_wait_start = now_seconds();
  double lock_hold_start = 0.0;
  {
    std::unique_lock<ChunkStore::Mutex> lock(chunk_store->mutex);
    log_dataflow_event("put_local_lock_wait", "end", ref, here, here, bytes, lock_wait_start);
    lock_hold_start = now_seconds();
    log_dataflow_marker("put_local_after_lock", ref, here, here, bytes);
    auto it = chunk_store->data.find(ref);
    if (it == chunk_store->data.end() || it->second.ready) {
      ChunkSlot slot = make_pending_slot();
      slot.ready = true;
      slot.value = shared_view;
      promise = slot.promise;
      chunk_store->data[ref] = std::move(slot);
    } else {
      it->second.ready = true;
      it->second.dataset_load_started = false;
      it->second.value = shared_view;
      promise = it->second.promise;
    }
    log_dataflow_marker("put_local_after_store_update", ref, here, here, bytes);
  }
  log_dataflow_event("put_local_lock_hold", "end", ref, here, here, bytes, lock_hold_start);

  log_dataflow_marker("put_local_promise_set_begin", ref, here, here, bytes);
  promise->set_value(std::move(shared_view));
  log_dataflow_marker("put_local_promise_set_end", ref, here, here, bytes);
}

hpx::future<HostView> DataServiceLocal::get_host(const ChunkRef& ref) {
  int target = home_rank(ref);
  int here = hpx::get_locality_id();
  const double start = now_seconds();
  if (target == here) {
    auto local = get_local_impl(ref);
    if (local.is_ready()) {
      HostView view = local.get();
      log_dataflow_fetch("get_host_local", ref, here, target, view.data.size());
      log_dataflow_event("get_host", "end", ref, here, target, view.data.size(), start);
      return hpx::make_ready_future(std::move(view));
    }
    return local.then([ref, here, target, start](auto&& result) mutable {
      try {
        HostView view = result.get();
        log_dataflow_fetch("get_host_local", ref, here, target, view.data.size());
        log_dataflow_event("get_host", "end", ref, here, target, view.data.size(), start);
        return view;
      } catch (...) {
        log_dataflow_event("get_host", "error", ref, here, target, 0, start);
        throw;
      }
    });
  }
  auto localities = hpx::find_all_localities();
  return hpx::unwrap(
             hpx::async<::kangaroo_data_get_local_action>(localities.at(target), run_id_, ref))
      .then([ref, here, target, start](auto&& result) mutable {
        try {
          HostView view = result.get();
          log_dataflow_fetch("get_host_remote", ref, here, target, view.data.size());
          log_dataflow_event("get_host", "end", ref, here, target, view.data.size(), start);
          return view;
        } catch (...) {
          log_dataflow_event("get_host", "error", ref, here, target, 0, start);
          throw;
        }
      });
}

std::vector<hpx::future<HostView>> DataServiceLocal::get_hosts(
    const std::vector<ChunkRef>& refs) {
  std::vector<hpx::future<HostView>> out(refs.size());
  if (refs.empty()) {
    return out;
  }

  const int here = hpx::get_locality_id();
  std::vector<std::size_t> local_indices;
  std::vector<ChunkRef> local_refs;
  std::vector<double> local_starts;
  local_indices.reserve(refs.size());
  local_refs.reserve(refs.size());
  local_starts.reserve(refs.size());

  for (std::size_t i = 0; i < refs.size(); ++i) {
    const int target = home_rank(refs[i]);
    if (target == here) {
      local_indices.push_back(i);
      local_refs.push_back(refs[i]);
      local_starts.push_back(now_seconds());
    } else {
      out[i] = get_host(refs[i]);
    }
  }

  if (!local_refs.empty()) {
    auto shared_futures = get_local_shared_batch_impl(local_refs);
    for (std::size_t j = 0; j < local_refs.size(); ++j) {
      const auto ref = local_refs[j];
      const int target = here;
      const double start = local_starts[j];
      auto ready = std::move(shared_futures[j]);
      if (ready.is_ready()) {
        HostView view = *ready.get();
        log_dataflow_fetch("get_host_local", ref, here, target, view.data.size());
        log_dataflow_event("get_host", "end", ref, here, target, view.data.size(), start);
        out[local_indices[j]] = hpx::make_ready_future(std::move(view));
      } else {
        out[local_indices[j]] = ready.then([ref, here, target, start](auto&& result) mutable {
          try {
            HostView view = *result.get();
            log_dataflow_fetch("get_host_local", ref, here, target, view.data.size());
            log_dataflow_event("get_host", "end", ref, here, target, view.data.size(), start);
            return view;
          } catch (...) {
            log_dataflow_event("get_host", "error", ref, here, target, 0, start);
            throw;
          }
        });
      }
    }
  }

  return out;
}

hpx::future<SubboxView> DataServiceLocal::get_subbox(const ChunkSubboxRef& ref) {
  return get_host(ref.chunk).then([ref](auto&& result) {
    HostView chunk = result.get();
    return build_subbox_view(chunk, ref);
  });
}

hpx::future<void> DataServiceLocal::put_host(const ChunkRef& ref, HostView view) {
  int target = home_rank(ref);
  int here = hpx::get_locality_id();
  const double start = now_seconds();
  const std::size_t bytes = view.data.size();
  log_dataflow_marker("put_host_enter", ref, here, target, bytes);
  log_dataflow_fetch("put_host", ref, here, target, bytes);
  if (target == here) {
    log_dataflow_marker("put_host_local_begin", ref, here, target, bytes);
    put_local_impl(ref, std::move(view));
    log_dataflow_marker("put_host_local_return", ref, here, target, bytes);
    log_dataflow_event("put_host", "end", ref, here, target, bytes, start);
    log_dataflow_marker("put_host_return", ref, here, target, bytes);
    return hpx::make_ready_future();
  }
  auto localities = hpx::find_all_localities();
  log_dataflow_marker("put_host_remote_begin", ref, here, target, bytes);
  return hpx::async<::kangaroo_data_put_local_action>(localities.at(target), run_id_, ref,
                                                      std::move(view))
      .then([ref, here, target, bytes, start](auto&& result) mutable {
        try {
          result.get();
          log_dataflow_marker("put_host_remote_return", ref, here, target, bytes);
          log_dataflow_event("put_host", "end", ref, here, target, bytes, start);
          log_dataflow_marker("put_host_return", ref, here, target, bytes);
          return;
        } catch (...) {
          log_dataflow_event("put_host", "error", ref, here, target, bytes, start);
          throw;
        }
      });
}

void DataServiceLocal::preload(const RunMeta& meta,
                               const DatasetHandle& dataset,
                               std::shared_ptr<ChunkStore> chunk_store,
                               const std::vector<int32_t>& fields) {
  if (fields.empty()) {
    return;
  }
  if (meta.steps.empty()) {
    return;
  }
  if (dataset.step < 0 || dataset.step >= static_cast<int32_t>(meta.steps.size())) {
    return;
  }
  const auto& step_meta = meta.steps.at(dataset.step);
  if (dataset.level < 0 || dataset.level >= static_cast<int16_t>(step_meta.levels.size())) {
    return;
  }
  const auto& level_meta = step_meta.levels.at(dataset.level);
  const int32_t nblocks = static_cast<int32_t>(level_meta.boxes.size());
  if (nblocks <= 0) {
    return;
  }

  DataServiceLocal local(0, &dataset, std::move(chunk_store));
  const int here = hpx::get_locality_id();

  for (int32_t block = 0; block < nblocks; ++block) {
    for (int32_t field : fields) {
      ChunkRef ref{dataset.step, dataset.level, field, 0, block};
      if (local.home_rank(ref) != here) {
        continue;
      }
      auto view = dataset.get_chunk(ref);
      if (!view.has_value()) {
        continue;
      }
      local.put_local_impl(ref, std::move(*view));
    }
  }
}

}  // namespace kangaroo
