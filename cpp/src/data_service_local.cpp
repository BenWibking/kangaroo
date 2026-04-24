#include "kangaroo/data_service_local.hpp"

#include "kangaroo/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
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

  auto in_index = [&](int32_t i, int32_t j, int32_t k) -> std::size_t {
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ny) +
            static_cast<std::size_t>(j)) *
               static_cast<std::size_t>(nz) +
           static_cast<std::size_t>(k);
  };
  auto out_index = [&](int32_t i, int32_t j, int32_t k) -> std::size_t {
    return (static_cast<std::size_t>(i) * static_cast<std::size_t>(ony) +
            static_cast<std::size_t>(j)) *
               static_cast<std::size_t>(onz) +
           static_cast<std::size_t>(k);
  };

  const auto* src = chunk.data.data();
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
        const std::size_t src_byte = in_index(li, lj, lk) * bytes_per;
        const std::size_t dst_byte = out_index(i, j, k) * bytes_per;
        std::memcpy(dst + dst_byte, src + src_byte, bytes_per);
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
    std::lock_guard<std::mutex> lock(chunk_store->mutex);
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

  promise->set_value(std::move(shared_view));
}

void fail_dataset_load(const std::shared_ptr<ChunkStore>& chunk_store,
                       const ChunkRef& ref,
                       std::exception_ptr error) {
  std::shared_ptr<hpx::promise<std::shared_ptr<HostView>>> promise;
  {
    std::lock_guard<std::mutex> lock(chunk_store->mutex);
    auto [it, inserted] = chunk_store->data.try_emplace(ref, make_pending_slot());
    (void)inserted;
    if (it->second.ready) {
      return;
    }
    it->second.ready = true;
    it->second.dataset_load_started = false;
    promise = it->second.promise;
  }

  promise->set_exception(error);
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
    std::lock_guard<std::mutex> lock(chunk_store->mutex);
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
    std::lock_guard<std::mutex> lock(chunk_store->mutex);
    auto [it, inserted] = chunk_store->data.try_emplace(ref, make_pending_slot());
    (void)inserted;
    future = it->second.future;
    can_probe_dataset = !it->second.ready && !it->second.dataset_load_started && dataset_has_chunk;
  }

  if (!can_probe_dataset) {
    return future;
  }

  bool start_load = false;
  {
    std::lock_guard<std::mutex> lock(chunk_store->mutex);
    auto it = chunk_store->data.find(ref);
    if (it != chunk_store->data.end() && !it->second.ready && !it->second.dataset_load_started &&
        dataset_has_chunk) {
      it->second.dataset_load_started = true;
      future = it->second.future;
      start_load = true;
    }
  }

  if (start_load) {
    hpx::async([ref, dataset, chunk_store, ctx]() {
      try {
        if (dataset == nullptr) {
          throw std::runtime_error("dataset not initialized");
        }
        auto view = dataset->get_chunk(ref);
        if (!view.has_value()) {
          throw std::runtime_error("dataset chunk disappeared during async load");
        }
        fulfill_dataset_load(chunk_store, ref, std::move(*view));
      } catch (...) {
        fail_dataset_load(chunk_store, ref, std::current_exception());
      }
    });
  }

  return future;
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

hpx::future<HostView> data_get_local_impl(int32_t run_id, const ChunkRef& ref) {
  DataServiceLocal data_service(run_id);
  return data_service.get_local_impl(ref);
}

void data_put_local_impl(int32_t run_id, const ChunkRef& ref, HostView view) {
  DataServiceLocal data_service(run_id);
  data_service.put_local_impl(ref, std::move(view));
}

void DataServiceLocal::put_local_impl(const ChunkRef& ref, HostView view) {
  auto chunk_store = resolve_chunk_store();
  if (chunk_store == nullptr) {
    throw std::runtime_error("chunk store not initialized");
  }
  auto shared_view = std::make_shared<HostView>(std::move(view));
  std::shared_ptr<hpx::promise<std::shared_ptr<HostView>>> promise;
  {
    std::lock_guard<std::mutex> lock(chunk_store->mutex);
    auto it = chunk_store->data.find(ref);
    if (it == chunk_store->data.end() || it->second.ready) {
      ChunkSlot slot = make_pending_slot();
      slot.ready = true;
      slot.value = shared_view;
      slot.promise->set_value(shared_view);
      chunk_store->data[ref] = std::move(slot);
      return;
    }
    it->second.ready = true;
    it->second.dataset_load_started = false;
    it->second.value = shared_view;
    promise = it->second.promise;
  }

  promise->set_value(std::move(shared_view));
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
  log_dataflow_fetch("put_host", ref, here, target, bytes);
  if (target == here) {
    put_local_impl(ref, std::move(view));
    log_dataflow_event("put_host", "end", ref, here, target, bytes, start);
    return hpx::make_ready_future();
  }
  auto localities = hpx::find_all_localities();
  return hpx::async<::kangaroo_data_put_local_action>(localities.at(target), run_id_, ref,
                                                      std::move(view))
      .then([ref, here, target, bytes, start](auto&& result) mutable {
        try {
          result.get();
          log_dataflow_event("put_host", "end", ref, here, target, bytes, start);
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
