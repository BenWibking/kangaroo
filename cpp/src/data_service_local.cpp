#include "kangaroo/data_service_local.hpp"

#include "kangaroo/runtime.hpp"

#include <functional>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>

HPX_PLAIN_ACTION(kangaroo::data_get_local_impl, kangaroo_data_get_local_action)
HPX_PLAIN_ACTION(kangaroo::data_put_local_impl, kangaroo_data_put_local_action)

namespace kangaroo {

DataServiceLocal::DataServiceLocal() = default;

std::mutex DataServiceLocal::mutex_;
DataServiceLocal::MapT DataServiceLocal::data_;
const DatasetHandle* DataServiceLocal::dataset_ = nullptr;

void DataServiceLocal::set_dataset(const DatasetHandle* dataset) {
  dataset_ = dataset;
}

int DataServiceLocal::home_rank(const ChunkRef& ref) const {
  std::size_t h = ChunkRefHash{}(ref);
  auto localities = hpx::find_all_localities();
  return static_cast<int>(h % localities.size());
}

HostView DataServiceLocal::alloc_host(const ChunkRef&, std::size_t bytes) {
  HostView view;
  view.data.resize(bytes);
  return view;
}

HostView DataServiceLocal::get_local_impl(const ChunkRef& ref) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = data_.find(ref);
    if (it != data_.end()) {
      return it->second;
    }
  }

  if (dataset_ != nullptr) {
    auto view = dataset_->get_chunk(ref);
    if (view.has_value()) {
      std::lock_guard<std::mutex> lock(mutex_);
      auto [it, inserted] = data_.emplace(ref, *view);
      if (!inserted) {
        it->second = *view;
      }
      return it->second;
    }
  }

  return HostView{};
}

void DataServiceLocal::put_local_impl(const ChunkRef& ref, HostView view) {
  std::lock_guard<std::mutex> lock(mutex_);
  data_[ref] = std::move(view);
}

HostView data_get_local_impl(const ChunkRef& ref) {
  return DataServiceLocal::get_local_impl(ref);
}

void data_put_local_impl(const ChunkRef& ref, HostView view) {
  DataServiceLocal::put_local_impl(ref, std::move(view));
}

hpx::future<HostView> DataServiceLocal::get_host(const ChunkRef& ref) {
  int target = home_rank(ref);
  int here = hpx::get_locality_id();
  if (target == here) {
    return hpx::make_ready_future(get_local_impl(ref));
  }
  auto localities = hpx::find_all_localities();
  return hpx::async<::kangaroo_data_get_local_action>(localities.at(target), ref);
}

hpx::future<void> DataServiceLocal::put_host(const ChunkRef& ref, HostView view) {
  int target = home_rank(ref);
  int here = hpx::get_locality_id();
  if (target == here) {
    put_local_impl(ref, std::move(view));
    return hpx::make_ready_future();
  }
  auto localities = hpx::find_all_localities();
  return hpx::async<::kangaroo_data_put_local_action>(localities.at(target), ref, std::move(view))
      .then(
      [](auto&&) { return; });
}

void DataServiceLocal::preload(const RunMeta& meta,
                               const DatasetHandle& dataset,
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

  DataServiceLocal local;
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
      std::lock_guard<std::mutex> lock(mutex_);
      data_[ref] = std::move(*view);
    }
  }
}

}  // namespace kangaroo
