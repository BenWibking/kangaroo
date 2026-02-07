#include "kangaroo/data_service_local.hpp"

#include "kangaroo/runtime.hpp"

#include <algorithm>
#include <cstring>
#include <functional>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>

HPX_PLAIN_ACTION(kangaroo::data_get_local_impl, kangaroo_data_get_local_action)
HPX_PLAIN_ACTION(kangaroo::data_get_subbox_local_impl, kangaroo_data_get_subbox_local_action)
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

SubboxView data_get_subbox_local_impl(const ChunkSubboxRef& ref) {
  SubboxView out;
  out.bytes_per_value = ref.bytes_per_value;
  out.box = ref.request_box;

  const HostView chunk = DataServiceLocal::get_local_impl(ref.chunk);
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

hpx::future<HostView> DataServiceLocal::get_host(const ChunkRef& ref) {
  int target = home_rank(ref);
  int here = hpx::get_locality_id();
  if (target == here) {
    return hpx::make_ready_future(get_local_impl(ref));
  }
  auto localities = hpx::find_all_localities();
  return hpx::async<::kangaroo_data_get_local_action>(localities.at(target), ref);
}

hpx::future<SubboxView> DataServiceLocal::get_subbox(const ChunkSubboxRef& ref) {
  const int target = home_rank(ref.chunk);
  const int here = hpx::get_locality_id();
  if (target == here) {
    return hpx::make_ready_future(data_get_subbox_local_impl(ref));
  }
  auto localities = hpx::find_all_localities();
  return hpx::async<::kangaroo_data_get_subbox_local_action>(localities.at(target), ref);
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
