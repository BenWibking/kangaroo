#include "kangaroo/data_service_local.hpp"

#include <functional>

#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>

HPX_PLAIN_ACTION(kangaroo::data_get_local_impl, kangaroo_data_get_local_action)
HPX_PLAIN_ACTION(kangaroo::data_put_local_impl, kangaroo_data_put_local_action)

namespace kangaroo {

DataServiceLocal::DataServiceLocal() = default;

std::mutex DataServiceLocal::mutex_;
DataServiceLocal::MapT DataServiceLocal::data_;

std::size_t DataServiceLocal::KeyHash::operator()(const ChunkRef& ref) const {
  std::size_t h = 0xcbf29ce484222325ull;
  auto mix = [&](auto v) {
    h ^= static_cast<std::size_t>(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  };
  mix(ref.step);
  mix(ref.level);
  mix(ref.field);
  mix(ref.version);
  mix(ref.block);
  return h;
}

bool DataServiceLocal::KeyEq::operator()(const ChunkRef& a, const ChunkRef& b) const {
  return a.step == b.step && a.level == b.level && a.field == b.field && a.version == b.version &&
         a.block == b.block;
}

int DataServiceLocal::home_rank(const ChunkRef& ref) const {
  std::size_t h = KeyHash{}(ref);
  auto localities = hpx::find_all_localities();
  return static_cast<int>(h % localities.size());
}

HostView DataServiceLocal::get_local_impl(const ChunkRef& ref) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(ref);
  if (it == data_.end()) {
    return HostView{};
  }
  return it->second;
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

}  // namespace kangaroo
