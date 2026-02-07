#pragma once

#include "kangaroo/data_service.hpp"

#include <mutex>
#include <unordered_map>
#include <vector>

#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>

namespace kangaroo {

HostView data_get_local_impl(const ChunkRef& ref);
SubboxView data_get_subbox_local_impl(const ChunkSubboxRef& ref);
void data_put_local_impl(const ChunkRef& ref, HostView view);
struct DatasetHandle;
struct RunMeta;

class DataServiceLocal : public DataService {
 public:
  DataServiceLocal();

  int home_rank(const ChunkRef& ref) const override;
  HostView alloc_host(const ChunkRef& ref, std::size_t bytes) override;
  hpx::future<HostView> get_host(const ChunkRef& ref) override;
  hpx::future<SubboxView> get_subbox(const ChunkSubboxRef& ref) override;
  hpx::future<void> put_host(const ChunkRef& ref, HostView view) override;
  static void set_dataset(const DatasetHandle* dataset);
  static void preload(const RunMeta& meta,
                      const DatasetHandle& dataset,
                      const std::vector<int32_t>& fields);

 private:
  friend HostView data_get_local_impl(const ChunkRef& ref);
  friend SubboxView data_get_subbox_local_impl(const ChunkSubboxRef& ref);
  friend void data_put_local_impl(const ChunkRef& ref, HostView view);

  using MapT = std::unordered_map<ChunkRef, HostView, ChunkRefHash, ChunkRefEq>;

  static HostView get_local_impl(const ChunkRef& ref);
  static void put_local_impl(const ChunkRef& ref, HostView view);

  static std::mutex mutex_;
  static MapT data_;
  static const DatasetHandle* dataset_;
};

}  // namespace kangaroo
