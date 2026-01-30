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
void data_put_local_impl(const ChunkRef& ref, HostView view);

class DataServiceLocal : public DataService {
 public:
  DataServiceLocal();

  int home_rank(const ChunkRef& ref) const override;
  HostView alloc_host(const ChunkRef& ref, std::size_t bytes) override;
  hpx::future<HostView> get_host(const ChunkRef& ref) override;
  hpx::future<void> put_host(const ChunkRef& ref, HostView view) override;

 private:
  friend HostView data_get_local_impl(const ChunkRef& ref);
  friend void data_put_local_impl(const ChunkRef& ref, HostView view);

  struct KeyHash {
    std::size_t operator()(const ChunkRef& ref) const;
  };

  struct KeyEq {
    bool operator()(const ChunkRef& a, const ChunkRef& b) const;
  };

  using MapT = std::unordered_map<ChunkRef, HostView, KeyHash, KeyEq>;

  static HostView get_local_impl(const ChunkRef& ref);
  static void put_local_impl(const ChunkRef& ref, HostView view);

  static std::mutex mutex_;
  static MapT data_;
};

}  // namespace kangaroo
