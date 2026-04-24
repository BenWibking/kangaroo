#pragma once

#include "kangaroo/data_service.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>

namespace kangaroo {

hpx::future<HostView> data_get_local_impl(int32_t run_id, const ChunkRef& ref);
void data_put_local_impl(int32_t run_id, const ChunkRef& ref, HostView view);
struct ChunkStore;
struct DatasetHandle;
struct RunMeta;

class DataServiceLocal : public DataService {
 public:
  explicit DataServiceLocal(int32_t run_id = 0,
                            const DatasetHandle* dataset = nullptr,
                            std::shared_ptr<ChunkStore> chunk_store = nullptr);

  int home_rank(const ChunkRef& ref) const override;
  HostView alloc_host(const ChunkRef& ref, std::size_t bytes) override;
  hpx::future<HostView> get_host(const ChunkRef& ref) override;
  hpx::future<SubboxView> get_subbox(const ChunkSubboxRef& ref) override;
  hpx::future<void> put_host(const ChunkRef& ref, HostView view) override;
  static void preload(const RunMeta& meta,
                      const DatasetHandle& dataset,
                      std::shared_ptr<ChunkStore> chunk_store,
                      const std::vector<int32_t>& fields);

 private:
  friend hpx::future<HostView> data_get_local_impl(int32_t run_id, const ChunkRef& ref);
  friend void data_put_local_impl(int32_t run_id, const ChunkRef& ref, HostView view);

  const DatasetHandle* resolve_dataset() const;
  std::shared_ptr<ChunkStore> resolve_chunk_store() const;
  hpx::shared_future<std::shared_ptr<HostView>> get_local_shared_impl(const ChunkRef& ref);
  hpx::future<HostView> get_local_impl(const ChunkRef& ref);
  void put_local_impl(const ChunkRef& ref, HostView view);

  int32_t run_id_ = 0;
  const DatasetHandle* dataset_ = nullptr;
  std::shared_ptr<ChunkStore> chunk_store_;
};

}  // namespace kangaroo
