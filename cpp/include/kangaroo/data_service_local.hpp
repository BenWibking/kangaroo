#pragma once

#include "kangaroo/data_service.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <hpx/include/actions.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime_local/get_locality_id.hpp>
#include <hpx/runtime_distributed/find_localities.hpp>

namespace kangaroo {

hpx::future<ChunkBuffer> data_get_local_impl(int32_t run_id, const ChunkRef& ref);
void data_put_local_impl(int32_t run_id, const ChunkRef& ref, ChunkBuffer view);
struct ChunkConsumerCount {
  ChunkRef ref;
  std::int64_t count = 0;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& ref& count;
  }
};
void data_register_input_consumers_local_impl(int32_t run_id,
                                              const std::vector<ChunkConsumerCount>& counts);
void data_release_consumed_inputs_local_impl(int32_t run_id,
                                             const std::vector<ChunkConsumerCount>& counts);
struct ChunkStore;
struct DatasetHandle;
struct RunMeta;

class DataServiceLocal : public DataService {
 public:
  explicit DataServiceLocal(int32_t run_id = 0,
                            const DatasetHandle* dataset = nullptr,
                            std::shared_ptr<ChunkStore> chunk_store = nullptr);

  int home_rank(const ChunkRef& ref) const override;
  std::size_t estimate_host_bytes(const ChunkRef& ref) const override;
  std::optional<BufferDesc> describe_host(const ChunkRef& ref) const override;
  std::optional<std::uint64_t> estimate_particle_chunk_records(
      const std::string& particle_type, std::int64_t chunk_index) const override;
  ChunkBuffer alloc_host(const ChunkRef& ref, const ResolvedBufferSpec& spec) override;
  hpx::shared_future<std::shared_ptr<ChunkBuffer>> get_host_shared(const ChunkRef& ref);
  std::vector<hpx::shared_future<std::shared_ptr<ChunkBuffer>>> get_hosts_shared(
      const std::vector<ChunkRef>& refs);
  hpx::future<ChunkBuffer> get_host(const ChunkRef& ref) override;
  std::vector<hpx::future<ChunkBuffer>> get_hosts(const std::vector<ChunkRef>& refs) override;
  hpx::future<SubboxView> get_subbox(const ChunkSubboxRef& ref) override;
  hpx::future<void> put_host(const ChunkRef& ref, ChunkBuffer view) override;
  hpx::future<void> register_input_consumers(const std::vector<ChunkConsumerCount>& counts);
  hpx::future<void> release_consumed_inputs(const std::vector<ChunkRef>& refs);
  static void preload(const RunMeta& meta,
                      const DatasetHandle& dataset,
                      std::shared_ptr<ChunkStore> chunk_store,
                      const std::vector<int32_t>& fields);

 private:
  friend hpx::future<ChunkBuffer> data_get_local_impl(int32_t run_id, const ChunkRef& ref);
  friend void data_put_local_impl(int32_t run_id, const ChunkRef& ref, ChunkBuffer view);
  friend void data_register_input_consumers_local_impl(
      int32_t run_id,
      const std::vector<ChunkConsumerCount>& counts);
  friend void data_release_consumed_inputs_local_impl(
      int32_t run_id,
      const std::vector<ChunkConsumerCount>& counts);

  const DatasetHandle* resolve_dataset() const;
  std::shared_ptr<ChunkStore> resolve_chunk_store() const;
  hpx::shared_future<std::shared_ptr<ChunkBuffer>> get_local_shared_impl(const ChunkRef& ref);
  std::vector<hpx::shared_future<std::shared_ptr<ChunkBuffer>>> get_local_shared_batch_impl(
      const std::vector<ChunkRef>& refs);
  hpx::future<ChunkBuffer> get_local_impl(const ChunkRef& ref);
  void put_local_impl(const ChunkRef& ref, ChunkBuffer view);

  int32_t run_id_ = 0;
  const DatasetHandle* dataset_ = nullptr;
  std::shared_ptr<ChunkStore> chunk_store_;
};

}  // namespace kangaroo
