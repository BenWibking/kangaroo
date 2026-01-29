#pragma once

#include "kangaroo/runmeta.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace kangaroo {

enum class Face : uint8_t { Xm = 0, Xp = 1, Ym = 2, Yp = 3, Zm = 4, Zp = 5 };

struct NeighborSpan {
  const int32_t* ptr = nullptr;
  int32_t n = 0;
};

class AdjacencyService {
 public:
  virtual ~AdjacencyService() = default;
  virtual NeighborSpan neighbors(int32_t step, int16_t level, int32_t block, Face face) = 0;
};

class AdjacencyServiceLocal final : public AdjacencyService {
 public:
  explicit AdjacencyServiceLocal(const RunMeta& meta);

  NeighborSpan neighbors(int32_t step, int16_t level, int32_t block, Face face) override;

 private:
  struct CacheKey {
    int32_t step = 0;
    int16_t level = 0;
    int32_t block = 0;
    Face face = Face::Xm;

    bool operator==(const CacheKey& other) const;
  };

  struct CacheKeyHash {
    std::size_t operator()(const CacheKey& key) const;
  };

  NeighborSpan compute_neighbors(int32_t step, int16_t level, int32_t block, Face face);

  const RunMeta& meta_;
  std::unordered_map<CacheKey, std::vector<int32_t>, CacheKeyHash> cache_;
  std::vector<int32_t> scratch_;
};

}  // namespace kangaroo
