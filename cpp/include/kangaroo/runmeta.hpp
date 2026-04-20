#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kangaroo {

struct Int3 {
  int32_t x = 0;
  int32_t y = 0;
  int32_t z = 0;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& x& y& z;
  }
};

struct BlockBox {
  Int3 lo;
  Int3 hi;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& lo& hi;
  }
};

struct LevelGeom {
  double dx[3] = {0.0, 0.0, 0.0};
  double x0[3] = {0.0, 0.0, 0.0};
  int32_t index_origin[3] = {0, 0, 0};
  bool is_periodic[3] = {false, false, false};
  int ref_ratio = 1;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    for (auto& v : dx) {
      ar& v;
    }
    for (auto& v : x0) {
      ar& v;
    }
    for (auto& v : index_origin) {
      ar& v;
    }
    for (auto& v : is_periodic) {
      ar& v;
    }
    ar& ref_ratio;
  }
};

struct LevelMeta {
  LevelGeom geom;
  std::vector<BlockBox> boxes;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& geom& boxes;
  }
};

struct StepMeta {
  int32_t step = 0;
  std::vector<LevelMeta> levels;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& step& levels;
  }
};

class StepList {
 public:
  using storage_type = std::vector<StepMeta>;
  using iterator = storage_type::iterator;
  using const_iterator = storage_type::const_iterator;

  void reserve(std::size_t n) {
    steps_.reserve(n);
    step_index_.reserve(n);
  }

  void push_back(StepMeta step) {
    const auto step_id = step.step;
    if (step_index_.contains(step_id)) {
      throw std::runtime_error("duplicate RunMeta step");
    }
    step_index_.emplace(step_id, steps_.size());
    steps_.push_back(std::move(step));
  }

  std::size_t size() const {
    return steps_.size();
  }

  bool empty() const {
    return steps_.empty();
  }

  bool contains(int32_t step) const {
    return step_index_.contains(step);
  }

  StepMeta& at(int32_t step) {
    const auto it = step_index_.find(step);
    if (it == step_index_.end()) {
      throw std::out_of_range("RunMeta does not contain requested step");
    }
    return steps_.at(it->second);
  }

  const StepMeta& at(int32_t step) const {
    const auto it = step_index_.find(step);
    if (it == step_index_.end()) {
      throw std::out_of_range("RunMeta does not contain requested step");
    }
    return steps_.at(it->second);
  }

  StepMeta& operator[](int32_t step) {
    return at(step);
  }

  const StepMeta& operator[](int32_t step) const {
    return at(step);
  }

  iterator begin() {
    return steps_.begin();
  }

  const_iterator begin() const {
    return steps_.begin();
  }

  const_iterator cbegin() const {
    return steps_.cbegin();
  }

  iterator end() {
    return steps_.end();
  }

  const_iterator end() const {
    return steps_.end();
  }

  const_iterator cend() const {
    return steps_.cend();
  }

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& steps_;
    rebuild_index();
  }

 private:
  void rebuild_index() {
    step_index_.clear();
    step_index_.reserve(steps_.size());
    for (std::size_t i = 0; i < steps_.size(); ++i) {
      const auto step_id = steps_[i].step;
      if (step_index_.contains(step_id)) {
        throw std::runtime_error("duplicate RunMeta step");
      }
      step_index_.emplace(step_id, i);
    }
  }

  storage_type steps_;
  std::unordered_map<int32_t, std::size_t> step_index_;
};

struct RunMeta {
  StepList steps;
  std::vector<std::pair<std::string, int64_t>> particle_species;

  template <typename Archive>
  void serialize(Archive& ar, unsigned) {
    ar& steps& particle_species;
  }
};

}  // namespace kangaroo
