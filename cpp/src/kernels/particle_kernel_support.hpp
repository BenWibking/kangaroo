#pragma once

#include "kangaroo/dataset_backend.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace kangaroo {

void append_particle_values_as_f64(const ParticleFieldChunk &data,
                                   const std::string &name,
                                   const std::string &context,
                                   std::vector<double> &out_vals);
std::unordered_map<double, int64_t>
decode_particle_value_counts(std::span<const std::uint8_t> bytes);
void encode_particle_value_counts(
    const std::unordered_map<double, int64_t> &counts,
    std::vector<std::uint8_t> &out);

} // namespace kangaroo
