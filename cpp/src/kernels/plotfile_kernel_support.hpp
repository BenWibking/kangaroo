#pragma once

#include "kangaroo/plotfile_reader.hpp"

#include <memory>
#include <string>

namespace kangaroo {

bool plotfile_zero_copy_reads_enabled();
std::shared_ptr<plotfile::PlotfileReader>
get_plotfile_reader(const std::string &path);

} // namespace kangaroo
