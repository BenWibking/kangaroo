#include "plotfile_kernel_support.hpp"

#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace kangaroo {

bool plotfile_zero_copy_reads_enabled() {
  static const bool enabled = [] {
    const char *env = std::getenv("KANGAROO_PLOTFILE_ZERO_COPY_READS");
    if (env == nullptr || *env == '\0') {
      return true;
    }
    const std::string value(env);
    return value != "0" && value != "false" && value != "FALSE" &&
           value != "off" && value != "OFF";
  }();
  return enabled;
}

std::shared_ptr<plotfile::PlotfileReader>
get_plotfile_reader(const std::string &path) {
  static std::mutex reader_mutex;
  static std::unordered_map<std::string,
                            std::weak_ptr<plotfile::PlotfileReader>>
      readers;

  std::lock_guard<std::mutex> lock(reader_mutex);
  auto it = readers.find(path);
  if (it != readers.end()) {
    if (auto shared = it->second.lock()) {
      return shared;
    }
  }
  auto shared = std::make_shared<plotfile::PlotfileReader>(path);
  readers[path] = shared;
  return shared;
}

} // namespace kangaroo
