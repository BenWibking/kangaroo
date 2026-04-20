#include "internal.hpp"

#include "kangaroo/plotfile_reader.hpp"

#include <msgpack.hpp>

#include <asio/post.hpp>

#include <hpx/io_service/io_service_pool.hpp>
#include <hpx/runtime_distributed.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace kangaroo {
namespace {

struct PlotfileLoadParams {
  std::string plotfile;
  int level = 0;
  int comp = 0;
  int bytes_per_value = 4;
};

std::optional<msgpack::object_handle> unpack_params(std::span<const std::uint8_t> params_msgpack) {
  if (params_msgpack.empty()) {
    return std::nullopt;
  }
  try {
    return msgpack::unpack(reinterpret_cast<const char*>(params_msgpack.data()), params_msgpack.size());
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::shared_ptr<plotfile::PlotfileReader> get_plotfile_reader(const std::string& path) {
  static std::mutex reader_mutex;
  static std::unordered_map<std::string, std::weak_ptr<plotfile::PlotfileReader>> readers;

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

template <typename InT, typename OutT>
void transpose_plotfile_axes(const InT* in, OutT* out, int nx, int ny, int nz) {
  for (int k = 0; k < nz; ++k) {
    for (int j = 0; j < ny; ++j) {
      for (int i = 0; i < nx; ++i) {
        const std::size_t in_idx = (static_cast<std::size_t>(k) * ny + j) * nx + i;
        const std::size_t out_idx = (static_cast<std::size_t>(i) * ny + j) * nz + k;
        out[out_idx] = static_cast<OutT>(in[in_idx]);
      }
    }
  }
}

PlotfileLoadParams unpack_plotfile_params(std::span<const std::uint8_t> params_msgpack) {
  PlotfileLoadParams params;
  const auto handle = unpack_params(params_msgpack);
  if (!handle) {
    return params;
  }

  auto root = handle->get();
  if (root.type != msgpack::type::MAP) {
    return params;
  }

  auto get_key = [&](const char* key) -> const msgpack::object* {
    for (uint32_t i = 0; i < root.via.map.size; ++i) {
      const auto& k = root.via.map.ptr[i].key;
      if (k.type == msgpack::type::STR && k.as<std::string>() == key) {
        return &root.via.map.ptr[i].val;
      }
    }
    return nullptr;
  };

  if (const auto* path = get_key("plotfile"); path && path->type == msgpack::type::STR) {
    params.plotfile = path->as<std::string>();
  }
  if (const auto* lvl = get_key("level"); lvl &&
                                          (lvl->type == msgpack::type::POSITIVE_INTEGER ||
                                           lvl->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.level = lvl->as<int>();
  }
  if (const auto* comp = get_key("comp"); comp &&
                                           (comp->type == msgpack::type::POSITIVE_INTEGER ||
                                            comp->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.comp = comp->as<int>();
  }
  if (const auto* bpv = get_key("bytes_per_value"); bpv &&
                                                  (bpv->type == msgpack::type::POSITIVE_INTEGER ||
                                                   bpv->type == msgpack::type::NEGATIVE_INTEGER)) {
    params.bytes_per_value = bpv->as<int>();
  }
  return params;
}

void load_plotfile_block(const std::shared_ptr<plotfile::PlotfileReader>& reader,
                         const PlotfileLoadParams& params,
                         int32_t block_index,
                         int nx,
                         int ny,
                         int nz,
                         HostView& output) {
  if (!reader || params.level < 0 || params.level >= reader->num_levels()) {
    return;
  }
  if (block_index >= reader->num_fabs(params.level)) {
    return;
  }

  auto data = reader->read_fab(params.level, block_index, params.comp, 1);
  if (data.ncomp < 1 || data.nx != nx || data.ny != ny || data.nz != nz) {
    return;
  }

  if (data.type == plotfile::RealType::kFloat32) {
    const auto* in = reinterpret_cast<const float*>(data.bytes.data());
    if (params.bytes_per_value == 4) {
      auto* out = reinterpret_cast<float*>(output.data.data());
      transpose_plotfile_axes(in, out, nx, ny, nz);
    } else if (params.bytes_per_value == 8) {
      auto* out = reinterpret_cast<double*>(output.data.data());
      transpose_plotfile_axes(in, out, nx, ny, nz);
    }
  } else if (data.type == plotfile::RealType::kFloat64) {
    const auto* in = reinterpret_cast<const double*>(data.bytes.data());
    if (params.bytes_per_value == 8) {
      auto* out = reinterpret_cast<double*>(output.data.data());
      transpose_plotfile_axes(in, out, nx, ny, nz);
    } else if (params.bytes_per_value == 4) {
      auto* out = reinterpret_cast<float*>(output.data.data());
      transpose_plotfile_axes(in, out, nx, ny, nz);
    }
  }
}

}  // namespace

namespace runtime_kernels {

KANGAROO_KERNEL(plotfile_load) {
  (void)self_inputs;
  (void)nbr_inputs;
  if (outputs.empty()) {
    return hpx::make_ready_future();
  }

  const auto params = unpack_plotfile_params(params_msgpack);

  if (params.plotfile.empty()) {
    return hpx::make_ready_future();
  }

  if (block_index < 0 || static_cast<std::size_t>(block_index) >= level.boxes.size()) {
    return hpx::make_ready_future();
  }

  const auto& box = level.boxes.at(static_cast<std::size_t>(block_index));
  const int nx = box.hi.x - box.lo.x + 1;
  const int ny = box.hi.y - box.lo.y + 1;
  const int nz = box.hi.z - box.lo.z + 1;
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    return hpx::make_ready_future();
  }

  std::size_t out_bytes = 0;
  if (params.bytes_per_value > 0) {
    out_bytes = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                static_cast<std::size_t>(nz) * static_cast<std::size_t>(params.bytes_per_value);
  }
  if (out_bytes == 0) {
    return hpx::make_ready_future();
  }

  if (outputs[0].data.size() != out_bytes) {
    outputs[0].data.assign(out_bytes, 0);
  }

  auto reader = get_plotfile_reader(params.plotfile);
  if (!reader) {
    return hpx::make_ready_future();
  }

  auto run_load = [reader = std::move(reader), params, block_index, nx, ny, nz, out = &outputs[0]]() mutable {
    load_plotfile_block(reader, params, block_index, nx, ny, nz, *out);
  };

  auto* runtime = hpx::get_runtime_distributed_ptr();
  auto* io_pool = runtime != nullptr ? runtime->get_thread_pool("io_pool") : nullptr;
  if (io_pool == nullptr) {
    run_load();
    return hpx::make_ready_future();
  }

  auto promise = std::make_shared<hpx::promise<void>>();
  auto future = promise->get_future();
  asio::post(io_pool->get_io_service(), [promise, run_load = std::move(run_load)]() mutable {
    try {
      run_load();
      promise->set_value();
    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  });
  return future;
}

}  // namespace runtime_kernels
}  // namespace kangaroo
