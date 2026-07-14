#include "kangaroo/dataset_backend.hpp"

#include "kangaroo/backend_memory.hpp"
#include "kangaroo/backend_plotfile.hpp"

#ifdef KANGAROO_USE_OPENPMD
#include "kangaroo/backend_openpmd.hpp"
#endif

#ifdef KANGAROO_USE_PARTHENON_HDF5
#include "kangaroo/backend_parthenon.hpp"
#endif

#include <stdexcept>
#include <string_view>

namespace kangaroo {

namespace {

bool starts_with(std::string_view value, std::string_view prefix) {
  return value.starts_with(prefix);
}

bool has_parthenon_suffix(std::string_view path) {
  return path.ends_with(".phdf") || path.ends_with(".h5") || path.ends_with(".hdf5");
}

std::string strip_scheme(const std::string& uri, std::string_view scheme) {
  return uri.substr(scheme.size());
}

}  // namespace

void DatasetBackend::set_chunk(const ChunkRef&, ChunkBuffer) {
  throw std::runtime_error("Cannot set_chunk on read-only backend");
}

void DatasetBackend::register_field(int32_t, const std::string&) {}

void DatasetBackend::register_field_component(int32_t, int32_t) {}

std::vector<std::string> DatasetBackend::list_meshes(int32_t) const { return {}; }

void DatasetBackend::select_mesh(const std::string&) {
  throw std::runtime_error("dataset backend does not support mesh selection");
}

std::vector<std::string> DatasetBackend::list_particle_types() const { return {}; }

std::vector<std::string> DatasetBackend::list_particle_fields(const std::string&) const {
  return {};
}

int64_t DatasetBackend::particle_chunk_count(const std::string&) const {
  throw std::runtime_error("dataset backend does not provide particle chunk metadata");
}

ParticleFieldChunk DatasetBackend::read_particle_field_chunk(
    const std::string&, const std::string&, int64_t) const {
  throw std::runtime_error("dataset backend does not support particle chunk field access");
}

ParticleFieldChunk DatasetBackend::read_particle_field_grid(
    const std::string&, const std::string&, int, int) const {
  throw std::runtime_error("dataset backend does not support particle grid field access");
}

DatasetBackendSnapshot DatasetBackend::snapshot() const {
  return DatasetBackendSnapshot{.kind = kind()};
}

std::shared_ptr<DatasetBackend> make_dataset_backend(const std::string& uri) {
  if (starts_with(uri, "memory://")) {
    return std::make_shared<MemoryBackend>();
  }
  if (starts_with(uri, "amrex://")) {
    return std::make_shared<PlotfileBackend>(strip_scheme(uri, "amrex://"));
  }
  if (starts_with(uri, "parthenon://")) {
#ifdef KANGAROO_USE_PARTHENON_HDF5
    return std::make_shared<ParthenonBackend>(strip_scheme(uri, "parthenon://"));
#else
    throw std::runtime_error("Parthenon backend not enabled in this build");
#endif
  }
  if (starts_with(uri, "openpmd://")) {
#ifdef KANGAROO_USE_OPENPMD
    return std::make_shared<OpenPMDBackend>(uri);
#else
    throw std::runtime_error("openPMD backend not enabled in this build");
#endif
  }
  if (starts_with(uri, "file://")) {
    const std::string path = strip_scheme(uri, "file://");
    if (has_parthenon_suffix(path)) {
#ifdef KANGAROO_USE_PARTHENON_HDF5
      return std::make_shared<ParthenonBackend>(path);
#else
      throw std::runtime_error("Parthenon backend not enabled in this build");
#endif
    }
    return std::make_shared<PlotfileBackend>(path);
  }
  throw std::runtime_error("unsupported dataset URI scheme: " + uri);
}

std::shared_ptr<DatasetBackend> restore_dataset_backend(
    const std::string& uri, const DatasetBackendSnapshot& snapshot) {
  auto backend = make_dataset_backend(uri);
  if (backend->kind() != snapshot.kind) {
    throw std::runtime_error(
        "serialized dataset backend kind does not match URI: expected " + snapshot.kind +
        ", got " + backend->kind());
  }
  for (const auto& [ref, chunk] : snapshot.memory_chunks) {
    backend->set_chunk(ref, chunk);
  }
  for (const auto& [field, component] : snapshot.component_fields) {
    backend->register_field_component(field, component);
  }
  return backend;
}

}  // namespace kangaroo
