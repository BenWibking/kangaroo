#include "kangaroo/runtime.hpp"

#ifdef KANGAROO_USE_NANOBIND
#include <nanobind/nanobind.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <chrono>

#include <hpx/version.hpp>

namespace nb = nanobind;

namespace {

std::atomic<std::uint64_t> g_py_event_counter{0};

nb::object require_key(const nb::dict& d, const char* key) {
  if (!d.contains(key)) {
    throw std::runtime_error(std::string("missing key: ") + key);
  }
  return d[nb::str(key)];
}

kangaroo::RunMeta parse_runmeta(const nb::object& obj) {
  kangaroo::RunMeta meta;
  auto steps_list = nb::cast<nb::list>(obj);
  meta.steps.reserve(steps_list.size());
  for (auto step_item : steps_list) {
    auto step_dict = nb::cast<nb::dict>(step_item);
    kangaroo::StepMeta step;
    step.step = nb::cast<int32_t>(require_key(step_dict, "step"));

    auto levels_list = nb::cast<nb::list>(require_key(step_dict, "levels"));
    step.levels.reserve(levels_list.size());
    for (auto level_item : levels_list) {
      auto level_dict = nb::cast<nb::dict>(level_item);
      kangaroo::LevelMeta level;

      auto geom_dict = nb::cast<nb::dict>(require_key(level_dict, "geom"));
      auto dx = nb::cast<nb::tuple>(require_key(geom_dict, "dx"));
      auto x0 = nb::cast<nb::tuple>(require_key(geom_dict, "x0"));
      level.geom.dx[0] = nb::cast<double>(dx[0]);
      level.geom.dx[1] = nb::cast<double>(dx[1]);
      level.geom.dx[2] = nb::cast<double>(dx[2]);
      level.geom.x0[0] = nb::cast<double>(x0[0]);
      level.geom.x0[1] = nb::cast<double>(x0[1]);
      level.geom.x0[2] = nb::cast<double>(x0[2]);
      if (geom_dict.contains("index_origin")) {
        auto origin = nb::cast<nb::tuple>(geom_dict["index_origin"]);
        level.geom.index_origin[0] = nb::cast<int32_t>(origin[0]);
        level.geom.index_origin[1] = nb::cast<int32_t>(origin[1]);
        level.geom.index_origin[2] = nb::cast<int32_t>(origin[2]);
      }
      level.geom.ref_ratio = nb::cast<int>(require_key(geom_dict, "ref_ratio"));

      auto boxes_list = nb::cast<nb::list>(require_key(level_dict, "boxes"));
      level.boxes.reserve(boxes_list.size());
      for (auto box_item : boxes_list) {
        auto box_pair = nb::cast<nb::tuple>(box_item);
        auto lo = nb::cast<nb::tuple>(box_pair[0]);
        auto hi = nb::cast<nb::tuple>(box_pair[1]);
        kangaroo::BlockBox box;
        box.lo = {nb::cast<int32_t>(lo[0]), nb::cast<int32_t>(lo[1]), nb::cast<int32_t>(lo[2])};
        box.hi = {nb::cast<int32_t>(hi[0]), nb::cast<int32_t>(hi[1]), nb::cast<int32_t>(hi[2])};
        level.boxes.push_back(box);
      }

      step.levels.push_back(std::move(level));
    }

    meta.steps.push_back(std::move(step));
  }

  return meta;
}

double now_seconds() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration<double>(now.time_since_epoch()).count();
}

}  // namespace

NB_MODULE(_core, m) {
  m.def("hpx_configuration_string", []() { return hpx::configuration_string(); });
  m.def("set_event_log_path", &kangaroo::set_event_log_path);
  m.def("set_global_dataset", &kangaroo::set_global_dataset);
  m.def("log_task_event",
        [](const std::string& name,
           const std::string& status,
           nb::object start_obj,
           nb::object end_obj,
           nb::object id_obj,
           nb::object worker_label_obj) {
          kangaroo::TaskEvent event;
          event.name = name;
          event.kernel = name;
          event.plane = "python";
          event.status = status;
          event.stage = -1;
          event.template_index = -1;
          event.block = -1;
          event.step = 0;
          event.level = 0;
          event.locality = -1;
          event.worker = -1;
          if (!worker_label_obj.is_none()) {
            event.worker_label = nb::cast<std::string>(worker_label_obj);
          } else {
            event.worker_label = "python";
          }

          if (!id_obj.is_none()) {
            event.id = nb::cast<std::string>(id_obj);
          } else {
            event.id = "py:" + std::to_string(++g_py_event_counter);
          }

          double start = start_obj.is_none() ? now_seconds() : nb::cast<double>(start_obj);
          double end = end_obj.is_none() ? start : nb::cast<double>(end_obj);
          event.ts = end;
          event.start = start;
          event.end = end;
          kangaroo::log_task_event(event);
        },
        nb::arg("name"),
        nb::arg("status"),
        nb::arg("start") = nb::none(),
        nb::arg("end") = nb::none(),
        nb::arg("id") = nb::none(),
        nb::arg("worker_label") = nb::none());

  nb::class_<kangaroo::Runtime>(m, "Runtime")
      .def(nb::init<>())
      .def("__init__",
           [](kangaroo::Runtime* self, nb::object config_obj, nb::object args_obj) {
             std::vector<std::string> cfg;
             std::vector<std::string> args;
             if (!config_obj.is_none()) {
               auto list = nb::cast<nb::list>(config_obj);
               cfg.reserve(list.size());
               for (auto item : list) {
                 cfg.push_back(nb::cast<std::string>(item));
               }
             }
             if (!args_obj.is_none()) {
               auto list = nb::cast<nb::list>(args_obj);
               args.reserve(list.size());
               for (auto item : list) {
                 args.push_back(nb::cast<std::string>(item));
               }
             }
             new (self) kangaroo::Runtime(cfg, args);
           },
           nb::arg("hpx_config") = nb::none(),
           nb::arg("hpx_args") = nb::none())
      .def("alloc_field_id", &kangaroo::Runtime::alloc_field_id)
      .def("mark_field_persistent", &kangaroo::Runtime::mark_field_persistent)
      .def("kernels", &kangaroo::Runtime::kernels, nb::rv_policy::reference)
      .def("set_event_log_path", &kangaroo::Runtime::set_event_log_path)
      .def("get_task_chunk",
           [](kangaroo::Runtime& self, int32_t step, int16_t level, int32_t field,
              int32_t version, int32_t block) {
             auto view = self.get_task_chunk(step, level, field, version, block);
             return nb::bytes(reinterpret_cast<const char*>(view.data.data()), view.data.size());
           })
      .def("preload_dataset", &kangaroo::Runtime::preload_dataset)
      .def("run_packed_plan", &kangaroo::Runtime::run_packed_plan)
      .def("run_packed_plan",
           [](kangaroo::Runtime& self, nb::bytes packed, kangaroo::RunMetaHandle& runmeta,
              kangaroo::DatasetHandle& dataset) {
             auto* data = static_cast<const std::uint8_t*>(packed.data());
             std::vector<std::uint8_t> buffer(data, data + packed.size());
             self.run_packed_plan(buffer, runmeta, dataset);
           });

  nb::class_<kangaroo::KernelRegistry>(m, "KernelRegistry")
      .def("list", &kangaroo::KernelRegistry::list_kernel_descs);

  nb::class_<kangaroo::RunMetaHandle>(m, "RunMetaHandle")
      .def("__init__", [](kangaroo::RunMetaHandle* self, nb::object payload) {
        new (self) kangaroo::RunMetaHandle();
        self->meta = parse_runmeta(payload);
      });

  nb::class_<kangaroo::DatasetHandle>(m, "DatasetHandle")
      .def("__init__", [](kangaroo::DatasetHandle* self, const std::string& uri, int32_t step,
                          int16_t level) {
        new (self) kangaroo::DatasetHandle();
        self->uri = uri;
        self->step = step;
        self->level = level;
        
        if (uri.rfind("amrex://", 0) == 0 || uri.rfind("file://", 0) == 0) {
            std::string path = uri;
            if (uri.rfind("amrex://", 0) == 0) {
                path = uri.substr(8);
            } else {
                path = uri.substr(7);
            }
            self->backend = std::make_shared<kangaroo::PlotfileBackend>(path);
#ifdef KANGAROO_USE_OPENPMD
        } else if (uri.rfind("openpmd://", 0) == 0) {
            self->backend = std::make_shared<kangaroo::OpenPMDBackend>(uri);
#endif
        } else if (uri == "memory://local") {
            self->backend = std::make_shared<kangaroo::MemoryBackend>();
        }
      })
      .def("register_field", [](kangaroo::DatasetHandle& self, int32_t field_id, int32_t component) {
          if (auto plt = std::dynamic_pointer_cast<kangaroo::PlotfileBackend>(self.backend)) {
              plt->register_field(field_id, component);
          }
      })
      .def("register_field", [](kangaroo::DatasetHandle& self, int32_t field_id,
                                const std::string& name) {
#ifdef KANGAROO_USE_OPENPMD
          if (auto opmd = std::dynamic_pointer_cast<kangaroo::OpenPMDBackend>(self.backend)) {
              opmd->register_field(field_id, name);
          }
#else
          (void)self;
          (void)field_id;
          (void)name;
#endif
      })
      .def("list_meshes", [](kangaroo::DatasetHandle& self) -> nb::list {
#ifdef KANGAROO_USE_OPENPMD
          if (auto opmd = std::dynamic_pointer_cast<kangaroo::OpenPMDBackend>(self.backend)) {
              nb::list out;
              for (const auto& name : opmd->list_meshes(self.step)) {
                  out.append(name);
              }
              return out;
          }
#endif
          return nb::list();
      })
      .def("select_mesh", [](kangaroo::DatasetHandle& self, const std::string& name) {
#ifdef KANGAROO_USE_OPENPMD
          if (auto opmd = std::dynamic_pointer_cast<kangaroo::OpenPMDBackend>(self.backend)) {
              opmd->select_mesh(name);
          }
#else
          (void)self;
          (void)name;
#endif
      })
      .def("metadata", [](kangaroo::DatasetHandle& self) -> nb::dict {
          if (!self.backend) return nb::dict();
          auto* reader = self.backend->get_plotfile_reader();
          if (reader) {

            const auto& hdr = reader->header();
            nb::dict d;
            d["var_names"] = hdr.var_names;
            d["finest_level"] = hdr.finest_level;
            d["prob_lo"] = hdr.prob_lo;
            d["prob_hi"] = hdr.prob_hi;
            d["ref_ratio"] = hdr.ref_ratio;
            d["cell_size"] = hdr.cell_size;
            
            auto to_tuple = [](const kangaroo::plotfile::IntVect& iv) {
              return nb::make_tuple(iv.x, iv.y, iv.z);
            };
            auto box_to_tuple = [&](const kangaroo::plotfile::Box& box) {
              return nb::make_tuple(to_tuple(box.lo), to_tuple(box.hi));
            };

            nb::list levels;
            for (int level = 0; level <= hdr.finest_level; ++level) {
              nb::list boxes;
              const auto& vismf = reader->vismf_header(level);
              for (const auto& box : vismf.box_array.boxes) {
                boxes.append(box_to_tuple(box));
              }
              levels.append(boxes);
            }
            d["level_boxes"] = levels;

            nb::list prob_domains;
            for (const auto& box : hdr.prob_domain) {
              prob_domains.append(box_to_tuple(box));
            }
            d["prob_domain"] = prob_domains;

            return d;
          }

#ifdef KANGAROO_USE_OPENPMD
          if (auto opmd = std::dynamic_pointer_cast<kangaroo::OpenPMDBackend>(self.backend)) {
            auto meta = opmd->metadata(self.step);
            nb::dict d;
            nb::list var_names;
            for (const auto& field : meta.fields) {
              var_names.append(field.name);
            }
            d["var_names"] = var_names;
            d["mesh_names"] = meta.mesh_names;
            d["selected_mesh"] = meta.selected_mesh;
            d["finest_level"] = meta.finest_level;
            d["prob_lo"] = nb::make_tuple(meta.prob_lo[0], meta.prob_lo[1], meta.prob_lo[2]);
            d["prob_hi"] = nb::make_tuple(meta.prob_hi[0], meta.prob_hi[1], meta.prob_hi[2]);

            nb::list ref_ratio;
            for (const auto& ratio : meta.ref_ratio) {
              ref_ratio.append(ratio[0]);
            }
            d["ref_ratio"] = ref_ratio;

            nb::list cell_size;
            for (const auto& size : meta.cell_size) {
              cell_size.append(nb::make_tuple(size[0], size[1], size[2]));
            }
            d["cell_size"] = cell_size;

            nb::list levels;
            for (const auto& level_boxes : meta.level_boxes) {
              nb::list boxes;
              for (const auto& box : level_boxes) {
                auto lo = nb::make_tuple(box.first[0], box.first[1], box.first[2]);
                auto hi = nb::make_tuple(box.second[0], box.second[1], box.second[2]);
                boxes.append(nb::make_tuple(lo, hi));
              }
              levels.append(boxes);
            }
            d["level_boxes"] = levels;

            nb::list prob_domains;
            for (const auto& box : meta.prob_domain) {
              auto lo = nb::make_tuple(box.first[0], box.first[1], box.first[2]);
              auto hi = nb::make_tuple(box.second[0], box.second[1], box.second[2]);
              prob_domains.append(nb::make_tuple(lo, hi));
            }
            d["prob_domain"] = prob_domains;

            return d;
          }
#endif

          return nb::dict();
      })
      .def("set_chunk",
           [](kangaroo::DatasetHandle& self, int32_t field, int32_t version, int32_t block,
              nb::bytes payload) {
             const auto* data = static_cast<const std::uint8_t*>(payload.data());
             std::vector<std::uint8_t> buffer(data, data + payload.size());
             kangaroo::HostView view;
             view.data = std::move(buffer);
             kangaroo::ChunkRef ref{self.step, self.level, field, version, block};
             self.set_chunk(ref, std::move(view));
           });

}
#endif
