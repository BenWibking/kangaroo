#include "kangaroo/runtime.hpp"

#ifdef KANGAROO_USE_NANOBIND
#include <nanobind/nanobind.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <hpx/version.hpp>

namespace nb = nanobind;

namespace {

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

}  // namespace

NB_MODULE(_core, m) {
  m.def("hpx_configuration_string", []() { return hpx::configuration_string(); });

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
