#include "kangaroo/plotfile_reader.hpp"

#ifdef KANGAROO_USE_NANOBIND
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(_plotfile, m) {
  nb::class_<kangaroo::plotfile::PlotfileReader>(m, "PlotfileReader")
      .def(nb::init<std::string>())
      .def("header", [](const kangaroo::plotfile::PlotfileReader& reader) {
        const auto& hdr = reader.header();
        nb::dict d;
        d["file_version"] = hdr.file_version;
        d["ncomp"] = hdr.ncomp;
        d["var_names"] = hdr.var_names;
        d["spacedim"] = hdr.spacedim;
        d["time"] = hdr.time;
        d["finest_level"] = hdr.finest_level;
        d["ref_ratio"] = hdr.ref_ratio;
        d["mf_name"] = hdr.mf_name;
        return d;
      })
      .def("num_levels", &kangaroo::plotfile::PlotfileReader::num_levels)
      .def("num_fabs", &kangaroo::plotfile::PlotfileReader::num_fabs)
      .def("read_fab",
           [](kangaroo::plotfile::PlotfileReader& reader, int level, int fab, int comp_start,
              int comp_count) {
             auto data = reader.read_fab(level, fab, comp_start, comp_count);
             nb::dict out;
             out["shape"] = nb::make_tuple(data.ncomp, data.nz, data.ny, data.nx);
             out["dtype"] =
                 (data.type == kangaroo::plotfile::RealType::kFloat32) ? "float32" : "float64";
             out["data"] =
                 nb::bytes(reinterpret_cast<const char*>(data.bytes.data()), data.bytes.size());
             return out;
           },
           nb::arg("level"),
           nb::arg("fab"),
           nb::arg("comp_start"),
           nb::arg("comp_count"));
}
#endif
