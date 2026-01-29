#include "kangaroo/executor.hpp"

#include <array>
#include <stdexcept>

#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/tuple.hpp>

namespace kangaroo {

Executor::Executor(const RunMeta& meta, DataService& data, AdjacencyService& adj, KernelRegistry& kr)
    : meta_(meta), data_(data), adj_(adj), kernels_(kr) {}

hpx::future<void> Executor::run(const PlanIR& plan) {
  hpx::future<void> chain = hpx::make_ready_future();
  for (const auto& stage : plan.stages) {
    chain = chain.then([this, stage](auto&&) { return run_stage(stage); });
  }
  return chain;
}

hpx::future<void> Executor::run_stage(const StageIR& stage) {
  if (stage.plane != ExecPlane::Chunk) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("only chunk plane is implemented"));
  }

  std::vector<hpx::future<void>> futures;
  for (const auto& tmpl : stage.templates) {
    const auto& level = meta_.steps.at(tmpl.domain.step).levels.at(tmpl.domain.level);
    int32_t nblocks = static_cast<int32_t>(level.boxes.size());

    if (tmpl.domain.blocks.has_value()) {
      for (int32_t block : tmpl.domain.blocks.value()) {
        futures.push_back(run_block_task(tmpl, block));
      }
    } else {
      for (int32_t block = 0; block < nblocks; ++block) {
        futures.push_back(run_block_task(tmpl, block));
      }
    }
  }

  return hpx::when_all(futures).then([](auto&&) { return; });
}

hpx::future<void> Executor::run_block_task(const TaskTemplateIR& tmpl, int32_t block) {
  if (tmpl.plane != ExecPlane::Chunk) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("only chunk plane is implemented"));
  }

  if (tmpl.inputs.empty() || tmpl.outputs.empty()) {
    return hpx::make_exceptional_future<void>(
        std::runtime_error("templates must specify inputs and outputs"));
  }

  std::vector<hpx::future<HostView>> input_futures;
  input_futures.reserve(tmpl.inputs.size());
  for (const auto& input : tmpl.inputs) {
    ChunkRef cref{tmpl.domain.step, tmpl.domain.level, input.field, input.version, block};
    input_futures.push_back(data_.get_host(cref));
  }

  auto fetch_face = [&](Face face, int32_t field, int32_t ver) -> hpx::future<std::vector<HostView>> {
    if (tmpl.deps.kind != "FaceNeighbors") {
      return hpx::make_ready_future(std::vector<HostView>{});
    }
    NeighborSpan ns = adj_.neighbors(tmpl.domain.step, tmpl.domain.level, block, face);
    std::vector<hpx::future<HostView>> futs;
    futs.reserve(ns.n);
    for (int i = 0; i < ns.n; ++i) {
      ChunkRef nbr{tmpl.domain.step, tmpl.domain.level, field, ver, ns.ptr[i]};
      futs.push_back(data_.get_host(nbr));
    }
    return hpx::when_all(futs).then([](auto&& all) {
      std::vector<HostView> out;
      out.reserve(all.get().size());
      for (auto& f : all.get()) {
        out.push_back(f.get());
      }
      return out;
    });
  };

  const auto& nbr_field = tmpl.inputs.front();
  auto f_xm = fetch_face(Face::Xm, nbr_field.field, nbr_field.version);
  auto f_xp = fetch_face(Face::Xp, nbr_field.field, nbr_field.version);
  auto f_ym = fetch_face(Face::Ym, nbr_field.field, nbr_field.version);
  auto f_yp = fetch_face(Face::Yp, nbr_field.field, nbr_field.version);
  auto f_zm = fetch_face(Face::Zm, nbr_field.field, nbr_field.version);
  auto f_zp = fetch_face(Face::Zp, nbr_field.field, nbr_field.version);

  auto all_inputs = hpx::when_all(input_futures);

  return hpx::when_all(std::move(all_inputs), std::move(f_xm), std::move(f_xp), std::move(f_ym),
                       std::move(f_yp), std::move(f_zm), std::move(f_zp))
      .then([this, tmpl, block](auto&& all) -> hpx::future<void> {
        auto results = all.get();
        auto input_pack = hpx::get<0>(results).get();
        auto xm = hpx::get<1>(results).get();
        auto xp = hpx::get<2>(results).get();
        auto ym = hpx::get<3>(results).get();
        auto yp = hpx::get<4>(results).get();
        auto zm = hpx::get<5>(results).get();
        auto zp = hpx::get<6>(results).get();

        std::vector<HostView> inputs;
        inputs.reserve(input_pack.size());
        for (auto& f : input_pack) {
          inputs.push_back(f.get());
        }

        std::vector<HostView> outputs;
        outputs.resize(tmpl.outputs.size());

        NeighborViews nbrs{xm, xp, ym, yp, zm, zp};
        const auto& level = meta_.steps.at(tmpl.domain.step).levels.at(tmpl.domain.level);
        auto& fn = kernels_.get_by_name(tmpl.kernel);

        return fn(level, block, inputs, nbrs, outputs, tmpl.params_msgpack)
            .then([this, tmpl, block, outputs = std::move(outputs)](auto&&) mutable {
              std::vector<hpx::future<void>> puts;
              puts.reserve(tmpl.outputs.size());
              for (std::size_t i = 0; i < tmpl.outputs.size(); ++i) {
                const auto& out = tmpl.outputs[i];
                ChunkRef cref{tmpl.domain.step, tmpl.domain.level, out.field, out.version, block};
                puts.push_back(data_.put_host(cref, std::move(outputs[i])));
              }
              return hpx::when_all(puts).then([](auto&&) { return; });
            });
      });
}

}  // namespace kangaroo
