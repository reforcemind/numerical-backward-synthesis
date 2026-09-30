#include "bw_syn/backends/tt_harness.hpp"
#include "bw_syn/cost_model.hpp"
#include "bw_syn/csv_io.hpp"
#include "bw_syn/detail/scale_eval.hpp"
#include "bw_syn/functions/sigmoid_bw.hpp"
#include "bw_syn/functions/tanh_bw.hpp"
#include "bw_syn/oracle.hpp"
#include "bw_syn/synthesize.hpp"
#include "bw_syn/transforms.hpp"
#include "bw_syn/verify.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void ablation_rows(std::vector<std::string>& rows,
                          bw_syn::BackwardKind kind,
                          const bw_syn::NumericalContract& c,
                          float x0,
                          float g0) {
  using namespace bw_syn;
  BF16 x = BF16::from_f64(x0), g = BF16::from_f64(g0);
  BF16 o = contract_reference(kind, c, x, g);
  BF16 sc = kind == BackwardKind::Sigmoid ? sigmoid_bw::scale_separated(x, g, c)
                                          : tanh_bw::scale_separated(x, g, c);
  BF16 b = kind == BackwardKind::Sigmoid ? sigmoid_bw::baseline_materialize(x, g, c)
                                         : tanh_bw::baseline_materialize(x, g, c);

  SynthParams mid;
  mid.eval = EvalStrategy::ScaleSeparated;
  mid.round = RoundStrategy::IntermediateAndFinal;
  Program pmid = build_candidate_program(kind, mid);
  BF16 m = detail::eval_ir_under_contract(pmid, x, g, c);

  auto push = [&](const char* method, BF16 y, bool fz) {
    rows.push_back(std::string("host,") + backward_kind_name(kind) + "," + method + "," +
                   csv_f32(x0) + "," + csv_f32(g0) + "," + hex16(o.bits) + "," + hex16(y.bits) +
                   "," + std::to_string(ulp_distance(y, o)) + "," + (fz ? "1" : "0"));
  };
  push("oracle", o, false);
  push("scale_sep", sc, false);
  push("baseline", b, b.is_zero() && !o.is_zero());
  push("mid_round_ir", m, m.is_zero() && !o.is_zero());
}

int main(int argc, char** argv) {
  using namespace bw_syn;
  std::string out = "results";
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--out" && i + 1 < argc)
      out = argv[++i];

  fs::create_directories(out + "/hw");
  fs::create_directories(out + "/paper");
  auto ct = NumericalContract::bf16_default("tanh_backward");
  auto cs = NumericalContract::bf16_default("sigmoid_backward");

  {
    std::vector<std::string> rows;
    for (const auto& s : tt_harness::critical_cases()) {
      BF16 x = BF16::from_f64(s.x), g = BF16::from_f64(s.g);
      BF16 o = contract_reference(BackwardKind::Tanh, ct, x, g);
      BF16 b = tanh_bw::baseline_materialize(x, g, ct);
      BF16 sc = tanh_bw::scale_separated(x, g, ct);
      rows.push_back(csv_f32(s.x) + "," + csv_f32(s.g) + "," + hex16(o.bits) + "," + hex16(b.bits) +
                     "," + hex16(sc.bits) + "," + std::to_string(ulp_distance(b, o)) + "," +
                     std::to_string(ulp_distance(sc, o)) + "," +
                     ((b.is_zero() && !o.is_zero()) ? "1" : "0"));
    }
    if (!write_csv(
            out + "/cases_critical.csv",
            "x,g,oracle_bits,baseline_bits,scale_bits,ulp_baseline,ulp_scale,false_zero_baseline",
            rows))
      return 1;
  }

  {
    BF16 x = BF16::from_f64(45), g = BF16::from_f64(4);
    auto tr = apply_default_scale_pipeline(tanh_bw::ir_direct());
    BF16 t = detail::eval_ir_under_contract(tr.program, x, g, ct);
    if (!write_csv(out + "/baselines_tanh.csv",
                   "method,bits_45_4,false_zero",
                   {"oracle," + hex16(contract_reference(BackwardKind::Tanh, ct, x, g).bits) + ",0",
                    "scale_sep," + hex16(tanh_bw::scale_separated(x, g, ct).bits) + ",0",
                    "baseline," + hex16(tanh_bw::baseline_materialize(x, g, ct).bits) + ",1",
                    "ir_pipeline," + hex16(t.bits) + ",0"}))
      return 1;
  }

  {
    std::vector<std::string> rows;
    ablation_rows(rows, BackwardKind::Tanh, ct, 45.f, 4.f);
    ablation_rows(rows, BackwardKind::Sigmoid, cs, 90.f, 8.f);
    if (!write_csv(out + "/paper/ablation_host.csv",
                   "label,kind,method,x,g,oracle_bits,got_bits,ulp,false_zero",
                   rows))
      return 1;
  }

  {
    VerifyConfig v;
    v.max_samples = 20000;
    std::vector<std::string> rows;
    struct J {
      const char* k;
      const char* impl;
      BackwardKind bk;
      KernelFn fn;
    };
    J jobs[] = {
        {"tanh_backward",
         "scale_sep",
         BackwardKind::Tanh,
         [&](BF16 x, BF16 g) { return tanh_bw::scale_separated(x, g, ct); }},
        {"tanh_backward",
         "baseline",
         BackwardKind::Tanh,
         [&](BF16 x, BF16 g) { return tanh_bw::baseline_materialize(x, g, ct); }},
        {"sigmoid_backward",
         "scale_sep",
         BackwardKind::Sigmoid,
         [&](BF16 x, BF16 g) { return sigmoid_bw::scale_separated(x, g, cs); }},
        {"sigmoid_backward",
         "baseline",
         BackwardKind::Sigmoid,
         [&](BF16 x, BF16 g) { return sigmoid_bw::baseline_materialize(x, g, cs); }},
    };
    for (auto& j : jobs) {
      v.kind = j.bk;
      v.contract = NumericalContract::bf16_default(j.k);
      auto r = verify_kernel(j.fn, v);
      rows.push_back(std::string(j.k) + "," + j.impl + "," + std::to_string(r.counters.tested) +
                     "," + std::to_string(r.counters.pass) + "," +
                     std::to_string(r.counters.false_zeros) + "," +
                     std::to_string(r.counters.max_ulp) + "," + (r.ok ? "1" : "0"));
    }
    if (!write_csv(out + "/verify.csv", "kind,impl,tested,pass,false_zeros,max_ulp,ok", rows))
      return 1;
  }

  {
    std::vector<std::string> detail;
    std::vector<std::string> summary;
    BF16 x = BF16::from_f64(45), g = BF16::from_f64(4);
    for (auto kind : {BackwardKind::Tanh, BackwardKind::Sigmoid}) {
      SynthConfig cfg;
      cfg.kind = kind;
      cfg.contract = NumericalContract::bf16_default(backward_kind_name(kind));
      cfg.verify_cfg.kind = kind;
      cfg.verify_cfg.contract = cfg.contract;
      cfg.verify_cfg.max_samples = 20000;
      auto syn = synthesize(cfg);
      const double motiv_x = kind == BackwardKind::Sigmoid ? 90.0 : 45.0;
      const double motiv_g = kind == BackwardKind::Sigmoid ? 8.0 : 4.0;
      x = BF16::from_f64(motiv_x);
      g = BF16::from_f64(motiv_g);
      for (const auto& c : syn.all) {
        BF16 y = c.params.eval == EvalStrategy::DirectMaterialize
                     ? baseline_materialize_derivative(kind, cfg.contract, x, g)
                     : detail::eval_ir_under_contract(c.program, x, g, cfg.contract);
        detail.push_back(std::string(backward_kind_name(kind)) + "," + c.id + "," +
                         (c.verified ? "1" : "0") + "," +
                         std::to_string(static_cast<int>(c.cost.scalar_score)) + "," +
                         std::to_string(c.verify.counters.max_ulp) + "," +
                         std::to_string(c.verify.counters.false_zeros) + "," + hex16(y.bits));
      }
      summary.push_back(
          std::string(backward_kind_name(kind)) + "," + std::to_string(syn.search_space_size) +
          "," + std::to_string(syn.candidates_built) + "," +
          std::to_string(syn.candidates_verified) + "," + std::to_string(syn.pareto.size()) + "," +
          (syn.found ? "1" : "0") + "," + (syn.found ? syn.best.id : "") + "," +
          (syn.found ? std::to_string(static_cast<int>(syn.best.cost.scalar_score)) : "") + "," +
          std::to_string(syn.synthesis_seconds));
    }
    if (!write_csv(out + "/synth.csv",
                   "kind,id,verified,score,max_ulp,false_zeros,bits_motivating",
                   detail))
      return 1;
    if (!write_csv(out + "/paper/synth_cost.csv",
                   "kind,search_space,built,verified,pareto,found,best_id,best_score,seconds",
                   summary))
      return 1;
  }

  {
    auto ctanh = estimate_cost(tanh_bw::ir_scale_separated(), {});
    auto csig = estimate_cost(sigmoid_bw::ir_scale_separated(), {});
    if (!write_csv(
            out + "/cost_host_proxy.csv",
            "label,program,insn,regs,latency,score",
            {"host_proxy,tanh_bw_scale_separated," + std::to_string(ctanh.instruction_count) + "," +
                 std::to_string(ctanh.register_estimate) + "," +
                 std::to_string(static_cast<int>(ctanh.latency)) + "," +
                 std::to_string(static_cast<int>(ctanh.scalar_score)),
             "host_proxy,sigmoid_bw_scale_separated," + std::to_string(csig.instruction_count) +
                 "," + std::to_string(csig.register_estimate) + "," +
                 std::to_string(static_cast<int>(csig.latency)) + "," +
                 std::to_string(static_cast<int>(csig.scalar_score))}))
      return 1;
  }

  {
    auto rep = tt_harness::run_critical_tanh(tt_harness::DeviceMode::HostSim);
    auto cost = estimate_cost(tanh_bw::ir_scale_separated(), {});
    std::vector<std::string> rows;
    for (const auto& c : rep.cases)
      rows.push_back("host_sim,host,unset," + csv_f32(c.spec.x) + "," + csv_f32(c.spec.g) + "," +
                     hex16(c.oracle.bits) + "," + hex16(c.device_or_sim.bits) + "," +
                     hex16(c.baseline_model.bits) + "," + std::to_string(c.ulp_vs_oracle) + "," +
                     (c.pass ? "1" : "0") + "," + (c.false_zero_baseline ? "1" : "0") + "," +
                     std::to_string(static_cast<int>(cost.latency)) + "," +
                     std::to_string(cost.instruction_count) + "," +
                     std::to_string(cost.register_estimate) + ",");
    if (!write_csv(out + "/hw/host_sim.csv",
                   "label,arch,tt_metal_commit,x,g,oracle_bits,got_bits,baseline_bits,ulp,pass,"
                   "false_zero_baseline,cycles_per_tile,insn,regs,ftz",
                   rows))
      return 1;
  }

  if (!write_csv(out + "/paper/device_status.csv",
                 "item,status",
                 {"wormhole_csv,unmeasured",
                  "blackhole_csv,unmeasured",
                  "cycles_per_tile,unmeasured",
                  "insn_dump,unmeasured",
                  "regs,unmeasured",
                  "device_ftz,unmeasured",
                  "tt_metal_pin,UNSET_until_board_run"}))
    return 1;

  return 0;
}
