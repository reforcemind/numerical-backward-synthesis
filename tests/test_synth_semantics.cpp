#include "bw_syn/detail/scale_eval.hpp"
#include "bw_syn/functions/tanh_bw.hpp"
#include "bw_syn/oracle.hpp"
#include "bw_syn/synthesize.hpp"

#include <cassert>
#include <iostream>

int main() {
  using namespace bw_syn;
  auto c = NumericalContract::bf16_default("tanh_backward");
  BF16 x = BF16::from_f64(45.0);
  BF16 g = BF16::from_f64(4.0);

  SynthParams good;
  good.eval = EvalStrategy::ScaleSeparated;
  good.round = RoundStrategy::FinalOnly;
  good.reconstruct = ReconstructStrategy::LdexpThenRound;
  good.norm = NormStrategy::BeforeMul;

  SynthParams bad;
  bad.eval = EvalStrategy::DirectMaterialize;
  bad.round = RoundStrategy::IntermediateAndFinal;

  Program pg = build_candidate_program(BackwardKind::Tanh, good);
  Program pb = build_candidate_program(BackwardKind::Tanh, bad);
  assert(pg.name.find("direct_scale_sep") != std::string::npos);

  BF16 yg = detail::eval_ir_under_contract(pg, x, g, c);
  BF16 yb = baseline_materialize_derivative(BackwardKind::Tanh, c, x, g);
  assert(yg.bits == 0x008f);
  assert(yb.is_zero());
  assert(yg.bits != yb.bits);
  assert(synth_params_key(good) != synth_params_key(bad));

  SynthParams neg = good;
  neg.reconstruct = ReconstructStrategy::RoundThenLdexp;
  Program pn = build_candidate_program(BackwardKind::Tanh, neg);
  assert(pn.name.find("_rtl") != std::string::npos);

  BF16 direct_ir = detail::eval_ir_under_contract(pb, x, g, c);
  assert(direct_ir.bits == yb.bits);

  const Program factored = tanh_bw::ir_factored();
  auto factored_candidate = [&](BF16 xi, BF16 gi) {
    return detail::eval_ir_under_contract(factored, xi, gi, c);
  };
  VerifyConfig factored_cfg;
  factored_cfg.kind = BackwardKind::Tanh;
  factored_cfg.contract = c;
  factored_cfg.include_specials = false;
  auto factored_report = verify_kernel(factored_candidate, factored_cfg);
  if (!factored_report.ok)
    std::cerr << factored_report.summary << "\n";
  assert(factored_report.ok);
  assert(factored_candidate(x, g).bits == 0x008f);
  assert(factored_candidate(BF16::max_finite(), BF16::inf(false)).is_nan());
  factored_cfg.exhaustive_x = true;
  factored_cfg.max_samples = 2 * 65536;
  factored_cfg.fixed_g_values = {BF16::zero(false), BF16::from_f64(4.0)};
  auto factored_x = verify_kernel(factored_candidate, factored_cfg);
  if (!factored_x.ok)
    std::cerr << factored_x.summary << "\n";
  assert(factored_x.ok);
  factored_cfg.exhaustive_x = false;
  factored_cfg.exhaustive_g = true;
  factored_cfg.max_samples = 3 * 65536;
  factored_cfg.fixed_g_values.clear();
  factored_cfg.fixed_x_values = {BF16::zero(false), BF16::from_f64(45.0), BF16::from_f64(90.0)};
  auto factored_g = verify_kernel(factored_candidate, factored_cfg);
  if (!factored_g.ok)
    std::cerr << factored_g.summary << "\n";
  assert(factored_g.ok);

  for (auto kind : {BackwardKind::Erf, BackwardKind::Elu}) {
    auto program = build_candidate_program(kind, good);
    assert(program.name.find("direct_scale_sep") != std::string::npos);
    const BF16 xi = BF16::from_f64(-1.0);
    const BF16 gi = BF16::from_f64(2.0);
    auto kind_contract = NumericalContract::bf16_default(backward_kind_name(kind));
    const auto got = detail::eval_ir_under_contract(program, xi, gi, kind_contract);
    const auto expected = contract_reference(kind, kind_contract, xi, gi);
    assert(check_sample(kind_contract, xi, gi, got, expected).verdict == ContractVerdict::Pass);
  }

  for (auto kind : {BackwardKind::Tanh, BackwardKind::Sigmoid}) {
    SynthConfig cfg;
    cfg.kind = kind;
    cfg.contract = NumericalContract::bf16_default(backward_kind_name(kind));
    auto result = synthesize(cfg);
    if (!result.found) {
      for (const auto& candidate : result.all)
        std::cerr << candidate.id << " " << candidate.verify.summary << "\n";
    }
    assert(result.found);
    assert(result.candidates_verified > 0);
    assert(result.best.verify.ok);
    assert(result.candidates_built < result.search_space_size);
    assert(result.best.program.name.find("direct_scale_sep") != std::string::npos);

    auto candidate = [&](BF16 xi, BF16 gi) {
      return detail::eval_ir_under_contract(result.best.program, xi, gi, cfg.contract);
    };
    const BF16 extreme = BF16::max_finite();
    assert(candidate(extreme, BF16::inf(false)).bits == BF16::inf(false).bits);
    assert(candidate(extreme, BF16::inf(true)).bits == BF16::inf(true).bits);

    VerifyConfig sweep = cfg.verify_cfg;
    sweep.kind = kind;
    sweep.contract = cfg.contract;
    sweep.exhaustive_x = true;
    sweep.max_samples = 3 * 65536;
    sweep.fixed_g_values = {BF16::zero(false), BF16::from_f64(4.0), BF16::inf(false)};
    auto x_report = verify_kernel(candidate, sweep);
    if (!x_report.ok)
      std::cerr << x_report.summary << "\n";
    assert(x_report.ok);
    assert(x_report.counters.tested == sweep.max_samples);

    sweep.exhaustive_x = false;
    sweep.exhaustive_g = true;
    sweep.fixed_g_values.clear();
    sweep.fixed_x_values = {BF16::zero(false),
                            BF16::from_f64(kind == BackwardKind::Tanh ? 45.0 : 90.0),
                            BF16::inf(false)};
    auto g_report = verify_kernel(candidate, sweep);
    if (!g_report.ok)
      std::cerr << g_report.summary << "\n";
    assert(g_report.ok);
    assert(g_report.counters.tested == sweep.max_samples);

    sweep.exhaustive_g = false;
    sweep.fixed_x_values.clear();
    sweep.paired_bit_permutations = 3;
    auto pair_report = verify_kernel(candidate, sweep);
    if (!pair_report.ok)
      std::cerr << pair_report.summary << "\n";
    assert(pair_report.ok);
    assert(pair_report.counters.tested == sweep.max_samples);

    sweep.max_samples = 65536;
    auto truncated = verify_kernel(candidate, sweep);
    assert(truncated.counters.hit_sample_cap);
    assert(!truncated.ok);
  }

  std::cout << "ok test_synth_semantics\n";
  return 0;
}
