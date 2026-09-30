#include "bw_syn/detail/scale_eval.hpp"
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

  std::cout << "ok test_synth_semantics\n";
  return 0;
}
