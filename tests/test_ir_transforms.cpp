#include "bw_syn/detail/scale_eval.hpp"
#include "bw_syn/functions/tanh_bw.hpp"
#include "bw_syn/ir.hpp"
#include "bw_syn/transforms.hpp"

#include <cassert>
#include <iostream>

int main() {
  using namespace bw_syn;

  auto empty = apply_scale_separate_mul(Program{});
  assert(!empty.applied);
  assert(!empty.sound_under_preconditions);

  auto direct = tanh_bw::ir_direct();
  auto sep = apply_scale_separate_mul(direct);
  assert(sep.applied);
  assert(!sep.sound_under_preconditions);

  auto deferred = apply_defer_rounding(direct);
  assert(deferred.applied);
  assert(deferred.sound_under_preconditions);
  assert(!has_intermediate_bf16_round(deferred.program));

  auto pipe = apply_default_scale_pipeline(direct);
  assert(pipe.applied);
  assert(pipe.sound_under_preconditions);

  auto contract = NumericalContract::bf16_default("tanh_backward");
  BF16 x = BF16::from_f64(45.0);
  BF16 g = BF16::from_f64(4.0);
  BF16 y = detail::eval_ir_under_contract(pipe.program, x, g, contract);
  BF16 ref = contract_reference(BackwardKind::Tanh, contract, x, g);
  assert(y.bits == ref.bits);
  assert(y.bits == 0x008f);

  auto already = tanh_bw::ir_scale_separated();
  auto ev = eval_program(already, 45.0, 4.0);
  assert(ev.value_bf16.bits == 0x008f);

  std::cout << "ok test_ir_transforms\n";
  return 0;
}
