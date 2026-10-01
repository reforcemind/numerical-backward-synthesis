#include "bw_syn/backends/sfpu_emit.hpp"
#include "bw_syn/cost_model.hpp"
#include "bw_syn/detail/scale_eval.hpp"
#include "bw_syn/functions/tanh_bw.hpp"
#include "bw_syn/ir.hpp"
#include "bw_syn/transforms.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

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

  Program plain;
  const int px = plain.add(Node{OpKind::InputX});
  const int pg = plain.add(Node{OpKind::InputG});
  plain.result = plain.add(Node{OpKind::Mul, {px, pg}});
  plain.add(Node{OpKind::Exp, {px}});
  assert(estimate_cost(plain, {}).instruction_count == 3);
  auto emitted = sfpu::emit_sfpi_cpp(plain, {"plain_product"});
  assert(emitted.find("v0 * v1") != std::string::npos);
  assert(emitted.find("plain_product") != std::string::npos);

  auto expect_unsupported = [](OpKind op) {
    Program p;
    const int a = p.add(Node{OpKind::InputX});
    const int b = p.add(Node{OpKind::InputG});
    p.result = p.add(Node{op, {a, b}});
    try {
      (void)sfpu::emit_sfpi_cpp(p);
      assert(false);
    } catch (const std::runtime_error& ex) {
      assert(std::string(ex.what()).find(op_kind_name(op)) != std::string::npos);
    }
  };
  for (auto op : {OpKind::Div,
                  OpKind::Exp,
                  OpKind::Select,
                  OpKind::Copysign,
                  OpKind::Frexp,
                  OpKind::FrexpMant,
                  OpKind::FrexpExp,
                  OpKind::Ldexp,
                  OpKind::ScaleMul,
                  OpKind::AddExp,
                  OpKind::Normalize,
                  OpKind::Reconstruct,
                  OpKind::RoundBF16,
                  OpKind::FlushBF16,
                  OpKind::PositiveDerivativeInf})
    expect_unsupported(op);
  try {
    (void)sfpu::emit_sfpi_cpp(already);
    assert(false);
  } catch (const std::runtime_error& ex) {
    assert(std::string(ex.what()).find("RoundBF16") != std::string::npos);
  }

  std::cout << "ok test_ir_transforms\n";
  return 0;
}
