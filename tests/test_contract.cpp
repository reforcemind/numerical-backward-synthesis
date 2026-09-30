#include "bw_syn/contract.hpp"
#include "bw_syn/oracle.hpp"

#include <cassert>
#include <iostream>

int main() {
  using namespace bw_syn;
  auto c = NumericalContract::bf16_default("tanh_backward");
  BF16 x = BF16::from_f64(45.0);
  BF16 g = BF16::from_f64(4.0);
  BF16 exp = contract_reference(BackwardKind::Tanh, c, x, g);
  assert(exp.bits == 0x008f);
  auto r = check_sample(c, x, g, exp, exp);
  assert(r.verdict == ContractVerdict::Pass);

  BF16 zero = BF16::zero();
  auto r2 = check_sample(c, x, g, zero, exp);
  assert(r2.verdict == ContractVerdict::FailFalseZero);

  BF16 gn = BF16::from_f64(-4.0);
  BF16 zn = contract_reference(BackwardKind::Tanh, c, BF16::from_f64(400.0), gn);
  assert(zn.is_zero() && zn.signbit());

  assert(contract_reference(BackwardKind::Tanh, c, BF16::qnan(), g).is_nan());
  assert(contract_reference(BackwardKind::Tanh, c, BF16::inf(false), g).is_zero());
  assert(contract_reference(BackwardKind::Tanh, c, BF16::inf(false), BF16::inf(false)).is_nan());

  BF16 sub = BF16::min_subnormal();
  assert(flush_subnormals(sub).is_zero());
  BF16 tiny = contract_reference(BackwardKind::Tanh, c, BF16::from_f64(0.0), sub);
  assert(tiny.is_finite());

  std::cout << "ok test_contract\n";
  return 0;
}
