#include "bw_syn/functions/elu_bw.hpp"
#include "bw_syn/oracle.hpp"

#include <cassert>
#include <iostream>

int main() {
  using namespace bw_syn;
  auto c = NumericalContract::bf16_default("elu_backward");
  BF16 g = BF16::from_f64(2.0);

  BF16 xpos = BF16::from_f64(1.0);
  assert(ulp_distance(elu_bw::scale_separated(xpos, g, c),
                      contract_reference(BackwardKind::Elu, c, xpos, g)) == 0);

  BF16 x0 = BF16::from_f64(0.0);
  BF16 y0 = elu_bw::scale_separated(x0, g, c);
  BF16 r0 = contract_reference(BackwardKind::Elu, c, x0, g);
  assert(ulp_distance(y0, r0) == 0);

  BF16 xneg = BF16::from_f64(-1.0);
  assert(ulp_distance(elu_bw::scale_separated(xneg, g, c),
                      contract_reference(BackwardKind::Elu, c, xneg, g)) == 0);

  auto ir = elu_bw::ir_scale_separated(1.0);
  auto ev = eval_program(ir, -1.0, 2.0);
  assert(ev.value_f64 > 0.0 && ev.value_f64 < 2.0);

  std::cout << "ok test_elu\n";
  return 0;
}
