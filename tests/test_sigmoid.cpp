#include "bw_syn/functions/sigmoid_bw.hpp"
#include "bw_syn/oracle.hpp"
#include <cassert>
#include <iostream>

int main() {
  using namespace bw_syn;
  auto c = NumericalContract::bf16_default("sigmoid_backward");

  BF16 x = BF16::from_f64(90.0);
  BF16 g = BF16::from_f64(8.0);
  BF16 o = contract_reference(BackwardKind::Sigmoid, c, x, g);
  BF16 s = sigmoid_bw::scale_separated(x, g, c);
  BF16 b = sigmoid_bw::baseline_materialize(x, g, c);
  assert(ulp_distance(s, o) == 0);
  assert(b.is_zero() && !o.is_zero());

  std::cout << "ok test_sigmoid\n";
  return 0;
}
