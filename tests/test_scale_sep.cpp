#include "bw_syn/functions/tanh_bw.hpp"
#include "bw_syn/oracle.hpp"
#include <cassert>
#include <iostream>

int main() {
  using namespace bw_syn;
  auto c = NumericalContract::bf16_default("tanh_backward");
  BF16 x = BF16::from_f64(45.0);
  BF16 g = BF16::from_f64(4.0);
  BF16 o = contract_reference(BackwardKind::Tanh, c, x, g);
  BF16 s = tanh_bw::scale_separated(x, g, c);
  BF16 b = tanh_bw::baseline_materialize(x, g, c);
  assert(ulp_distance(s, o) == 0);
  assert(b.is_zero());
  assert(!o.is_zero());
  std::cout << "ok test_scale_sep\n";
  return 0;
}
