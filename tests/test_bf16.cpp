#include "bw_syn/bf16.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
  using namespace bw_syn;
  assert(BF16::from_f64(0.0).is_zero());
  assert(BF16::from_f64(1.0).bits == 0x3f80);
  assert(BF16::min_normal().to_f64() > 0.0);
  assert(ulp_distance(BF16::from_f64(1.0), BF16::from_f64(1.0)) == 0);

  BF16 p = round_to_bf16(4.0 * 3.277605e-39);
  std::cout << "product bf16=" << p.to_string() << "\n";
  assert(p.is_normal() || p.is_subnormal());

  BF16 d = flush_subnormals(round_to_bf16(3.277605e-39));
  assert(d.is_zero());
  std::cout << "ok test_bf16\n";
  return 0;
}
