#include "bw_syn/oracle.hpp"
#include <cassert>
#include <iostream>

int main() {
  using namespace bw_syn;
  auto r = analyze_tanh_motivating_case(45.0, 4.0);
  assert(r.derivative_underflows_normal);
  assert(r.baseline_is_false_zero);
  assert(r.product_is_normal);
  assert(r.product_bf16.bits == 0x008f);
  std::cout << "ok test_tanh_oracle bits=0x" << std::hex << r.product_bf16.bits << std::dec << "\n";
  return 0;
}
