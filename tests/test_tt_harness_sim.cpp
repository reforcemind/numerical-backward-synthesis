#include "bw_syn/backends/tt_harness.hpp"
#include <cassert>
#include <iostream>

int main() {
  using namespace bw_syn::tt_harness;
  auto probe = probe_device();
  assert(probe.available);
  auto rep = run_critical_tanh(DeviceMode::HostSim);
  assert(rep.all_pass);

  bool saw_false = false;
  for (const auto& c : rep.cases) {
    if (c.spec.x == 45.f && c.spec.g == 4.f) {
      assert(c.pass);
      assert(c.oracle.bits == 0x008f);
      assert(c.false_zero_baseline);
      saw_false = true;
    }
  }
  assert(saw_false);
  std::cout << "ok test_tt_harness_sim\n";
  return 0;
}
