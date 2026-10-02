#include "bw_syn/backends/tt_harness.hpp"
#include "bw_syn/functions/tanh_bw.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

int main() {
  using namespace bw_syn::tt_harness;
  auto probe = probe_device();
  assert(probe.available);
  auto rep = run_critical_tanh(DeviceMode::HostSim);
  assert(rep.all_pass);

  bool saw_false = false;
  bool saw_infinite_gradient = false;
  for (const auto& c : rep.cases) {
    if (c.spec.x == 45.f && c.spec.g == 4.f) {
      assert(c.pass);
      assert(c.oracle.bits == 0x008f);
      assert(c.false_zero_baseline);
      saw_false = true;
    }
    if (c.spec.x == bw_syn::BF16::max_finite().to_f32() && std::isinf(c.spec.g)) {
      assert(c.pass);
      assert(c.oracle.bits == bw_syn::BF16::inf(false).bits);
      saw_infinite_gradient = true;
    }
  }
  assert(saw_false);
  assert(saw_infinite_gradient);

  const auto contract = bw_syn::NumericalContract::bf16_default("tanh_backward");
  const auto extreme_x = bw_syn::BF16::from_f64(88.5);
  const auto extreme_g = bw_syn::BF16::max_finite();
  const auto extreme_ref =
      bw_syn::contract_reference(bw_syn::BackwardKind::Tanh, contract, extreme_x, extreme_g);
  assert(extreme_ref.is_normal());
  assert(std::exp(-extreme_x.to_f32()) < std::numeric_limits<float>::min());
  const auto extreme_got = bw_syn::tanh_bw::tail_split4_host(extreme_x, extreme_g, contract);
  assert(bw_syn::check_sample(contract, extreme_x, extreme_g, extreme_got, extreme_ref).verdict ==
         bw_syn::ContractVerdict::Pass);

  const std::vector<bw_syn::BF16> gradients = {bw_syn::BF16::from_f64(1),
                                               bw_syn::BF16::from_f64(4),
                                               bw_syn::BF16::from_f64(256),
                                               bw_syn::BF16::from_f64(4294967296.0),
                                               bw_syn::BF16::max_finite(),
                                               bw_syn::BF16::from_bits(0xff7f)};
  std::size_t checked = 0;
  for (std::uint32_t bits = bw_syn::BF16::from_f64(4).bits;
       bits <= bw_syn::BF16::from_f64(88.5).bits;
       ++bits) {
    const auto xi = bw_syn::BF16::from_bits(static_cast<std::uint16_t>(bits));
    for (const auto gi : gradients) {
      const auto expected =
          bw_syn::contract_reference(bw_syn::BackwardKind::Tanh, contract, xi, gi);
      if (!expected.is_normal())
        continue;
      const auto actual = bw_syn::tanh_bw::tail_split4_host(xi, gi, contract);
      const auto verdict = bw_syn::check_sample(contract, xi, gi, actual, expected).verdict;
      if (verdict != bw_syn::ContractVerdict::Pass) {
        std::cerr << "tail split host failure x=" << xi.to_string() << " g=" << gi.to_string()
                  << " got=" << actual.to_string() << " expected=" << expected.to_string() << "\n";
        return 1;
      }
      ++checked;
    }
  }
  assert(checked > 1000);
  const auto tail = run_tail_tanh(DeviceMode::HostSim);
  assert(tail.cases.size() == 3402);
  assert(tail.normal_outputs > 2900);
  assert(tail.all_normal_outputs_pass);
  assert(!tail.vendor_normal_passes);
  const auto format = run_format_probe(DeviceMode::HostSim);
  assert(format.rows.size() == 8);
  for (const auto& row : format.rows)
    assert(row.expected.bits == row.observed.bits);
  std::cout << "ok test_tt_harness_sim\n";
  return 0;
}
