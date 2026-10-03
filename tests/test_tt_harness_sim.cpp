#include "bw_syn/backends/tt_harness.hpp"
#include "bw_syn/detail/tail_exp2_coefficients.hpp"
#include "bw_syn/detail/tail_exp_host.hpp"
#include "bw_syn/functions/tanh_bw.hpp"
#include "bw_syn/verify.hpp"
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
  assert(tail.cases.size() == 6774);
  assert(tail.normal_outputs > 2900);
  assert(tail.all_normal_outputs_pass);
  assert(!tail.vendor_normal_passes);
  for (const auto [xb, gb] : std::vector<std::pair<std::uint16_t, std::uint16_t>>{
           {0x4080, 0x053a}, {0x4083, 0x0560}, {0x4234, 0x4080}, {0x42b1, 0x7f7f}}) {
    for (const std::uint16_t sx : {0, 0x8000})
      for (const std::uint16_t sg : {0, 0x8000}) {
        const auto xi = bw_syn::BF16::from_bits(xb | sx);
        const auto gi = bw_syn::BF16::from_bits(gb | sg);
        const auto ref = bw_syn::contract_reference(bw_syn::BackwardKind::Tanh, contract, xi, gi);
        assert(ref.is_normal());
        for (int degree : {2, 3, 4}) {
          const auto got = bw_syn::tanh_bw::tail_fused_host(xi, gi, contract, degree);
          assert(bw_syn::check_sample(contract, xi, gi, got, ref).verdict ==
                 bw_syn::ContractVerdict::Pass);
        }
      }
  }
  const auto boundary_x = bw_syn::BF16::from_bits(0xc29d);
  const auto boundary_g = bw_syn::BF16::from_bits(0xf0b5);
  const auto boundary_ref =
      bw_syn::contract_reference(bw_syn::BackwardKind::Tanh, contract, boundary_x, boundary_g);
  assert(boundary_ref.bits == 0x8080);
  const auto uncorrected = bw_syn::detail::tail_exp2_product<bw_syn::detail::HostTailOps, false>(
      boundary_x.to_f32(), boundary_g.to_f32(), bw_syn::detail::kTailExp2Quadratic);
  assert(uncorrected == 0.0f);
  assert(bw_syn::tanh_bw::tail_fused_host(boundary_x, boundary_g, contract).bits ==
         boundary_ref.bits);
  bw_syn::VerifyConfig zero_cfg;
  zero_cfg.contract = contract;
  zero_cfg.exhaustive_g = true;
  zero_cfg.fixed_x_values = {boundary_x};
  zero_cfg.contract.finite_inputs_only = true;
  zero_cfg.contract.normal_reference_output_only = true;
  const auto zero_rep = bw_syn::verify_kernel(
      [](bw_syn::BF16, bw_syn::BF16) { return bw_syn::BF16::zero(); }, zero_cfg);
  assert(zero_rep.counters.false_zeros > 0 && zero_rep.counters.max_ulp >= 128);
  const auto format = run_format_probe(DeviceMode::HostSim);
  assert(format.rows.size() == 8);
  for (const auto& row : format.rows)
    assert(row.expected.bits == row.observed.bits);
  std::cout << "ok test_tt_harness_sim\n";
  return 0;
}
