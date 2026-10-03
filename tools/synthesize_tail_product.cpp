#include "bw_syn/csv_io.hpp"
#include "bw_syn/detail/tail_exp_host.hpp"
#include "bw_syn/verify.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace bw_syn;
using detail::HostTailOps;
using detail::TailQuadratic;

struct Interval {
  float r;
  double lower;
  double upper;
};

std::vector<Interval> product_intervals() {
  std::vector<Interval> intervals;
  for (const auto kind : {BackwardKind::Tanh, BackwardKind::Sigmoid}) {
    const int rate = kind == BackwardKind::Tanh ? 2 : 1;
    const int scale = kind == BackwardKind::Tanh ? 2 : 0;
    const auto contract = NumericalContract::bf16_default(backward_kind_name(kind));
    const auto first = BF16::from_f64(8.0 / rate).bits;
    const auto last = BF16::from_f64(177.0 / rate).bits;
    for (unsigned xb = first; xb <= last; ++xb) {
      const auto x = BF16::from_bits(static_cast<std::uint16_t>(xb));
      const float z = (-rate * x.to_f32()) * 1.4426950408889634f;
      int k_int;
      const float k = HostTailOps::round_int(z, k_int);
      Interval constraint{z - k, 0.0, std::numeric_limits<double>::infinity()};
      for (unsigned mantissa = 0; mantissa < 128; ++mantissa) {
        // The largest finite exponent exposes each significand's interior output
        // interval. The independent acceptance sweep checks every exponent.
        const auto g = BF16::from_bits(static_cast<std::uint16_t>(0x7f00 | mantissa));
        const auto ref = contract_reference(kind, contract, x, g);
        if (!ref.is_normal())
          continue;
        auto value = [&](int offset) {
          return BF16::from_bits(static_cast<std::uint16_t>(ref.bits + offset)).to_f64();
        };
        const double low = (value(-2) + value(-1)) * 0.5;
        const double high = (value(1) + value(2)) * 0.5;
        const double m = 1.0 + mantissa / 128.0;
        const double lower = std::ldexp(low, -(k_int + scale + 127)) / m;
        const double upper = std::ldexp(high, -(k_int + scale + 127)) / m;
        // Stay inside midpoint endpoints and leave room for FP32 evaluation.
        constraint.lower = std::max(constraint.lower, lower * (1.0 + 1e-6));
        constraint.upper = std::min(constraint.upper, upper * (1.0 - 1e-6));
      }
      if (!std::isfinite(constraint.upper))
        continue;
      if (constraint.lower >= constraint.upper)
        throw std::runtime_error("empty final-product approximation interval");
      intervals.push_back(constraint);
    }
  }
  return intervals;
}

TailQuadratic fit(const std::vector<Interval>& intervals, int& sweeps) {
  constexpr double ln2 = 0.693147180559945309417;
  std::array<double, 3> c{1.0, ln2, ln2 * ln2 / 2};
  // Cyclic projections onto linear interval constraints; no optimality claim.
  for (sweeps = 1; sweeps <= 20000; ++sweeps) {
    for (const auto& constraint : intervals) {
      const double r = constraint.r;
      const std::array<double, 3> a{1.0, r, r * r};
      const double y = c[0] + c[1] * a[1] + c[2] * a[2];
      const double delta = (std::clamp(y, constraint.lower, constraint.upper) - y) /
                           (1.0 + a[1] * a[1] + a[2] * a[2]);
      for (int j = 0; j < 3; ++j)
        c[j] += delta * a[j];
    }
    bool feasible = true;
    for (const auto& constraint : intervals) {
      const double y = c[0] + constraint.r * (c[1] + constraint.r * c[2]);
      feasible &= y >= constraint.lower - 1e-12 && y <= constraint.upper + 1e-12;
    }
    if (feasible)
      return {static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2])};
  }
  throw std::runtime_error("quadratic search exhausted; no candidate emitted");
}

template <bool Saturate>
VerifyReport validate(BackwardKind kind, const TailQuadratic& c, bool quick) {
  VerifyConfig cfg;
  cfg.kind = kind;
  cfg.contract = NumericalContract::bf16_default(backward_kind_name(kind));
  cfg.contract.finite_inputs_only = true;
  cfg.contract.normal_reference_output_only = true;
  cfg.exhaustive_g = true;
  cfg.max_samples = 100000000;
  const int rate = kind == BackwardKind::Tanh ? 2 : 1;
  const auto first = BF16::from_f64(8.0 / rate).bits;
  const auto last = BF16::from_f64(177.0 / rate).bits;
  for (unsigned xb = first; xb <= last; ++xb) {
    const auto boundary_regression = BF16::from_f64(157.0 / rate).bits;
    if (quick && xb != first && xb != last && xb != boundary_regression && (xb - first) % 71 != 0)
      continue;
    cfg.fixed_x_values.push_back(BF16::from_bits(static_cast<std::uint16_t>(xb)));
    cfg.fixed_x_values.push_back(BF16::from_bits(static_cast<std::uint16_t>(xb | 0x8000)));
  }
  return verify_kernel(
      [&](BF16 x, BF16 g) {
        const float y =
            kind == BackwardKind::Tanh
                ? detail::tail_exp2_product<HostTailOps, Saturate, 2, 2>(x.to_f32(), g.to_f32(), c)
                : detail::tail_exp2_product<HostTailOps, Saturate, 1, 0>(x.to_f32(), g.to_f32(), c);
        return BF16::from_f32(y);
      },
      cfg);
}
} // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--quick")) {
    std::cerr << "usage: synthesize_tail_product OUT_DIR [--quick]\n";
    return 2;
  }
  try {
    const bool quick = argc == 3;
    const std::filesystem::path out(argv[1]);
    std::filesystem::create_directories(out);
    std::filesystem::remove(out / "tail_exp2_coefficients.hpp");
    const auto start = std::chrono::steady_clock::now();
    const auto intervals = product_intervals();
    int sweeps = 0;
    const auto c = fit(intervals, sweeps);
    const double fit_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << std::setprecision(10) << "quadratic: " << c.c0 << ", " << c.c1 << ", " << c.c2
              << "; projection sweeps=" << sweeps << std::endl;
    std::vector<std::string> rows;
    bool accepted = true;
    bool legacy_accepted = true;
    for (bool saturate : {false, true}) {
      for (const auto kind : {BackwardKind::Tanh, BackwardKind::Sigmoid}) {
        const auto rep =
            saturate ? validate<true>(kind, c, quick) : validate<false>(kind, c, quick);
        const auto& v = rep.counters;
        rows.push_back("host_model," + backward_kind_name(kind) + "," +
                       (saturate ? "normal_saturate" : "wh4_boundary") + "," +
                       (quick ? "sampled_x_all_g_both_signs" : "all_x_all_g_both_signs") + "," +
                       std::to_string(v.tested) + "," + std::to_string(v.tested - v.pass) + "," +
                       std::to_string(v.false_zeros) + "," + std::to_string(v.max_ulp) + "," +
                       hex16(v.worst_x.bits) + "," + hex16(v.worst_g.bits) + "," +
                       hex16(v.worst_actual.bits) + "," + hex16(v.worst_expected.bits));
        std::cout << rows.back() << std::endl;
        if (saturate)
          accepted &= rep.ok;
        else
          legacy_accepted &= rep.ok;
      }
    }
    if (!write_csv((out / "verification.csv").string(),
                   "label,function,reconstruction,domain,checked,failed,false_zeros,max_ulp,"
                   "worst_x_bits,worst_g_bits,worst_actual_bits,worst_expected_bits",
                   rows))
      throw std::runtime_error("cannot write verification CSV");
    const double total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ofstream manifest(out / "manifest.txt");
    manifest << std::setprecision(10) << "label=host_model\nalgorithm=exp2_quadratic_product\n"
             << "arithmetic=IEEE_FP32_std_fma\noutput=BF16_RNE\nmax_ulp=1\n"
             << "scope=normal_reference_outputs_finite_normal_g\n"
             << "tanh_abs_x=4..88.5\nsigmoid_abs_x=8..177\n"
             << "intervals=" << intervals.size() << "\nprojection_sweeps=" << sweeps
             << "\nfit_seconds=" << fit_seconds
             << "\nverification_seconds=" << total_seconds - fit_seconds
             << "\ntotal_seconds=" << total_seconds << "\nc0=" << c.c0 << "\nc1=" << c.c1
             << "\nc2=" << c.c2 << "\nselected_reconstruction=normal_saturate\n"
             << "wh4_boundary_accepted=" << legacy_accepted
             << "\nnormal_saturate_accepted=" << accepted << '\n'
             << "claimed_sound=false\noptimality_proved=false\ndevice_verified=false\n"
             << "full_host_accepted=" << (accepted && !quick) << '\n';
    if (!manifest)
      throw std::runtime_error("cannot write manifest");
    if (accepted && !quick) {
      std::ofstream header(out / "tail_exp2_coefficients.hpp");
      header << "#pragma once\n\n#include \"tail_exp_product.hpp\"\n\n"
             << "namespace bw_syn::detail {\n"
             << "// Generated by synthesize_tail_product; see results/host/tail_synthesis.\n"
             << "inline constexpr TailQuadratic kTailExp2Quadratic{" << std::hexfloat << c.c0
             << "f, " << c.c1 << "f, " << c.c2 << "f};\n"
             << "} // namespace bw_syn::detail\n";
      if (!header)
        throw std::runtime_error("cannot write selected coefficients");
    }
    return accepted ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 3;
  }
}
