#include "bw_syn/verify.hpp"

#include <cmath>
#include <sstream>

namespace bw_syn {

std::vector<BF16> special_bf16_values() {
  return {
      BF16::zero(false),
      BF16::zero(true),
      BF16::inf(false),
      BF16::inf(true),
      BF16::qnan(),
      BF16::min_subnormal(),
      BF16::min_normal(),
      BF16::max_finite(),
      BF16::from_f64(1.0),
      BF16::from_f64(-1.0),
  };
}

std::vector<BF16> boundary_bf16_values() {
  std::vector<BF16> v = special_bf16_values();

  for (double x : {0.0, 1.0, 2.0, 8.0, 16.0, 20.0, 40.0, 44.0, 45.0, 46.0, 50.0, 60.0, 80.0}) {
    v.push_back(BF16::from_f64(x));
    v.push_back(BF16::from_f64(-x));
  }
  for (double g : {0.0, 1.0, 2.0, 4.0, 8.0, 0.5, 16.0, 100.0, 1e-3, 1e3}) {
    v.push_back(BF16::from_f64(g));
    v.push_back(BF16::from_f64(-g));
  }
  return v;
}

ReducedDomain build_reduced_domain(BackwardKind kind,
                                   const NumericalContract& contract,
                                   int significand_samples) {
  ReducedDomain d;
  d.claimed_sound = false;
  d.rationale = "Hypothesis: for f' even and positive (tanh/sigmoid), and P ≈ round(g*f'(x)), "
                "fixing a grid of significands for |x| and |g| plus exponent boundary bands "
                "may cover failure modes of scale-separated implementations. "
                "NOT proved for arbitrary IR with intermediate rounding; treat as experimental "
                "reduction pending appendix proof obligations.";
  (void)kind;
  (void)contract;

  std::vector<double> sigs;
  for (int i = 0; i < significand_samples; ++i) {

    sigs.push_back(1.0 + static_cast<double>(i) / significand_samples);
  }

  const int exps_x[] = {-10, -2, 0, 2, 4, 5, 6, 8, 10};
  const int exps_g[] = {-20, -10, -5, 0, 2, 5, 10, 20};

  for (int ex : exps_x) {
    for (double s : sigs) {
      const double xv = std::ldexp(s, ex);
      if (!std::isfinite(xv) || xv > 1e8)
        continue;
      BF16 x = BF16::from_f64(xv);
      for (int eg : exps_g) {
        for (double sg : {1.0, 1.5}) {
          const double gv = std::ldexp(sg, eg);
          if (!std::isfinite(gv))
            continue;
          d.pairs.emplace_back(x, BF16::from_f64(gv));
        }
      }
    }
  }

  d.pairs.emplace_back(BF16::from_f64(45.0), BF16::from_f64(4.0));
  return d;
}

VerifyReport verify_kernel(const KernelFn& kernel, const VerifyConfig& cfg) {
  VerifyReport rep;
  auto& c = rep.counters;

  auto consider = [&](BF16 x, BF16 g) {
    if (cfg.contract.max_abs_x && std::fabs(x.to_f64()) > *cfg.contract.max_abs_x) {
      ++c.skipped;
      return;
    }
    BF16 expected = contract_reference(cfg.kind, cfg.contract, x, g, cfg.alpha);
    BF16 actual = kernel(x, g);
    auto chk = check_sample(cfg.contract, x, g, actual, expected);
    ++c.tested;
    switch (chk.verdict) {
    case ContractVerdict::Pass:
      ++c.pass;
      c.sum_ulp += chk.ulp;
      if (chk.ulp > c.max_ulp) {
        c.max_ulp = chk.ulp;
        c.worst_x = x;
        c.worst_g = g;
        c.worst_actual = actual;
        c.worst_expected = expected;
      }
      break;
    case ContractVerdict::Skip:
      ++c.skipped;
      --c.tested;
      break;
    case ContractVerdict::FailUlp:
      ++c.fail_ulp;
      c.sum_ulp += chk.ulp;
      if (chk.ulp > c.max_ulp) {
        c.max_ulp = chk.ulp;
        c.worst_x = x;
        c.worst_g = g;
        c.worst_actual = actual;
        c.worst_expected = expected;
      }
      break;
    case ContractVerdict::FailFalseZero:
      ++c.false_zeros;
      c.worst_x = x;
      c.worst_g = g;
      c.worst_actual = actual;
      c.worst_expected = expected;
      break;
    case ContractVerdict::FailFalseInf:
      ++c.false_infs;
      break;
    case ContractVerdict::FailSign:
      ++c.fail_sign;
      break;
    case ContractVerdict::FailException:
      ++c.fail_exception;
      break;
    case ContractVerdict::FailDomain:
      ++c.fail_exception;
      break;
    }
  };

  if (cfg.use_reduced_significand_domain) {
    auto red = build_reduced_domain(cfg.kind, cfg.contract, cfg.significand_samples);
    for (auto& [x, g] : red.pairs) {
      if (c.tested >= cfg.max_samples) {
        c.hit_sample_cap = true;
        break;
      }
      consider(x, g);
    }
  } else if (cfg.exhaustive_x && !cfg.fixed_g_values.empty()) {
    for (BF16 g : cfg.fixed_g_values) {
      for (std::uint32_t bits = 0; bits < 0x10000u; ++bits) {
        if (c.tested >= cfg.max_samples) {
          c.hit_sample_cap = true;
          break;
        }
        BF16 x = BF16::from_bits(static_cast<std::uint16_t>(bits));
        consider(x, g);
      }
      if (c.hit_sample_cap)
        break;
    }
  } else if (cfg.exhaustive_g && !cfg.fixed_x_values.empty()) {
    for (BF16 x : cfg.fixed_x_values) {
      for (std::uint32_t bits = 0; bits < 0x10000u; ++bits) {
        if (c.tested >= cfg.max_samples) {
          c.hit_sample_cap = true;
          break;
        }
        BF16 g = BF16::from_bits(static_cast<std::uint16_t>(bits));
        consider(x, g);
      }
      if (c.hit_sample_cap)
        break;
    }
  } else {
    auto xs = boundary_bf16_values();
    auto gs = boundary_bf16_values();
    if (cfg.include_specials) {
      for (auto s : special_bf16_values()) {
        xs.push_back(s);
        gs.push_back(s);
      }
    }
    for (BF16 x : xs) {
      for (BF16 g : gs) {
        if (c.tested >= cfg.max_samples) {
          c.hit_sample_cap = true;
          break;
        }
        consider(x, g);
      }
      if (c.hit_sample_cap)
        break;
    }
  }

  const std::uint64_t fails =
      c.fail_ulp + c.false_zeros + c.false_infs + c.fail_sign + c.fail_exception;

  const bool exhaustive = cfg.exhaustive_x || cfg.exhaustive_g;
  rep.ok = (fails == 0) && (c.tested > 0) && !(exhaustive && c.hit_sample_cap);

  std::ostringstream os;
  os << "verify kind=" << backward_kind_name(cfg.kind) << " tested=" << c.tested
     << " pass=" << c.pass << " skipped=" << c.skipped << " fail_ulp=" << c.fail_ulp
     << " false_zeros=" << c.false_zeros << " false_infs=" << c.false_infs
     << " fail_sign=" << c.fail_sign << " fail_exc=" << c.fail_exception
     << " max_ulp=" << c.max_ulp;
  if (c.hit_sample_cap)
    os << " HIT_SAMPLE_CAP";
  if (c.pass > 0)
    os << " mean_ulp=" << (c.sum_ulp / static_cast<double>(c.pass + c.fail_ulp));
  if (!rep.ok) {
    os << " worst_x=" << c.worst_x.to_string() << " worst_g=" << c.worst_g.to_string()
       << " got=" << c.worst_actual.to_string() << " exp=" << c.worst_expected.to_string();
  }
  rep.summary = os.str();
  return rep;
}

} // namespace bw_syn
