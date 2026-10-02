#include "bw_syn/contract.hpp"

#include <cmath>

namespace bw_syn {

NumericalContract NumericalContract::bf16_default(const std::string& fname) {
  NumericalContract c;
  c.function_name = fname;
  c.output_round = RoundMode::ToNearestEven;
  c.flush_output_subnormals = false;
  c.baseline_flush_derivative_subnormals = true;
  c.max_ulp_error = 1;
  c.require_signed_zero = true;
  c.require_ieee_exceptions = true;
  c.notes = "P within max_ulp_error of round(exact_product_f64(...)) "
            "(f64 scaled host reference), not correctly-rounded reals. "
            "baseline_flush_derivative_subnormals is a host model of unsafe "
            "intermediate FTZ, not device measurement.";
  return c;
}

std::string verdict_name(ContractVerdict v) {
  switch (v) {
  case ContractVerdict::Pass:
    return "pass";
  case ContractVerdict::FailUlp:
    return "fail_ulp";
  case ContractVerdict::FailFalseZero:
    return "fail_false_zero";
  case ContractVerdict::FailFalseInf:
    return "fail_false_inf";
  case ContractVerdict::FailSign:
    return "fail_sign";
  case ContractVerdict::FailException:
    return "fail_exception";
  case ContractVerdict::FailDomain:
    return "fail_domain";
  case ContractVerdict::Skip:
    return "skip";
  }
  return "?";
}

ContractCheckResult
check_sample(const NumericalContract& c, BF16 x, BF16 g, BF16 actual, BF16 expected_ref) {
  ContractCheckResult r;
  r.expected = expected_ref;
  r.actual = actual;

  if (c.finite_inputs_only && (!x.is_finite() || !g.is_finite())) {
    r.verdict = ContractVerdict::Skip;
    r.detail = "nonfinite input outside contract domain";
    return r;
  }
  if (c.min_abs_x && std::fabs(x.to_f64()) < *c.min_abs_x) {
    r.verdict = ContractVerdict::Skip;
    r.detail = "below |x| domain";
    return r;
  }
  if (c.normal_reference_output_only && !expected_ref.is_normal()) {
    r.verdict = ContractVerdict::Skip;
    r.detail = "reference output outside normal range";
    return r;
  }

  if (c.max_abs_x.has_value()) {
    if (std::fabs(x.to_f64()) > *c.max_abs_x) {
      r.verdict = ContractVerdict::Skip;
      r.detail = "outside |x| domain";
      return r;
    }
  }

  if (!x.is_finite() || !g.is_finite() || !expected_ref.is_finite() || !actual.is_finite()) {
    if (!c.require_ieee_exceptions) {
      r.verdict = ContractVerdict::Skip;
      return r;
    }
    if (expected_ref.is_nan() || x.is_nan() || g.is_nan()) {
      r.verdict = actual.is_nan() ? ContractVerdict::Pass : ContractVerdict::FailException;
      r.detail = "NaN handling";
      return r;
    }
    if (actual.is_nan() && expected_ref.is_finite()) {
      r.verdict = ContractVerdict::FailException;
      r.detail = "NaN actual vs finite expected";
      return r;
    }
    if (expected_ref.is_inf()) {
      if (!actual.is_inf()) {
        r.verdict = ContractVerdict::FailFalseInf;
        r.detail = "expected Inf";
        return r;
      }
      if (expected_ref.signbit() != actual.signbit()) {
        r.verdict = ContractVerdict::FailSign;
        return r;
      }
      r.verdict = ContractVerdict::Pass;
      return r;
    }
    if (actual.is_inf() && !expected_ref.is_inf()) {
      r.verdict = ContractVerdict::FailFalseInf;
      return r;
    }

    if (!actual.is_finite() || !expected_ref.is_finite()) {
      r.verdict = ContractVerdict::FailException;
      r.detail = "exceptional mismatch";
      return r;
    }
  }

  if (actual.is_zero() && !expected_ref.is_zero() && expected_ref.is_finite()) {
    r.verdict = ContractVerdict::FailFalseZero;
    r.ulp = ulp_distance(actual, expected_ref);
    r.detail = "false zero vs representable reference";
    return r;
  }

  if (c.require_signed_zero && actual.is_zero() && expected_ref.is_zero()) {
    if (actual.signbit() != expected_ref.signbit()) {
      r.verdict = ContractVerdict::FailSign;
      r.detail = "signed zero mismatch";
      return r;
    }
  }

  r.ulp = ulp_distance(actual, expected_ref);
  if (r.ulp <= c.max_ulp_error) {
    r.verdict = ContractVerdict::Pass;
  } else {
    r.verdict = ContractVerdict::FailUlp;
    r.detail = "ULP exceeds contract";
  }
  return r;
}

} // namespace bw_syn
