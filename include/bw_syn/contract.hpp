#pragma once

#include "bw_syn/bf16.hpp"

#include <optional>
#include <string>

namespace bw_syn {

struct NumericalContract {
  std::string function_name;
  RoundMode output_round = RoundMode::ToNearestEven;
  bool flush_output_subnormals = false;
  // Host model of flushing a materialized BF16 derivative; not device FTZ.
  bool baseline_flush_derivative_subnormals = true;
  std::uint32_t max_ulp_error = 1;
  std::optional<double> max_abs_x;
  std::optional<double> min_abs_x;
  bool finite_inputs_only = false;
  bool normal_reference_output_only = false;
  bool require_signed_zero = true;
  bool require_ieee_exceptions = true;
  std::string notes;

  static NumericalContract bf16_default(const std::string& fname);
};

enum class ContractVerdict {
  Pass,
  FailUlp,
  FailFalseZero,
  FailFalseInf,
  FailSign,
  FailException,
  FailDomain,
  Skip,
};

struct ContractCheckResult {
  ContractVerdict verdict{ContractVerdict::Skip};
  std::uint32_t ulp{0};
  BF16 expected{};
  BF16 actual{};
  std::string detail;
};

std::string verdict_name(ContractVerdict v);

ContractCheckResult
check_sample(const NumericalContract& c, BF16 x, BF16 g, BF16 actual, BF16 expected_ref);

} // namespace bw_syn
