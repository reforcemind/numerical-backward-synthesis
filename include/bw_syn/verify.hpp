#pragma once

#include "bw_syn/bf16.hpp"
#include "bw_syn/contract.hpp"
#include "bw_syn/oracle.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace bw_syn {

using KernelFn = std::function<BF16(BF16 x, BF16 g)>;

struct VerifyConfig {
  NumericalContract contract;
  BackwardKind kind{BackwardKind::Tanh};
  double alpha{1.0};
  bool exhaustive_x{false};
  bool exhaustive_g{false};
  std::vector<BF16> fixed_g_values;
  std::vector<BF16> fixed_x_values;
  bool use_reduced_significand_domain{false};
  int significand_samples{64};
  std::uint64_t max_samples{200000};
  bool include_specials{true};
};

struct VerifyCounters {
  std::uint64_t tested{0};
  std::uint64_t skipped{0};
  std::uint64_t pass{0};
  std::uint64_t fail_ulp{0};
  std::uint64_t false_zeros{0};
  std::uint64_t false_infs{0};
  std::uint64_t fail_sign{0};
  std::uint64_t fail_exception{0};
  std::uint32_t max_ulp{0};
  double sum_ulp{0.0};
  bool hit_sample_cap{false};
  BF16 worst_x{};
  BF16 worst_g{};
  BF16 worst_actual{};
  BF16 worst_expected{};
};

struct VerifyReport {
  VerifyCounters counters;
  std::string summary;
  bool ok{false};
};

VerifyReport verify_kernel(const KernelFn& kernel, const VerifyConfig& cfg);

std::vector<BF16> special_bf16_values();
std::vector<BF16> boundary_bf16_values();

struct ReducedDomain {
  std::vector<std::pair<BF16, BF16>> pairs;
  std::string rationale;
  bool claimed_sound{false};
};

ReducedDomain build_reduced_domain(BackwardKind kind,
                                   const NumericalContract& contract,
                                   int significand_samples = 64);

} // namespace bw_syn
