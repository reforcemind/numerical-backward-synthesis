#pragma once

#include "bw_syn/bf16.hpp"
#include "bw_syn/contract.hpp"

#include <string>

namespace bw_syn {

enum class BackwardKind {
  Tanh,
  Sigmoid,
  Erf,
  Elu,
};

std::string backward_kind_name(BackwardKind k);

double exact_derivative_f64(BackwardKind kind, double x, double alpha = 1.0);

double exact_product_f64(BackwardKind kind, double x, double g, double alpha = 1.0);

BF16 contract_reference(
    BackwardKind kind, const NumericalContract& contract, BF16 x, BF16 g, double alpha = 1.0);

BF16 baseline_materialize_derivative(
    BackwardKind kind, const NumericalContract& contract, BF16 x, BF16 g, double alpha = 1.0);

struct MotivatingCaseReport {
  double x;
  double g;
  double sech2;
  double product;
  BF16 product_bf16;
  BF16 derivative_bf16;
  BF16 baseline_bf16;
  BF16 min_normal;
  bool derivative_underflows_normal;
  bool product_is_normal;
  bool baseline_is_false_zero;
};

MotivatingCaseReport analyze_tanh_motivating_case(double x = 45.0, double g = 4.0);

} // namespace bw_syn
