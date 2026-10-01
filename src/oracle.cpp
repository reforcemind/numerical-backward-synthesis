#include "bw_syn/oracle.hpp"

#include <cmath>
#include <stdexcept>

namespace bw_syn {

std::string backward_kind_name(BackwardKind k) {
  switch (k) {
  case BackwardKind::Tanh:
    return "tanh_backward";
  case BackwardKind::Sigmoid:
    return "sigmoid_backward";
  case BackwardKind::Erf:
    return "erf_backward";
  case BackwardKind::Elu:
    return "elu_backward";
  }
  return "unknown";
}

double exact_derivative_f64(BackwardKind kind, double x, double alpha) {
  if (kind == BackwardKind::Elu && !std::isfinite(alpha))
    throw std::invalid_argument("ELU alpha must be finite");
  switch (kind) {
  case BackwardKind::Tanh: {

    const double ax = std::fabs(x);
    if (ax > 40.0) {

      return 4.0 * std::exp(-2.0 * ax);
    }
    const double e = std::exp(-2.0 * ax);
    const double d = 1.0 + e;
    return 4.0 * e / (d * d);
  }
  case BackwardKind::Sigmoid: {

    const double ax = std::fabs(x);
    if (ax > 40.0)
      return std::exp(-ax);
    const double e = std::exp(-ax);
    const double d = 1.0 + e;
    return e / (d * d);
  }
  case BackwardKind::Erf: {
    const double k = 2.0 / std::sqrt(3.14159265358979323846);
    return k * std::exp(-x * x);
  }
  case BackwardKind::Elu: {
    if (x > 0.0)
      return 1.0;
    return alpha * std::exp(x);
  }
  }
  return 0.0;
}

double exact_product_f64(BackwardKind kind, double x, double g, double alpha) {
  if (kind == BackwardKind::Elu && !std::isfinite(alpha))
    throw std::invalid_argument("ELU alpha must be finite");
  constexpr double kLog2 = 0.693147180559945309417;
  constexpr double kExpUnder = -700.0;

  switch (kind) {
  case BackwardKind::Tanh: {
    const double ax = std::fabs(x);
    // f64 sech^2 path collapses for extreme |x|; reference becomes signed zero.
    if (ax > 300.0)
      return std::copysign(0.0, g);
    const double e = std::exp(-2.0 * ax);
    const double d = 1.0 + e;
    return g * (4.0 * e) / (d * d);
  }
  case BackwardKind::Sigmoid: {
    const double ax = std::fabs(x);
    if (ax > 700.0)
      return std::copysign(0.0, g);
    const double e = std::exp(-ax);
    const double d = 1.0 + e;
    return g * e / (d * d);
  }
  case BackwardKind::Erf: {
    const double k = 2.0 / std::sqrt(3.14159265358979323846);
    int eg = 0;
    const double mg = std::frexp(g, &eg);
    const double t = -x * x + static_cast<double>(eg) * kLog2;
    if (t < kExpUnder)
      return std::copysign(0.0, g);
    return mg * k * std::exp(t);
  }
  case BackwardKind::Elu: {
    if (x > 0.0)
      return g;
    if (alpha == 0.0)
      return std::copysign(0.0, g * alpha);
    int eg = 0;
    const double mg = std::frexp(g, &eg);
    const double t = x + static_cast<double>(eg) * kLog2 + std::log(std::fabs(alpha));
    if (t < kExpUnder)
      return std::copysign(0.0, g * alpha);
    return std::copysign(mg * std::exp(t), g * alpha);
  }
  }
  return 0.0;
}

BF16 contract_reference(
    BackwardKind kind, const NumericalContract& contract, BF16 x, BF16 g, double alpha) {
  if (kind == BackwardKind::Elu && !std::isfinite(alpha))
    throw std::invalid_argument("ELU alpha must be finite");
  if (x.is_nan() || g.is_nan())
    return BF16::qnan();

  if (x.is_inf() || g.is_inf()) {
    if (kind == BackwardKind::Elu && x.is_inf() && !x.signbit())
      return g;
    if (g.is_inf() && x.is_finite()) {
      if (kind == BackwardKind::Elu) {
        if (x.to_f64() > 0.0)
          return g;
        if (alpha == 0.0)
          return BF16::qnan();
        return BF16::inf(g.signbit() != std::signbit(alpha));
      }
      return g;
    }
    // Avoid IEEE 0·∞ → NaN when the mathematical derivative vanishes at infinity.
    if (x.is_inf() && g.is_finite()) {
      BF16 out =
          BF16::zero(kind == BackwardKind::Elu && std::signbit(alpha) ? !g.signbit() : g.signbit());
      if (contract.flush_output_subnormals)
        out = flush_subnormals(out);
      return out;
    }
    return BF16::qnan();
  }

  const double prod = exact_product_f64(kind, x.to_f64(), g.to_f64(), alpha);
  BF16 out = round_to_bf16(prod, contract.output_round);
  if (contract.flush_output_subnormals)
    out = flush_subnormals(out);
  return out;
}

BF16 baseline_materialize_derivative(
    BackwardKind kind, const NumericalContract& contract, BF16 x, BF16 g, double alpha) {
  // Host model of unsafe intermediate FTZ of f'; not device measurement.
  if (x.is_nan() || g.is_nan())
    return BF16::qnan();
  if (x.is_inf() || g.is_inf())
    return contract_reference(kind, contract, x, g, alpha);
  const double d = exact_derivative_f64(kind, x.to_f64(), alpha);
  BF16 d_bf = round_to_bf16(d, contract.output_round);
  if (contract.baseline_flush_derivative_subnormals)
    d_bf = flush_subnormals(d_bf);
  const double prod = d_bf.to_f64() * g.to_f64();
  BF16 out = round_to_bf16(prod, contract.output_round);
  if (contract.flush_output_subnormals)
    out = flush_subnormals(out);
  return out;
}

MotivatingCaseReport analyze_tanh_motivating_case(double x, double g) {
  MotivatingCaseReport r;
  r.x = x;
  r.g = g;
  r.sech2 = exact_derivative_f64(BackwardKind::Tanh, x);
  r.product = exact_product_f64(BackwardKind::Tanh, x, g);
  r.product_bf16 = round_to_bf16(r.product);
  r.derivative_bf16 = round_to_bf16(r.sech2);
  r.derivative_bf16 = flush_subnormals(r.derivative_bf16);
  r.baseline_bf16 = round_to_bf16(r.derivative_bf16.to_f64() * g);
  r.min_normal = BF16::min_normal();
  r.derivative_underflows_normal = r.sech2 < r.min_normal.to_f64();
  r.product_is_normal = r.product_bf16.is_normal();
  r.baseline_is_false_zero =
      r.baseline_bf16.is_zero() && !r.product_bf16.is_zero() && r.product_bf16.is_finite();
  return r;
}

} // namespace bw_syn
