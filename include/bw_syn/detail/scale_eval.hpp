#pragma once

#include "bw_syn/bf16.hpp"
#include "bw_syn/contract.hpp"
#include "bw_syn/ir.hpp"
#include "bw_syn/oracle.hpp"

#include <cmath>

namespace bw_syn {
namespace detail {

inline BF16
eval_ir_under_contract(const Program& prog, BF16 x, BF16 g, const NumericalContract& c) {
  if (x.is_nan() || g.is_nan())
    return BF16::qnan();
  const EvalResult r = eval_program(prog, x.to_f64(), g.to_f64(), c.output_round);
  BF16 out = r.value_bf16;
  if (c.flush_output_subnormals)
    out = flush_subnormals(out);
  return out;
}

inline BF16 scale_separated_product(
    BackwardKind kind, BF16 x, BF16 g, const NumericalContract& c, double alpha = 1.0) {
  if (x.is_nan() || g.is_nan())
    return BF16::qnan();
  if (x.is_inf() || g.is_inf())
    return contract_reference(kind, c, x, g, alpha);

  const double d = exact_derivative_f64(kind, x.to_f64(), alpha);
  const double gd = g.to_f64();
  int eg = 0;
  int ed = 0;
  const double mg = std::frexp(gd, &eg);
  const double md = std::frexp(d, &ed);
  double mm = mg * md;
  int ee = eg + ed;
  int e2 = 0;
  mm = std::frexp(mm, &e2);
  ee += e2;
  const double prod = std::ldexp(mm, ee);
  BF16 out = round_to_bf16(prod, c.output_round);
  if (c.flush_output_subnormals)
    out = flush_subnormals(out);
  return out;
}

} // namespace detail
} // namespace bw_syn
