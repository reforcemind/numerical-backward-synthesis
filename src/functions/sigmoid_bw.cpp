#include "bw_syn/functions/sigmoid_bw.hpp"

#include "bw_syn/detail/ir_build.hpp"
#include "bw_syn/detail/scale_eval.hpp"

namespace bw_syn {
namespace sigmoid_bw {

BF16 baseline_materialize(BF16 x, BF16 g, const NumericalContract& c) {
  return baseline_materialize_derivative(BackwardKind::Sigmoid, c, x, g);
}

BF16 scale_separated(BF16 x, BF16 g, const NumericalContract& c) {
  return detail::scale_separated_product(BackwardKind::Sigmoid, x, g, c);
}

static int sigmoid_deriv(Program& p, int x) {
  const int ax = p.add(Node{OpKind::Abs, {x}});
  const int e = p.add(Node{OpKind::Exp, {p.add(Node{OpKind::Neg, {ax}})}});
  const int one = p.add(Node{OpKind::ConstF64, {}, 1.0});
  const int den = p.add(Node{OpKind::Add, {one, e}});
  return p.add(Node{OpKind::Div, {e, p.add(Node{OpKind::Mul, {den, den}})}});
}

Program ir_direct() {
  Program p;
  p.name = "sigmoid_bw_direct";
  const int x = p.add(Node{OpKind::InputX});
  const int g = p.add(Node{OpKind::InputG});
  p.result = detail::add_direct_product(p, g, sigmoid_deriv(p, x));
  return p;
}

Program ir_scale_separated() {
  Program p;
  p.name = "sigmoid_bw_scale_separated";
  const int x = p.add(Node{OpKind::InputX});
  const int g = p.add(Node{OpKind::InputG});
  p.result = detail::add_scale_product(p, g, sigmoid_deriv(p, x));
  return p;
}

} // namespace sigmoid_bw
} // namespace bw_syn
