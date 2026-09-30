#include "bw_syn/functions/tanh_bw.hpp"

#include "bw_syn/detail/ir_build.hpp"
#include "bw_syn/detail/scale_eval.hpp"

namespace bw_syn {
namespace tanh_bw {

BF16 baseline_materialize(BF16 x, BF16 g, const NumericalContract& c) {
  return baseline_materialize_derivative(BackwardKind::Tanh, c, x, g);
}

BF16 scale_separated(BF16 x, BF16 g, const NumericalContract& c) {
  return detail::scale_separated_product(BackwardKind::Tanh, x, g, c);
}

static int sech2(Program& p, int x) {
  const int ax = p.add(Node{OpKind::Abs, {x}});
  const int two = p.add(Node{OpKind::ConstF64, {}, 2.0});
  const int e =
      p.add(Node{OpKind::Exp, {p.add(Node{OpKind::Neg, {p.add(Node{OpKind::Mul, {two, ax}})}})}});
  const int one = p.add(Node{OpKind::ConstF64, {}, 1.0});
  const int den = p.add(Node{OpKind::Add, {one, e}});
  const int four = p.add(Node{OpKind::ConstF64, {}, 4.0});
  return p.add(Node{OpKind::Div,
                    {p.add(Node{OpKind::Mul, {four, e}}), p.add(Node{OpKind::Mul, {den, den}})}});
}

Program ir_direct() {
  Program p;
  p.name = "tanh_bw_direct";
  const int x = p.add(Node{OpKind::InputX});
  const int g = p.add(Node{OpKind::InputG});
  p.result = detail::add_direct_product(p, g, sech2(p, x));
  return p;
}

Program ir_scale_separated() {
  Program p;
  p.name = "tanh_bw_scale_separated";
  const int x = p.add(Node{OpKind::InputX});
  const int g = p.add(Node{OpKind::InputG});
  p.result = detail::add_scale_product(p, g, sech2(p, x));
  return p;
}

} // namespace tanh_bw
} // namespace bw_syn
