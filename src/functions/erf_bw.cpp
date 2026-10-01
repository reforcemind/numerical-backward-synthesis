#include "bw_syn/functions/erf_bw.hpp"

#include "bw_syn/detail/ir_build.hpp"
#include "bw_syn/detail/scale_eval.hpp"

namespace bw_syn {
namespace erf_bw {

BF16 baseline_materialize(BF16 x, BF16 g, const NumericalContract& c) {
  return baseline_materialize_derivative(BackwardKind::Erf, c, x, g);
}

BF16 scale_separated(BF16 x, BF16 g, const NumericalContract& c) {
  return detail::scale_separated_product(BackwardKind::Erf, x, g, c);
}

static int erf_deriv(Program& p, int x) {
  const int e =
      p.add(Node{OpKind::Exp, {p.add(Node{OpKind::Neg, {p.add(Node{OpKind::Mul, {x, x}})}})}});
  const int k = p.add(Node{OpKind::ConstF64, {}, 2.0 / 1.772453850905516});
  return p.add(Node{OpKind::Mul, {k, e}});
}

Program ir_direct() {
  Program p;
  p.name = "erf_bw_direct";
  const int x = p.add(Node{OpKind::InputX});
  const int g = p.add(Node{OpKind::InputG});
  p.result = detail::add_direct_product(p, g, erf_deriv(p, x));
  return p;
}

Program ir_scale_separated() {
  Program p;
  p.name = "erf_bw_scale_separated";
  const int x = p.add(Node{OpKind::InputX});
  const int g = p.add(Node{OpKind::InputG});
  p.result = detail::add_scale_product(p, g, erf_deriv(p, x));
  return p;
}

} // namespace erf_bw
} // namespace bw_syn
