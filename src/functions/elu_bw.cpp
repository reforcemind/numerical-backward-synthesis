#include "bw_syn/functions/elu_bw.hpp"

#include "bw_syn/detail/ir_build.hpp"
#include "bw_syn/detail/scale_eval.hpp"

namespace bw_syn {
namespace elu_bw {

BF16 baseline_materialize(BF16 x, BF16 g, const NumericalContract& c, double alpha) {
  return baseline_materialize_derivative(BackwardKind::Elu, c, x, g, alpha);
}

BF16 scale_separated(BF16 x, BF16 g, const NumericalContract& c, double alpha) {
  return detail::scale_separated_product(BackwardKind::Elu, x, g, c, alpha);
}

Program ir_scale_separated(double alpha) {
  Program p;
  p.name = "elu_bw_scale_separated";
  const int x = p.add(Node{OpKind::InputX});
  const int g = p.add(Node{OpKind::InputG});
  const int a = p.add(Node{OpKind::ConstF64, {}, alpha});
  const int zero = p.add(Node{OpKind::ConstF64, {}, 0.0});
  const int one = p.add(Node{OpKind::ConstF64, {}, 1.0});
  const int aex = p.add(Node{OpKind::Mul, {a, p.add(Node{OpKind::Exp, {x}})}});
  const int xpos = p.add(Node{OpKind::Max, {x, zero}});
  p.result = detail::add_scale_product(p, g, p.add(Node{OpKind::Select, {xpos, one, aex}}));
  return p;
}

} // namespace elu_bw
} // namespace bw_syn
