#include "bw_syn/functions/tanh_bw.hpp"

#include "bw_syn/detail/ir_build.hpp"
#include "bw_syn/detail/scale_eval.hpp"
#include "bw_syn/detail/tail_exp2_coefficients.hpp"
#include "bw_syn/detail/tail_exp_host.hpp"

#include <cmath>
#include <stdexcept>

namespace bw_syn {
namespace tanh_bw {

BF16 baseline_materialize(BF16 x, BF16 g, const NumericalContract& c) {
  return baseline_materialize_derivative(BackwardKind::Tanh, c, x, g);
}

BF16 scale_separated(BF16 x, BF16 g, const NumericalContract& c) {
  return detail::scale_separated_product(BackwardKind::Tanh, x, g, c);
}

BF16 tail_split4_host(BF16 x, BF16 g, const NumericalContract& c) {
  const float q = std::exp(-0.5f * std::fabs(x.to_f32()));
  float value = g.to_f32() * q;
  value *= 4.0f;
  value *= q;
  value *= q;
  value *= q;
  BF16 out = round_to_bf16(value, c.output_round);
  if (c.flush_output_subnormals)
    out = flush_subnormals(out);
  return out;
}

BF16 tail_fused_host(BF16 x, BF16 g, const NumericalContract& c, int degree) {
  const float ax = std::fabs(x.to_f32());
  if (!x.is_finite() || ax < 4.0f || ax > 88.5f || !g.is_normal())
    throw std::invalid_argument("fused tail requires 4 <= |x| <= 88.5 and normal finite g");
  if (c.output_round != RoundMode::ToNearestEven)
    throw std::invalid_argument("fused tail requires BF16 round-to-nearest-even");
  float value;
  if (degree == 2)
    value = detail::tail_exp2_product<detail::HostTailOps>(
        x.to_f32(), g.to_f32(), detail::kTailExp2Quadratic);
  else if (degree == 3)
    value = detail::tail_exp_product<detail::HostTailOps, 3>(x.to_f32(), g.to_f32());
  else if (degree == 4)
    value = detail::tail_exp_product<detail::HostTailOps, 4>(x.to_f32(), g.to_f32());
  else
    throw std::invalid_argument("tail polynomial degree must be 2, 3 or 4");
  BF16 out = round_to_bf16(value, c.output_round);
  return c.flush_output_subnormals ? flush_subnormals(out) : out;
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

Program ir_factored() {
  Program p;
  p.name = "tanh_bw_factored_g_h_h";
  const int x = p.add(Node{OpKind::InputX});
  const int g = p.add(Node{OpKind::InputG});
  const int ax = p.add(Node{OpKind::Abs, {x}});
  const int e = p.add(Node{OpKind::Exp, {p.add(Node{OpKind::Neg, {ax}})}});
  const int one = p.add(Node{OpKind::ConstF64, {}, 1.0});
  const int two = p.add(Node{OpKind::ConstF64, {}, 2.0});
  const int den = p.add(Node{OpKind::Add, {one, p.add(Node{OpKind::Mul, {e, e}})}});
  const int h = p.add(Node{OpKind::Div, {p.add(Node{OpKind::Mul, {two, e}}), den}});
  const int gh = p.add(Node{OpKind::Mul, {g, h}});
  const int product = p.add(Node{OpKind::Mul, {gh, h}});
  p.result = p.add(Node{OpKind::RoundBF16, {product}});
  return p;
}

Program ir_tail_split4() {
  Program p;
  p.name = "tanh_bw_tail_split4";
  const int x = p.add(Node{OpKind::InputX});
  const int g = p.add(Node{OpKind::InputG});
  const int half = p.add(Node{OpKind::ConstF64, {}, 0.5});
  const int four = p.add(Node{OpKind::ConstF64, {}, 4.0});
  const int ax = p.add(Node{OpKind::Abs, {x}});
  const int q =
      p.add(Node{OpKind::Exp, {p.add(Node{OpKind::Neg, {p.add(Node{OpKind::Mul, {half, ax}})}})}});
  int value = p.add(Node{OpKind::Mul, {g, q}});
  value = p.add(Node{OpKind::Mul, {value, four}});
  for (int i = 0; i < 3; ++i)
    value = p.add(Node{OpKind::Mul, {value, q}});
  p.result = p.add(Node{OpKind::RoundBF16, {value}});
  return p;
}

} // namespace tanh_bw
} // namespace bw_syn
