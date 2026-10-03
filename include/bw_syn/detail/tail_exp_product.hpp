#pragma once

namespace bw_syn::detail {

#if defined(TRISC_MATH)
#define BW_SYN_TAIL_INLINE inline __attribute__((always_inline))
#else
#define BW_SYN_TAIL_INLINE inline
#endif

struct TailQuadratic {
  float c0;
  float c1;
  float c2;
};

// Coefficients are fitted to final-product intervals, then independently checked.
template <class Ops, bool Saturate = true, int Rate = 2, int Scale = 2>
BW_SYN_TAIL_INLINE typename Ops::Float
tail_exp2_product(typename Ops::Float x, typename Ops::Float g, const TailQuadratic& coefficients) {
  using F = typename Ops::Float;
  using I = typename Ops::Int;
  const F z = Ops::mul(Ops::mul(Ops::abs(x), F(-Rate)), F(1.4426950408889634f));
  I k_int;
  const F k = Ops::round_int(z, k_int);
  const F r = z - k;
  const F p = Ops::mad(Ops::mad(F(coefficients.c2), r, F(coefficients.c1)), r, F(coefficients.c0));
  const F product = Ops::round_bf16(Ops::mul(Ops::set_exponent(g, 127), p));
  const I exponent = Ops::exponent(product) + Ops::exponent(g) + k_int + (Scale - 127);
  if constexpr (Saturate)
    return Ops::reconstruct_normal(product, exponent);
  else
    return Ops::reconstruct(product, exponent);
}

// Approximate g * 2^Scale * exp(-Rate * abs(x)). Callers supply a bounded tail
// domain and normal finite g. Ops fixes arithmetic and reconstruction semantics.
template <class Ops, int Degree, int Rate = 2, int Scale = 2>
BW_SYN_TAIL_INLINE typename Ops::Float tail_exp_product(typename Ops::Float x,
                                                        typename Ops::Float g) {
  static_assert(Degree == 3 || Degree == 4);
  using F = typename Ops::Float;
  using I = typename Ops::Int;
  const F t = Ops::mul(Ops::abs(x), F(-Rate));
  I k_int;
  const F k = Ops::round_int(Ops::mul(t, F(1.4426950408889634f)), k_int);
  F r = Ops::mad(k, F(-0.693115234375f), t);
  r = Ops::mad(k, F(-3.19461832987e-05f), r);
  F p = F(0.1666666666666667f);
  if constexpr (Degree == 4)
    p = Ops::mad(r, F(0.0416666666666667f), p);
  p = Ops::mad(p, r, F(0.5f));
  p = Ops::mad(p, r, F(1.0f));
  // On |r| <= .347 these offsets exceed the Taylor remainder bounds.
  // A one-sided approximation protects the normal-output boundary.
  p = Ops::mad(p, r, F(Degree == 3 ? 1.0009f : 1.00006f));
  // Rounding after reconstruction would lose values that round up to min-normal.
  const F product = Ops::round_bf16(Ops::mul(Ops::set_exponent(g, 127), p));
  const I exponent = Ops::exponent(product) + Ops::exponent(g) + k_int + (Scale - 127);
  return Ops::reconstruct(product, exponent);
}

#undef BW_SYN_TAIL_INLINE

} // namespace bw_syn::detail
