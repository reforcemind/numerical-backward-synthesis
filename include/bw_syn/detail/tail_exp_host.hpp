#pragma once

#include "bw_syn/detail/tail_exp_product.hpp"

#include <bit>
#include <cmath>
#include <cstdint>

namespace bw_syn::detail {

// IEEE FP32 model, not a bit-exact model of Wormhole's partially fused SFPMAD.
struct HostTailOps {
  using Float = float;
  using Int = int;
  static Float abs(Float x) { return std::fabs(x); }
  static Float mul(Float a, Float b) { return a * b; }
  static Float mad(Float a, Float b, Float c) { return std::fma(a, b, c); }
  static Float round_bf16(Float x) {
    const auto bits = std::bit_cast<std::uint32_t>(x);
    return std::bit_cast<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
  }
  static Float round_int(Float z, Int& k) {
    const float shifted = z + 12582912.0f;
    k = static_cast<int>(std::bit_cast<std::uint32_t>(shifted)) - 0x4b400000;
    return shifted - 12582912.0f;
  }
  static Int exponent(Float x) {
    return static_cast<int>((std::bit_cast<std::uint32_t>(x) >> 23) & 255u);
  }
  static Float set_exponent(Float x, Int e) {
    return std::bit_cast<float>((std::bit_cast<std::uint32_t>(x) & 0x807fffffu) |
                                (static_cast<std::uint32_t>(e) << 23));
  }
  static Float reconstruct(Float x, Int e) {
    if (e == 0 && std::fabs(set_exponent(x, 127)) >= 1.9921875f)
      return std::bit_cast<float>((std::bit_cast<std::uint32_t>(x) & 0x80000000u) | 0x00800000u);
    if (e <= 0 || e >= 255)
      return 0.0f;
    return set_exponent(x, e);
  }
  static Float reconstruct_normal(Float x, Int e) {
    if (e <= 0)
      return std::bit_cast<float>((std::bit_cast<std::uint32_t>(x) & 0x80000000u) | 0x00800000u);
    return reconstruct(x, e);
  }
};

} // namespace bw_syn::detail
