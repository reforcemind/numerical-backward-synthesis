#pragma once

#include <cmath>
#include <cstdint>
#include <string>

namespace bw_syn {

struct BF16 {
  std::uint16_t bits{0};

  static constexpr int kExpBits = 8;
  static constexpr int kFracBits = 7;
  static constexpr int kBias = 127;
  static constexpr int kExpMax = 255;
  static constexpr std::uint16_t kSignMask = 0x8000;
  static constexpr std::uint16_t kExpMask = 0x7F80;
  static constexpr std::uint16_t kFracMask = 0x007F;

  constexpr BF16() = default;
  explicit constexpr BF16(std::uint16_t b) : bits(b) {}

  static BF16 from_bits(std::uint16_t b) { return BF16{b}; }
  static BF16 from_f32(float x);
  static BF16 from_f64(double x);

  float to_f32() const;
  double to_f64() const;

  bool is_nan() const;
  bool is_inf() const;
  bool is_finite() const;
  bool is_zero() const;
  bool is_subnormal() const;
  bool is_normal() const;
  bool signbit() const;

  int unbiased_exponent() const;

  double significand() const;

  static BF16 zero(bool neg = false);
  static BF16 inf(bool neg = false);
  static BF16 qnan();
  static BF16 min_normal();
  static BF16 min_subnormal();
  static BF16 max_finite();

  std::string to_string() const;
};

enum class RoundMode {
  ToNearestEven,
  TowardZero,
  TowardPosInf,
  TowardNegInf,
};

BF16 round_to_bf16(double x, RoundMode mode = RoundMode::ToNearestEven);

BF16 flush_subnormals(BF16 x);

std::uint32_t ulp_distance(BF16 a, BF16 b);

std::string round_mode_name(RoundMode m);

} // namespace bw_syn
