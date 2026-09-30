#include "bw_syn/bf16.hpp"

#include <cstdio>
#include <cstring>

namespace bw_syn {
namespace {

float bits_f32(std::uint32_t u) {
  float x;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

} // namespace

BF16 BF16::from_f32(float x) {
  return round_to_bf16(static_cast<double>(x));
}

BF16 BF16::from_f64(double x) {
  return round_to_bf16(x);
}

float BF16::to_f32() const {

  return bits_f32(static_cast<std::uint32_t>(bits) << 16);
}

double BF16::to_f64() const {
  return static_cast<double>(to_f32());
}

bool BF16::is_nan() const {
  return ((bits & kExpMask) == kExpMask) && ((bits & kFracMask) != 0);
}

bool BF16::is_inf() const {
  return ((bits & kExpMask) == kExpMask) && ((bits & kFracMask) == 0);
}

bool BF16::is_finite() const {
  return (bits & kExpMask) != kExpMask;
}

bool BF16::is_zero() const {
  return (bits & (kExpMask | kFracMask)) == 0;
}

bool BF16::is_subnormal() const {
  return ((bits & kExpMask) == 0) && ((bits & kFracMask) != 0);
}

bool BF16::is_normal() const {
  const auto e = bits & kExpMask;
  return e != 0 && e != kExpMask;
}

bool BF16::signbit() const {
  return (bits & kSignMask) != 0;
}

int BF16::unbiased_exponent() const {
  const int e = (bits & kExpMask) >> kFracBits;
  if (e == 0) {

    return 1 - kBias;
  }
  return e - kBias;
}

double BF16::significand() const {
  if (is_zero() || !is_finite())
    return 0.0;
  const int frac = bits & kFracMask;
  if (is_subnormal()) {
    return static_cast<double>(frac) / static_cast<double>(1 << kFracBits);
  }
  return 1.0 + static_cast<double>(frac) / static_cast<double>(1 << kFracBits);
}

BF16 BF16::zero(bool neg) {
  return BF16{static_cast<std::uint16_t>(neg ? kSignMask : 0)};
}
BF16 BF16::inf(bool neg) {
  return BF16{static_cast<std::uint16_t>((neg ? kSignMask : 0) | kExpMask)};
}
BF16 BF16::qnan() {
  return BF16{static_cast<std::uint16_t>(kExpMask | 0x40)};
}
BF16 BF16::min_normal() {
  return BF16{0x0080};
}
BF16 BF16::min_subnormal() {
  return BF16{0x0001};
}
BF16 BF16::max_finite() {
  return BF16{0x7F7F};
}

std::string BF16::to_string() const {
  char buf[64];
  if (is_nan()) {
    std::snprintf(buf, sizeof(buf), "nan(0x%04x)", bits);
  } else if (is_inf()) {
    std::snprintf(buf, sizeof(buf), "%sinf", signbit() ? "-" : "+");
  } else {
    std::snprintf(buf, sizeof(buf), "%.8g(0x%04x)", to_f64(), bits);
  }
  return buf;
}

BF16 flush_subnormals(BF16 x) {
  if (x.is_subnormal())
    return BF16::zero(x.signbit());
  return x;
}

std::string round_mode_name(RoundMode m) {
  switch (m) {
  case RoundMode::ToNearestEven:
    return "RNE";
  case RoundMode::TowardZero:
    return "RTZ";
  case RoundMode::TowardPosInf:
    return "RU";
  case RoundMode::TowardNegInf:
    return "RD";
  }
  return "?";
}

BF16 round_to_bf16(double x, RoundMode mode) {
  if (std::isnan(x))
    return BF16::qnan();
  if (std::isinf(x))
    return BF16::inf(std::signbit(x));

  const bool neg = std::signbit(x);
  double ax = std::fabs(x);
  if (ax == 0.0)
    return BF16::zero(neg);

  const double max_fin = std::ldexp(2.0 - std::ldexp(1.0, -7), 127);
  const double min_sub = std::ldexp(1.0, -133);

  auto pack = [&](int exp_unbiased, int frac_bits, bool n) -> BF16 {
    if (exp_unbiased > 127)
      return BF16::inf(n);
    if (exp_unbiased < -126) {

      return BF16::zero(n);
    }
    const int e = exp_unbiased + BF16::kBias;
    std::uint16_t bits =
        static_cast<std::uint16_t>((e << BF16::kFracBits) | (frac_bits & BF16::kFracMask));
    if (n)
      bits = static_cast<std::uint16_t>(bits | BF16::kSignMask);
    return BF16{bits};
  };

  if (ax < min_sub / 2.0) {

    if (mode == RoundMode::TowardPosInf && !neg)
      return BF16::min_subnormal();
    if (mode == RoundMode::TowardNegInf && neg) {
      auto z = BF16::min_subnormal();
      z.bits = static_cast<std::uint16_t>(z.bits | BF16::kSignMask);
      return z;
    }
    return BF16::zero(neg);
  }

  if (ax > max_fin) {
    const double mid = max_fin + std::ldexp(1.0, 119);

    auto signed_max = [&]() {
      auto m = BF16::max_finite();
      if (neg)
        m.bits = static_cast<std::uint16_t>(m.bits | BF16::kSignMask);
      return m;
    };

    switch (mode) {
    case RoundMode::TowardZero:
      return signed_max();
    case RoundMode::TowardNegInf:
      return neg ? BF16::inf(true) : BF16::max_finite();
    case RoundMode::TowardPosInf:
      return neg ? signed_max() : BF16::inf(false);
    case RoundMode::ToNearestEven:

      if (ax < mid)
        return signed_max();
      return BF16::inf(neg);
    }
    return BF16::inf(neg);
  }

  int exp = 0;
  double m = std::frexp(ax, &exp);

  double m2 = m * 2.0;
  int e2 = exp - 1;

  if (e2 < -126) {
    double scaled = std::ldexp(ax, 126 + BF16::kFracBits);
    double fl = std::floor(scaled);
    double frac_part = scaled - fl;
    int frac = static_cast<int>(fl);
    auto round_up = [&]() {
      ++frac;
      if (frac >= (1 << BF16::kFracBits)) {

        return pack(-126, 0, neg);
      }
      std::uint16_t bits = static_cast<std::uint16_t>(frac & BF16::kFracMask);
      if (neg)
        bits = static_cast<std::uint16_t>(bits | BF16::kSignMask);
      return BF16{bits};
    };

    bool up = false;
    switch (mode) {
    case RoundMode::ToNearestEven:
      if (frac_part > 0.5)
        up = true;
      else if (frac_part == 0.5 && (frac & 1))
        up = true;
      break;
    case RoundMode::TowardZero:
      break;
    case RoundMode::TowardPosInf:
      up = !neg && frac_part > 0.0;
      break;
    case RoundMode::TowardNegInf:
      up = neg && frac_part > 0.0;
      break;
    }
    if (up)
      return round_up();
    if (frac == 0)
      return BF16::zero(neg);
    std::uint16_t bits = static_cast<std::uint16_t>(frac & BF16::kFracMask);
    if (neg)
      bits = static_cast<std::uint16_t>(bits | BF16::kSignMask);
    return BF16{bits};
  }

  double frac_f = (m2 - 1.0) * static_cast<double>(1 << BF16::kFracBits);
  double fl = std::floor(frac_f);
  double rem = frac_f - fl;
  int frac = static_cast<int>(fl);

  bool up = false;
  switch (mode) {
  case RoundMode::ToNearestEven:
    if (rem > 0.5)
      up = true;
    else if (rem == 0.5 && (frac & 1))
      up = true;
    break;
  case RoundMode::TowardZero:
    break;
  case RoundMode::TowardPosInf:
    up = !neg && rem > 0.0;
    break;
  case RoundMode::TowardNegInf:
    up = neg && rem > 0.0;
    break;
  }

  if (up) {
    ++frac;
    if (frac >= (1 << BF16::kFracBits)) {
      frac = 0;
      ++e2;
    }
  }
  return pack(e2, frac, neg);
}

std::uint32_t ulp_distance(BF16 a, BF16 b) {
  if (a.is_nan() || b.is_nan())
    return 0xFFFFFFFFu;

  auto to_ord = [](BF16 v) -> std::int32_t {
    std::uint16_t u = v.bits;
    if (u & BF16::kSignMask)
      return -static_cast<std::int32_t>(u & 0x7FFF);
    return static_cast<std::int32_t>(u);
  };
  const std::int32_t ia = to_ord(a);
  const std::int32_t ib = to_ord(b);
  return static_cast<std::uint32_t>(ia >= ib ? ia - ib : ib - ia);
}

} // namespace bw_syn
