#pragma once

#include "bw_syn/ir.hpp"

namespace bw_syn {
namespace detail {

inline int add_scale_product(Program& p, int g, int d) {
  const int fg = p.add(Node{OpKind::Frexp, {g}});
  const int mg = p.add(Node{OpKind::FrexpMant, {fg}});
  const int eg = p.add(Node{OpKind::FrexpExp, {fg}});
  const int fd = p.add(Node{OpKind::Frexp, {d}});
  const int md = p.add(Node{OpKind::FrexpMant, {fd}});
  const int ed = p.add(Node{OpKind::FrexpExp, {fd}});
  const int mm = p.add(Node{OpKind::ScaleMul, {mg, md}});
  const int ee = p.add(Node{OpKind::AddExp, {eg, ed}});
  const int nm = p.add(Node{OpKind::Normalize, {mm, ee}});
  const int m2 = p.add(Node{OpKind::FrexpMant, {nm}});
  const int e2 = p.add(Node{OpKind::FrexpExp, {nm}});
  const int rec = p.add(Node{OpKind::Reconstruct, {m2, e2}});
  return p.add(Node{OpKind::RoundBF16, {rec}});
}

inline int add_direct_product(Program& p, int g, int d) {
  const int dr = p.add(Node{OpKind::RoundBF16, {d}});
  const int prod = p.add(Node{OpKind::Mul, {g, dr}});
  return p.add(Node{OpKind::RoundBF16, {prod}});
}

} // namespace detail
} // namespace bw_syn
