#pragma once

#include "bw_syn/bf16.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace bw_syn {

enum class OpKind {
  InputX,
  InputG,
  ConstF64,
  ConstI32,
  Abs,
  Neg,
  Add,
  Sub,
  Mul,
  Div,
  Max,
  Min,
  Exp,
  Exp2,
  Log,
  Log2,
  Sqrt,
  Tanh,
  Select,
  Frexp,
  FrexpMant,
  FrexpExp,
  Ldexp,
  ScaleMul,
  AddExp,
  Normalize,
  Reconstruct,
  RoundBF16,
  FlushBF16,
  // Valid only when the analytic derivative is strictly positive at finite x.
  PositiveDerivativeInf,
  Clamp,
  Copysign,
};

struct Node {
  OpKind op{OpKind::ConstF64};
  std::vector<int> args;
  double c_f64{0.0};
  std::int32_t c_i32{0};
  std::string name;

  Node() = default;
  Node(OpKind o, std::vector<int> a = {}, double cf = 0.0, std::int32_t ci = 0, std::string n = {})
      : op(o), args(std::move(a)), c_f64(cf), c_i32(ci), name(std::move(n)) {}
};

struct Program {
  std::string name;
  std::vector<Node> nodes;
  int result{-1};

  int add(Node n);
  std::string dump() const;
};

std::string op_kind_name(OpKind k);
std::vector<int> reachable_nodes(const Program& p);

struct EvalResult {
  double value_f64{0.0};
  BF16 value_bf16{};
};

EvalResult
eval_program(const Program& p, double x, double g, RoundMode round = RoundMode::ToNearestEven);

} // namespace bw_syn
