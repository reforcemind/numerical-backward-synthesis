#include "bw_syn/ir.hpp"

#include <cmath>
#include <functional>
#include <sstream>
#include <stdexcept>

namespace bw_syn {

std::string op_kind_name(OpKind k) {
  switch (k) {
  case OpKind::InputX:
    return "InputX";
  case OpKind::InputG:
    return "InputG";
  case OpKind::ConstF64:
    return "ConstF64";
  case OpKind::ConstI32:
    return "ConstI32";
  case OpKind::Abs:
    return "Abs";
  case OpKind::Neg:
    return "Neg";
  case OpKind::Add:
    return "Add";
  case OpKind::Sub:
    return "Sub";
  case OpKind::Mul:
    return "Mul";
  case OpKind::Div:
    return "Div";
  case OpKind::Max:
    return "Max";
  case OpKind::Min:
    return "Min";
  case OpKind::Exp:
    return "Exp";
  case OpKind::Exp2:
    return "Exp2";
  case OpKind::Log:
    return "Log";
  case OpKind::Log2:
    return "Log2";
  case OpKind::Sqrt:
    return "Sqrt";
  case OpKind::Tanh:
    return "Tanh";
  case OpKind::Select:
    return "Select";
  case OpKind::Frexp:
    return "Frexp";
  case OpKind::FrexpMant:
    return "FrexpMant";
  case OpKind::FrexpExp:
    return "FrexpExp";
  case OpKind::Ldexp:
    return "Ldexp";
  case OpKind::ScaleMul:
    return "ScaleMul";
  case OpKind::AddExp:
    return "AddExp";
  case OpKind::Normalize:
    return "Normalize";
  case OpKind::Reconstruct:
    return "Reconstruct";
  case OpKind::RoundBF16:
    return "RoundBF16";
  case OpKind::FlushBF16:
    return "FlushBF16";
  case OpKind::PositiveDerivativeInf:
    return "PositiveDerivativeInf";
  case OpKind::Clamp:
    return "Clamp";
  case OpKind::Copysign:
    return "Copysign";
  }
  return "?";
}

int Program::add(Node n) {
  nodes.push_back(std::move(n));
  return static_cast<int>(nodes.size() - 1);
}

std::string Program::dump() const {
  std::ostringstream os;
  os << "Program " << name << " result=v" << result << "\n";
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& n = nodes[i];
    os << "  v" << i << " = " << op_kind_name(n.op);
    if (n.op == OpKind::ConstF64)
      os << " " << n.c_f64;
    if (n.op == OpKind::ConstI32)
      os << " " << n.c_i32;
    for (int a : n.args)
      os << " v" << a;
    if (!n.name.empty())
      os << "  ; " << n.name;
    os << "\n";
  }
  return os.str();
}

std::vector<int> reachable_nodes(const Program& p) {
  if (p.result < 0 || p.result >= static_cast<int>(p.nodes.size()))
    throw std::invalid_argument("Program has invalid result index");
  std::vector<unsigned char> state(p.nodes.size(), 0);
  std::vector<int> order;
  std::function<void(int)> visit = [&](int index) {
    if (index < 0 || index >= static_cast<int>(p.nodes.size()))
      throw std::invalid_argument("Program has invalid argument index");
    auto& s = state[static_cast<size_t>(index)];
    if (s == 1)
      throw std::invalid_argument("Program contains a cycle");
    if (s == 2)
      return;
    s = 1;
    for (int arg : p.nodes[static_cast<size_t>(index)].args)
      visit(arg);
    s = 2;
    order.push_back(index);
  };
  visit(p.result);
  return order;
}

namespace {

struct Slot {
  double f{0.0};
  int e{0};
  bool is_exp{false};
};

Slot eval_node(const Program& p,
               int idx,
               double x,
               double g,
               RoundMode round,
               std::vector<Slot>& memo,
               std::vector<char>& seen) {
  if (seen[static_cast<size_t>(idx)])
    return memo[static_cast<size_t>(idx)];
  const Node& n = p.nodes[static_cast<size_t>(idx)];
  Slot out;

  auto arg = [&](int k) {
    return eval_node(p, n.args[static_cast<size_t>(k)], x, g, round, memo, seen);
  };

  switch (n.op) {
  case OpKind::InputX:
    out.f = x;
    break;
  case OpKind::InputG:
    out.f = g;
    break;
  case OpKind::ConstF64:
    out.f = n.c_f64;
    break;
  case OpKind::ConstI32:
    out.f = static_cast<double>(n.c_i32);
    out.e = n.c_i32;
    out.is_exp = true;
    break;
  case OpKind::Abs:
    out.f = std::fabs(arg(0).f);
    break;
  case OpKind::Neg:
    out.f = -arg(0).f;
    break;
  case OpKind::Add:
    out.f = arg(0).f + arg(1).f;
    break;
  case OpKind::Sub:
    out.f = arg(0).f - arg(1).f;
    break;
  case OpKind::Mul:
    out.f = arg(0).f * arg(1).f;
    break;
  case OpKind::Div:
    out.f = arg(0).f / arg(1).f;
    break;
  case OpKind::Max:
    out.f = std::fmax(arg(0).f, arg(1).f);
    break;
  case OpKind::Min:
    out.f = std::fmin(arg(0).f, arg(1).f);
    break;
  case OpKind::Exp:
    out.f = std::exp(arg(0).f);
    break;
  case OpKind::Exp2:
    out.f = std::exp2(arg(0).f);
    break;
  case OpKind::Log:
    out.f = std::log(arg(0).f);
    break;
  case OpKind::Log2:
    out.f = std::log2(arg(0).f);
    break;
  case OpKind::Sqrt:
    out.f = std::sqrt(arg(0).f);
    break;
  case OpKind::Tanh:
    out.f = std::tanh(arg(0).f);
    break;
  case OpKind::Select:
    out.f = (arg(0).f != 0.0) ? arg(1).f : arg(2).f;
    break;
  case OpKind::Frexp: {
    int e = 0;
    double m = std::frexp(arg(0).f, &e);

    out.f = m;
    out.e = e;
    break;
  }
  case OpKind::FrexpMant: {
    Slot s = arg(0);
    out.f = s.f;
    break;
  }
  case OpKind::FrexpExp: {
    Slot s = arg(0);
    out.f = static_cast<double>(s.e);
    out.e = s.e;
    out.is_exp = true;
    break;
  }
  case OpKind::Ldexp: {
    Slot m = arg(0);
    Slot e = arg(1);
    out.f = std::ldexp(m.f, static_cast<int>(e.f));
    break;
  }
  case OpKind::ScaleMul: {

    out.f = arg(0).f * arg(1).f;
    break;
  }
  case OpKind::AddExp: {
    Slot a = arg(0);
    Slot b = arg(1);
    out.e = static_cast<int>(a.f) + static_cast<int>(b.f);
    out.f = static_cast<double>(out.e);
    out.is_exp = true;
    break;
  }
  case OpKind::Normalize: {

    Slot m = arg(0);
    Slot e = arg(1);
    int ee = 0;
    double mm = std::frexp(m.f, &ee);
    out.f = mm;
    out.e = static_cast<int>(e.f) + ee;
    break;
  }
  case OpKind::Reconstruct: {

    Slot m = arg(0);
    Slot e = arg(1);
    int ee = e.is_exp ? e.e : static_cast<int>(e.f);

    out.f = std::ldexp(m.f, ee);
    break;
  }
  case OpKind::RoundBF16: {
    out.f = round_to_bf16(arg(0).f, round).to_f64();
    break;
  }
  case OpKind::FlushBF16: {
    out.f = flush_subnormals(BF16::from_f64(arg(0).f)).to_f64();
    break;
  }
  case OpKind::PositiveDerivativeInf: {
    const Slot sx = arg(0);
    const Slot sg = arg(1);
    out.f = std::isfinite(sx.f) && std::isinf(sg.f) ? sg.f : arg(2).f;
    break;
  }
  case OpKind::Clamp: {
    out.f = std::fmin(arg(2).f, std::fmax(arg(0).f, arg(1).f));
    break;
  }
  case OpKind::Copysign: {
    out.f = std::copysign(arg(0).f, arg(1).f);
    break;
  }
  }

  memo[static_cast<size_t>(idx)] = out;
  seen[static_cast<size_t>(idx)] = 1;
  return out;
}

} // namespace

EvalResult eval_program(const Program& p, double x, double g, RoundMode round) {
  if (p.result < 0 || p.result >= static_cast<int>(p.nodes.size())) {
    throw std::runtime_error("Program has invalid result index");
  }
  std::vector<Slot> memo(p.nodes.size());
  std::vector<char> seen(p.nodes.size(), 0);
  Slot s = eval_node(p, p.result, x, g, round, memo, seen);
  EvalResult r;
  r.value_f64 = s.f;
  r.value_bf16 = round_to_bf16(s.f, round);
  return r;
}

} // namespace bw_syn
