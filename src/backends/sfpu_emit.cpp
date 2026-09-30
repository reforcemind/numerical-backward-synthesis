#include "bw_syn/backends/sfpu_emit.hpp"

#include <sstream>
#include <stdexcept>
#include <vector>

namespace bw_syn {
namespace sfpu {
namespace {

std::string v(int i) {
  return "v" + std::to_string(i);
}

int frexp_src(const Program& p, int i) {
  if (i < 0 || i >= static_cast<int>(p.nodes.size()))
    return -1;
  const Node& n = p.nodes[static_cast<size_t>(i)];
  if ((n.op == OpKind::FrexpMant || n.op == OpKind::FrexpExp) && !n.args.empty()) {
    const int f = n.args[0];
    if (f >= 0 && p.nodes[static_cast<size_t>(f)].op == OpKind::Frexp &&
        !p.nodes[static_cast<size_t>(f)].args.empty())
      return p.nodes[static_cast<size_t>(f)].args[0];
  }
  return -1;
}

int peel(const Program& p, int i) {
  while (i >= 0 && i < static_cast<int>(p.nodes.size())) {
    const Node& n = p.nodes[static_cast<size_t>(i)];
    if ((n.op == OpKind::RoundBF16 || n.op == OpKind::Normalize || n.op == OpKind::FrexpMant ||
         n.op == OpKind::Frexp || n.op == OpKind::Reconstruct) &&
        !n.args.empty()) {
      i = n.args[0];
      continue;
    }
    return i;
  }
  return i;
}

Program lower(Program p) {
  for (auto& n : p.nodes) {
    if (n.op == OpKind::ScaleMul && n.args.size() == 2) {
      const int a = frexp_src(p, n.args[0]);
      const int b = frexp_src(p, n.args[1]);
      if (a >= 0 && b >= 0) {
        n.op = OpKind::Mul;
        n.args = {a, b};
      }
    }
  }
  for (auto& n : p.nodes) {
    if (n.op == OpKind::Reconstruct && n.args.size() == 2) {
      const int m = peel(p, n.args[0]);
      if (m >= 0 && (p.nodes[static_cast<size_t>(m)].op == OpKind::Mul ||
                     p.nodes[static_cast<size_t>(m)].op == OpKind::ScaleMul))
        n = p.nodes[static_cast<size_t>(m)];
    }
  }
  p.result = peel(p, p.result);
  return p;
}

std::string emit_expr(const Program& p, int i, std::vector<char>& done, std::ostringstream& out) {
  if (done[static_cast<size_t>(i)])
    return v(i);
  const Node& n = p.nodes[static_cast<size_t>(i)];
  auto a = [&](size_t k) { return emit_expr(p, n.args.at(k), done, out); };

  std::string rhs;
  switch (n.op) {
  case OpKind::InputX:
    rhs = "x";
    break;
  case OpKind::InputG:
    rhs = "g";
    break;
  case OpKind::ConstF64:
    rhs = "vFloat(" + std::to_string(static_cast<float>(n.c_f64)) + "f)";
    break;
  case OpKind::ConstI32:
    rhs = "vFloat(" + std::to_string(static_cast<float>(n.c_i32)) + "f)";
    break;
  case OpKind::Abs:
    rhs = "sfpi::abs(" + a(0) + ")";
    break;
  case OpKind::Neg:
    rhs = "-(" + a(0) + ")";
    break;
  case OpKind::Add:
    rhs = a(0) + " + " + a(1);
    break;
  case OpKind::Sub:
    rhs = a(0) + " - " + a(1);
    break;
  case OpKind::Mul:
  case OpKind::ScaleMul:
    rhs = a(0) + " * " + a(1);
    break;
  case OpKind::Div:
    rhs = a(0) + " / " + a(1);
    break;
  case OpKind::Max:
    rhs = "sfpi::max(" + a(0) + ", " + a(1) + ")";
    break;
  case OpKind::Min:
    rhs = "sfpi::min(" + a(0) + ", " + a(1) + ")";
    break;
  case OpKind::Exp:
    rhs = "sfpi::exp(" + a(0) + ")";
    break;
  case OpKind::Exp2:
    rhs = "sfpi::exp2(" + a(0) + ")";
    break;
  case OpKind::Sqrt:
    rhs = "sfpi::sqrt(" + a(0) + ")";
    break;
  case OpKind::Tanh:
    rhs = "sfpi::tanh(" + a(0) + ")";
    break;
  case OpKind::Log:
    rhs = "sfpi::log(" + a(0) + ")";
    break;
  case OpKind::Log2:
    rhs = "sfpi::log2(" + a(0) + ")";
    break;
  case OpKind::Select:
    rhs = "((" + a(0) + " != 0) ? (" + a(1) + ") : (" + a(2) + "))";
    break;
  case OpKind::Clamp:
    rhs = "sfpi::min(" + a(2) + ", sfpi::max(" + a(0) + ", " + a(1) + "))";
    break;
  case OpKind::Copysign:
    rhs = "sfpi::copysign(" + a(0) + ", " + a(1) + ")";
    break;
  case OpKind::Frexp:
  case OpKind::FrexpMant:
  case OpKind::Normalize:
  case OpKind::RoundBF16:
  case OpKind::Ldexp:
  case OpKind::Reconstruct:
    rhs = a(0);
    break;
  case OpKind::FrexpExp:
  case OpKind::AddExp:
    rhs = "vFloat(0.0f)";
    break;
  default:
    throw std::runtime_error(std::string("sfpu: ") + op_kind_name(n.op));
  }
  out << "  vFloat " << v(i) << " = " << rhs << ";\n";
  done[static_cast<size_t>(i)] = 1;
  return v(i);
}

} // namespace

std::string emit_sfpi_cpp(const Program& p, const EmitOptions& opt) {
  const Program q = lower(p);
  std::vector<char> done(q.nodes.size(), 0);
  std::ostringstream body;
  const std::string ret = emit_expr(q, q.result, done, body);
  std::ostringstream os;
  os << "// SKETCH: not linked into tt-metal. Device kernels live under tt/kernels/compute/.\n"
     << "#include \"sfpi.h\"\n\nusing namespace sfpi;\n\n"
     << "inline vFloat " << opt.kernel_name << "(vFloat x, vFloat g) {\n"
     << body.str() << "  return " << ret << ";\n}\n";
  return os.str();
}

} // namespace sfpu
} // namespace bw_syn
