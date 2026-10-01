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
    rhs = a(0) + " * " + a(1);
    break;
  case OpKind::Div:
  case OpKind::Max:
  case OpKind::Min:
  case OpKind::Exp:
  case OpKind::Exp2:
  case OpKind::Sqrt:
  case OpKind::Tanh:
  case OpKind::Log:
  case OpKind::Log2:
  case OpKind::Select:
  case OpKind::Clamp:
  case OpKind::Copysign:
  case OpKind::Frexp:
  case OpKind::FrexpMant:
  case OpKind::FrexpExp:
  case OpKind::ScaleMul:
  case OpKind::AddExp:
  case OpKind::Normalize:
  case OpKind::RoundBF16:
  case OpKind::FlushBF16:
  case OpKind::PositiveDerivativeInf:
  case OpKind::Ldexp:
  case OpKind::Reconstruct:
    throw std::runtime_error("sfpu: unsupported " + op_kind_name(n.op) + " at v" +
                             std::to_string(i));
  default:
    throw std::runtime_error(std::string("sfpu: ") + op_kind_name(n.op));
  }
  out << "  vFloat " << v(i) << " = " << rhs << ";\n";
  done[static_cast<size_t>(i)] = 1;
  return v(i);
}

} // namespace

std::string emit_sfpi_cpp(const Program& p, const EmitOptions& opt) {
  (void)reachable_nodes(p);
  std::vector<char> done(p.nodes.size(), 0);
  std::ostringstream body;
  const std::string ret = emit_expr(p, p.result, done, body);
  std::ostringstream os;
  os << "// SKETCH: not linked into tt-metal. Device kernels live under tt/kernels/compute/.\n"
     << "#include \"sfpi.h\"\n\nusing namespace sfpi;\n\n"
     << "inline vFloat " << opt.kernel_name << "(vFloat x, vFloat g) {\n"
     << body.str() << "  return " << ret << ";\n}\n";
  return os.str();
}

} // namespace sfpu
} // namespace bw_syn
