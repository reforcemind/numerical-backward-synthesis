#include "bw_syn/cost_model.hpp"

#include <algorithm>

namespace bw_syn {

CostEstimate estimate_cost(const Program& p, const CostModel& model) {
  CostEstimate e;
  e.instruction_count = static_cast<int>(p.nodes.size());

  e.register_estimate = std::min(8, 2 + static_cast<int>(p.nodes.size() / 3));

  double lat = 0.0;
  for (const auto& n : p.nodes) {
    switch (n.op) {
    case OpKind::Add:
    case OpKind::Sub:
    case OpKind::Abs:
    case OpKind::Neg:
    case OpKind::Max:
    case OpKind::Min:
    case OpKind::Copysign:
    case OpKind::AddExp:
      lat += model.latency_add;
      break;
    case OpKind::Mul:
    case OpKind::ScaleMul:
    case OpKind::Ldexp:
    case OpKind::Frexp:
    case OpKind::FrexpMant:
    case OpKind::FrexpExp:
    case OpKind::Normalize:
    case OpKind::Reconstruct:
      lat += model.latency_mul;
      break;
    case OpKind::Div:
      lat += model.latency_div;
      break;
    case OpKind::Exp:
    case OpKind::Exp2:
    case OpKind::Tanh:
    case OpKind::Log:
    case OpKind::Log2:
    case OpKind::Sqrt:
      lat += model.latency_exp;
      break;
    case OpKind::RoundBF16:
      lat += model.latency_round;
      break;
    default:
      lat += model.latency_misc;
      break;
    }
  }
  e.latency = lat;
  e.scalar_score = scalarize(e, model);
  e.detail = "target=" + model.target_name + " (model, not measured hardware)";
  return e;
}

double scalarize(const CostEstimate& e, const CostModel& model) {
  return model.insn_weight * static_cast<double>(e.instruction_count) + e.latency +
         model.reg_pressure_weight * static_cast<double>(e.register_estimate);
}

bool dominates(double ulp_a, double score_a, double ulp_b, double score_b) {
  const bool le = (ulp_a <= ulp_b) && (score_a <= score_b);
  const bool strict = (ulp_a < ulp_b) || (score_a < score_b);
  return le && strict;
}

} // namespace bw_syn
