#pragma once

#include "bw_syn/ir.hpp"

#include <cstdint>
#include <string>

namespace bw_syn {

struct CostModel {
  double latency_add{1.0};
  double latency_mul{1.0};
  double latency_div{4.0};
  double latency_exp{8.0};
  double latency_misc{1.0};
  double latency_round{1.0};
  double reg_pressure_weight{0.25};
  double insn_weight{1.0};
  std::string target_name{"generic-sfpu-model"};
};

struct CostEstimate {
  double latency{0.0};
  int instruction_count{0};
  int register_estimate{0};
  double scalar_score{0.0};
  std::string detail;
};

CostEstimate estimate_cost(const Program& p, const CostModel& model);

double scalarize(const CostEstimate& e, const CostModel& model);

bool dominates(double ulp_a, double score_a, double ulp_b, double score_b);

} // namespace bw_syn
