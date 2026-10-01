#pragma once

#include "bw_syn/cost_model.hpp"
#include "bw_syn/ir.hpp"
#include "bw_syn/oracle.hpp"
#include "bw_syn/verify.hpp"

#include <string>
#include <vector>

namespace bw_syn {

enum class RangeReduction {
  None,
  AbsSymmetry,
  ClampDomain,
};

enum class EvalStrategy {
  DirectMaterialize, // IR rounds and flushes the derivative before multiplication
  ScaleSeparated,
};

enum class NormStrategy {
  None,
  BeforeMul,
  AfterMul,
};

enum class ReconstructStrategy {
  LdexpThenRound,
  RoundThenLdexp,
};

enum class RoundStrategy {
  FinalOnly,
  IntermediateAndFinal,
};

struct SynthParams {
  RangeReduction range_red{RangeReduction::None};
  EvalStrategy eval{EvalStrategy::ScaleSeparated};
  NormStrategy norm{NormStrategy::BeforeMul};
  ReconstructStrategy reconstruct{ReconstructStrategy::LdexpThenRound};
  RoundStrategy round{RoundStrategy::FinalOnly};
};

struct Candidate {
  SynthParams params;
  Program program;
  CostEstimate cost;
  VerifyReport verify;
  bool verified{false};
  std::string id;
};

struct SynthConfig {
  BackwardKind kind{BackwardKind::Tanh};
  NumericalContract contract;
  VerifyConfig verify_cfg;
  CostModel cost_model;
  double alpha{1.0};

  bool full_grammar{true};
  std::uint32_t max_ulp_to_accept{1};
};

struct SynthResult {
  std::vector<Candidate> all;
  std::vector<Candidate> pareto;
  Candidate best;
  bool found{false};
  std::uint64_t search_space_size{0};
  std::uint64_t candidates_built{0};
  std::uint64_t candidates_verified{0};
  double synthesis_seconds{0.0};
  std::string summary;
};

Program build_candidate_program(BackwardKind kind, const SynthParams& p, double alpha = 1.0);

SynthResult synthesize(const SynthConfig& cfg);

std::string synth_params_key(const SynthParams& p);

} // namespace bw_syn
