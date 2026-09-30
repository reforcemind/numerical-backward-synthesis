#pragma once

#include "bw_syn/ir.hpp"

namespace bw_syn {

enum class TransformKind {
  ScaleSeparateMul,
  InsertNormalize,
  DeferRounding,
  DefaultScalePipeline,
};

struct TransformPreconditions {
  bool inputs_finite_nonzero{true};
  bool no_intermediate_overflow_in_working_prec{true};
  bool significands_normalized_before_mul{true};
  bool exponent_sum_within_i32{true};
  bool final_round_matches_contract{true};
  bool allow_subnormal_significands{false};
  bool no_prior_bf16_of_derivative{true};
};

struct TransformResult {
  TransformKind kind{TransformKind::ScaleSeparateMul};
  bool applied{false};
  // Local structural flag (appendix B sketch): not a soundness proof.
  // Unchecked assumptions remain in TransformPreconditions unless gated.
  bool sound_under_preconditions{false};
  TransformPreconditions preconditions;
  Program program;
};

TransformResult apply_scale_separate_mul(const Program& input);
TransformResult apply_insert_normalize(const Program& input);
TransformResult apply_defer_rounding(const Program& input);
TransformResult apply_default_scale_pipeline(const Program& input);

bool has_intermediate_bf16_round(const Program& p);

} // namespace bw_syn
