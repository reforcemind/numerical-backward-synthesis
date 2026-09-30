#pragma once

#include "bw_syn/bf16.hpp"
#include "bw_syn/contract.hpp"
#include "bw_syn/ir.hpp"

namespace bw_syn {

namespace erf_bw {

BF16 baseline_materialize(BF16 x, BF16 g, const NumericalContract& c);
BF16 scale_separated(BF16 x, BF16 g, const NumericalContract& c);
Program ir_scale_separated();

} // namespace erf_bw
} // namespace bw_syn
