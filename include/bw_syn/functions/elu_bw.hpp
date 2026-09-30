#pragma once

#include "bw_syn/bf16.hpp"
#include "bw_syn/contract.hpp"
#include "bw_syn/ir.hpp"

namespace bw_syn {

namespace elu_bw {

BF16 baseline_materialize(BF16 x, BF16 g, const NumericalContract& c, double alpha = 1.0);
BF16 scale_separated(BF16 x, BF16 g, const NumericalContract& c, double alpha = 1.0);
Program ir_scale_separated(double alpha = 1.0);

} // namespace elu_bw
} // namespace bw_syn
