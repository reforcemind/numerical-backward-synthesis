#pragma once

#include "bw_syn/bf16.hpp"
#include "bw_syn/contract.hpp"
#include "bw_syn/ir.hpp"

namespace bw_syn {
namespace tanh_bw {

BF16 baseline_materialize(BF16 x, BF16 g, const NumericalContract& c);
BF16 scale_separated(BF16 x, BF16 g, const NumericalContract& c);
BF16 tail_split4_host(BF16 x, BF16 g, const NumericalContract& c);
BF16 tail_fused_host(BF16 x, BF16 g, const NumericalContract& c, int degree = 3);
Program ir_direct();
Program ir_scale_separated();
Program ir_factored();
Program ir_tail_split4();

} // namespace tanh_bw
} // namespace bw_syn
