#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/recip.h"

// Input x is in DST tile 0; tile 2 is scratch. Output is h(x) in tile 0.
inline void bw_syn_tanh_half_factor_tile() {
  ckernel::abs_tile_init();
  ckernel::abs_tile(0);
  ckernel::negative_tile_init();
  ckernel::negative_tile(0);
  ckernel::exp_tile_init();
  ckernel::exp_tile(0);

  ckernel::mul_binary_tile_init();
  ckernel::mul_binary_tile(0, 0, 2);
  ckernel::binop_with_scalar_tile_init();
  ckernel::add_unary_tile(2, 0x3f800000u);
  ckernel::recip_tile_init();
  ckernel::recip_tile(2);

  ckernel::binop_with_scalar_tile_init();
  ckernel::mul_unary_tile(0, 0x40000000u);
  ckernel::mul_binary_tile_init();
  ckernel::mul_binary_tile(0, 2, 0);
}
