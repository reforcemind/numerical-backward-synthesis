#include "../common/cb_indices.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/tile_move_copy.h"
#include <cstdint>
#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
  const uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_x = BW_SYN_CB_X;
  constexpr uint32_t cb_g = BW_SYN_CB_G;
  constexpr uint32_t cb_y = BW_SYN_CB_Y;

  compute_kernel_hw_startup(cb_x, cb_y);
  DeviceZoneScopedN("BW_SYN_TANH_TAIL_SPLIT4");
  for (uint32_t t = 0; t < n_tiles; ++t) {
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_g, 1);
    tile_regs_acquire();
    copy_init(cb_x);
    copy_tile(cb_x, 0, 0);
    copy_init(cb_g);
    copy_tile(cb_g, 0, 1);

    ckernel::abs_tile_init();
    ckernel::abs_tile(0);
    ckernel::binop_with_scalar_tile_init();
    ckernel::mul_unary_tile(0, 0x3f000000u);
    ckernel::negative_tile_init();
    ckernel::negative_tile(0);
    ckernel::exp_tile_init();
    ckernel::exp_tile(0);

    ckernel::mul_binary_tile_init();
    ckernel::mul_binary_tile(1, 0, 1);
    ckernel::binop_with_scalar_tile_init();
    ckernel::mul_unary_tile(1, 0x40800000u);
    ckernel::mul_binary_tile_init();
    ckernel::mul_binary_tile(1, 0, 1);
    ckernel::mul_binary_tile(1, 0, 1);
    ckernel::mul_binary_tile(1, 0, 1);

    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_y, 1);
    pack_tile(1, cb_y);
    tile_regs_release();
    cb_push_back(cb_y, 1);
    cb_pop_front(cb_x, 1);
    cb_pop_front(cb_g, 1);
  }
}
