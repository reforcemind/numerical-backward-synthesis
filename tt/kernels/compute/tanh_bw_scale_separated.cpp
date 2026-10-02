#include "../common/cb_indices.h"
#include "../common/tanh_factor.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/isinf_isnan.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/tile_move_copy.h"
#include <cstdint>
#include <tools/profiler/kernel_profiler.hpp>

void kernel_main() {
  const uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_x = BW_SYN_CB_X;
  constexpr uint32_t cb_g = BW_SYN_CB_G;
  constexpr uint32_t cb_y = BW_SYN_CB_Y;

  compute_kernel_hw_startup(cb_x, cb_y);
  DeviceZoneScopedN("BW_SYN_TANH_FACTORED");
  for (uint32_t t = 0; t < n_tiles; ++t) {
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_g, 1);
    tile_regs_acquire();
    copy_init(cb_x);
    copy_tile(cb_x, 0, 0);
    copy_tile(cb_x, 0, 3);
    copy_init(cb_g);
    copy_tile(cb_g, 0, 1);
    bw_syn_tanh_half_factor_tile();
    ckernel::mul_binary_tile_init();
    ckernel::mul_binary_tile(1, 0, 2);
    ckernel::mul_binary_tile(2, 0, 0);

    copy_init(cb_g);
    copy_tile(cb_g, 0, 2);
    ckernel::isfinite_tile_init();
    ckernel::isfinite_tile(3);
    ckernel::isinf_tile_init();
    ckernel::isinf_tile(2);
    ckernel::mul_binary_tile_init();
    ckernel::mul_binary_tile(2, 3, 2);
    ckernel::where_tile_init();
    ckernel::where_tile<DataFormat::Float32>(2, 1, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_y, 1);
    pack_tile(0, cb_y);
    tile_regs_release();
    cb_push_back(cb_y, 1);
    cb_pop_front(cb_x, 1);
    cb_pop_front(cb_g, 1);
  }
}
