#include "../common/cb_indices.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include <cstdint>

void kernel_main() {
  constexpr uint32_t mode = get_compile_time_arg_val(0);
  const uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_x = BW_SYN_CB_X;
  constexpr uint32_t cb_g = BW_SYN_CB_G;
  constexpr uint32_t cb_y = BW_SYN_CB_Y;

  compute_kernel_hw_startup(cb_x, cb_y);
  for (uint32_t t = 0; t < n_tiles; ++t) {
    cb_wait_front(cb_x, 1);
    if constexpr (mode == 1)
      cb_wait_front(cb_g, 1);
    tile_regs_acquire();
    copy_init(cb_x);
    copy_tile(cb_x, 0, 0);
    if constexpr (mode == 1) {
      copy_init(cb_g);
      copy_tile(cb_g, 0, 1);
      ckernel::mul_binary_tile_init();
      ckernel::mul_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_y, 1);
    pack_tile(0, cb_y);
    tile_regs_release();
    cb_push_back(cb_y, 1);
    cb_pop_front(cb_x, 1);
    if constexpr (mode == 1)
      cb_pop_front(cb_g, 1);
  }
}
