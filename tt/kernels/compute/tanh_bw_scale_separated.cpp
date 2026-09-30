#include "../common/cb_indices.h"
#include <cstdint>

#if defined(BW_SYN_TT_DEVICE)
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "sfpi.h"
#endif

#if defined(BW_SYN_TT_DEVICE)
namespace {
using namespace sfpi;

inline vFloat tanh_bw_scale_separated(vFloat x, vFloat g) {
  vFloat ax = sfpi::abs(x);
  vFloat e = sfpi::exp(-(ax + ax));
  vFloat den = vFloat(1.0f) + e;
  return g * ((vFloat(4.0f) * e) / (den * den));
}
} // namespace

namespace NAMESPACE {
void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_x = BW_SYN_CB_X;
  constexpr uint32_t cb_g = BW_SYN_CB_G;
  constexpr uint32_t cb_y = BW_SYN_CB_Y;

  init_sfpu(cb_x);
  for (uint32_t t = 0; t < n_tiles; ++t) {
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_g, 1);
    tile_regs_acquire();
    copy_tile_init(cb_x);
    copy_tile(cb_x, 0, 0);
    copy_tile_init(cb_g);
    copy_tile(cb_g, 0, 1);
    dst_reg[0] = tanh_bw_scale_separated(dst_reg[0], dst_reg[1]);
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
} // namespace NAMESPACE
#endif
