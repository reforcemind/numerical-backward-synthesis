#include "../common/cb_indices.h"
#include <cstdint>

#if defined(BW_SYN_TT_DEVICE)
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "sfpi.h"
#endif

#if defined(BW_SYN_TT_DEVICE)
namespace {
using namespace sfpi;

// Materialize sech^2 through pack/unpack (BF16 round + FTZ), then multiply by g.
inline vFloat sech2_fp32(vFloat x) {
  vFloat ax = sfpi::abs(x);
  vFloat e = sfpi::exp(-(ax + ax));
  vFloat den = vFloat(1.0f) + e;
  return (vFloat(4.0f) * e) / (den * den);
}
} // namespace

namespace NAMESPACE {
void MAIN {
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr uint32_t cb_x = BW_SYN_CB_X;
  constexpr uint32_t cb_g = BW_SYN_CB_G;
  constexpr uint32_t cb_tmp = BW_SYN_CB_TMP;
  constexpr uint32_t cb_y = BW_SYN_CB_Y;

  init_sfpu(cb_x);
  binary_op_init_common(cb_tmp, cb_g, cb_y);

  for (uint32_t t = 0; t < n_tiles; ++t) {
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_g, 1);

    tile_regs_acquire();
    copy_tile_init(cb_x);
    copy_tile(cb_x, 0, 0);
    dst_reg[0] = sech2_fp32(dst_reg[0]);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_tmp, 1);
    pack_tile(0, cb_tmp);
    tile_regs_release();
    cb_push_back(cb_tmp, 1);
    cb_pop_front(cb_x, 1);

    cb_wait_front(cb_tmp, 1);
    tile_regs_acquire();
    mul_tiles_init(cb_tmp, cb_g);
    mul_tiles(cb_tmp, cb_g, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_y, 1);
    pack_tile(0, cb_y);
    tile_regs_release();
    cb_push_back(cb_y, 1);
    cb_pop_front(cb_tmp, 1);
    cb_pop_front(cb_g, 1);
  }
}
} // namespace NAMESPACE
#endif
