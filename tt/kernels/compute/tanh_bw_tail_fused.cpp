#include "../../../include/bw_syn/detail/tail_exp_product.hpp"
#include "../common/cb_indices.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include <cstdint>
#include <tools/profiler/kernel_profiler.hpp>

#ifdef TRISC_MATH
#include "ckernel_sfpu_tanh_derivative.h"
namespace ckernel::sfpu {

struct TailOps {
  using Float = sfpi::vFloat;
  using Int = sfpi::vInt;
  static sfpi_inline Float abs(Float x) { return sfpi::abs(x); }
  static sfpi_inline Float mul(Float a, Float b) { return a * b; }
  static sfpi_inline Float mad(Float a, Float b, Float c) { return a * b + c; }
  static sfpi_inline Float round_bf16(Float x) {
    return sfpi::convert<sfpi::vFloat16b>(x, sfpi::RoundMode::Nearest);
  }
  static sfpi_inline Float round_int(Float z, Int& k) {
    return _sfpu_round_to_nearest_int32_(z, k);
  }
  static sfpi_inline Int exponent(Float x) { return sfpi::exexp(x, sfpi::ExponentMode::Biased); }
  static sfpi_inline Float set_exponent(Float x, int e) { return sfpi::setexp(x, e); }
  static sfpi_inline Float reconstruct(Float x, Int e) {
    Float result = 0.0f;
    v_if(e > 0) {
      result = sfpi::setexp(x, e);
    }
    v_endif;
    v_if(e == 0) {
      v_if(sfpi::abs(sfpi::setexp(x, 127)) >= 1.9921875f) {
        result = sfpi::setexp(sfpi::setman(x, 0), 1);
      }
      v_endif;
    }
    v_endif;
    return result;
  }
};

template <int Degree> inline void tail_product_face(uint32_t ix, uint32_t ig, uint32_t iy) {
  for (int lane_vector = 0; lane_vector < 8; ++lane_vector) {
    const sfpi::vFloat x = sfpi::dst_reg[ix * 32];
    const sfpi::vFloat g = sfpi::dst_reg[ig * 32];
    if constexpr (Degree == 0) {
      const sfpi::vFloat product = inline_exp_sech2_tail(sfpi::abs(x)) * g;
      const sfpi::vFloat rounded =
          sfpi::convert<sfpi::vFloat16b>(product, sfpi::RoundMode::Nearest);
      sfpi::dst_reg[iy * 32] = rounded;
    } else {
      sfpi::dst_reg[iy * 32] = bw_syn::detail::tail_exp_product<TailOps, Degree>(x, g);
    }
    sfpi::dst_reg++;
  }
}

} // namespace ckernel::sfpu
#endif

namespace ckernel {
template <int Degree> inline void tail_product_tile() {
  MATH((SFPU_BINARY_CALL(
      DST_SYNC_MODE, DST_ACCUM_MODE, tail_product_face, (Degree), 0, 1, 0, VectorMode::RC)));
}
} // namespace ckernel

template <int Degree> inline void run_tiles(uint32_t n_tiles) {
  constexpr uint32_t cb_x = BW_SYN_CB_X;
  constexpr uint32_t cb_g = BW_SYN_CB_G;
  constexpr uint32_t cb_y = BW_SYN_CB_Y;
  for (uint32_t t = 0; t < n_tiles; ++t) {
    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_g, 1);
    tile_regs_acquire();
    copy_init(cb_x);
    copy_tile(cb_x, 0, 0);
    copy_init(cb_g);
    copy_tile(cb_g, 0, 1);
    ckernel::mul_binary_tile_init();
    ckernel::tail_product_tile<Degree>();
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

void kernel_main() {
  const uint32_t n_tiles = get_arg_val<uint32_t>(0);
  constexpr int degree = get_compile_time_arg_val(0);
  compute_kernel_hw_startup(BW_SYN_CB_X, BW_SYN_CB_Y);
  if constexpr (degree == 0) {
    DeviceZoneScopedN("BW_SYN_TANH_VENDOR_FUSED");
    run_tiles<0>(n_tiles);
  } else {
    DeviceZoneScopedN("BW_SYN_TANH_TAIL_FUSED");
    run_tiles<degree>(n_tiles);
  }
}
