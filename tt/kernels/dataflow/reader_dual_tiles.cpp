#include "dataflow_api.h"

void kernel_main() {
  const uint32_t src0_addr = get_arg_val<uint32_t>(0);
  const uint32_t src1_addr = get_arg_val<uint32_t>(1);
  const uint32_t n_tiles = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_x = get_compile_time_arg_val(0);
  constexpr uint32_t cb_g = get_compile_time_arg_val(1);
  constexpr uint32_t onetile = 1;

  const uint32_t tile_bytes = get_tile_size(cb_x);
  const InterleavedAddrGenFast<true> src0 = {
      .bank_base_address = src0_addr, .page_size = tile_bytes, .data_format = get_dataformat(cb_x)};
  const InterleavedAddrGenFast<true> src1 = {
      .bank_base_address = src1_addr, .page_size = tile_bytes, .data_format = get_dataformat(cb_g)};

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_reserve_back(cb_x, onetile);
    noc_async_read_tile(i, src0, get_write_ptr(cb_x));
    noc_async_read_barrier();
    cb_push_back(cb_x, onetile);

    cb_reserve_back(cb_g, onetile);
    noc_async_read_tile(i, src1, get_write_ptr(cb_g));
    noc_async_read_barrier();
    cb_push_back(cb_g, onetile);
  }
}
