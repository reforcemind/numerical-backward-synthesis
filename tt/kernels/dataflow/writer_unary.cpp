#include "dataflow_api.h"

void kernel_main() {
  const uint32_t dst_addr = get_arg_val<uint32_t>(0);
  const uint32_t n_tiles = get_arg_val<uint32_t>(1);

  constexpr uint32_t cb_out = get_compile_time_arg_val(0);
  constexpr uint32_t onetile = 1;
  const uint32_t tile_bytes = get_tile_size(cb_out);
  const InterleavedAddrGenFast<true> dst = {.bank_base_address = dst_addr,
                                            .page_size = tile_bytes,
                                            .data_format = get_dataformat(cb_out)};

  for (uint32_t i = 0; i < n_tiles; ++i) {
    cb_wait_front(cb_out, onetile);
    noc_async_write_tile(i, dst, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, onetile);
  }
}
