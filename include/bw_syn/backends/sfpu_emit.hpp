#pragma once

#include "bw_syn/ir.hpp"

#include <string>

namespace bw_syn {
namespace sfpu {

struct EmitOptions {
  std::string kernel_name{"bw_syn_kernel"};
};

std::string emit_sfpi_cpp(const Program& p, const EmitOptions& opt = {});

} // namespace sfpu
} // namespace bw_syn
