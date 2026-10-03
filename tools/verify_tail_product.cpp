#include "bw_syn/contract.hpp"
#include "bw_syn/csv_io.hpp"
#include "bw_syn/detail/tail_exp_host.hpp"
#include "bw_syn/oracle.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

template <int Degree, int Rate, int Scale>
bool verify(bw_syn::BackwardKind kind, bool quick, std::vector<std::string>& rows) {
  using namespace bw_syn;
  const auto contract = NumericalContract::bf16_default(backward_kind_name(kind));
  const auto first = BF16::from_f64(8.0 / Rate).bits;
  const auto last = BF16::from_f64(177.0 / Rate).bits;
  std::uint64_t checked = 0, failed = 0;
  std::uint32_t max_ulp = 0;
  BF16 worst_x{}, worst_g{};
  for (std::uint32_t xb = first; xb <= last; ++xb) {
    const BF16 x = BF16::from_bits(static_cast<std::uint16_t>(xb));
    for (std::uint32_t gb = 0x0080; gb <= 0x7f7f; gb += quick ? 127 : 1) {
      const BF16 g = BF16::from_bits(static_cast<std::uint16_t>(gb));
      const BF16 ref = contract_reference(kind, contract, x, g);
      if (!ref.is_normal())
        continue;
      const auto value = detail::tail_exp_product<detail::HostTailOps, Degree, Rate, Scale>(
          x.to_f32(), g.to_f32());
      const BF16 got = BF16::from_f32(value);
      const auto check = check_sample(contract, x, g, got, ref);
      ++checked;
      if (check.verdict != ContractVerdict::Pass)
        ++failed;
      if (check.ulp > max_ulp) {
        max_ulp = check.ulp;
        worst_x = x;
        worst_g = g;
      }
    }
  }
  rows.push_back("host_model," + backward_kind_name(kind) + "," + std::to_string(Degree) + "," +
                 (quick ? "sampled_positive_normal_g" : "all_positive_normal_g") + "," +
                 std::to_string(checked) + "," + std::to_string(failed) + "," +
                 std::to_string(max_ulp) + "," + hex16(worst_x.bits) + "," + hex16(worst_g.bits));
  std::cout << rows.back() << '\n';
  return checked != 0 && failed == 0;
}

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--quick")) {
    std::cerr << "usage: verify_tail_product OUT_CSV [--quick]\n";
    return 2;
  }
  std::vector<std::string> rows;
  bool pass = true;
  const bool quick = argc == 3;
  pass &= verify<3, 2, 2>(bw_syn::BackwardKind::Tanh, quick, rows);
  pass &= verify<4, 2, 2>(bw_syn::BackwardKind::Tanh, quick, rows);
  pass &= verify<3, 1, 0>(bw_syn::BackwardKind::Sigmoid, quick, rows);
  pass &= verify<4, 1, 0>(bw_syn::BackwardKind::Sigmoid, quick, rows);
  const std::filesystem::path path(argv[1]);
  if (!path.parent_path().empty())
    std::filesystem::create_directories(path.parent_path());
  if (!bw_syn::write_csv(path.string(),
                         "label,function,degree,gradient_domain,checked,failed,"
                         "max_ulp,worst_x_bits,worst_g_bits",
                         rows))
    return 2;
  return pass ? 0 : 1;
}
