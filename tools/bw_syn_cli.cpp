#include "bw_syn/backends/sfpu_emit.hpp"
#include "bw_syn/backends/tt_harness.hpp"
#include "bw_syn/functions/sigmoid_bw.hpp"
#include "bw_syn/functions/tanh_bw.hpp"
#include "bw_syn/synthesize.hpp"
#include "bw_syn/version.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  using namespace bw_syn;
  if (argc < 2) {
    std::cerr << "bw_syn <version|doctor|reproduce|verify|synth|emit|tt>\n";
    return 2;
  }
  const std::string cmd = argv[1];

  if (cmd == "version") {
    std::cout << BW_SYN_VERSION_STRING << "\n";
    return 0;
  }
  if (cmd == "doctor") {
    auto d = tt_harness::probe_device();
    std::cout << BW_SYN_VERSION_STRING << " mode=" << tt_harness::mode_name(d.mode)
              << " available=" << (d.available ? "1" : "0") << " arch=" << d.arch
              << " home=" << (d.tt_metal_home.empty() ? "unset" : d.tt_metal_home)
              << " commit=" << d.tt_metal_commit << "\n";
    return d.available ? 0 : 1;
  }
  if (cmd == "reproduce") {
    auto r = analyze_tanh_motivating_case(45.0, 4.0);
    return (r.baseline_is_false_zero && r.product_bf16.bits == 0x008f) ? 0 : 1;
  }
  if (cmd == "verify") {
    BackwardKind kind = BackwardKind::Tanh;
    for (int i = 2; i < argc; ++i)
      if (std::string(argv[i]) == "--sigmoid")
        kind = BackwardKind::Sigmoid;
    VerifyConfig cfg;
    cfg.kind = kind;
    cfg.contract = NumericalContract::bf16_default(backward_kind_name(kind));
    auto fn = [&](BF16 x, BF16 g) {
      return kind == BackwardKind::Sigmoid ? sigmoid_bw::scale_separated(x, g, cfg.contract)
                                           : tanh_bw::scale_separated(x, g, cfg.contract);
    };
    return verify_kernel(fn, cfg).ok ? 0 : 1;
  }
  if (cmd == "synth") {
    BackwardKind kind = BackwardKind::Tanh;
    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--sigmoid")
        kind = BackwardKind::Sigmoid;
      if (a == "--erf")
        kind = BackwardKind::Erf;
      if (a == "--elu")
        kind = BackwardKind::Elu;
    }
    SynthConfig cfg;
    cfg.kind = kind;
    cfg.contract = NumericalContract::bf16_default(backward_kind_name(kind));
    cfg.verify_cfg.kind = kind;
    cfg.verify_cfg.contract = cfg.contract;
    cfg.verify_cfg.max_samples = 20000;
    auto r = synthesize(cfg);
    if (!r.found)
      return 2;
    namespace fs = std::filesystem;
    fs::create_directories("kernels/generated");
    std::ofstream ofs(std::string("kernels/generated/") + backward_kind_name(kind) + "_best.cpp");
    ofs << sfpu::emit_sfpi_cpp(r.best.program, {backward_kind_name(kind) + "_syn"});
    return ofs ? 0 : 1;
  }
  if (cmd == "emit") {
    std::string which = argc > 2 ? argv[2] : "tanh";
    Program p =
        which == "sigmoid" ? sigmoid_bw::ir_scale_separated() : tanh_bw::ir_scale_separated();
    namespace fs = std::filesystem;
    fs::create_directories("kernels/generated");
    const std::string path = "kernels/generated/" + which + "_bw_scale_separated.cpp";
    std::ofstream ofs(path);
    ofs << sfpu::emit_sfpi_cpp(p, {which + "_bw_scale_separated"});
    return ofs ? 0 : 1;
  }
  if (cmd == "tt") {
    auto mode = tt_harness::DeviceMode::HostSim;
    for (int i = 2; i < argc; ++i)
      if (std::string(argv[i]) == "--device")
        mode = tt_harness::DeviceMode::TtMetal;
    try {
      return tt_harness::run_critical_tanh(mode).all_pass ? 0 : 1;
    } catch (const std::exception& ex) {
      std::cerr << ex.what() << "\n";
      return 3;
    }
  }
  return 2;
}
