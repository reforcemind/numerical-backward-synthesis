#include "bw_syn/backends/tt_harness.hpp"
#include "bw_syn/cost_model.hpp"
#include "bw_syn/csv_io.hpp"
#include "bw_syn/functions/tanh_bw.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  using namespace bw_syn;
  using namespace bw_syn::tt_harness;
  namespace fs = std::filesystem;

  DeviceMode prefer = DeviceMode::HostSim;
  std::string out = "results/hw/host_sim.csv";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--device") {
      prefer = DeviceMode::TtMetal;
      if (const char* arch = std::getenv("BW_SYN_ARCH"))
        out = std::string("results/hw/") + arch + ".csv";
      else
        out = "results/hw/device.csv";
    } else if (a == "--csv" && i + 1 < argc)
      out = argv[++i];
  }

  HardwareRunReport rep;
  try {
    rep = run_critical_tanh(prefer);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 3;
  }

  auto cost = estimate_cost(tanh_bw::ir_scale_separated(), {});
  const char* label = prefer == DeviceMode::TtMetal ? "device" : "host_sim";
  const bool host = prefer == DeviceMode::HostSim;
  const std::string cycles = host ? std::to_string(static_cast<int>(cost.latency)) : "";
  const std::string insn = host ? std::to_string(cost.instruction_count) : "";
  const std::string regs = host ? std::to_string(cost.register_estimate) : "";
  std::string ftz;
  if (!host && rep.ftz_observed.has_value())
    ftz = *rep.ftz_observed ? "1" : "0";

  fs::create_directories(fs::path(out).parent_path());
  std::vector<std::string> rows;
  for (const auto& c : rep.cases)
    rows.push_back(std::string(label) + "," + rep.device.arch + "," + rep.device.tt_metal_commit +
                   "," + csv_f32(c.spec.x) + "," + csv_f32(c.spec.g) + "," + hex16(c.oracle.bits) +
                   "," + hex16(c.device_or_sim.bits) + "," + hex16(c.baseline_model.bits) + "," +
                   std::to_string(c.ulp_vs_oracle) + "," + (c.pass ? "1" : "0") + "," +
                   (c.false_zero_baseline ? "1" : "0") + "," + cycles + "," + insn + "," + regs +
                   "," + ftz);
  if (!write_csv(out,
                 "label,arch,tt_metal_commit,x,g,oracle_bits,got_bits,baseline_bits,ulp,pass,"
                 "false_zero_baseline,cycles_per_tile,insn,regs,ftz",
                 rows))
    return 1;

  if (!host) {
    fs::create_directories("results/paper");
    std::ofstream st("results/paper/device_status.csv", std::ios::binary);
    if (st) {
      st << "item,status\n";
      st << "device_csv," << out << "\n";
      st << "arch," << rep.device.arch << "\n";
      st << "tt_metal_commit," << rep.device.tt_metal_commit << "\n";
      st << "all_pass," << (rep.all_pass ? "1" : "0") << "\n";
      st << "ftz_observed_baseline," << (rep.ftz_observed.value_or(false) ? "1" : "0") << "\n";
      st << "cycles_per_tile,unmeasured\n";
      st << "insn_dump,unmeasured\n";
      st << "regs,unmeasured\n";
    }
  }

  return rep.all_pass ? 0 : 1;
}
