#include "bw_syn/backends/tt_harness.hpp"
#include "bw_syn/csv_io.hpp"

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
  bool tail_sweep = false;
  bool format_probe = false;
  bool timed = false;
  int degree = 2;
  std::string out;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--device")
      prefer = DeviceMode::TtMetal;
    else if (a == "--tail-sweep")
      tail_sweep = true;
    else if (a == "--format-probe")
      format_probe = true;
    else if (a == "--timed")
      timed = true;
    else if (a == "--tail-degree" && i + 1 < argc) {
      const std::string value = argv[++i];
      if (value != "2" && value != "3" && value != "4") {
        std::cerr << "--tail-degree must be 2, 3 or 4\n";
        return 2;
      }
      degree = std::stoi(value);
    } else if (a == "--csv" && i + 1 < argc)
      out = argv[++i];
    else {
      std::cerr << "unknown or incomplete argument: " << a << "\n";
      return 2;
    }
  }
  if (tail_sweep && format_probe) {
    std::cerr << "choose one of --tail-sweep and --format-probe\n";
    return 2;
  }
  if (timed && !tail_sweep) {
    std::cerr << "--timed applies only to --tail-sweep\n";
    return 2;
  }

  if (out.empty()) {
    const char* arch = std::getenv("BW_SYN_ARCH");
    const std::string suffix =
        prefer == DeviceMode::TtMetal ? (arch ? arch : "device") : "host_sim";
    out = "results/hw/" + std::string(tail_sweep ? "tail_" : (format_probe ? "format_" : "")) +
          suffix + ".csv";
  }
  if (format_probe) {
    FormatProbeReport rep;
    try {
      rep = run_format_probe(prefer);
    } catch (const std::exception& ex) {
      std::cerr << ex.what() << "\n";
      return 3;
    }
    const fs::path path(out);
    if (!path.parent_path().empty())
      fs::create_directories(path.parent_path());
    std::vector<std::string> rows;
    for (const auto& row : rep.rows)
      rows.push_back(std::string(prefer == DeviceMode::TtMetal ? "device_probe" : "host_model") +
                     "," + rep.device.arch + "," + rep.device.tt_metal_commit + "," + row.stage +
                     "," + hex16(row.x.bits) + "," + hex16(row.g.bits) + "," +
                     hex16(row.expected.bits) + "," + hex16(row.observed.bits) + "," +
                     (row.expected.bits == row.observed.bits ? "1" : "0"));
    if (!write_csv(
            out,
            "label,arch,tt_metal_commit,stage,x_bits,g_bits,expected_bits,observed_bits,equal",
            rows))
      return 1;
    std::cout << "format probe: " << rep.rows.size() << " observations written to " << out << "\n";
    return 0;
  }
  if (tail_sweep) {
    TailRunReport rep;
    try {
      rep = run_tail_tanh(prefer, timed, degree);
    } catch (const std::exception& ex) {
      std::cerr << ex.what() << "\n";
      return 3;
    }
    const fs::path path(out);
    if (!path.parent_path().empty())
      fs::create_directories(path.parent_path());
    std::vector<std::string> rows;
    for (const auto& c : rep.cases)
      rows.push_back(
          std::string(prefer == DeviceMode::TtMetal ? "device_probe" : "host_model") + "," +
          rep.device.arch + "," + rep.device.tt_metal_commit + "," + hex16(c.x.bits) + "," +
          hex16(c.g.bits) + "," + hex16(c.oracle.bits) + "," + hex16(c.observed.bits) + "," +
          (c.normal_output ? "finite_normal_output" : "outside_scope") + "," +
          (c.normal_output ? (c.pass ? "1" : "0") : "") + "," + hex16(c.baseline.bits) + "," +
          (c.normal_output ? (c.baseline_pass ? "1" : "0") : "") + "," +
          (c.timed_input ? "1" : "0") + "," + (c.vendor ? hex16(c.vendor->bits) : "") + "," +
          (c.vendor_pass ? (*c.vendor_pass ? "1" : "0") : "") + "," + hex16(c.split4.bits) + "," +
          (c.normal_output ? (c.split4_pass ? "1" : "0") : "") + "," +
          std::to_string(rep.polynomial_degree) + "," +
          (c.vendor_fused ? hex16(c.vendor_fused->bits) : "") + "," +
          (c.vendor_fused_pass ? (*c.vendor_fused_pass ? "1" : "0") : "") + "," +
          hex16(c.cubic.bits) + "," + (c.normal_output ? (c.cubic_pass ? "1" : "0") : "") + "," +
          (degree == 2 ? "exp2_quadratic_normal_saturate" : "exp_polynomial_wh4_boundary"));
    if (!write_csv(out,
                   "label,arch,tt_metal_commit,x_bits,g_bits,oracle_bits,observed_bits,scope,pass,"
                   "baseline_bits,baseline_pass,timed_input,vendor_bits,vendor_pass,"
                   "split4_bits,split4_pass,polynomial_degree,vendor_fused_bits,vendor_fused_pass,"
                   "cubic_bits,cubic_pass,recipe",
                   rows))
      return 1;
    std::cout << "fused tail degree " << degree << ": " << rep.normal_outputs
              << " normal-output cases of " << rep.cases.size()
              << "; all pass=" << rep.all_normal_outputs_pass << "; materialized baseline passes "
              << rep.baseline_normal_passes << "\n";
    std::cout << "split4 passes " << rep.split4_normal_passes << "\n";
    std::cout << "wh4 cubic passes " << rep.cubic_normal_passes << "\n";
    if (rep.vendor_normal_passes)
      std::cout << "vendor derivative-times-gradient passes " << *rep.vendor_normal_passes << "\n";
    if (rep.vendor_fused_normal_passes)
      std::cout << "fused vendor tail passes " << *rep.vendor_fused_normal_passes << "\n";
    if (rep.timing_tiles != 0)
      std::cout << "tail timing: " << rep.timing_tiles << " tiles, " << rep.device_warmup_runs
                << " warmup + " << rep.device_measured_runs << " measured launches per kernel\n";
    return rep.all_normal_outputs_pass ? 0 : 1;
  }

  HardwareRunReport rep;
  try {
    rep = run_critical_tanh(prefer);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 3;
  }

  const char* label = prefer == DeviceMode::TtMetal ? "device" : "host_sim";
  const bool host = prefer == DeviceMode::HostSim;
  if (!fs::path(out).parent_path().empty())
    fs::create_directories(fs::path(out).parent_path());
  std::vector<std::string> rows;
  for (const auto& c : rep.cases)
    rows.push_back(std::string(label) + "," + rep.device.arch + "," + rep.device.tt_metal_commit +
                   "," + csv_f32(c.spec.x) + "," + csv_f32(c.spec.g) + "," + hex16(c.oracle.bits) +
                   "," + hex16(c.device_or_sim.bits) + "," + hex16(c.baseline_model.bits) + "," +
                   std::to_string(c.ulp_vs_oracle) + "," + (c.pass ? "1" : "0") + "," +
                   (c.false_zero_baseline ? "1" : "0") + ",,,,");
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
      st << "baseline_false_zero_observed,"
         << (rep.baseline_false_zero_observed.value_or(false) ? "1" : "0") << "\n";
      st << "device_warmup_runs," << rep.device_warmup_runs << "\n";
      st << "device_measured_runs," << rep.device_measured_runs << "\n";
      st << "insn_dump,unmeasured\n";
      st << "regs,unmeasured\n";
    }
  }

  return rep.all_pass ? 0 : 1;
}
