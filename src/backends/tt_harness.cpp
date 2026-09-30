#include "bw_syn/backends/tt_harness.hpp"

#include "bw_syn/backends/tt_device.hpp"
#include "bw_syn/functions/tanh_bw.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bw_syn {
namespace tt_harness {
namespace {

#if defined(BW_SYN_WITH_TTMETAL)
std::string read_pin_sha() {
  std::string root = ".";
#if defined(BW_SYN_ROOT)
  root = BW_SYN_ROOT;
#else
  if (const char* r = std::getenv("BW_SYN_ROOT"))
    root = r;
#endif
  std::ifstream in(root + "/tt/pin/tt-metal.COMMIT");
  if (!in)
    return {};
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#')
      continue;
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
      line.pop_back();
    std::size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t'))
      ++i;
    return line.substr(i);
  }
  return {};
}

void require_device_pin(const DeviceInfo& d) {
  const std::string pin = read_pin_sha();
  if (pin.empty() || pin == "UNSET")
    throw std::runtime_error("--device requires a SHA in tt/pin/tt-metal.COMMIT");
  if (d.tt_metal_commit == "unset" || d.tt_metal_commit.empty())
    throw std::runtime_error("--device requires TT_METAL_COMMIT matching the pin");
  if (d.tt_metal_commit != pin) {
    std::ostringstream os;
    os << "--device TT_METAL_COMMIT (" << d.tt_metal_commit << ") != pin (" << pin << ")";
    throw std::runtime_error(os.str());
  }
  if (const char* arch = std::getenv("BW_SYN_ARCH")) {
    if (std::string(arch) != "wormhole" && std::string(arch) != "blackhole")
      throw std::runtime_error("BW_SYN_ARCH must be wormhole or blackhole");
  } else {
    throw std::runtime_error("--device requires BW_SYN_ARCH=wormhole|blackhole");
  }
}
#endif

} // namespace

std::string mode_name(DeviceMode m) {
  return m == DeviceMode::TtMetal ? "tt_metal" : "host_sim";
}

DeviceInfo probe_device() {
  const auto p = tt_probe();
  DeviceInfo info;
  info.mode = p.linked ? DeviceMode::TtMetal : DeviceMode::HostSim;
  info.arch = p.arch;
  if (const char* arch = std::getenv("BW_SYN_ARCH"))
    info.arch = arch;
  info.tt_metal_home = p.tt_metal_home;
  info.tt_metal_commit = p.tt_metal_commit.empty() ? "unset" : p.tt_metal_commit;
  info.available = p.linked ? p.available : true;
  return info;
}

HardwareRunReport run_critical_tanh(DeviceMode prefer) {
  HardwareRunReport rep;
  rep.device = probe_device();
  DeviceMode mode = DeviceMode::HostSim;
  if (prefer == DeviceMode::TtMetal) {
#if defined(BW_SYN_WITH_TTMETAL)
    if (!rep.device.available)
      throw std::runtime_error("--device requires TT_METAL_HOME");
    require_device_pin(rep.device);
    mode = DeviceMode::TtMetal;
#else
    throw std::runtime_error("--device requires -DBW_SYN_WITH_TTMETAL=ON");
#endif
  }
  rep.device.mode = mode;

  auto contract = NumericalContract::bf16_default("tanh_backward");
  rep.all_pass = true;
  const auto t0 = std::chrono::steady_clock::now();

#if defined(BW_SYN_WITH_TTMETAL)
  std::unique_ptr<TtDeviceSession> session;
  std::vector<BF16> xs, gs, got, base_dev;
  if (mode == DeviceMode::TtMetal) {
    session = std::make_unique<TtDeviceSession>(tt_probe().device_id);
    for (const auto& spec : critical_cases()) {
      xs.push_back(BF16::from_f64(spec.x));
      gs.push_back(BF16::from_f64(spec.g));
    }
    got = session->eval_batch(TtKernelKind::ScaleSeparated, xs, gs);
    base_dev = session->eval_batch(TtKernelKind::BaselineMaterialize, xs, gs);
  }
#endif

#if defined(BW_SYN_WITH_TTMETAL)
  std::size_t idx = 0;
#endif
  for (const auto& spec : critical_cases()) {
    CaseResult cr;
    cr.spec = spec;
    BF16 x = BF16::from_f64(spec.x);
    BF16 g = BF16::from_f64(spec.g);
    cr.oracle = contract_reference(BackwardKind::Tanh, contract, x, g);
#if defined(BW_SYN_WITH_TTMETAL)
    if (mode == DeviceMode::TtMetal) {
      cr.device_or_sim = got[idx];
      cr.baseline_model = base_dev[idx];
      ++idx;
    } else {
      cr.device_or_sim = tanh_bw::scale_separated(x, g, contract);
      cr.baseline_model = tanh_bw::baseline_materialize(x, g, contract);
    }
#else
    cr.device_or_sim = tanh_bw::scale_separated(x, g, contract);
    cr.baseline_model = tanh_bw::baseline_materialize(x, g, contract);
#endif
    cr.ulp_vs_oracle = ulp_distance(cr.device_or_sim, cr.oracle);
    cr.false_zero_baseline =
        cr.baseline_model.is_zero() && !cr.oracle.is_zero() && cr.oracle.is_finite();
    cr.pass = cr.ulp_vs_oracle <= contract.max_ulp_error;
    if (!cr.pass)
      rep.all_pass = false;
    if (cr.false_zero_baseline)
      rep.ftz_observed = true;
    rep.cases.push_back(cr);
  }

  if (mode == DeviceMode::TtMetal) {
    bool saw_motiv = false;
    for (const auto& c : rep.cases) {
      if (c.spec.x == 45.f && c.spec.g == 4.f) {
        saw_motiv = true;
        if (c.oracle.bits != 0x008f)
          throw std::runtime_error("device run: host oracle for (45,4) is not 0x008f");
        if (!c.pass)
          throw std::runtime_error(
              "device run: scale-separated kernel failed motivating case (45,4)");
      }
    }
    if (!saw_motiv)
      throw std::runtime_error("device run: missing motivating case (45,4)");
    if (!rep.ftz_observed.has_value())
      rep.ftz_observed = false;
  }

  rep.host_wall_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return rep;
}

} // namespace tt_harness
} // namespace bw_syn
