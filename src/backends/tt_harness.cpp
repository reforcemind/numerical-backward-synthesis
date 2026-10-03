#include "bw_syn/backends/tt_harness.hpp"

#include "bw_syn/backends/tt_device.hpp"
#include "bw_syn/functions/tanh_bw.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

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
    got = session->eval_batch(TtKernelKind::Factored, xs, gs);
    base_dev = session->eval_batch(TtKernelKind::BaselineMaterialize, xs, gs);
    rep.device_warmup_runs = kDeviceWarmupRuns;
    rep.device_measured_runs = kDeviceMeasuredRuns;
    for (int i = 0; i < rep.device_warmup_runs + rep.device_measured_runs; ++i) {
      session->eval_batch(TtKernelKind::Factored, xs, gs);
      session->eval_batch(TtKernelKind::BaselineMaterialize, xs, gs);
    }
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
    cr.pass =
        check_sample(contract, x, g, cr.device_or_sim, cr.oracle).verdict == ContractVerdict::Pass;
    if (!cr.pass)
      rep.all_pass = false;
    if (cr.false_zero_baseline)
      rep.baseline_false_zero_observed = true;
    rep.cases.push_back(cr);
  }

  if (mode == DeviceMode::TtMetal) {
    bool saw_motiv = false;
    for (const auto& c : rep.cases) {
      if (c.spec.x == 45.f && c.spec.g == 4.f) {
        saw_motiv = true;
        if (c.oracle.bits != 0x008f)
          throw std::runtime_error("device run: host oracle for (45,4) is not 0x008f");
      }
    }
    if (!saw_motiv)
      throw std::runtime_error("device run: missing motivating case (45,4)");
    if (!rep.baseline_false_zero_observed.has_value())
      rep.baseline_false_zero_observed = false;
  }

  rep.host_wall_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return rep;
}

std::vector<std::pair<BF16, BF16>> tail_tanh_cases() {
  const BF16 gradients[] = {BF16::from_f64(1),
                            BF16::from_f64(4),
                            BF16::from_f64(256),
                            BF16::from_f64(4294967296.0),
                            BF16::max_finite(),
                            BF16::from_bits(0xff7f)};
  std::vector<std::pair<BF16, BF16>> cases;
  const std::uint32_t first = BF16::from_f64(4).bits;
  const std::uint32_t last = BF16::from_f64(88.5).bits;
  for (std::uint32_t bits = first; bits <= last; ++bits) {
    const BF16 x = BF16::from_bits(static_cast<std::uint16_t>(bits));
    for (const BF16 g : gradients)
      cases.emplace_back(x, g);
    const double midpoint = BF16::min_normal().to_f64() - 0.5 * BF16::min_subnormal().to_f64();
    const BF16 threshold =
        BF16::from_f64(midpoint / exact_derivative_f64(BackwardKind::Tanh, x.to_f64()));
    for (int delta : {-1, 0, 1}) {
      const auto gb = static_cast<std::uint16_t>(threshold.bits + delta);
      for (const std::uint16_t sign : {0, 0x8000})
        cases.emplace_back(x, BF16::from_bits(gb | sign));
    }
  }
  for (const BF16 x : {BF16::from_f64(45),
                       BF16::from_f64(88.5),
                       BF16::from_f64(-4),
                       BF16::from_f64(-45),
                       BF16::from_f64(-88.5)})
    for (const BF16 g : gradients)
      cases.emplace_back(x, g);
  return cases;
}

TailRunReport run_tail_tanh(DeviceMode prefer, bool timed, int degree) {
  if (degree != 2 && degree != 3 && degree != 4)
    throw std::invalid_argument("tail polynomial degree must be 2, 3 or 4");
  TailRunReport rep;
  rep.polynomial_degree = degree;
  rep.device = probe_device();
  const auto contract = NumericalContract::bf16_default("tanh_backward");
  const auto cases = tail_tanh_cases();
  std::vector<BF16> oracles;
  oracles.reserve(cases.size());
  for (const auto& [x, g] : cases)
    oracles.push_back(contract_reference(BackwardKind::Tanh, contract, x, g));
  std::vector<BF16> observed, baseline, vendor, split4, vendor_fused, cubic;
  observed.reserve(cases.size());
  baseline.reserve(cases.size());
#if defined(BW_SYN_WITH_TTMETAL)
  std::unique_ptr<TtDeviceSession> session;
  const auto selected_kind =
      degree == 2 ? TtKernelKind::TanhTailFused2
                  : (degree == 3 ? TtKernelKind::TanhTailFused3 : TtKernelKind::TanhTailFused4);
#endif

  if (prefer == DeviceMode::TtMetal) {
#if defined(BW_SYN_WITH_TTMETAL)
    if (!rep.device.available)
      throw std::runtime_error("--device requires TT_METAL_HOME");
    require_device_pin(rep.device);
    session = std::make_unique<TtDeviceSession>(tt_probe().device_id);
    constexpr std::size_t batch_size = 128;
    auto run_all = [&](TtKernelKind kind, std::vector<BF16>& out) {
      for (std::size_t base = 0; base < cases.size(); base += batch_size) {
        const auto end = std::min(base + batch_size, cases.size());
        std::vector<BF16> xs, gs;
        xs.reserve(end - base);
        gs.reserve(end - base);
        for (std::size_t i = base; i < end; ++i) {
          xs.push_back(cases[i].first);
          gs.push_back(cases[i].second);
        }
        auto batch = session->eval_batch(kind, xs, gs);
        out.insert(out.end(), batch.begin(), batch.end());
      }
    };
    run_all(selected_kind, observed);
    if (degree != 3)
      run_all(TtKernelKind::TanhTailFused3, cubic);
    else
      cubic = observed;
    run_all(TtKernelKind::TanhTailSplit4, split4);
    run_all(TtKernelKind::BaselineMaterialize, baseline);
    run_all(TtKernelKind::VendorTanhDerivative, vendor);
    run_all(TtKernelKind::VendorTailFused, vendor_fused);
    rep.vendor_normal_passes = 0;
    rep.vendor_fused_normal_passes = 0;
#else
    (void)timed;
    throw std::runtime_error("--device requires -DBW_SYN_WITH_TTMETAL=ON");
#endif
  } else {
    if (timed)
      throw std::runtime_error("tail timing requires --device");
    for (const auto& [x, g] : cases) {
      observed.push_back(tanh_bw::tail_fused_host(x, g, contract, degree));
      cubic.push_back(tanh_bw::tail_fused_host(x, g, contract, 3));
      split4.push_back(tanh_bw::tail_split4_host(x, g, contract));
      baseline.push_back(tanh_bw::baseline_materialize(x, g, contract));
    }
  }

  rep.all_normal_outputs_pass = true;
  for (std::size_t i = 0; i < cases.size(); ++i) {
    const auto [x, g] = cases[i];
    const BF16 oracle = oracles[i];
    const bool in_scope = oracle.is_normal();
    auto passes = [&](BF16 got) {
      return in_scope && check_sample(contract, x, g, got, oracle).verdict == ContractVerdict::Pass;
    };
    const bool pass = passes(observed[i]);
    const bool baseline_pass = passes(baseline[i]);
    const bool split4_pass = passes(split4[i]);
    const BF16 cubic_result = cubic[i];
    const bool cubic_pass = passes(cubic_result);
    std::optional<BF16> vendor_result;
    std::optional<bool> vendor_pass;
    std::optional<BF16> vendor_fused_result;
    std::optional<bool> vendor_fused_pass;
    if (!vendor.empty()) {
      vendor_result = vendor[i];
      vendor_pass = passes(vendor[i]);
      vendor_fused_result = vendor_fused[i];
      vendor_fused_pass = passes(vendor_fused[i]);
    }
    if (in_scope) {
      ++rep.normal_outputs;
      if (!pass)
        rep.all_normal_outputs_pass = false;
      if (baseline_pass)
        ++rep.baseline_normal_passes;
      if (split4_pass)
        ++rep.split4_normal_passes;
      if (cubic_pass)
        ++rep.cubic_normal_passes;
      if (vendor_pass.value_or(false))
        ++*rep.vendor_normal_passes;
      if (vendor_fused_pass.value_or(false))
        ++*rep.vendor_fused_normal_passes;
    }
    rep.cases.push_back(TailCaseResult{x,
                                       g,
                                       oracle,
                                       observed[i],
                                       baseline[i],
                                       vendor_result,
                                       in_scope,
                                       pass,
                                       baseline_pass,
                                       vendor_pass,
                                       false,
                                       split4[i],
                                       split4_pass,
                                       vendor_fused_result,
                                       vendor_fused_pass,
                                       cubic_result,
                                       cubic_pass});
  }
  if (rep.normal_outputs == 0)
    throw std::runtime_error("tail sweep has no normal-output cases");

#if defined(BW_SYN_WITH_TTMETAL)
  if (timed && rep.all_normal_outputs_pass) {
    std::vector<std::size_t> eligible;
    for (std::size_t i = 0; i < rep.cases.size(); ++i)
      if (rep.cases[i].normal_output)
        eligible.push_back(i);
    if (eligible.size() < kTailTimingTiles)
      throw std::runtime_error("tail timing needs more normal-output cases");
    std::vector<BF16> xs, gs, expected;
    xs.reserve(kTailTimingTiles);
    gs.reserve(kTailTimingTiles);
    for (std::size_t j = 0; j < kTailTimingTiles; ++j) {
      const auto i = eligible[j * (eligible.size() - 1) / (kTailTimingTiles - 1)];
      rep.cases[i].timed_input = true;
      xs.push_back(cases[i].first);
      gs.push_back(cases[i].second);
      expected.push_back(observed[i]);
    }
    rep.timing_tiles = xs.size();
    rep.device_warmup_runs = kDeviceWarmupRuns;
    rep.device_measured_runs = kDeviceMeasuredRuns;
    for (int i = 0; i < rep.device_warmup_runs + rep.device_measured_runs; ++i) {
      const auto got = session->eval_batch(selected_kind, xs, gs);
      if (!std::equal(got.begin(), got.end(), expected.begin(), expected.end(), [](BF16 a, BF16 b) {
            return a.bits == b.bits;
          }))
        throw std::runtime_error("fused tail output changed during timing; reject this run");
      if (degree != 3)
        session->eval_batch(TtKernelKind::TanhTailFused3, xs, gs);
      session->eval_batch(TtKernelKind::TanhTailSplit4, xs, gs);
      session->eval_batch(TtKernelKind::BaselineMaterialize, xs, gs);
      session->eval_batch(TtKernelKind::VendorTanhDerivative, xs, gs);
      session->eval_batch(TtKernelKind::VendorTailFused, xs, gs);
    }
  }
#endif
  return rep;
}

FormatProbeReport run_format_probe(DeviceMode prefer) {
  FormatProbeReport rep;
  rep.device = probe_device();
#if defined(BW_SYN_WITH_TTMETAL)
  std::unique_ptr<TtDeviceSession> session;
#endif
  if (prefer == DeviceMode::TtMetal) {
#if defined(BW_SYN_WITH_TTMETAL)
    if (!rep.device.available)
      throw std::runtime_error("--device requires TT_METAL_HOME");
    require_device_pin(rep.device);
    session = std::make_unique<TtDeviceSession>(tt_probe().device_id);
#else
    throw std::runtime_error("--device requires -DBW_SYN_WITH_TTMETAL=ON");
#endif
  }

  auto record = [&](const std::string& stage, TtKernelKind kind, BF16 x, BF16 g) {
    const BF16 expected =
        kind == TtKernelKind::ProbeComputeMul ? BF16::from_f64(x.to_f64() * g.to_f64()) : x;
    BF16 observed = expected;
#if defined(BW_SYN_WITH_TTMETAL)
    if (session)
      observed = session->eval(kind, x, g);
#endif
    rep.rows.push_back(FormatProbeRow{stage, x, g, expected, observed});
  };
  const BF16 one = BF16::from_f64(1);
  const BF16 half = BF16::from_f64(0.5);
  const BF16 sub = BF16::min_subnormal();
  const BF16 normal = BF16::min_normal();
  record("raw_copy", TtKernelKind::ProbeRawCopy, sub, one);
  record("raw_copy", TtKernelKind::ProbeRawCopy, normal, one);
  record("compute_copy", TtKernelKind::ProbeComputeCopy, sub, one);
  record("compute_copy", TtKernelKind::ProbeComputeCopy, normal, one);
  record("compute_mul", TtKernelKind::ProbeComputeMul, normal, half);
  record("compute_mul", TtKernelKind::ProbeComputeMul, BF16::from_bits(0x8080), half);
  record("compute_mul", TtKernelKind::ProbeComputeMul, normal, one);
  record("compute_mul", TtKernelKind::ProbeComputeMul, BF16::from_bits(0x0100), half);
  return rep;
}

} // namespace tt_harness
} // namespace bw_syn
