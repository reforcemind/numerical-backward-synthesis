#pragma once

#include "bw_syn/bf16.hpp"
#include "bw_syn/contract.hpp"
#include "bw_syn/oracle.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace bw_syn {
namespace tt_harness {

enum class DeviceMode { HostSim, TtMetal };

inline constexpr int kDeviceWarmupRuns = 5;
inline constexpr int kDeviceMeasuredRuns = 20;

struct DeviceInfo {
  DeviceMode mode{DeviceMode::HostSim};
  std::string arch{"host"};
  std::string tt_metal_home;
  std::string tt_metal_commit{"unset"};
  bool available{true};
};

DeviceInfo probe_device();
std::string mode_name(DeviceMode m);

struct CaseSpec {
  float x;
  float g;
};

inline std::vector<CaseSpec> critical_cases() {
  return {{45.f, 4.f},
          {44.f, 4.f},
          {46.f, 4.f},
          {45.f, 2.f},
          {45.f, 8.f},
          {-45.f, 4.f},
          {45.f, -4.f},
          {0.f, 1.f},
          {20.f, 1.f},
          {BF16::max_finite().to_f32(), std::numeric_limits<float>::infinity()}};
}

struct CaseResult {
  CaseSpec spec;
  BF16 oracle;
  BF16 device_or_sim;
  BF16 baseline_model;
  std::uint32_t ulp_vs_oracle;
  bool false_zero_baseline;
  bool pass;
};

struct HardwareRunReport {
  DeviceInfo device;
  BackwardKind kind{BackwardKind::Tanh};
  std::vector<CaseResult> cases;
  bool all_pass{false};
  double host_wall_ms{0.0};
  std::optional<std::uint64_t> cycles_per_tile;
  std::optional<int> insn_count;
  std::optional<int> reg_count;
  std::optional<bool> baseline_false_zero_observed;
  int device_warmup_runs{0};
  int device_measured_runs{0};
};

HardwareRunReport run_critical_tanh(DeviceMode prefer = DeviceMode::HostSim);

struct TailCaseResult {
  BF16 x;
  BF16 g;
  BF16 oracle;
  BF16 observed;
  bool normal_output{false};
  bool pass{false};
};

struct TailRunReport {
  DeviceInfo device;
  std::vector<TailCaseResult> cases;
  std::size_t normal_outputs{0};
  bool all_normal_outputs_pass{false};
};

std::vector<std::pair<BF16, BF16>> tail_tanh_cases();
TailRunReport run_tail_tanh(DeviceMode prefer = DeviceMode::HostSim);

struct FormatProbeRow {
  std::string stage;
  BF16 x;
  BF16 g;
  BF16 expected;
  BF16 observed;
};

struct FormatProbeReport {
  DeviceInfo device;
  std::vector<FormatProbeRow> rows;
};

FormatProbeReport run_format_probe(DeviceMode prefer = DeviceMode::HostSim);

} // namespace tt_harness
} // namespace bw_syn
