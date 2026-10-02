#pragma once

#include "bw_syn/bf16.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace bw_syn {

enum class TtKernelKind {
  Factored,
  BaselineMaterialize,
  TanhTailSplit4,
  ProbeRawCopy,
  ProbeComputeCopy,
  ProbeComputeMul,
};

struct TtDeviceInfo {
  bool linked{false};
  bool available{false};
  int device_id{0};
  std::string arch;
  std::string tt_metal_home;
  std::string tt_metal_commit;
};

TtDeviceInfo tt_probe();

#if defined(BW_SYN_WITH_TTMETAL)

class TtDeviceSession {
public:
  explicit TtDeviceSession(int device_id = 0);
  ~TtDeviceSession();

  TtDeviceSession(const TtDeviceSession&) = delete;
  TtDeviceSession& operator=(const TtDeviceSession&) = delete;

  BF16 eval(TtKernelKind kind, BF16 x, BF16 g);
  std::vector<BF16>
  eval_batch(TtKernelKind kind, const std::vector<BF16>& xs, const std::vector<BF16>& gs);

private:
  void* device_{nullptr};
  int device_id_{0};
};

#endif

} // namespace bw_syn
