#include "bw_syn/backends/tt_device.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(BW_SYN_WITH_TTMETAL)

#include "cb_indices.h"

#if __has_include(<tt-metalium/host_api.hpp>)
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#define BW_SYN_METALIUM_HEADERS 1
#elif __has_include("tt_metal/host_api.hpp")
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#else
#error "BW_SYN_WITH_TTMETAL=ON but tt-metal headers not found under TT_METAL_HOME"
#endif

namespace bw_syn {
namespace {

namespace fs = std::filesystem;
namespace metal = tt::tt_metal;

#if defined(BW_SYN_METALIUM_HEADERS)
using MetalDevice = metal::IDevice;
#else
using MetalDevice = metal::Device;
#endif

constexpr std::uint32_t kTileBytes = 2048;

std::string repo_root() {
#if defined(BW_SYN_ROOT)
  return BW_SYN_ROOT;
#else
  if (const char* r = std::getenv("BW_SYN_ROOT"))
    return r;
  return ".";
#endif
}

std::string kernel_file(const char* rel) {
  const fs::path p = fs::path(repo_root()) / rel;
  if (!fs::exists(p))
    throw std::runtime_error("missing kernel: " + p.string() + " (set BW_SYN_ROOT)");
  return fs::absolute(p).string();
}

std::string compute_kernel_path(TtKernelKind kind) {
  switch (kind) {
  case TtKernelKind::Factored:
    return kernel_file("tt/kernels/compute/tanh_bw_scale_separated.cpp");
  case TtKernelKind::BaselineMaterialize:
    return kernel_file("tt/kernels/compute/tanh_bw_baseline.cpp");
  case TtKernelKind::TanhTailSplit4:
    return kernel_file("tt/kernels/compute/tanh_bw_tail_split4.cpp");
  case TtKernelKind::TanhTailFused3:
  case TtKernelKind::TanhTailFused4:
  case TtKernelKind::VendorTailFused:
    return kernel_file("tt/kernels/compute/tanh_bw_tail_fused.cpp");
  case TtKernelKind::VendorTanhDerivative:
    return kernel_file("tt/kernels/compute/tanh_bw_vendor_derivative.cpp");
  case TtKernelKind::ProbeComputeCopy:
  case TtKernelKind::ProbeComputeMul:
    return kernel_file("tt/kernels/compute/format_probe.cpp");
  case TtKernelKind::ProbeRawCopy:
    break;
  }
  throw std::runtime_error("unknown TtKernelKind");
}

// detail::LaunchProgram / WriteToBuffer / ReadFromBuffer are slow-dispatch APIs;
// TtDeviceSession opens the device in slow-dispatch mode to match.
void enqueue_or_launch(MetalDevice* device, metal::Program& program) {
  metal::detail::LaunchProgram(device, program);
}

std::vector<BF16> run_on_device(MetalDevice* device,
                                TtKernelKind kind,
                                const std::vector<BF16>& xs,
                                const std::vector<BF16>& gs) {
  if (xs.size() != gs.size() || xs.empty())
    throw std::runtime_error("tt batch size mismatch");
  if ((kind == TtKernelKind::ProbeRawCopy || kind == TtKernelKind::ProbeComputeCopy) &&
      xs.size() != 1)
    throw std::runtime_error("copy probes require one tile");

  const std::uint32_t n_tiles = static_cast<std::uint32_t>(xs.size());
  metal::Program program = metal::CreateProgram();
  const metal::CoreCoord core{0, 0};

  metal::InterleavedBufferConfig cfg{
      .device = device,
      .size = kTileBytes * n_tiles,
      .page_size = kTileBytes,
      .buffer_type = metal::BufferType::DRAM,
  };
  auto x_buf = metal::CreateBuffer(cfg);
  auto g_buf = metal::CreateBuffer(cfg);
  auto y_buf = metal::CreateBuffer(cfg);

  std::vector<std::uint32_t> x_words(kTileBytes * n_tiles / sizeof(std::uint32_t), 0);
  std::vector<std::uint32_t> g_words(kTileBytes * n_tiles / sizeof(std::uint32_t), 0);
  auto* xb = reinterpret_cast<std::uint16_t*>(x_words.data());
  auto* gb = reinterpret_cast<std::uint16_t*>(g_words.data());
  const std::uint32_t elems = kTileBytes / sizeof(std::uint16_t);
  for (std::uint32_t t = 0; t < n_tiles; ++t) {
    for (std::uint32_t i = 0; i < elems; ++i) {
      xb[t * elems + i] = xs[t].bits;
      gb[t * elems + i] = gs[t].bits;
    }
  }
  metal::detail::WriteToBuffer(x_buf, x_words);
  metal::detail::WriteToBuffer(g_buf, g_words);

  const tt::DataFormat df = tt::DataFormat::Float16_b;
  std::uint32_t cb_tiles = 2;
  if (const char* value = std::getenv("BW_SYN_CB_TILES")) {
    if (std::string(value) != "1" && std::string(value) != "2")
      throw std::runtime_error("BW_SYN_CB_TILES must be 1 or 2");
    cb_tiles = std::string(value) == "1" ? 1 : 2;
  }
  auto make_cb = [&](std::uint32_t idx) {
    metal::CircularBufferConfig cbc(kTileBytes * (idx == BW_SYN_CB_TMP ? 1 : cb_tiles),
                                    {{idx, df}});
    cbc.set_page_size(idx, kTileBytes);
    metal::CreateCircularBuffer(program, core, cbc);
  };
  make_cb(BW_SYN_CB_X);
  make_cb(BW_SYN_CB_G);
  make_cb(BW_SYN_CB_TMP);
  make_cb(BW_SYN_CB_Y);

  auto reader = metal::CreateKernel(
      program,
      kernel_file("tt/kernels/dataflow/reader_dual_tiles.cpp"),
      core,
      metal::DataMovementConfig{.processor = metal::DataMovementProcessor::RISCV_0,
                                .noc = metal::NOC::RISCV_0_default,
                                .compile_args = {BW_SYN_CB_X, BW_SYN_CB_G}});
  auto writer = metal::CreateKernel(
      program,
      kernel_file("tt/kernels/dataflow/writer_unary.cpp"),
      core,
      metal::DataMovementConfig{
          .processor = metal::DataMovementProcessor::RISCV_1,
          .noc = metal::NOC::RISCV_1_default,
          .compile_args = {kind == TtKernelKind::ProbeRawCopy ? BW_SYN_CB_X : BW_SYN_CB_Y}});

  metal::SetRuntimeArgs(program,
                        reader,
                        core,
                        {static_cast<std::uint32_t>(x_buf->address()),
                         static_cast<std::uint32_t>(g_buf->address()),
                         n_tiles});
  metal::SetRuntimeArgs(
      program, writer, core, {static_cast<std::uint32_t>(y_buf->address()), n_tiles});
  if (kind != TtKernelKind::ProbeRawCopy) {
    std::vector<std::uint32_t> compute_args;
    if (kind == TtKernelKind::TanhTailFused3 || kind == TtKernelKind::TanhTailFused4)
      compute_args.push_back(kind == TtKernelKind::TanhTailFused3 ? 3 : 4);
    if (kind == TtKernelKind::VendorTailFused)
      compute_args.push_back(0);
    if (kind == TtKernelKind::ProbeComputeCopy || kind == TtKernelKind::ProbeComputeMul)
      compute_args.push_back(kind == TtKernelKind::ProbeComputeMul ? 1 : 0);
    auto compute =
        metal::CreateKernel(program,
                            compute_kernel_path(kind),
                            core,
                            metal::ComputeConfig{.math_fidelity = metal::MathFidelity::HiFi4,
                                                 .fp32_dest_acc_en = true,
                                                 .math_approx_mode = false,
                                                 .compile_args = compute_args});
    metal::SetRuntimeArgs(program, compute, core, {n_tiles});
  }

  enqueue_or_launch(device, program);

  std::vector<std::uint32_t> y_words(kTileBytes * n_tiles / sizeof(std::uint32_t), 0);
  metal::detail::ReadFromBuffer(y_buf, y_words);
  const auto* yb = reinterpret_cast<const std::uint16_t*>(y_words.data());
  std::vector<BF16> out;
  out.reserve(n_tiles);
  for (std::uint32_t t = 0; t < n_tiles; ++t) {
    const auto first = yb[t * elems];
    for (std::uint32_t i = 1; i < elems; ++i)
      if (yb[t * elems + i] != first)
        throw std::runtime_error("nonuniform output in constant-input device tile " +
                                 std::to_string(t));
    out.push_back(BF16::from_bits(first));
  }
  return out;
}

MetalDevice* as_device(void* p) {
  return static_cast<MetalDevice*>(p);
}

} // namespace

TtDeviceInfo tt_probe() {
  TtDeviceInfo info;
  info.linked = true;
  if (const char* home = std::getenv("TT_METAL_HOME"))
    info.tt_metal_home = home;
  if (const char* commit = std::getenv("TT_METAL_COMMIT"))
    info.tt_metal_commit = commit;
  if (const char* id = std::getenv("BW_SYN_TT_DEVICE_ID"))
    info.device_id = std::atoi(id);
  if (const char* arch = std::getenv("BW_SYN_ARCH"))
    info.arch = arch;
  else
    info.arch = "unprobed";
  info.available = !info.tt_metal_home.empty();
  return info;
}

TtDeviceSession::TtDeviceSession(int device_id) : device_id_(device_id) {
  // Read by tt-metal when CreateDevice builds its runtime options; mixing fast and
  // slow dispatch aborts at CloseDevice.
  setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", /*overwrite=*/0);
  auto* dev = metal::CreateDevice(device_id_);
  if (!dev)
    throw std::runtime_error("CreateDevice failed");
  device_ = dev;
}

TtDeviceSession::~TtDeviceSession() {
  if (device_) {
    metal::CloseDevice(as_device(device_));
    device_ = nullptr;
  }
}

BF16 TtDeviceSession::eval(TtKernelKind kind, BF16 x, BF16 g) {
  auto v = eval_batch(kind, {x}, {g});
  return v[0];
}

std::vector<BF16> TtDeviceSession::eval_batch(TtKernelKind kind,
                                              const std::vector<BF16>& xs,
                                              const std::vector<BF16>& gs) {
  return run_on_device(as_device(device_), kind, xs, gs);
}

} // namespace bw_syn

#else

namespace bw_syn {

TtDeviceInfo tt_probe() {
  TtDeviceInfo info;
  info.linked = false;
  if (const char* home = std::getenv("TT_METAL_HOME"))
    info.tt_metal_home = home;
  if (const char* commit = std::getenv("TT_METAL_COMMIT"))
    info.tt_metal_commit = commit;
  info.available = false;
  info.arch = "host";
  return info;
}

} // namespace bw_syn

#endif
