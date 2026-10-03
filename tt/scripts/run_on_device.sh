#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export BW_SYN_ROOT="$ROOT"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: commit tracked source changes before a pinned device run" >&2
  exit 3
fi
if [[ "$#" -gt 1 || ( "${1:-}" != "" && "${1:-}" != "--explore" && "${1:-}" != "--tail-timing" && "${1:-}" != "--critical" ) ]]; then
  echo "ERROR: usage: run_on_device.sh [--tail-timing | --explore | --critical]" >&2
  exit 2
fi
EXPLORE=1
TAIL_TIMING=1
if [[ "${1:-}" == "--explore" ]]; then
  TAIL_TIMING=0
elif [[ "${1:-}" == "--critical" ]]; then
  EXPLORE=0
  TAIL_TIMING=0
fi
TAIL_DEGREE="${BW_SYN_TAIL_DEGREE:-3}"
if [[ "$TAIL_DEGREE" != "3" && "$TAIL_DEGREE" != "4" ]]; then
  echo "ERROR: BW_SYN_TAIL_DEGREE must be 3 or 4" >&2
  exit 2
fi

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  echo "ERROR: TT_METAL_HOME unset" >&2
  exit 3
fi
# Newer tt-metal resolves its runtime root from this variable, not TT_METAL_HOME.
export TT_METAL_RUNTIME_ROOT="${TT_METAL_RUNTIME_ROOT:-$TT_METAL_HOME}"
ARCH="${BW_SYN_ARCH:-}"
if [[ "$ARCH" != "wormhole" && "$ARCH" != "blackhole" ]]; then
  echo "ERROR: set BW_SYN_ARCH=wormhole or blackhole" >&2
  exit 3
fi

HEAD="$(git -C "$TT_METAL_HOME" rev-parse HEAD 2>/dev/null || true)"
PIN="$(sed -e 's/#.*$//' -e 's/[[:space:]]//g' tt/pin/tt-metal.COMMIT | sed '/^$/d' | tail -n1)"
if [[ -z "$HEAD" ]]; then
  echo "ERROR: cannot read HEAD from TT_METAL_HOME" >&2
  exit 3
fi
if [[ -n "$(git -C "$TT_METAL_HOME" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: TT_METAL_HOME has tracked source changes; a HEAD SHA alone would not pin this run" >&2
  exit 3
fi
if [[ "$PIN" == "UNSET" ]]; then
  printf '%s\n' "$HEAD" > tt/pin/tt-metal.COMMIT
  PIN="$HEAD"
  echo "Pinned tt-metal at $HEAD"
elif [[ ! "$PIN" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: tt/pin/tt-metal.COMMIT must contain UNSET or a 40-char SHA" >&2
  exit 3
fi
if [[ "$HEAD" != "$PIN" ]]; then
  echo "ERROR: TT_METAL_HOME HEAD ($HEAD) != pin ($PIN)" >&2
  exit 3
fi

export TT_METAL_COMMIT="$HEAD"

BUILD="${BUILD_DIR:-build-tt}"
mkdir -p results/hw results/paper
if [[ "$EXPLORE" == "1" ]]; then
  RUN_DIR="results/runs/${ARCH}_$(git rev-parse --short HEAD)_$(date -u +%Y%m%dT%H%M%SZ)_d${TAIL_DEGREE}"
  mkdir -p "$RUN_DIR"
  BUILD_LOG="$RUN_DIR/build.log"
  PROVENANCE="$RUN_DIR/provenance.txt"
  PROFILE_MARKER=""
  finish_run() {
    RUN_STATUS=$?
    printf 'exit_status=%s\n' "$RUN_STATUS" >> "$PROVENANCE"
    if [[ -n "$PROFILE_MARKER" ]]; then
      rm -f "$PROFILE_MARKER"
    fi
    echo "Evidence saved in $RUN_DIR (git add results/runs to include it in the run commit)"
  }
  trap finish_run EXIT
else
  BUILD_LOG="results/paper/build_${ARCH}_${HEAD}.log"
  PROVENANCE="results/paper/device_provenance_${ARCH}.txt"
fi
{
  printf 'bw_syn_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'tt_metal_commit=%s\n' "$HEAD"
  printf 'arch=%s\n' "$ARCH"
  printf 'tail_polynomial_degree=%s\n' "$TAIL_DEGREE"
  printf 'cb_tiles=%s\n' "${BW_SYN_CB_TILES:-2}"
  printf 'device_id=%s\n' "${BW_SYN_TT_DEVICE_ID:-0}"
  printf 'utc_start=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  sha256sum tt/kernels/common/tanh_factor.h tt/kernels/compute/tanh_bw_scale_separated.cpp \
    tt/kernels/compute/tanh_bw_baseline.cpp tt/kernels/compute/tanh_bw_tail_split4.cpp \
    tt/kernels/compute/tanh_bw_vendor_derivative.cpp tt/kernels/compute/tanh_bw_tail_fused.cpp \
    include/bw_syn/detail/tail_exp_product.hpp tt/kernels/compute/format_probe.cpp \
    tt/kernels/dataflow/reader_dual_tiles.cpp tt/kernels/dataflow/writer_unary.cpp \
    tt/host/tt_metal_backend.cpp src/backends/tt_harness.cpp
} > "$PROVENANCE"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBW_SYN_WITH_TTMETAL=ON 2>&1 | tee "$BUILD_LOG"
cmake --build "$BUILD" -j "${BW_SYN_BUILD_JOBS:-2}" 2>&1 | tee -a "$BUILD_LOG"
if [[ "$TAIL_TIMING" == "1" ]]; then
  echo "Running tail correctness before diagnostic timing"
  PROFILE_LOG="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
  PROFILE_MARKER="$(mktemp)"
  TAIL_CSV="$RUN_DIR/tail.csv"
  if TT_METAL_DEVICE_PROFILER=1 TT_METAL_SLOW_DISPATCH_MODE=1 "$BUILD/run_tt_harness" --device \
    --tail-sweep --tail-degree "$TAIL_DEGREE" --timed --csv "$TAIL_CSV" 2>&1 | \
    tee "$RUN_DIR/tail.log"; then
    TAIL_STATUS=0
  else
    TAIL_STATUS=$?
  fi
  if [[ "$TAIL_STATUS" -gt 1 ]]; then
    exit "$TAIL_STATUS"
  fi
  if [[ -f "$PROFILE_LOG" && "$PROFILE_LOG" -nt "$PROFILE_MARKER" ]]; then
    RAW_PROFILE="$RUN_DIR/raw_profiler.csv"
    cp "$PROFILE_LOG" "$RAW_PROFILE"
  elif [[ "$TAIL_STATUS" == "0" ]]; then
    echo "ERROR: no fresh device profiler CSV at $PROFILE_LOG" >&2
    exit 4
  fi
  if [[ "$TAIL_STATUS" == "0" ]]; then
    CASE_COUNT="$(($(wc -l < "$TAIL_CSV") - 1))"
    PREFIX_LAUNCHES="$(((CASE_COUNT + 127) / 128))"
    "$BUILD/summarize_tt_profiler" "$RAW_PROFILE" "$RUN_DIR/cycles.csv" "$ARCH" \
      "$HEAD" 128 "$PREFIX_LAUNCHES" BW_SYN_TANH_TAIL_FUSED BW_SYN_TANH_TAIL_SPLIT4 BW_SYN_TANH_MATERIALIZED \
      BW_SYN_TANH_VENDOR_DERIVATIVE BW_SYN_TANH_VENDOR_FUSED || exit 4
    cat "$RUN_DIR/cycles.csv"
  else
    echo "Tail candidate failed the normal-output contract; timing summary withheld" >&2
  fi
  TT_METAL_SLOW_DISPATCH_MODE=1 "$BUILD/run_tt_harness" --device --format-probe \
    --csv "$RUN_DIR/format.csv" 2>&1 | tee "$RUN_DIR/format.log"
  exit "$TAIL_STATUS"
fi
if [[ "$EXPLORE" == "1" ]]; then
  unset TT_METAL_DEVICE_PROFILER
  echo "Running exploratory correctness probes; these are not paper measurements"
  if TT_METAL_SLOW_DISPATCH_MODE=1 "$BUILD/run_tt_harness" --device --tail-sweep \
    --tail-degree "$TAIL_DEGREE" --csv "$RUN_DIR/tail.csv" 2>&1 | tee "$RUN_DIR/tail.log"; then
    TAIL_STATUS=0
  else
    TAIL_STATUS=$?
  fi
  if [[ "$TAIL_STATUS" -gt 1 ]]; then
    exit "$TAIL_STATUS"
  fi
  if TT_METAL_SLOW_DISPATCH_MODE=1 "$BUILD/run_tt_harness" --device --format-probe \
    --csv "$RUN_DIR/format.csv" 2>&1 | tee "$RUN_DIR/format.log"; then
    FORMAT_STATUS=0
  else
    FORMAT_STATUS=$?
  fi
  if [[ "$TAIL_STATUS" -ne 0 ]]; then
    exit "$TAIL_STATUS"
  fi
  exit "$FORMAT_STATUS"
fi
"$BUILD/export_results" --out results
CSV="results/hw/${ARCH}.csv"
PROFILE_LOG="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
PROFILE_MARKER="$(mktemp)"
trap 'rm -f "$PROFILE_MARKER"' EXIT
export TT_METAL_DEVICE_PROFILER=1
if TT_METAL_SLOW_DISPATCH_MODE=1 "$BUILD/run_tt_harness" --device --csv "$CSV" 2>&1 | \
  tee "results/paper/device_run_${ARCH}_${HEAD}.log"; then
  HARNESS_STATUS=0
else
  HARNESS_STATUS=$?
fi
if [[ "$HARNESS_STATUS" -gt 1 ]]; then
  exit "$HARNESS_STATUS"
fi
echo "$TT_METAL_COMMIT" > results/device_tt_metal_commit.txt
if [[ -f "$PROFILE_LOG" && "$PROFILE_LOG" -nt "$PROFILE_MARKER" ]]; then
  RAW_PROFILE="results/paper/raw_device_profiler_${ARCH}_${HEAD}.csv"
  cp "$PROFILE_LOG" "$RAW_PROFILE"
  if "$BUILD/summarize_tt_profiler" "$RAW_PROFILE" "results/paper/device_cycles_${ARCH}.csv" "$ARCH" "$HEAD"; then
    PROFILE_STATUS=0
  else
    PROFILE_STATUS=4
  fi
else
  echo "ERROR: no fresh device profiler CSV at $PROFILE_LOG" >&2
  PROFILE_STATUS=4
fi
unset TT_METAL_DEVICE_PROFILER
if python3 tt/scripts/ttnn_tanh_baseline.py --device-csv "$CSV" 2>&1 | \
  tee "results/paper/ttnn_run_${ARCH}_${HEAD}.log"; then
  TTNN_STATUS=0
else
  TTNN_STATUS=5
fi
if [[ "$HARNESS_STATUS" -ne 0 ]]; then
  exit "$HARNESS_STATUS"
fi
if [[ "$PROFILE_STATUS" -ne 0 ]]; then
  exit "$PROFILE_STATUS"
fi
exit "$TTNN_STATUS"
