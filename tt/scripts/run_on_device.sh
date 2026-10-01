#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export BW_SYN_ROOT="$ROOT"

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  echo "ERROR: TT_METAL_HOME unset" >&2
  exit 3
fi
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
BUILD_LOG="results/paper/build_${ARCH}_${HEAD}.log"
{
  printf 'bw_syn_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'tt_metal_commit=%s\n' "$HEAD"
  printf 'arch=%s\n' "$ARCH"
  printf 'device_id=%s\n' "${BW_SYN_TT_DEVICE_ID:-0}"
  printf 'utc_start=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  sha256sum tt/kernels/common/tanh_factor.h tt/kernels/compute/tanh_bw_scale_separated.cpp \
    tt/kernels/compute/tanh_bw_baseline.cpp
} > "results/paper/device_provenance_${ARCH}.txt"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBW_SYN_WITH_TTMETAL=ON 2>&1 | tee "$BUILD_LOG"
cmake --build "$BUILD" -j "${BW_SYN_BUILD_JOBS:-2}" 2>&1 | tee -a "$BUILD_LOG"
"$BUILD/export_results" --out results
CSV="results/hw/${ARCH}.csv"
PROFILE_LOG="$TT_METAL_HOME/generated/profiler/.logs/profile_log_device.csv"
PROFILE_MARKER="$(mktemp)"
trap 'rm -f "$PROFILE_MARKER"' EXIT
export TT_METAL_DEVICE_PROFILER=1
if "$BUILD/run_tt_harness" --device --csv "$CSV" 2>&1 | tee "results/paper/device_run_${ARCH}_${HEAD}.log"; then
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
