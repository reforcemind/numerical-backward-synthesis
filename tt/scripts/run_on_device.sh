#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export BW_SYN_ROOT="$ROOT"

if [[ -z "${TT_METAL_HOME:-}" ]]; then
  echo "ERROR: TT_METAL_HOME unset" >&2
  exit 3
fi

HEAD="$(git -C "$TT_METAL_HOME" rev-parse HEAD 2>/dev/null || true)"
PIN="$(tr -d '[:space:]' < tt/pin/tt-metal.COMMIT | grep -v '^#' | tail -n1 || true)"
if [[ -z "$HEAD" ]]; then
  echo "ERROR: cannot read HEAD from TT_METAL_HOME" >&2
  exit 3
fi
if [[ -z "$PIN" || "$PIN" == "UNSET" ]]; then
  echo "ERROR: set a 40-char SHA in tt/pin/tt-metal.COMMIT before device runs" >&2
  exit 3
fi
if [[ "$HEAD" != "$PIN" ]]; then
  echo "ERROR: TT_METAL_HOME HEAD ($HEAD) != pin ($PIN)" >&2
  exit 3
fi

export TT_METAL_COMMIT="$HEAD"
mkdir -p results
echo "$TT_METAL_COMMIT" > results/device_tt_metal_commit.txt

BUILD="${BUILD_DIR:-build-tt}"
if [[ ! -x "$BUILD/run_tt_harness" ]]; then
  cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DBW_SYN_WITH_TTMETAL=ON
  cmake --build "$BUILD" -j
fi

ARCH="${BW_SYN_ARCH:-}"
if [[ -z "$ARCH" ]]; then
  echo "ERROR: set BW_SYN_ARCH=wormhole or blackhole (do not invent arch)" >&2
  exit 3
fi
CSV="results/hw/${ARCH}.csv"
mkdir -p results/hw
"$BUILD/run_tt_harness" --device --csv "$CSV"
"$BUILD/export_results" --out results
