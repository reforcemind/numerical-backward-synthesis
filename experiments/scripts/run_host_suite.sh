#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-results}"
BIN="${BIN_DIR:-build}"
mkdir -p "$OUT"
"$BIN/export_results" --out "$OUT"
