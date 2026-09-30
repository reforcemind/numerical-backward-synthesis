#!/usr/bin/env bash
# Code Standards agent end-of-turn checks (C++ primary + optional Python).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "== clang-format (dry-run) =="
if command -v clang-format >/dev/null 2>&1; then
  mapfile -t CXX_FILES < <(find include src tools tests tt -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) 2>/dev/null | sort)
  if ((${#CXX_FILES[@]})); then
    clang-format --dry-run --Werror "${CXX_FILES[@]}"
  fi
else
  echo "skip: clang-format not installed"
fi

echo "== ctest =="
if [[ -d build ]]; then
  ctest --test-dir build --output-on-failure
else
  echo "skip: build/ missing - configure first"
fi

echo "== ruff (Python) =="
if command -v uvx >/dev/null 2>&1; then
  uvx ruff check .
  uvx ruff format --check .
elif command -v ruff >/dev/null 2>&1; then
  ruff check .
  ruff format --check .
else
  echo "skip: uvx/ruff not installed"
fi

echo "== pytest =="
if [[ -n "$(find experiments -name 'test_*.py' 2>/dev/null | head -1)" ]]; then
  if command -v uvx >/dev/null 2>&1; then
    uvx pytest -q
  elif command -v pytest >/dev/null 2>&1; then
    pytest -q
  else
    echo "skip: pytest not installed"
  fi
else
  echo "skip: no Python tests"
fi

echo "check_standards: ok"
