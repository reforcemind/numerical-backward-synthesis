# Code Standards agent end-of-turn checks (C++ primary + optional Python).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

Write-Host "== clang-format (dry-run) =="
$clangFormat = Get-Command clang-format -ErrorAction SilentlyContinue
if ($clangFormat) {
  $files = Get-ChildItem -Path include,src,tools,tests,tt -Recurse -Include *.cpp,*.hpp,*.h -ErrorAction SilentlyContinue
  if ($files) {
    & clang-format --dry-run --Werror @($files.FullName)
    if ($LASTEXITCODE -ne 0) { throw "clang-format check failed" }
  }
} else {
  Write-Host "skip: clang-format not installed"
}

Write-Host "== ctest =="
if (Test-Path build) {
  ctest --test-dir build --output-on-failure
  if ($LASTEXITCODE -ne 0) { throw "ctest failed" }
} else {
  Write-Host "skip: build/ missing - configure first"
}

Write-Host "== ruff (Python) =="
$uvx = Get-Command uvx -ErrorAction SilentlyContinue
$ruff = Get-Command ruff -ErrorAction SilentlyContinue
if ($uvx) {
  uvx ruff check .
  if ($LASTEXITCODE -ne 0) { throw "ruff check failed" }
  uvx ruff format --check .
  if ($LASTEXITCODE -ne 0) { throw "ruff format check failed" }
} elseif ($ruff) {
  ruff check .
  if ($LASTEXITCODE -ne 0) { throw "ruff check failed" }
  ruff format --check .
  if ($LASTEXITCODE -ne 0) { throw "ruff format check failed" }
} else {
  Write-Host "skip: uvx/ruff not installed"
}

Write-Host "== pytest =="
$pyTests = Get-ChildItem -Path experiments -Recurse -Filter test_*.py -ErrorAction SilentlyContinue
if ($pyTests) {
  if ($uvx) { uvx pytest -q }
  elseif (Get-Command pytest -ErrorAction SilentlyContinue) { pytest -q }
  else { Write-Host "skip: pytest not installed" }
} else {
  Write-Host "skip: no Python tests"
}

Write-Host "check_standards: ok"
