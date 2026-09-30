param([string]$OutDir = "results")
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
& ".\build\export_results.exe" --out $OutDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
