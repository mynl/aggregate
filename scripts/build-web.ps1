# Build the aggregate SPA into src/aggregate/api/static/.
#
# Usage:
#   .\scripts\build-web.ps1                  # same-origin build
#   .\scripts\build-web.ps1 -ApiBase https://api.mynl.com   # split-origin
#
# The script does not assume the working directory; it resolves
# everything relative to its own location so it's callable from
# anywhere (CI, package step, ad-hoc terminal).

[CmdletBinding()]
param(
    [string]$ApiBase = ""
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$webDir    = Join-Path $scriptDir "..\web" | Resolve-Path

Push-Location $webDir
try {
    if ($ApiBase) {
        $env:VITE_API_BASE_URL = $ApiBase
        Write-Host "Building with VITE_API_BASE_URL=$ApiBase"
    }
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing npm dependencies..."
        npm install --no-fund --no-audit
    }
    Write-Host "Running vite build..."
    npm run build
} finally {
    Pop-Location
    if ($ApiBase) { Remove-Item Env:VITE_API_BASE_URL -ErrorAction SilentlyContinue }
}
