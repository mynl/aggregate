# combine.ps1 - merge the individual cheat-sheet PDFs into one Cheat_Sheets.pdf.
#
# Uses pdfunite (poppler, ships with the MiKTeX/TeX distribution and is on PATH).
# Run .\make.ps1 first so the per-sheet PDFs are current.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Order in the combined document: the DecL language first, then the classes
# roughly in dependency order (Underwriter -> Severity -> Aggregate ->
# BivariateAggregate -> Portfolio -> PnL -> Distortion -> Bounds). Each class
# follows the one it is built out of, so the reader never meets a name before
# its card.
$order = @(
    'DecL_Cheat_Sheet.pdf',
    'Underwriter_Cheat_Sheet.pdf',
    'Severity_Cheat_Sheet.pdf',
    'Aggregate_Cheat_Sheet.pdf',
    'BivariateAggregate_Cheat_Sheet.pdf',
    'Portfolio_Cheat_Sheet.pdf',
    'PnL_Cheat_Sheet.pdf',
    'Distortion_Cheat_Sheet.pdf',
    'Bounds_Cheat_Sheet.pdf'
)

$missing = $order | Where-Object { -not (Test-Path $_) }
if ($missing) { throw "Missing PDFs (run .\make.ps1 first): $($missing -join ', ')" }

if (-not (Get-Command pdfunite -ErrorAction SilentlyContinue)) {
    throw "pdfunite not found on PATH (it ships with poppler / MiKTeX)."
}

pdfunite @order 'Cheat_Sheets.pdf'
if ($LASTEXITCODE -ne 0) { throw "pdfunite failed" }

Write-Host "Wrote Cheat_Sheets.pdf ($($order.Count) source PDFs)." -ForegroundColor Green
