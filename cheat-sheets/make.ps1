# make.ps1 - build the aggregate class / DecL cheat sheets with Tectonic.
#
#   .\make.ps1            # build all sheets
#   .\make.ps1 Distortion # build only sheets whose name matches (substring)
#
# Tectonic replaces the old lualatex/xelatex make.bat: it is faster, gives
# clearer errors, auto-fetches any missing fonts/packages, and leaves no
# .aux/.log clutter. cheat_sheet_macros.tex is \input by every sheet and is
# never compiled on its own.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$sheets = @(
    'Underwriter_Cheat_Sheet',
    'Severity_Cheat_Sheet',
    'Aggregate_Cheat_Sheet',
    'Portfolio_Cheat_Sheet',
    'Distortion_Cheat_Sheet',
    'DecL_Cheat_Sheet'
)

# Optional filter: keep only sheets matching any argument (substring, case-insensitive).
if ($args.Count -gt 0) {
    $sheets = $sheets | Where-Object { $s = $_; ($args | Where-Object { $s -like "*$_*" }).Count -gt 0 }
    if ($sheets.Count -eq 0) { throw "No cheat sheet matched: $($args -join ', ')" }
}

if (-not (Get-Command tectonic -ErrorAction SilentlyContinue)) {
    throw "tectonic not found on PATH. Install from https://tectonic-typesetting.github.io/"
}

# Stamp the footer version from pyproject.toml so it never drifts. Writes the
# generated (gitignored) aggversion.tex that cheat_sheet_macros.tex \input's.
$pyproject = Join-Path $PSScriptRoot '..\pyproject.toml'
$verMatch = Select-String -Path $pyproject -Pattern '^\s*version\s*=\s*"([^"]+)"' |
            Select-Object -First 1
if (-not $verMatch) { throw "Could not read version from $pyproject" }
$version = $verMatch.Matches[0].Groups[1].Value
"\newcommand{\aggversion}{$version}" |
    Set-Content -Path (Join-Path $PSScriptRoot 'aggversion.tex') -Encoding utf8
Write-Host "version $version (from pyproject.toml)" -ForegroundColor DarkGray

foreach ($s in $sheets) {
    Write-Host "==> tectonic $s.tex" -ForegroundColor Cyan
    tectonic "$s.tex"
    if ($LASTEXITCODE -ne 0) { throw "tectonic failed on $s" }
}

Write-Host "Done. Run .\combine.ps1 to merge into Cheat_Sheets.pdf." -ForegroundColor Green
