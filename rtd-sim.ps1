# =============================================================================
# rtd-sim.ps1 : rehearse the Read the Docs build the way RTD really runs it
#
# WHY THIS EXISTS, GIVEN doc-test-uv.ps1 ALREADY BUILDS THE DOCS
# --------------------------------------------------------------
# `doc-test-uv.ps1` answers "did my .rst edits break the build?". It builds in
# place, against the working tree, in a long lived `.doc-venv` that has
# accumulated whatever was installed into it over months. That is the right
# tool for the edit loop and the wrong tool for the question this script
# answers:
#
#     "will Read the Docs build this, on their machine, from a clean clone?"
#
# Three things make that a different question:
#
#   1. RTD builds a COMMIT, not a working tree. A file never committed, or one
#      an editor left behind, is invisible to RTD. This script clones, so only
#      committed content takes part.
#   2. RTD builds in an EMPTY environment. `.doc-venv` here holds five extras'
#      worth of packages, so a doc page that quietly leans on one of them
#      builds locally and fails on RTD. This script installs exactly what
#      `.readthedocs.yaml` asks for and nothing else.
#   3. RTD uses `pip install`, not `uv sync`. The two differ in a way that
#      matters: `uv sync` is an EXACT sync and prunes anything not named, which
#      is why the other script must pass `--all-extras`; `pip install` is
#      additive and prunes nothing, which is why RTD naming `dev` alone is
#      correct. Mirroring RTD means mirroring pip.
#
# WHAT IT DOES NOT REHEARSE
# -------------------------
# The pdf COMPILE. `.readthedocs.yaml` may list `formats: [pdf]`, and RTD
# builds that in two stages: `sphinx -b latex` writes the .tex sources, then
# latexmk compiles them. latexmk is a perl script, there is no perl on this
# machine, and the local PDF path goes through tectonic instead, so the compile
# itself cannot be rehearsed. The `-Tex` switch (off by default) rehearses the
# first stage, which is where a good share of pdf-only failures start: markup
# only the latex writer trips on, images with no latex-usable format, the
# writer erroring outright. Failures inside latexmk (font and unicode coverage
# under xelatex, overfull boxes that escalate to errors, list nesting past
# LaTeX's cap) still show up only on RTD. The report says exactly what a green
# run covered whenever pdf is configured, rather than letting it imply more
# than it earned.
#
# HOW IT STAYS HONEST
# -------------------
# Every RTD input is READ FROM `.readthedocs.yaml`, never hardcoded: the Python
# version, the extras, the conf.py location, the formats. Change that file and
# this script changes with it. That is the point. A rehearsal that can drift
# from the thing it rehearses is worse than no rehearsal, because it says the
# build is fine right up until it is not.
#
# USAGE
# -----
#   .\rtd-sim.ps1                       # build the REFACTOR branch, local clone
#   .\rtd-sim.ps1 -Ref master           # after the merge, this is the real one
#   .\rtd-sim.ps1 -FromGitHub           # clone from GitHub: tests what is PUSHED
#   .\rtd-sim.ps1 -Mode refresh         # reuse the venv, re-clone and rebuild
#   .\rtd-sim.ps1 -Strict               # stricter than RTD: warnings are fatal
#   .\rtd-sim.ps1 -Tex                  # also run sphinx's latex builder, the
#                                       # first stage of RTD's pdf build
#
# `-Strict` adds `-W --keep-going`, which RTD does NOT do: RTD fails only on a
# non zero sphinx exit code. Use plain mode to answer "will RTD go green", and
# `-Strict` for the release gate in plan-for-v1.md, which asks for a build with
# warnings fatal. Two different bars, both worth having.
# =============================================================================

param(
    # Branch or tag to build. RTD builds whatever its default version points
    # at, which is `master` here. Default is REFACTOR because that is where the
    # content lives until the merge lands.
    [string]$Ref = "REFACTOR",

    # Clone from github.com/mynl/aggregate instead of this local repository.
    # Local is the default so the build can be rehearsed BEFORE pushing; use
    # this switch afterwards to confirm what actually reached the remote.
    [switch]$FromGitHub,

    # `refresh` keeps the existing venv and only re-clones and rebuilds, saving
    # the install, which is most of the wall clock. Use it while iterating on
    # .rst content; use `new` (the default) for anything you will act on.
    [ValidateSet("new", "refresh")]
    [string]$Mode = "new",

    # Warnings become errors. Stricter than RTD. See the header note.
    [switch]$Strict,

    # Also run `sphinx -b latex`, the first stage of RTD's pdf build. Catches
    # writer-side pdf failures without perl or a TeX distribution; the latexmk
    # compile that turns the .tex into a .pdf still runs only on RTD. Off by
    # default because it adds a second full sphinx pass.
    [switch]$Tex,

    # Where the throwaway build lives. One directory per ref and interpreter,
    # so a REFACTOR rehearsal and a master rehearsal do not overwrite each
    # other.
    [string]$BuildRoot = "C:\tmp",

    # Port suggested in the serve command printed at the end.
    [int]$Port = 19334
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot

# Per CLAUDE.md: uv's default hardlink mode falls back with a warning on this
# path, so copy mode is the supported choice.
$env:UV_LINK_MODE = "copy"

function Assert-LastExit {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { Write-Error "$What failed (exit $LASTEXITCODE)."; exit $LASTEXITCODE }
}

# ---- Read the RTD config ----------------------------------------------------
# Parsed with pyyaml in a throwaway uv environment rather than by regex here.
# The config is real YAML with nested lists, and a regex that gets it subtly
# wrong is exactly the drift this script exists to prevent. `--no-project`
# stops uv resolving this repo just to run six lines of Python.
$yamlPath = Join-Path $repoRoot ".readthedocs.yaml"
if (-not (Test-Path $yamlPath)) { Write-Error "No .readthedocs.yaml at $yamlPath."; exit 1 }

$parser = @'
import pathlib, sys, yaml
d = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
py = str(((d.get("build") or {}).get("tools") or {}).get("python", ""))
extras = []
for item in ((d.get("python") or {}).get("install") or []):
    if isinstance(item, dict):
        extras += item.get("extra_requirements") or []
conf = str((d.get("sphinx") or {}).get("configuration") or "")
fmts = d.get("formats") or []
if isinstance(fmts, str):
    fmts = [fmts]
print("PYTHON=" + py)
print("EXTRAS=" + ",".join(extras))
print("CONF=" + conf)
print("FORMATS=" + ",".join(str(f) for f in fmts))
'@

Write-Host "Reading .readthedocs.yaml..." -ForegroundColor Cyan
$parsed = uv run --no-project --quiet --with pyyaml python -c $parser $yamlPath
Assert-LastExit "Parsing .readthedocs.yaml"

$cfg = @{}
foreach ($line in $parsed) {
    if ($line -match '^([A-Z]+)=(.*)$') { $cfg[$Matches[1]] = $Matches[2] }
}

$pyVersion = $cfg['PYTHON']
$extras    = @($cfg['EXTRAS'] -split ',' | Where-Object { $_ })
$confFile  = $cfg['CONF']
$formats   = @($cfg['FORMATS'] -split ',' | Where-Object { $_ })

if (-not $pyVersion) { Write-Error "No build.tools.python in .readthedocs.yaml."; exit 1 }
if (-not $confFile)  { Write-Error "No sphinx.configuration in .readthedocs.yaml."; exit 1 }

# Warn on an extra pyproject.toml does not declare. pip only WARNS on an
# unknown extra, so a stale name sits in the config indefinitely doing nothing
# while looking like it does something.
$declared = Select-String -Path (Join-Path $repoRoot "pyproject.toml") `
                          -Pattern '^\s*([a-zA-Z0-9_.\-]+)\s*=\s*\[' -AllMatches |
            ForEach-Object { $_.Matches[0].Groups[1].Value }
foreach ($e in $extras) {
    if ($declared -notcontains $e) {
        Write-Warning ".readthedocs.yaml asks for extra '$e', which pyproject.toml does not declare. pip warns and ignores it, so it is doing nothing."
    }
}

Write-Host ""
Write-Host "RTD configuration in force:" -ForegroundColor Cyan
Write-Host "  python  : $pyVersion"
Write-Host "  extras  : $(if ($extras) { $extras -join ', ' } else { '(none)' })"
Write-Host "  conf    : $confFile"
Write-Host "  formats : $(if ($formats) { $formats -join ', ' } else { 'html only' })"
Write-Host "  ref     : $Ref  (from $(if ($FromGitHub) { 'github.com/mynl/aggregate' } else { 'this repository' }))"
Write-Host ""

# ---- Clone ------------------------------------------------------------------
$safeRef  = $Ref -replace '[\\/:*?"<>|]', '_'
$buildDir = Join-Path $BuildRoot "aggregate_rtd_${safeRef}_$pyVersion"
$venvDir  = Join-Path $buildDir "venv"
$srcDir   = Join-Path $buildDir "src-clone"
$htmlOut  = Join-Path $buildDir "html"
# A plain path here makes git use its local optimization, which prints
# "warning: --depth is ignored in local clones" and copies the whole history.
# Spelling the same path as a `file://` URL sends it through the normal
# transport, so `--depth 1` is honored and the log stays free of a warning that
# is not about anything. Backslashes have to become forward slashes for the URL
# to parse: `file:///T:/worktrees/aggregate_REFACTOR`.
$localUrl = "file:///" + ($repoRoot -replace '\\', '/')
$origin   = if ($FromGitHub) { "https://github.com/mynl/aggregate.git" } else { $localUrl }

if ($Mode -ieq 'new' -and (Test-Path $buildDir)) {
    Write-Host "Removing previous build at $buildDir..." -ForegroundColor Cyan
    Remove-Item -Path $buildDir -Recurse -Force
}
# The clone is always fresh, even on `refresh`. Reusing it would defeat the
# whole point, which is that only committed content takes part.
if (Test-Path $srcDir) { Remove-Item -Path $srcDir -Recurse -Force }
New-Item -Path $buildDir -ItemType Directory -Force | Out-Null

Write-Host "Cloning $Ref (depth 1) into $srcDir..." -ForegroundColor Cyan
git clone --depth 1 --branch $Ref -- $origin $srcDir
Assert-LastExit "git clone"

$sha     = (git -C $srcDir rev-parse --short HEAD).Trim()
$subject = (git -C $srcDir log -1 --format=%s).Trim()
Write-Host "  at $sha  $subject"

# ---- Environment ------------------------------------------------------------
# `uv venv` then `uv pip install`, deliberately NOT `uv sync`: the exact sync
# semantics are precisely what makes the local build unlike RTD's.
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating an empty venv on Python $pyVersion..." -ForegroundColor Cyan
    uv venv $venvDir --python $pyVersion
    Assert-LastExit "uv venv"
}
$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) { Write-Error "No interpreter at $venvPython."; exit 1 }

if ($Mode -ieq 'new') {
    # RTD upgrades the build tooling first, then installs the project. Same
    # order here, and `--no-cache-dir` for the same reason RTD passes it: a
    # cached wheel from an earlier version hides a packaging change.
    Write-Host "Upgrading pip and setuptools..." -ForegroundColor Cyan
    uv pip install --python $venvPython --upgrade --no-cache-dir pip setuptools wheel
    Assert-LastExit "pip and setuptools upgrade"

    $target = if ($extras) { ".[" + ($extras -join ",") + "]" } else { "." }
    Write-Host "Installing $target ..." -ForegroundColor Cyan
    Push-Location $srcDir
    try {
        uv pip install --python $venvPython --upgrade --no-cache-dir $target
        $installExit = $LASTEXITCODE
    } finally { Pop-Location }
    if ($installExit -ne 0) {
        Write-Error "Install failed (exit $installExit). If the message mentions a Python version, build.tools.python in .readthedocs.yaml disagrees with requires-python in pyproject.toml, or with a dependency's own floor."
        exit $installExit
    }
}

# The version stamped on every page comes from the INSTALLED metadata
# (`aggregate.__version__` is `importlib.metadata.version`), not from
# pyproject.toml. Print it: a surprise here is the whole class of bug where a
# bumped pyproject and a stale install disagree.
$builtVersion = (& $venvPython -c "import importlib.metadata as m; print(m.version('aggregate'))").Trim()
Write-Host "  installed aggregate: $builtVersion" -ForegroundColor Green

# ---- Build ------------------------------------------------------------------
# RTD runs sphinx from the directory holding conf.py, with `.` as the source
# dir. Matched here, because conf.py resolves `../src` and its bibliography
# paths relative to that location.
$confDir = Split-Path -Parent (Join-Path $srcDir $confFile)

$sphinxArgs = @('-T', '-b', 'html', '-d', '_build\doctrees', '-D', 'language=en', '.', $htmlOut)
if ($Strict) { $sphinxArgs = @('-W', '--keep-going') + $sphinxArgs }

Write-Host ""
$strictNote = if ($Strict) { " (strict: warnings are errors)" } else { "" }
Write-Host "Building HTML$strictNote..." -ForegroundColor Cyan
Push-Location $confDir
try {
    & $venvPython -m sphinx @sphinxArgs
    $sphinxExit = $LASTEXITCODE
} finally { Pop-Location }

if ($sphinxExit -ne 0) {
    Write-Host ""
    $bar = if ($Strict) { ", though -Strict is a higher bar than RTD applies" } else { "" }
    Write-Error "Sphinx failed (exit $sphinxExit). This is what RTD would report$bar."
    exit $sphinxExit
}

# ---- LaTeX stage (opt-in, -Tex) ----------------------------------------------
# The first half of RTD's pdf build: the same sphinx invocation with the latex
# builder. RTD then compiles the result with latexmk, which needs perl, so the
# compile stays out of reach here; this stage still catches everything the
# latex WRITER can get wrong on the way to the .tex. The doctrees directory is
# shared with the HTML pass, as RTD shares it, so the second pass skips the
# read stage and costs only the write.
$texOut = Join-Path $buildDir "latex"
if ($Tex) {
    $texArgs = @('-T', '-b', 'latex', '-d', '_build\doctrees', '-D', 'language=en', '.', $texOut)
    if ($Strict) { $texArgs = @('-W', '--keep-going') + $texArgs }
    Write-Host ""
    Write-Host "Building LaTeX$strictNote..." -ForegroundColor Cyan
    Push-Location $confDir
    try {
        & $venvPython -m sphinx @texArgs
        $texExit = $LASTEXITCODE
    } finally { Pop-Location }
    if ($texExit -ne 0) {
        Write-Host ""
        Write-Error "The sphinx latex builder failed (exit $texExit). RTD's pdf build dies at this same stage, and a pdf failure fails the whole RTD build, HTML included."
        exit $texExit
    }
}

# ---- Report -----------------------------------------------------------------
Write-Host ""
Write-Host "RTD rehearsal PASSED." -ForegroundColor Green
Write-Host "  ref      : $Ref at $sha"
Write-Host "  python   : $pyVersion"
Write-Host "  version  : $builtVersion"
Write-Host "  html     : $htmlOut"
if ($Tex) {
    Write-Host "  latex    : $texOut"
}
if ($formats -contains 'pdf') {
    Write-Host ""
    if ($Tex) {
        Write-Warning "The latex stage passed, which covers the sphinx half of RTD's pdf build. The latexmk compile itself (perl plus xelatex) still runs only on RTD, so font, unicode, and box errors inside TeX remain unrehearsed."
    } else {
        Write-Warning ".readthedocs.yaml also asks for the pdf format, which this run did not touch. A pdf failure on RTD fails the whole build, HTML included. Re-run with -Tex to rehearse the sphinx latex stage; the latexmk compile itself runs only on RTD."
    }
}
Write-Host ""
Write-Host "To serve locally:" -ForegroundColor Cyan
Write-Host "  python -m http.server $Port --directory `"$htmlOut`""
Write-Host "  Start-Process http://localhost:$Port"
