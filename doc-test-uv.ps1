# =============================================================================
# doc-test-uv.ps1 — build Sphinx HTML docs locally via uv
#
# This is the slim, uv-managed replacement for doc-test.ps1. Use it to sanity-
# check that the docs build cleanly *before* merging to master and triggering
# the Read-the-Docs (RTD) build.
#
# WHAT CHANGED VS. doc-test.ps1
# -----------------------------
# The original script went to a lot of trouble:
#   • cleaned and recreated C:\tmp\<project>_rtd_build_<pyver>\
#   • git-cloned the repo into it (depth 1)
#   • fetched + checked out origin/master
#   • created a venv with `uv venv` and activated it
#   • ran `uv pip install setuptools sphinx .[dev]` by hand
#   • finally invoked `python -m sphinx ...`
# All of that was a hand-rolled approximation of an RTD-clean environment.
#
# With `uv` it isn't needed:
#   • the project's pyproject.toml already declares Sphinx and friends under
#     the `[dev]` extra, so `uv sync --all-extras` brings them in;
#   • `uv run sphinx-build` invokes Sphinx from that venv with zero activation
#     ceremony;
#   • the build runs in place against your current working tree — no clone
#     necessary. RTD itself does a clean clone on its own infrastructure, so
#     replicating that locally was always belt-and-braces.
#
# If you want the "clean-clone" guarantee, run `git stash && this script`
# first, or run it from a temporary worktree. The 99% case is "did my latest
# .rst edits break the build?", which doesn't need that.
#
# WHERE THE VENV LIVES
# --------------------
# This script uses a doc-specific environment in `.doc-venv\` (set via the
# UV_PROJECT_ENVIRONMENT env var below). That keeps it separate from your
# main `.venv` — so passing a different `-PythonVersion` here will NOT clobber
# whatever Python your day-to-day dev venv is using.
#
# Add `.doc-venv/` to your local .gitignore if it isn't already.
#
# USAGE EXAMPLES
# --------------
#   .\doc-test-uv.ps1                          # build with default Python (3.13)
#   .\doc-test-uv.ps1 -PythonVersion 3.12      # build under a specific Python
#   .\doc-test-uv.ps1 -Clean                   # wipe doctrees + html first
#   .\doc-test-uv.ps1 -NoSync                  # skip dependency sync (fast iteration)
#   .\doc-test-uv.ps1 -Lenient                 # don't abort on Sphinx errors;
#                                              # let nbsphinx render error cells
#                                              # instead of failing the build
#   .\doc-test-uv.ps1 -Clean -Lenient          # typical first-pass after a big
#                                              # refactor — shows everything
#   .\doc-test-uv.ps1 -Clean -PythonVersion 3.14
#   .\doc-test-uv.ps1 -Format pdf              # PDF via LaTeX + tectonic, to
#                                              # docs\_build\latex\aggregate.pdf
#   .\doc-test-uv.ps1 -Format html,pdf         # both, one sync, one pass each
#   .\doc-test-uv.ps1 -Format text             # plain-text build, output to
#                                              # docs\_build\text (handy for
#                                              # cross-branch numerical diffs)
#   .\doc-test-uv.ps1 -Text -Lenient `
#       -OutputDir T:\doc-diff\agg-doc-diff\text
#                                              # text build into a custom dir,
#                                              # warnings non-fatal
#
# PDF OUTPUT
# ----------
# `-Format pdf` runs Sphinx's `latex` builder and then compiles the generated
# `aggregate.tex` with **tectonic**, not latexmk. latexmk is a Perl script and
# there is no Perl on this machine, so both `sphinx-build -M latexpdf` and the
# generated `make.bat` fail with "MiKTeX could not find the script engine
# 'perl'". tectonic is a single self-contained binary that runs the rerun loop
# itself and fetches whatever packages it needs, so it needs no MiKTeX package
# juggling either.
#
# Two conf.py settings are load-bearing for the PDF and must not be dropped:
# `verbatimforcewraps=true` (long ipython output lines) and the `enumitem`
# preamble raising the list-depth cap. Both are commented at their definitions.
#
# When the build finishes, the script prints the command to serve the result
# locally — just copy-paste.
# =============================================================================

param(
    # Python version for the doc build. uv will use an existing install if it
    # has one; otherwise uv downloads the requested Python automatically (one
    # of the nicer uv features — no manual pyenv juggling). Project's
    # pyproject.toml says requires-python = ">=3.10".
    [string]$PythonVersion = "3.13",

    # Which outputs to build. Accepts more than one: `-Format html,pdf` runs a
    # single dependency sync and then one Sphinx pass per format. Each format
    # gets its own output dir and its own doctrees cache, so they never tread
    # on each other.
    [ValidateSet('html', 'pdf', 'text')]
    [string[]]$Format = @('html'),

    # Where to write the build. Only meaningful for a single -Format; with
    # several, each uses its own default (docs\_build\{html,latex,text}).
    [string]$OutputDir = "docs\_build\html",

    # Skip the `uv sync` step. Use when iterating on .rst edits and you know
    # the doc venv is already up to date — shaves a few seconds.
    [switch]$NoSync,

    # Remove the output dir AND the doctrees cache before building.
    # Equivalent to `make clean && make html`. Use this when you suspect a
    # stale doctrees cache is hiding a real problem.
    [switch]$Clean,

    # Don't abort on Sphinx warnings/errors. Passes `--keep-going` to
    # sphinx-build so it collects every warning instead of stopping at the
    # first; also sets `nbsphinx_allow_errors=1` so a notebook cell that
    # raises an exception renders the traceback inline instead of failing
    # the build. Typical use: first pass after a big refactor when you
    # expect lots of broken cross-references and stale examples.
    [switch]$Lenient,

    # Back-compat alias for `-Format text`, kept so existing muscle memory and
    # any saved command lines keep working. Prefer -Format for new use.
    [switch]$Text,

    # Port to suggest when printing the local-serve command at the end.
    [int]$Port = 19333
)

$ErrorActionPreference = 'Stop'

# ---- Builder selection ------------------------------------------------------
# `-Text` is the old spelling of `-Format text`; honour it unless the caller
# also passed -Format explicitly.
if ($Text -and -not $PSBoundParameters.ContainsKey('Format')) { $Format = @('text') }
$Format = $Format | Select-Object -Unique

# Per-format Sphinx builder, output dir and doctrees cache. The caches are kept
# separate so a text or latex pass never invalidates the HTML one. -OutputDir
# overrides the default only when a single format was asked for; with several
# there is no single directory it could sensibly mean.
$plan = @{
    html = @{ Builder = 'html';  Out = 'docs\_build\html';  Doctrees = 'docs\_build\doctrees' }
    pdf  = @{ Builder = 'latex'; Out = 'docs\_build\latex'; Doctrees = 'docs\_build\doctrees-latex' }
    text = @{ Builder = 'text';  Out = 'docs\_build\text';  Doctrees = 'docs\_build\doctrees-text' }
}
if ($PSBoundParameters.ContainsKey('OutputDir')) {
    if ($Format.Count -eq 1) {
        $plan[$Format[0]].Out = $OutputDir
    } else {
        Write-Warning "-OutputDir ignored: it cannot apply to $($Format.Count) formats at once. Using the defaults."
    }
}

# tectonic compiles the PDF (see the header note on why not latexmk). Fail here
# with something readable rather than deep inside a LaTeX run.
if ($Format -contains 'pdf' -and -not (Get-Command tectonic -ErrorAction SilentlyContinue)) {
    Write-Error ("PDF output needs 'tectonic' on PATH and it was not found. " +
                 "Install from https://tectonic-typesetting.github.io/, or drop 'pdf' from -Format.")
    exit 1
}

# ---- uv environment knobs ---------------------------------------------------
# Per CLAUDE.md: the repo lives on a path where uv's default hardlink mode
# falls back with a warning. Copy mode is the supported choice here.
$env:UV_LINK_MODE = "copy"

# Tell uv to use a doc-specific venv directory instead of the default `.venv`.
# Both `uv sync` and `uv run` honour this env var. Result: doc builds and
# day-to-day development stay out of each other's way, and they can use
# different Python versions independently.
$env:UV_PROJECT_ENVIRONMENT = ".doc-venv"

# ---- Optional clean --------------------------------------------------------
# Only the requested formats are cleaned, so `-Format pdf -Clean` does not
# throw away an HTML build you still wanted.
if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Cyan
    foreach ($f in $Format) {
        foreach ($p in @($plan[$f].Out, $plan[$f].Doctrees)) {
            if (-not (Test-Path $p)) { continue }
            # Empty the directory rather than delete it. Something holding a
            # handle on the folder itself (an editor, a file watcher, a stray
            # http.server, the indexer) makes a recursive delete of the folder
            # fail, and that should not abort the build: emptying it achieves
            # the same thing, and a leftover file is not fatal either.
            try {
                Remove-Item -Path (Join-Path $p '*') -Recurse -Force -ErrorAction Stop
                Write-Host "  emptied $p"
            } catch {
                Write-Warning "could not fully clean $p ($($_.Exception.Message.Split([Environment]::NewLine)[0])); continuing"
            }
        }
    }
}

# ---- Sync dependencies into the doc venv -----------------------------------
# `uv sync --all-extras` installs every `[project.optional-dependencies]` block
# of pyproject.toml, plus the project itself in editable mode. The doc build
# needs `dev` (Sphinx, myst-parser, nbsphinx, sphinx-design, sphinx-rtd-theme),
# but it must be `--all-extras`, not `--extra dev`.
#
# `uv sync` is an EXACT sync: naming a subset of extras makes uv PRUNE the
# packages belonging to the unselected ones. The optional deps here are split
# five ways (`dev` / `notebook` / `numba` / `massive` / `viz`), so `--extra dev`
# deletes `zarr`, `holoviews`, `datashader` and `numba`. Any page that imports
# them then fails to build, and the pruning churns the venv on every run.
# `--all-extras` selects them all, so nothing gets pruned. See CLAUDE.md.
#
# `--python 3.X` pins the venv's interpreter. If the doc venv already uses
# that version this is essentially a no-op (uv just verifies the lockfile);
# if not, uv recreates `.doc-venv\` with the requested Python.
if (-not $NoSync) {
    Write-Host "Syncing dependencies (--all-extras) under Python $PythonVersion (into .doc-venv\)..." -ForegroundColor Cyan
    uv sync --all-extras --python $PythonVersion
    if ($LASTEXITCODE -ne 0) {
        Write-Error "uv sync failed."
        exit $LASTEXITCODE
    }
}

# ---- Build HTML docs --------------------------------------------------------
# Same Sphinx flags RTD uses internally:
#   -T               show full tracebacks on errors (useful when an extension
#                    blows up during doctree generation)
#   -b html          HTML builder
#   -d <dir>         doctrees cache directory
#   -D language=en   set conf.py's language to English
# `docs` is the source directory; $OutputDir is where the .html files land.
#
# `uv run sphinx-build` finds Sphinx in `.doc-venv` and invokes it — no
# manual activation needed. The same command works on Windows, macOS, Linux.
#
# -Lenient prepends `--keep-going` (collect all warnings, don't stop at the
# first) and `-D nbsphinx_allow_errors=1` (notebook cell exceptions render
# inline instead of aborting). It also sets the AGG_DOCS_LENIENT env var,
# which docs/conf.py reads to monkey-patch the IPython sphinx directive
# (``.. ipython::`` blocks) into permissive mode — exceptions in those
# blocks render as inline tracebacks instead of aborting the build, since
# the directive has no global ``okexcept`` config knob. jupyter-sphinx
# already renders cell errors inline by default, so nothing extra needed
# there.
if ($Lenient) { $env:AGG_DOCS_LENIENT = "1" }

$results = @()
foreach ($f in $Format) {
    $builder     = $plan[$f].Builder
    $outDir      = $plan[$f].Out
    $doctreesDir = $plan[$f].Doctrees

    $sphinxArgs = @('-T', '-b', $builder,
                    '-d', $doctreesDir,
                    '-D', 'language=en',
                    'docs', $outDir)
    if ($Lenient) {
        $sphinxArgs = @('--keep-going', '-D', 'nbsphinx_allow_errors=1') + $sphinxArgs
    }

    Write-Host "`nBuilding $f documentation (sphinx -b $builder)..." -ForegroundColor Cyan
    uv run sphinx-build @sphinxArgs
    $sphinxExit = $LASTEXITCODE

    if ($sphinxExit -ne 0) {
        if ($Lenient) {
            Write-Warning "Sphinx reported warnings/errors for $f (exit $sphinxExit); continuing because -Lenient is set."
        } else {
            Write-Error "$f build failed."
            exit $sphinxExit
        }
    }

    # PDF is a two-stage build: Sphinx writes aggregate.tex above, tectonic
    # turns it into the PDF here. tectonic runs its own rerun loop, so one
    # invocation resolves the TOC and cross-references.
    if ($f -eq 'pdf') {
        $tex = Join-Path $outDir 'aggregate.tex'
        if (-not (Test-Path $tex)) {
            Write-Error "Expected $tex but the latex builder did not write it."
            exit 1
        }
        Write-Host "Compiling PDF with tectonic..." -ForegroundColor Cyan
        Push-Location $outDir
        try {
            $tectonicArgs = @('aggregate.tex')
            # -Lenient lets a recoverable LaTeX error through so a PDF still
            # lands; without it a LaTeX error stops the run, like Sphinx's.
            if ($Lenient) { $tectonicArgs = @('-Z', 'continue-on-errors') + $tectonicArgs }
            tectonic @tectonicArgs
            $tectonicExit = $LASTEXITCODE
        } finally {
            Pop-Location
        }
        if ($tectonicExit -ne 0) {
            Write-Error "tectonic failed (exit $tectonicExit). See $outDir\aggregate.log."
            exit $tectonicExit
        }
        $results += [pscustomobject]@{ Format = $f; Path = (Join-Path $outDir 'aggregate.pdf') }
        continue
    }

    $results += [pscustomobject]@{ Format = $f; Path = $outDir }
}

# ---- Done -------------------------------------------------------------------
Write-Host ""
foreach ($r in $results) {
    Write-Host "$($r.Format) documentation built successfully in: $($r.Path)" -ForegroundColor Green
}
if ($Format -contains 'html') {
    $htmlOut = $plan['html'].Out
    Write-Host ""
    Write-Host "To serve locally and open in a browser:" -ForegroundColor Cyan
    Write-Host "  uv run python -m http.server $Port --directory $htmlOut"
    Write-Host "  Start-Process http://localhost:$Port"
    Write-Host ""
    Write-Host "(Ctrl-C in the serving terminal to stop the server.)"
}
