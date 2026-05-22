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
#     the `[dev]` extra, so `uv sync --extra dev` brings them in;
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
#   .\doc-test-uv.ps1 -Clean -PythonVersion 3.14
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

    # Where to write the built HTML. Default mirrors the `make html` layout
    # so you can also use `cd docs && make html` if you prefer.
    [string]$OutputDir = "docs\_build\html",

    # Skip the `uv sync` step. Use when iterating on .rst edits and you know
    # the doc venv is already up to date — shaves a few seconds.
    [switch]$NoSync,

    # Remove the output dir AND the doctrees cache before building.
    # Equivalent to `make clean && make html`. Use this when you suspect a
    # stale doctrees cache is hiding a real problem.
    [switch]$Clean,

    # Port to suggest when printing the local-serve command at the end.
    [int]$Port = 9800
)

$ErrorActionPreference = 'Stop'

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
if ($Clean) {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Cyan
    foreach ($p in @($OutputDir, "docs\_build\doctrees")) {
        if (Test-Path $p) {
            Remove-Item -Path $p -Recurse -Force
            Write-Host "  removed $p"
        }
    }
}

# ---- Sync dev dependencies into the doc venv -------------------------------
# `uv sync --extra dev` installs everything in the `[project.optional-
# dependencies] dev` block of pyproject.toml — Sphinx, myst-parser, nbsphinx,
# sphinx-design, sphinx-rtd-theme, etc. — plus the project itself in editable
# mode.
#
# `--python 3.X` pins the venv's interpreter. If the doc venv already uses
# that version this is essentially a no-op (uv just verifies the lockfile);
# if not, uv recreates `.doc-venv\` with the requested Python.
if (-not $NoSync) {
    Write-Host "Syncing dev dependencies under Python $PythonVersion (into .doc-venv\)..." -ForegroundColor Cyan
    uv sync --extra dev --python $PythonVersion
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
Write-Host "Building HTML documentation..." -ForegroundColor Cyan
uv run sphinx-build -T -b html -d docs\_build\doctrees -D language=en docs $OutputDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "HTML build failed."
    exit $LASTEXITCODE
}

# ---- Done -------------------------------------------------------------------
Write-Host "`nHTML documentation built successfully in: $OutputDir" -ForegroundColor Green
Write-Host ""
Write-Host "To serve locally and open in a browser:" -ForegroundColor Cyan
Write-Host "  uv run python -m http.server $Port --directory $OutputDir"
Write-Host "  Start-Process http://localhost:$Port"
Write-Host ""
Write-Host "(Ctrl-C in the serving terminal to stop the server.)"
