# Build script converted from batch to PowerShell 7
# Date: 2026-02-14

param (
    [Parameter(Mandatory = $true, HelpMessage = "Python version (e.g., 3.13)")]
    [string]$PythonVersion,

    [ValidateSet("new", "refresh")]
    [string]$Mode = "new"
)

# --- Configuration ---
$ProjectName = "aggregate"
$ProjectRepo = "C:\S\TELOS\Python\aggregate_project"
$BuildDir    = "C:\tmp\${ProjectName}_rtd_build_$PythonVersion"
$VenvDir     = Join-Path $BuildDir "venv"
$HtmlOutputDir = Join-Path $BuildDir "html"
$Port        = 9800

# --- Prepare Environment and Clone Repository ---
if ($Mode -ieq "new") {
    Write-Host "Cleaning previous build directory..." -ForegroundColor Cyan
    if (Test-Path $BuildDir) {
        Remove-Item -Path $BuildDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    New-Item -Path $BuildDir -ItemType Directory -Force

    Write-Host "Cloning repository..." -ForegroundColor Cyan
    git clone --depth 1 $ProjectRepo $BuildDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Git clone failed."
        exit $LASTEXITCODE
    }
} else {
    Write-Host "Reusing existing build directory at '$BuildDir'" -ForegroundColor Yellow
}

# Change location to build directory (Push-Location is the PS equivalent of pushd)
Push-Location $BuildDir

# --- Fetch latest changes ---
Write-Host "Fetching latest changes..." -ForegroundColor Cyan
git fetch origin --force --prune --prune-tags --depth 50 refs/heads/master:refs/remotes/origin/master
if ($LASTEXITCODE -ne 0) {
    Write-Error "Git fetch failed."
    exit $LASTEXITCODE
}

# --- Checkout master branch ---
Write-Host "Checking out master branch..." -ForegroundColor Cyan
git checkout --force origin/master
if ($LASTEXITCODE -ne 0) {
    Write-Error "Git checkout failed."
    exit $LASTEXITCODE
}

# --- Setup Virtual Environment ---
if ($Mode -ieq "new") {
    Write-Host "Creating virtual environment for Python $PythonVersion..." -ForegroundColor Cyan
    uv venv $VenvDir --python $PythonVersion
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit $LASTEXITCODE
    }
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    Write-Error "Virtual environment not found at '$VenvDir'. Run with '-Mode new' first."
    exit 1
}

# Dot-source the PowerShell activation script
. $ActivateScript

if ($Mode -ieq "new") {
    # --- Install Dependencies ---
    Write-Host "Upgrading setuptools..." -ForegroundColor Cyan
    uv pip install --upgrade setuptools
    
    Write-Host "Installing Sphinx..." -ForegroundColor Cyan
    uv pip install --upgrade sphinx

    Write-Host "Installing project dependencies from pyproject.toml..." -ForegroundColor Cyan
    uv pip install --upgrade --no-cache-dir .[dev]
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Dependency installation failed."
        exit $LASTEXITCODE
    }
}

# --- Build HTML Documentation ---
Write-Host "Building HTML documentation..." -ForegroundColor Cyan
# Using python -m sphinx to ensure the venv version is used
python -m sphinx -T -b html -d _build\doctrees -D language=en docs $HtmlOutputDir
if ($LASTEXITCODE -ne 0) {
    Write-Error "HTML build failed."
    exit $LASTEXITCODE
}

Write-Host "`nHTML documentation built successfully in: $HtmlOutputDir" -ForegroundColor Green
Write-Host "To serve: cd '$HtmlOutputDir'; python -m http.server $Port"

# Return to original location
Pop-Location
