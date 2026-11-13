param(
    [ValidateSet('native', 'browser', 'python', 'all')]
    [string]$Profile = 'native',
    [switch]$Gym,
    [switch]$Neural,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonManifest = Get-Content (Join-Path $projectRoot 'adapters/python/pyproject.toml') -Raw
$browserManifest = Get-Content (Join-Path $projectRoot 'adapters/browser/Cargo.toml') -Raw
$venvPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
$missing = New-Object System.Collections.Generic.List[string]

function Get-PinnedVersion([string]$Text, [string]$Name) {
    $match = [regex]::Match($Text, [regex]::Escape($Name) + '==([0-9.]+)')
    if (-not $match.Success) { throw "Missing $Name version in the Python manifest." }
    return $match.Groups[1].Value
}

function Require-Command([string]$Name, [string]$Instructions) {
    if (Get-Command $Name -ErrorAction SilentlyContinue) { return $true }
    $missing.Add("$Name`: $Instructions")
    return $false
}

function Invoke-Tool([string]$Description, [scriptblock]$Command) {
    Write-Host $Description
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Description failed (exit code $LASTEXITCODE)." }
}

function Check-PythonPackage([string]$Name, [string]$Version) {
    if (-not (Test-Path -LiteralPath $venvPython)) { return }
    & $venvPython -c 'import importlib.metadata, sys; sys.exit(importlib.metadata.version(sys.argv[1]) != sys.argv[2])' $Name $Version
    if ($LASTEXITCODE -ne 0) { $missing.Add(".venv: install $Name==$Version with .\tools\setup.ps1 -Profile python") }
}

if (($Gym -or $Neural) -and $Profile -notin @('python', 'all')) {
    throw '-Gym and -Neural require -Profile python or -Profile all.'
}

$needsRust = $Profile -in @('native', 'browser', 'python', 'all')
$needsBrowser = $Profile -in @('browser', 'all')
$needsPython = $Profile -in @('python', 'all')
$rustReady = $true
if ($needsRust) {
    $rustReady = (Require-Command 'cargo' 'Install stable Rust from https://rust-lang.org/tools/install/ and reopen PowerShell.') -and $rustReady
    $rustReady = (Require-Command 'rustc' 'Install stable Rust from https://rust-lang.org/tools/install/ and reopen PowerShell.') -and $rustReady
}
if ($needsBrowser) {
    $rustReady = (Require-Command 'rustup' 'Install Rust through rustup from https://rust-lang.org/tools/install/.') -and $rustReady
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        $missing.Add('python: Install Python and make it available on PATH to run serve-browser.ps1.')
    }
}
if ($needsPython) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        $missing.Add('py: Install CPython 3.14 from https://www.python.org/downloads/windows/ and reopen PowerShell.')
    } else {
        & py -3.14 --version *> $null
        if ($LASTEXITCODE -ne 0) { $missing.Add('Python 3.14: run pymanager install 3.14 in your own terminal.') }
    }
}

if ($Check) {
    if ($needsBrowser -and $rustReady) {
        $targets = & rustup target list --installed
        if ($LASTEXITCODE -ne 0 -or $targets -notcontains 'wasm32-unknown-unknown') {
            $missing.Add('Wasm target: run rustup target add wasm32-unknown-unknown.')
        }
        $wasmVersion = [regex]::Match($browserManifest, 'wasm-bindgen\s*=\s*"=([0-9.]+)"').Groups[1].Value
        if (-not $wasmVersion) { throw 'Missing wasm-bindgen version in the browser manifest.' }
        $bindgen = Get-Command wasm-bindgen -ErrorAction SilentlyContinue
        if (-not $bindgen -or (& wasm-bindgen --version) -notmatch ('\b' + [regex]::Escape($wasmVersion) + '\b')) {
            $missing.Add("wasm-bindgen: run cargo install wasm-bindgen-cli --version $wasmVersion --locked.")
        }
    }
    if ($needsPython) {
        if (-not (Test-Path -LiteralPath $venvPython)) {
            $missing.Add('.venv: run .\tools\setup.ps1 -Profile python to create the project environment.')
        } else {
            & $venvPython -c 'import sys; sys.exit(sys.version_info[:2] != (3, 14))'
            if ($LASTEXITCODE -ne 0) { $missing.Add('.venv: use CPython 3.14 for this project environment.') }
            foreach ($name in @('numpy', 'maturin')) { Check-PythonPackage $name (Get-PinnedVersion $pythonManifest $name) }
            if ($Gym) { Check-PythonPackage 'gymnasium' (Get-PinnedVersion $pythonManifest 'gymnasium') }
            if ($Neural) { Check-PythonPackage 'torch' (Get-PinnedVersion $pythonManifest 'torch') }
            & $venvPython -c 'import tag_rust; print(tag_rust.registered_games())'
            if ($LASTEXITCODE -ne 0) { $missing.Add('Python adapter: run .\tools\setup.ps1 -Profile python to build it.') }
        }
    }
    if ($missing.Count -gt 0) {
        Write-Host 'Missing setup requirements:'
        $missing | ForEach-Object { Write-Host "  $_" }
        exit 1
    }
    Write-Host 'Selected tools and Python packages are available. Build artifacts were not checked.'
    exit 0
}

if ($missing.Count -gt 0) {
    $missing | ForEach-Object { Write-Host "Missing: $_" }
    throw 'Install the missing host tools in your own terminal, then rerun setup.'
}

Push-Location -LiteralPath $projectRoot
try {
    if ($Profile -in @('native', 'all')) {
        Invoke-Tool 'Building native games and agents...' { cargo build --release --features mcts-json,replay,inference }
        Invoke-Tool 'Checking a native match...' { cargo run --release --features mcts-json,replay --example match_play -- examples/matches/random-vs-osla.json }
    }
    if ($needsBrowser) {
        $wasmVersion = [regex]::Match($browserManifest, 'wasm-bindgen\s*=\s*"=([0-9.]+)"').Groups[1].Value
        if (-not $wasmVersion) { throw 'Missing wasm-bindgen version in the browser manifest.' }
        $targets = & rustup target list --installed
        if ($LASTEXITCODE -ne 0) { throw 'Could not read installed Rust targets.' }
        if ($targets -notcontains 'wasm32-unknown-unknown') {
            Invoke-Tool 'Installing the Wasm target...' { rustup target add wasm32-unknown-unknown }
        }
        $bindgen = Get-Command wasm-bindgen -ErrorAction SilentlyContinue
        if (-not $bindgen -or (& wasm-bindgen --version) -notmatch ('\b' + [regex]::Escape($wasmVersion) + '\b')) {
            Invoke-Tool 'Installing the matching wasm-bindgen CLI...' { cargo install wasm-bindgen-cli --version $wasmVersion --locked --force }
        }
        Invoke-Tool 'Building the browser adapter...' { & (Join-Path $projectRoot 'tools/browser/build.ps1') }
        Write-Host 'To play, run .\serve-browser.ps1 and open the printed address.'
    }
    if ($needsPython) {
        if (-not (Test-Path -LiteralPath $venvPython)) {
            Invoke-Tool 'Creating .venv with Python 3.14...' { py -3.14 -m venv .venv }
        }
        & $venvPython -c 'import sys; sys.exit(sys.version_info[:2] != (3, 14))'
        if ($LASTEXITCODE -ne 0) { throw '.venv must use CPython 3.14. Move or remove the existing environment yourself before retrying.' }
        foreach ($name in @('numpy', 'maturin')) {
            $version = Get-PinnedVersion $pythonManifest $name
            Invoke-Tool "Installing $name==$version..." { & $venvPython -m pip install "$name==$version" }
        }
        if ($Gym) {
            $version = Get-PinnedVersion $pythonManifest 'gymnasium'
            Invoke-Tool "Installing gymnasium==$version..." { & $venvPython -m pip install "gymnasium==$version" }
        }
        if ($Neural) {
            $version = Get-PinnedVersion $pythonManifest 'torch'
            Invoke-Tool "Installing CPU PyTorch==$version..." { & $venvPython -m pip install "torch==$version" --index-url https://download.pytorch.org/whl/cpu }
        }
        $env:VIRTUAL_ENV = (Resolve-Path .venv).Path
        $env:PYO3_PYTHON = $venvPython
        Invoke-Tool 'Building and installing the Python adapter...' { & $venvPython -m maturin develop --release --manifest-path adapters/python/Cargo.toml }
        Invoke-Tool 'Checking the Python adapter import...' { & $venvPython -c 'import tag_rust; print(tag_rust.registered_games())' }
    }
} finally {
    Pop-Location
}
