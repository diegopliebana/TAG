$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location -LiteralPath $projectRoot
try {
    & cargo build --offline --release -p tag-browser --target wasm32-unknown-unknown
    if ($LASTEXITCODE -ne 0) { throw 'Browser Rust build failed.' }
    & wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/tag_browser.wasm
    if ($LASTEXITCODE -ne 0) { throw 'Browser binding generation failed.' }
} finally { Pop-Location }
