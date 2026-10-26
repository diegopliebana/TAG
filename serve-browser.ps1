param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8080
)

$ErrorActionPreference = 'Stop'
$webDirectory = Join-Path $PSScriptRoot 'web'

Write-Host "Open http://127.0.0.1:$Port in your browser."
Write-Host 'Leave this terminal running. Press Ctrl+C to stop the server.'

& python -m http.server $Port --bind 127.0.0.1 --directory $webDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Browser server exited with code $LASTEXITCODE."
}
