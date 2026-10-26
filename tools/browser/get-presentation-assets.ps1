# Download the published candidate packs for local inspection and licence verification.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$destination = Join-Path $projectRoot '.scratch/presentation/incoming'
New-Item -ItemType Directory -Path $destination -Force | Out-Null

$packs = @(
    @{
        File = 'kenney_boardgame-pack.zip'
        Url = 'https://kenney.nl/media/pages/assets/boardgame-pack/a1656828d2-1677667644/kenney_boardgame-pack.zip'
    },
    @{
        File = 'kenney_interface-sounds.zip'
        Url = 'https://kenney.nl/media/pages/assets/interface-sounds/fa43c1dd4d-1677589452/kenney_interface-sounds.zip'
    },
    @{
        File = 'kenney_impact-sounds.zip'
        Url = 'https://kenney.nl/media/pages/assets/impact-sounds/87b4ddecda-1677589768/kenney_impact-sounds.zip'
    },
    @{
        File = 'kenney_ui-pack.zip'
        Url = 'https://kenney.nl/media/pages/assets/ui-pack/f651646eab-1718203990/kenney_ui-pack.zip'
    }
)
foreach ($pack in $packs) {
    $archivePath = Join-Path $destination $pack.File
    if ((Test-Path -LiteralPath $archivePath) -and (Get-Item -LiteralPath $archivePath).Length -gt 0) {
        Write-Host "Keeping existing $($pack.File)."
        continue
    }
    Write-Host "Downloading $($pack.File)..."
    Invoke-WebRequest -Uri $pack.Url -OutFile $archivePath -TimeoutSec 120 -UseBasicParsing
    Write-Host "Saved $archivePath"
}
Write-Host 'Candidate downloads finished. Codex can now inspect the files and licences.'
