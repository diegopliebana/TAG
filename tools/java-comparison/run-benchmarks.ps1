param(
    [ValidateRange(1, 10000)][int]$Passes = 500,
    [ValidateRange(1, 10)][int]$Forks = 3,
    [string]$OutputDirectory = '.scratch/java-comparison/benchmark-results'
)

$benchmarkWorkspace = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
Set-Location -LiteralPath $benchmarkWorkspace
$benchmarkOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $benchmarkWorkspace $OutputDirectory))
if (-not $benchmarkOutputRoot.StartsWith($benchmarkWorkspace + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Output directory must be inside the project workspace.'
}
$benchmarkRoot = Join-Path $benchmarkWorkspace '.scratch/java-comparison'
$benchmarkDependencyFile = Join-Path $benchmarkRoot 'target/comparison-classpath.txt'
if (-not (Test-Path -LiteralPath $benchmarkDependencyFile)) {
    throw 'The disposable Java build and dependency classpath are required; see README.md.'
}
$benchmarkDependencies = (Get-Content -LiteralPath $benchmarkDependencyFile -Raw).Trim()
$benchmarkClasspath = "$benchmarkRoot/target/classes;$benchmarkDependencies"

cargo build --offline --release --example java_comparison_measure
if ($LASTEXITCODE -ne 0) { throw 'Rust benchmark build failed.' }
javac -cp $benchmarkClasspath -d "$benchmarkRoot/target/classes" tools/java-comparison/BenchmarkRules.java
if ($LASTEXITCODE -ne 0) { throw 'Java benchmark compilation failed.' }
New-Item -ItemType Directory -Path $benchmarkOutputRoot -Force | Out-Null

for ($benchmarkFork = 1; $benchmarkFork -le $Forks; $benchmarkFork++) {
    $benchmarkOrder = if ($benchmarkFork % 2 -eq 0) { @('java', 'rust') } else { @('rust', 'java') }
    foreach ($benchmarkEngine in $benchmarkOrder) {
        $benchmarkOutput = Join-Path $benchmarkOutputRoot "$benchmarkEngine-fork-$benchmarkFork.tsv"
        if ($benchmarkEngine -eq 'rust') {
            & target/release/examples/java_comparison_measure.exe $Passes | Out-File -FilePath $benchmarkOutput -Encoding utf8
        } else {
            & java -Xms256m -Xmx512m -cp $benchmarkClasspath BenchmarkRules tools/java-comparison/benchmark-corpus.tsv $Passes | Out-File -FilePath $benchmarkOutput -Encoding utf8
        }
        if ($LASTEXITCODE -ne 0) { throw "$benchmarkEngine fork $benchmarkFork failed; inspect $benchmarkOutput" }
        Write-Output "Completed $benchmarkEngine fork $benchmarkFork"
    }
}
Write-Output "Saved benchmark results to $benchmarkOutputRoot"
