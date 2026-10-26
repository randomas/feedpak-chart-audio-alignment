[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$MaterialsRoot = "",

    [Parameter(Position = 1)]
    [string]$OutputRoot = "",

    [ValidateSet(
        "nominal",
        "offset",
        "linear",
        "dtw",
        "dtw-checkpoint",
        "checkpoint-linear",
        "checkpoint-dtw-diagnostic",
        "checkpoint-dtw-selective",
        "continuous-anchor-linear",
        "continuous-anchor-dtw",
        "continuous-anchor-structural",
        "bar-landmark-regularized"
    )]
    [string]$AlignmentMode = "continuous-anchor-dtw",

    [ValidateSet("tempos", "beats", "both")]
    [string]$TimelineMode = "both",

    [int]$CheckpointMeasures = 4,
    [double]$CheckpointSearchRadius = 1.0,

    [switch]$IncludeVocals,
    [switch]$ReuseVocals,
    [string]$HFToken = $env:HF_TOKEN,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cpu",

    [switch]$SkipUnitTests,
    [switch]$KeepGoing = $true,
    [switch]$AllowSevereAlignment,
    [switch]$BuildUnsafeContinuousComparison,
    [switch]$Phase3DVariants,
    [ValidateRange(0, 10)] [int]$Phase3DSafeCandidates = 2,
    [ValidateRange(0, 10)] [int]$Phase3DStructuralCandidates = 1,
    [switch]$Force,
    [switch]$StructuralSecondPass,
    [switch]$TranscriptionInvestigator,
    [switch]$ForceMidiAuthority,
    [switch]$LightDiagnostics,

    [string[]]$Only = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Builder = Join-Path $ProjectRoot "build_feedpak.py"

if (-not (Test-Path -LiteralPath $Builder -PathType Leaf)) {
    throw "build_feedpak.py was not found beside this script: $Builder"
}

$transcriptionMethods = @(
    "dtw-checkpoint",
    "checkpoint-linear",
    "checkpoint-dtw-diagnostic",
    "checkpoint-dtw-selective",
    "continuous-anchor-linear",
    "continuous-anchor-dtw",
    "continuous-anchor-structural",
    "bar-landmark-regularized"
)
if ($ForceMidiAuthority -and -not $TranscriptionInvestigator) { throw "-ForceMidiAuthority requires -TranscriptionInvestigator" }
if ($ForceMidiAuthority -and -not $StructuralSecondPass) { throw "-ForceMidiAuthority requires -StructuralSecondPass" }
if ($TranscriptionInvestigator -and $transcriptionMethods -notcontains $AlignmentMode) {
    throw "-TranscriptionInvestigator is not available for alignment mode '$AlignmentMode'. Supported methods: $($transcriptionMethods -join ', ')"
}

if ([string]::IsNullOrWhiteSpace($MaterialsRoot)) {
    $candidate = Join-Path $ProjectRoot "test_materials"
    if (Test-Path -LiteralPath $candidate -PathType Container) {
        $MaterialsRoot = $candidate
    }
    else {
        throw "No MaterialsRoot was supplied and '$candidate' does not exist. Pass the folder that contains the existing test-song directories."
    }
}

$MaterialsRoot = (Resolve-Path -LiteralPath $MaterialsRoot).Path

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "test_outputs"
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $OutputRoot "run_$RunStamp"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
$DiagnosticsRoot = Join-Path $RunRoot $(if ($LightDiagnostics) { "Diagnostics_light" } else { "Diagnostics" })
New-Item -ItemType Directory -Path $DiagnosticsRoot -Force | Out-Null
$LogRoot = Join-Path $DiagnosticsRoot "logs"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
$DiagnosticsCollector = Join-Path $ProjectRoot "collect_run_diagnostics.py"
if (-not (Test-Path -LiteralPath $DiagnosticsCollector -PathType Leaf)) {
    throw "collect_run_diagnostics.py was not found beside this script: $DiagnosticsCollector"
}

function Get-SafeName([string]$Name) {
    $safe = $Name -replace '[^A-Za-z0-9._-]+', '_'
    if ([string]::IsNullOrWhiteSpace($safe)) { return "material" }
    return $safe.Trim('_')
}

function Test-NameSelected([System.IO.DirectoryInfo]$Directory) {
    if ($Only.Count -eq 0) { return $true }
    foreach ($pattern in $Only) {
        if ($Directory.Name -like $pattern -or $Directory.FullName -like $pattern) {
            return $true
        }
    }
    return $false
}

function Get-TestMaterialFolders {
    $chartExtensions = @(".gp3", ".gp4", ".gp5", ".gp")
    $audioExtensions = @(".ogg", ".wav", ".flac")

    Get-ChildItem -LiteralPath $MaterialsRoot -Directory -Recurse |
        Where-Object {
            $directory = $_
            $files = @(Get-ChildItem -LiteralPath $directory.FullName -File)
            $hasChart = @($files | Where-Object { $chartExtensions -contains $_.Extension.ToLowerInvariant() }).Count -gt 0
            $hasAudio = @($files | Where-Object { $audioExtensions -contains $_.Extension.ToLowerInvariant() }).Count -gt 0
            $hasChart -and $hasAudio -and (Test-NameSelected $directory)
        } |
        Sort-Object FullName -Unique
}

function Protect-SensitiveText {
    param(
        [AllowNull()]
        [string]$Text,
        [AllowNull()]
        [string[]]$Secrets = @()
    )

    if ($null -eq $Text) { return $Text }
    $safe = $Text
    foreach ($secret in @($Secrets)) {
        if (-not [string]::IsNullOrWhiteSpace($secret)) {
            $safe = $safe.Replace($secret, "<redacted>")
        }
    }
    # Defense in depth for common Hugging Face token forms, even when a token
    # reaches output through a dependency rather than the command formatter.
    $safe = [regex]::Replace($safe, '(?i)\b(hf_[A-Za-z0-9]{8,})\b', '<redacted-hf-token>')
    $safe = [regex]::Replace($safe, '(?i)(--hf-token\s+)("[^"]+"|\S+)', '$1<redacted>')
    return $safe
}

function Invoke-LoggedProcess {
    param(
        [Parameter(Mandatory)]
        [string]$Executable,

        [Parameter(Mandatory)]
        [string[]]$Arguments,

        [Parameter(Mandatory)]
        [string]$LogPath
    )

    $displayArguments = [System.Collections.Generic.List[string]]::new()
    for ($i = 0; $i -lt $Arguments.Count; $i++) {
        $value = $Arguments[$i]
        if ($i -gt 0 -and $Arguments[$i - 1] -eq "--hf-token") {
            $value = "<redacted>"
        }
        $value = Protect-SensitiveText -Text $value -Secrets @($HFToken)
        if ($value -match '\s') { $value = '"' + $value + '"' }
        $displayArguments.Add($value)
    }
    $printable = @($Executable) + @($displayArguments)
    $header = Protect-SensitiveText -Text ("COMMAND: " + ($printable -join " ")) -Secrets @($HFToken)

    [System.IO.File]::WriteAllText(
        $LogPath,
        $header + [Environment]::NewLine,
        $utf8
    )

    # Native applications legitimately write traceback and diagnostic lines
    # to stderr. Do not let PowerShell convert the first stderr line into a
    # terminating error and discard the remainder.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    try {
        & $Executable @Arguments 2>&1 |
            ForEach-Object {
                $line = Protect-SensitiveText -Text $_.ToString() -Secrets @($HFToken)

                Write-Host $line

                [System.IO.File]::AppendAllText(
                    $LogPath,
                    $line + [Environment]::NewLine,
                    $utf8
                )
            }

        $processExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    return $processExitCode
}

Write-Host "Project:        $ProjectRoot" -ForegroundColor Cyan
Write-Host "Test materials: $MaterialsRoot" -ForegroundColor Cyan
Write-Host "Run output:     $RunRoot" -ForegroundColor Cyan
Write-Host "Alignment mode: $AlignmentMode" -ForegroundColor Cyan
Write-Host "Vocals:         $($IncludeVocals.IsPresent)" -ForegroundColor Cyan
Write-Host ""

if (-not $SkipUnitTests) {
    Write-Host "Running unit/regression tests..." -ForegroundColor Yellow
    $testLog = Join-Path $LogRoot "pytest.log"
    $testCode = Invoke-LoggedProcess -Executable "python" -Arguments @("-m", "pytest", "-q") -LogPath $testLog
    if ($testCode -ne 0) {
        throw "Unit tests failed with exit code $testCode. See $testLog"
    }
    Write-Host "Unit tests passed." -ForegroundColor Green
    Write-Host ""
}

$Materials = @(Get-TestMaterialFolders)
if ($Materials.Count -eq 0) {
    throw "No test-material folders containing both a Guitar Pro chart and audio were found under $MaterialsRoot"
}

Write-Host "Found $($Materials.Count) test material folder(s)." -ForegroundColor Green
$Materials | ForEach-Object { Write-Host "  $($_.FullName)" }
Write-Host ""

$Results = [System.Collections.Generic.List[object]]::new()
$index = 0

foreach ($Material in $Materials) {
    $index++
    $safeName = Get-SafeName $Material.Name
    $materialOutput = Join-Path $RunRoot ("{0:D2}_{1}" -f $index, $safeName)
    $logPath = Join-Path $LogRoot ("{0:D2}_{1}.log" -f $index, $safeName)

    $modernCharts = @(Get-ChildItem -LiteralPath $Material.FullName -File -Filter "*.gp")
    $legacyCharts = @(Get-ChildItem -LiteralPath $Material.FullName -File |
        Where-Object { $_.Extension.ToLowerInvariant() -in @(".gp3", ".gp4", ".gp5") })

    # Prefer a genuine modern .gp chart when both modern and legacy files exist.
    $parser = if ($modernCharts.Count -gt 0) { "alphatab" } else { "gp5" }
    $selectedChart = if ($parser -eq "alphatab") { $modernCharts[0].FullName } else { $legacyCharts[0].FullName }
    Write-Host "  Parser selected: $parser" -ForegroundColor Cyan
    Write-Host "  Chart selected:  $selectedChart" -ForegroundColor Cyan

    if ($Force -and (Test-Path -LiteralPath $materialOutput)) {
        Remove-Item -LiteralPath $materialOutput -Recurse -Force
    }
    New-Item -ItemType Directory -Path $materialOutput -Force | Out-Null

    $arguments = @(
        $Builder,
        $Material.FullName,
        $materialOutput,
        "--alignment-mode", $AlignmentMode,
        "--checkpoint-measures", $CheckpointMeasures.ToString(),
        "--checkpoint-search-radius", $CheckpointSearchRadius.ToString([Globalization.CultureInfo]::InvariantCulture),
        "--timeline-mode", $TimelineMode,
        "--keep-work-dir",
        "--device", $Device
    )

    if (-not $IncludeVocals) {
        $arguments += "--skip-vocals"
    }
    elseif ($ReuseVocals) {
        $arguments += "--reuse-vocals"
    }
    if ($IncludeVocals -and -not [string]::IsNullOrWhiteSpace($HFToken)) {
        $arguments += @("--hf-token", $HFToken)
    }

    if ($AllowSevereAlignment) {
        $arguments += "--allow-severe-alignment"
    }
	
    if ($BuildUnsafeContinuousComparison) { $arguments += "--build-unsafe-continuous-comparison" }
    if ($Phase3DVariants) { $arguments += @("--phase3d-variants","--phase3d-safe-candidates",$Phase3DSafeCandidates.ToString(),"--phase3d-structural-candidates",$Phase3DStructuralCandidates.ToString()) }
    if ($StructuralSecondPass) { $arguments += "--structural-second-pass" }
    if ($TranscriptionInvestigator) { $arguments += "--transcription-investigator" }
    if ($ForceMidiAuthority) { $arguments += "--force-midi-authority" }

    if ($parser -eq "alphatab") {
        $arguments += @("--gp-parser", "alphatab")
        $scoreJson = @(Get-ChildItem -LiteralPath $Material.FullName -File |
            Where-Object { $_.Name -like "*.alphatab.json" -or $_.Name -like "*schema4*.json" } |
            Select-Object -First 1)
        if ($scoreJson.Count -gt 0) {
            $arguments += @("--score-json", $scoreJson[0].FullName)
        }
        else {
            $extractor = Join-Path $ProjectRoot "alphatab-extractor\extract-score.mjs"
            if (-not (Test-Path -LiteralPath $extractor -PathType Leaf)) {
                $message = "Modern .gp material requires a schema-v4 score JSON or $extractor"
                Write-Warning "$($Material.Name): $message"
                $Results.Add([pscustomobject]@{
                    Material = $Material.Name
                    Folder = $Material.FullName
                    Parser = $parser
                    Status = "SKIPPED"
                    ExitCode = $null
                    Output = $materialOutput
                    Log = $logPath
                    Message = $message
                    DurationSeconds = 0
                })
                continue
            }
            $arguments += @("--alphatab-extractor", $extractor)
        }
    }

    Write-Host "[$index/$($Materials.Count)] $($Material.Name)" -ForegroundColor Yellow
    $started = Get-Date
    $exitCode = 1
    $message = ""

    try {
        $exitCode = Invoke-LoggedProcess -Executable "python" -Arguments $arguments -LogPath $logPath
        $message = if ($exitCode -eq 0) { "Build completed" } else { "Builder returned exit code $exitCode" }
    }
    catch {
        $message = $_.Exception.Message
        Add-Content -LiteralPath $logPath -Value ("`nPOWERSHELL ERROR: " + $message)
    }

    $duration = [Math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    $status = if ($exitCode -eq 0) { "PASS" } else { "FAIL" }

    $Results.Add([pscustomobject]@{
        Material = $Material.Name
        Folder = $Material.FullName
        Parser = $parser
        Status = $status
        ExitCode = $exitCode
        Output = $materialOutput
        Log = $logPath
        Diagnostics = (Join-Path $DiagnosticsRoot $safeName)
        Message = $message
        DurationSeconds = $duration
    })

    # Copy all text evidence into one song-scoped diagnostics folder. This runs
    # for PASS and FAIL alike, so partial reports and tracebacks are preserved.
    $collectorArguments = @(
        $DiagnosticsCollector,
        "--song-dir", $Material.FullName,
        "--output-dir", $materialOutput,
        "--log", $logPath,
        "--diagnostics-dir", $DiagnosticsRoot,
        "--material-name", $Material.Name,
        "--method", $AlignmentMode
    )
    if ($LightDiagnostics) { $collectorArguments += "--light" }
    $collectorCode = Invoke-LoggedProcess -Executable "python" -Arguments $collectorArguments -LogPath (Join-Path $LogRoot ("{0:D2}_{1}.diagnostics.log" -f $index, $safeName))
    if ($collectorCode -ne 0) {
        Write-Warning "Diagnostics collection failed for $($Material.Name), exit code $collectorCode"
    }

    if ($status -eq "PASS") {
        Write-Host "PASS: $($Material.Name) ($duration s)" -ForegroundColor Green
    }
    else {
        Write-Host "FAIL: $($Material.Name) ($duration s). See $logPath" -ForegroundColor Red
        if (-not $KeepGoing) { break }
    }
    Write-Host ""
}

$summaryCsv = Join-Path $RunRoot "test_material_summary.csv"
$summaryJson = Join-Path $RunRoot "test_material_summary.json"
$Results | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation -Encoding UTF8
$Results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryJson -Encoding UTF8
Copy-Item -LiteralPath $summaryCsv -Destination (Join-Path $DiagnosticsRoot "test_material_summary.csv") -Force
Copy-Item -LiteralPath $summaryJson -Destination (Join-Path $DiagnosticsRoot "test_material_summary.json") -Force

$passed = @($Results | Where-Object Status -eq "PASS").Count
$failed = @($Results | Where-Object Status -eq "FAIL").Count
$skipped = @($Results | Where-Object Status -eq "SKIPPED").Count

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Test-material run complete" -ForegroundColor Cyan
Write-Host "Passed:  $passed" -ForegroundColor Green
Write-Host "Failed:  $failed" -ForegroundColor $(if ($failed) { "Red" } else { "Green" })
Write-Host "Skipped: $skipped" -ForegroundColor $(if ($skipped) { "Yellow" } else { "Green" })
Write-Host "Summary: $summaryCsv"
Write-Host "Details: $summaryJson"
Write-Host "Logs:    $LogRoot"
Write-Host "Diagnostics: $DiagnosticsRoot"

if ($failed -gt 0) { exit 1 }
exit 0
