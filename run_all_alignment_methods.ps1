[CmdletBinding()]
param(
    [string]$MaterialsRoot = "",
    [string]$OutputRoot = "",
    [string[]]$Only = @(),
    [string[]]$Methods = @(
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
    ),
    [int]$CheckpointMeasures = 4,
    [double]$CheckpointSearchRadius = 1.0,
    [ValidateSet("tempos", "beats", "both")]
    [string]$TimelineMode = "both",
    [switch]$IncludeVocals,
    [string]$HFToken = $env:HF_TOKEN,
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cpu",
    [switch]$SkipUnitTests,
    [switch]$AllowSevereAlignment,
    [switch]$Force,
    [switch]$IncludePhase3DVariants,
    [ValidateRange(0,10)] [int]$Phase3DSafeCandidates = 2,
    [ValidateRange(0,10)] [int]$Phase3DStructuralCandidates = 1,
    [switch]$IncludeUnsafeContinuousComparison,
    [switch]$StructuralSecondPass,
    [switch]$TranscriptionInvestigator,
    [switch]$MidiSectionFit,
    [switch]$ForceMidiSectionFit,
    [ValidateSet("conservative","permissive","diagnostic-unsafe")]
    [string]$MidiSectionFitMaterialization,
    [switch]$LightDiagnostics,
    [switch]$KeepGoing,
    [switch]$FourSongCorpus
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CorpusRunner = Join-Path $ProjectRoot "run_all_test_materials.ps1"
if (-not (Test-Path -LiteralPath $CorpusRunner -PathType Leaf)) {
    throw "run_all_test_materials.ps1 was not found beside this script"
}
if ([string]::IsNullOrWhiteSpace($MaterialsRoot)) { $MaterialsRoot = $ProjectRoot }
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $MaterialsRoot "all_method_comparison_outputs"
}
if ($FourSongCorpus) {
    $Only = @("my_test_song_2_gp","my_test_song_3_gp","my_test_song_3b_gp","my_test_song_7_gp")
}
if ($ForceMidiSectionFit -and [string]::IsNullOrWhiteSpace($MidiSectionFitMaterialization)) {
    $MidiSectionFitMaterialization = "permissive"
}
if (-not [string]::IsNullOrWhiteSpace($MidiSectionFitMaterialization) -and -not $MidiSectionFit) {
    throw "-MidiSectionFitMaterialization requires -MidiSectionFit"
}
$allowed = @(
    "nominal","offset","linear","dtw","dtw-checkpoint","checkpoint-linear",
    "checkpoint-dtw-diagnostic","checkpoint-dtw-selective",
    "continuous-anchor-linear","continuous-anchor-dtw","continuous-anchor-structural",
    "bar-landmark-regularized"
)
foreach ($method in $Methods) {
    if ($method -notin $allowed) { throw "Unknown alignment method: $method" }
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$ComparisonDiagnostics = Join-Path $OutputRoot "Diagnostics"
New-Item -ItemType Directory -Path $ComparisonDiagnostics -Force | Out-Null
$summary = [System.Collections.Generic.List[object]]::new()
$failedMethods = [System.Collections.Generic.List[string]]::new()
$methodIndex = 0
foreach ($method in $Methods) {
    $methodIndex++
    $methodOutput = Join-Path $OutputRoot $method
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "All-method comparison: $method ($methodIndex/$($Methods.Count))" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    $params = @{
        MaterialsRoot = $MaterialsRoot
        OutputRoot = $methodOutput
        AlignmentMode = $method
        TimelineMode = $TimelineMode
        CheckpointMeasures = $CheckpointMeasures
        CheckpointSearchRadius = $CheckpointSearchRadius
        Device = $Device
        SkipUnitTests = ($SkipUnitTests -or $methodIndex -gt 1)
        AllowSevereAlignment = $AllowSevereAlignment
        Force = $Force
        Only = $Only
    }
    if ($LightDiagnostics) { $params.LightDiagnostics = $true }
    if ($KeepGoing) { $params.KeepGoing = $true }
    if ($IncludeVocals) {
        $params.IncludeVocals = $true
        # First method runs WhisperX/CREPE. Every later method reuses each song's
        # hash-validated intermediate_vocals.json created by the first method.
        if ($methodIndex -gt 1) { $params.ReuseVocals = $true }
        if (-not [string]::IsNullOrWhiteSpace($HFToken)) { $params.HFToken = $HFToken }
    }
    if ($IncludePhase3DVariants -and $method -eq "continuous-anchor-structural") {
        $params.Phase3DVariants = $true
        $params.Phase3DSafeCandidates = $Phase3DSafeCandidates
        $params.Phase3DStructuralCandidates = $Phase3DStructuralCandidates
    }
    if ($IncludeUnsafeContinuousComparison -and $method -eq "continuous-anchor-dtw") {
        $params.BuildUnsafeContinuousComparison = $true
    }
    if ($StructuralSecondPass -and $method -eq "continuous-anchor-structural") {
        $params.StructuralSecondPass = $true
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
    if ($TranscriptionInvestigator -and $transcriptionMethods -contains $method) {
        $params.TranscriptionInvestigator = $true
    }
    elseif ($TranscriptionInvestigator) {
        Write-Host "Transcription investigator skipped for unsupported method: $method" -ForegroundColor DarkGray
    }
    $midiEnabled = $transcriptionMethods -contains $method
    if ($MidiSectionFit -and $midiEnabled) {
        $params.MidiSectionFit = $true
        if ($ForceMidiSectionFit) { $params.ForceMidiSectionFit = $true }
        if (-not [string]::IsNullOrWhiteSpace($MidiSectionFitMaterialization)) {
            $params.MidiSectionFitMaterialization = $MidiSectionFitMaterialization
        }
    }
    elseif ($MidiSectionFit) {
        Write-Host "MIDI section fit skipped for unsupported method: $method" -ForegroundColor DarkGray
    }
    $started = Get-Date
    & $CorpusRunner @params
    $code = $LASTEXITCODE
    $summary.Add([pscustomobject]@{
        Method = $method
        ExitCode = $code
        Seconds = [Math]::Round(((Get-Date) - $started).TotalSeconds, 1)
        Output = $methodOutput
        Materials = ($Only -join ";")
        MidiSectionFit = if ($MidiSectionFit -and $midiEnabled) { "enabled" } else { "unsupported_or_disabled" }
        MidiMaterialization = if ($MidiSectionFit -and $midiEnabled -and -not [string]::IsNullOrWhiteSpace($MidiSectionFitMaterialization)) { $MidiSectionFitMaterialization } else { "none" }
        Vocals = if (-not $IncludeVocals) { "skipped" } elseif ($methodIndex -eq 1) { "analysed_once" } else { "hash_validated_cache" }
    })
    if ($code -ne 0) {
        $failedMethods.Add($method)
        Write-Warning "Method $method completed with one or more failures (exit code $code). Continuing with the remaining methods."
    }
}
$csv = Join-Path $OutputRoot "all_methods_summary.csv"
$json = Join-Path $OutputRoot "all_methods_summary.json"
$summary | Export-Csv -LiteralPath $csv -NoTypeInformation -Encoding UTF8
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $json -Encoding UTF8
Copy-Item -LiteralPath $csv -Destination (Join-Path $ComparisonDiagnostics "all_methods_summary.csv") -Force
Copy-Item -LiteralPath $json -Destination (Join-Path $ComparisonDiagnostics "all_methods_summary.json") -Force
Write-Host ""
Write-Host "All requested methods were attempted." -ForegroundColor Green
Write-Host "Methods passed: $($Methods.Count - $failedMethods.Count)" -ForegroundColor Green
Write-Host "Methods with failures: $($failedMethods.Count)" -ForegroundColor $(if ($failedMethods.Count) { "Yellow" } else { "Green" })
if ($failedMethods.Count) { Write-Host ("Failed methods: " + ($failedMethods -join ", ")) -ForegroundColor Yellow }
Write-Host "Summary CSV:  $csv"
Write-Host "Summary JSON: $json"
Write-Host "Comparison diagnostics: $ComparisonDiagnostics"
if ($failedMethods.Count -gt 0) { exit 1 }
exit 0
