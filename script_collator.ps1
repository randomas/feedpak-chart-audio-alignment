param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$Files,

    [string]$OutputFile = "python_scripts_dump.txt"
)

# Create/overwrite output file
"" | Set-Content -Path $OutputFile

foreach ($File in $Files) {
    if (-not (Test-Path $File)) {
        Write-Warning "File not found: $File"
        continue
    }

    Add-Content -Path $OutputFile -Value "===== $(Split-Path $File -Leaf) ====="
    Add-Content -Path $OutputFile -Value (Get-Content $File -Raw)
    Add-Content -Path $OutputFile -Value "`r`n"
}

Write-Host "Exported $($Files.Count) file(s) to $OutputFile"


