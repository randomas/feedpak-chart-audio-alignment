[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$OutputFile,

    [Parameter(Mandatory = $true, Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Files
)

$ErrorActionPreference = "Stop"

if (-not $Files -or $Files.Count -eq 0) {
    throw "No input files were supplied."
}

$outputPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputFile))
$outputDirectory = Split-Path -Parent $outputPath

if ($outputDirectory -and -not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

$resolvedFiles = New-Object System.Collections.Generic.List[string]
$missingFiles = New-Object System.Collections.Generic.List[string]

foreach ($file in $Files) {
    if ([string]::IsNullOrWhiteSpace($file)) {
        continue
    }

    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        $missingFiles.Add($file)
        continue
    }

    $resolved = (Resolve-Path -LiteralPath $file).Path

    if ([System.StringComparer]::OrdinalIgnoreCase.Equals($resolved, $outputPath)) {
        throw "The output file cannot also be an input file: $file"
    }

    if (-not $resolvedFiles.Contains($resolved)) {
        $resolvedFiles.Add($resolved)
    }
}

if ($missingFiles.Count -gt 0) {
    throw ("The following input files do not exist:`n  " + ($missingFiles -join "`n  "))
}

if ($resolvedFiles.Count -eq 0) {
    throw "No valid input files were supplied."
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$writer = New-Object System.IO.StreamWriter($outputPath, $false, $utf8NoBom)

try {
    $writer.WriteLine("PROJECT DUMP")
    $writer.WriteLine(("Generated: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss K")))
    $writer.WriteLine(("File count: {0}" -f $resolvedFiles.Count))
    $writer.WriteLine()

    foreach ($resolved in $resolvedFiles) {
        $relative = Resolve-Path -LiteralPath $resolved -Relative
        $relative = $relative -replace "\\", "/"
        $extension = [System.IO.Path]::GetExtension($resolved).TrimStart(".")

        if ([string]::IsNullOrWhiteSpace($extension)) {
            $extension = "text"
        }

        $writer.WriteLine("================================================================================")
        $writer.WriteLine(("FILE: {0}" -f $relative))
        $writer.WriteLine("================================================================================")
        $writer.WriteLine(('```{0}' -f $extension))

        $text = [System.IO.File]::ReadAllText($resolved)
        $writer.Write($text)

        if (-not $text.EndsWith("`n")) {
            $writer.WriteLine()
        }

        $writer.WriteLine('```')
        $writer.WriteLine()
    }
}
finally {
    $writer.Dispose()
}

$size = (Get-Item -LiteralPath $outputPath).Length
Write-Host (("Wrote {0} file(s) to {1} ({2:N0} bytes)" -f $resolvedFiles.Count, $OutputFile, $size))
