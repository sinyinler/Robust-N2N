param(
    [string]$TargetDir = "E:\SIDD"
)

$ErrorActionPreference = "Stop"
$Url = "http://130.63.97.225/share/SIDD_Medium_Srgb.zip"
$ExpectedBytes = 13234744070
$ZipPath = Join-Path $TargetDir "SIDD_Medium_Srgb.zip"
$LogPath = Join-Path $TargetDir "SIDD_Medium_Srgb.download.stdout.log"
$ErrorLogPath = Join-Path $TargetDir "SIDD_Medium_Srgb.download.stderr.log"
$Aria2Path = Join-Path $TargetDir "tools\aria2-1.37.0-win-64bit-build1\aria2c.exe"

New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null

$existingProcess = Get-CimInstance Win32_Process |
    Where-Object {
        ($_.Name -eq "curl.exe" -or $_.Name -eq "aria2c.exe") -and
        $_.CommandLine -like "*SIDD_Medium_Srgb.zip*"
    }
if ($existingProcess) {
    Write-Output "Download already running (PID $($existingProcess.ProcessId)): $ZipPath"
    exit 0
}

if (Test-Path -LiteralPath $Aria2Path) {
    $arguments = @(
        "--continue=true",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--max-tries=0",
        "--retry-wait=10",
        "--auto-file-renaming=false",
        "--allow-overwrite=true",
        "--summary-interval=5",
        "--dir=$TargetDir",
        "--out=SIDD_Medium_Srgb.zip",
        $Url
    )
    $process = Start-Process -FilePath $Aria2Path -ArgumentList $arguments `
        -RedirectStandardOutput $LogPath -RedirectStandardError $ErrorLogPath `
        -WindowStyle Hidden -PassThru
}
else {
    $arguments = @(
        "--location", "--fail", "--retry", "20", "--retry-all-errors",
        "--retry-delay", "10", "--continue-at", "-", "--output", $ZipPath, $Url
    )
    $process = Start-Process -FilePath "curl.exe" -ArgumentList $arguments `
        -RedirectStandardError $ErrorLogPath -WindowStyle Hidden -PassThru
}

Start-Sleep -Seconds 5
$download = Get-Item -LiteralPath $ZipPath -ErrorAction SilentlyContinue
[PSCustomObject]@{
    Target = $ZipPath
    ExpectedBytes = $ExpectedBytes
    ExistingBytes = if ($download) { $download.Length } else { 0 }
    DownloadPid = $process.Id
    Log = $LogPath
    ErrorLog = $ErrorLogPath
}
