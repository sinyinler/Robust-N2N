param(
    [string]$TargetDir = "E:\SIDD"
)

$ErrorActionPreference = "Stop"
$ZipPath = Join-Path $TargetDir "SIDD_Medium_Srgb.zip"
$ExtractDir = Join-Path $TargetDir "SIDD_Medium_Srgb"
$ExpectedBytes = 13234744070
$ExpectedMD5 = "F95B4BC9EC1DD3FE4EBD61AEACAD3991"
$ExpectedSHA1 = "B0F895258112DB896D6ADE0A8DDAFC8CFC9BD54D"

if (Test-Path -LiteralPath "$ZipPath.aria2") {
    throw "aria2 control file still exists; download is incomplete: $ZipPath.aria2"
}
$archive = Get-Item -LiteralPath $ZipPath
if ($archive.Length -ne $ExpectedBytes) {
    throw "Archive size mismatch: expected $ExpectedBytes, got $($archive.Length)"
}

$md5 = (Get-FileHash -LiteralPath $ZipPath -Algorithm MD5).Hash
$sha1 = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA1).Hash
if ($md5 -ne $ExpectedMD5 -or $sha1 -ne $ExpectedSHA1) {
    throw "Checksum mismatch: MD5=$md5 SHA1=$sha1"
}

if (-not (Test-Path -LiteralPath $ExtractDir)) {
    New-Item -ItemType Directory -Path $ExtractDir | Out-Null
    tar -xf $ZipPath -C $ExtractDir
    if ($LASTEXITCODE -ne 0) {
        throw "Archive extraction failed with exit code $LASTEXITCODE"
    }
}

$pngFiles = Get-ChildItem -LiteralPath $ExtractDir -Recurse -File -Filter "*.PNG"
$noisyFiles = $pngFiles | Where-Object { $_.Name -like "*_NOISY_SRGB_*.PNG" }
$gtFiles = $pngFiles | Where-Object { $_.Name -like "*_GT_SRGB_*.PNG" }
if ($noisyFiles.Count -ne 320 -or $gtFiles.Count -ne 320) {
    throw "Unexpected pair count after extraction: noisy=$($noisyFiles.Count), gt=$($gtFiles.Count)"
}

[PSCustomObject]@{
    Archive = $ZipPath
    Extracted = $ExtractDir
    Bytes = $archive.Length
    MD5 = $md5
    SHA1 = $sha1
    NoisyImages = $noisyFiles.Count
    GroundTruthImages = $gtFiles.Count
}
