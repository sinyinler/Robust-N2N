param(
    [string]$LevelDir = "D:\Desktop\5x5x4",
    [string]$PythonPath = "D:\Anaconda\envs\denoise\python.exe",
    [int]$Batch = 8,
    [int]$Seed = 187,
    [Parameter(Mandatory = $true)]
    [int]$ExpectedScenes,
    [int]$ExpectedFramesPerScene = 0,
    [string]$ResumeCheckpoint = "",
    [int]$Progress = 0,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedLevelDir = (Resolve-Path -LiteralPath $LevelDir).Path
$LevelName = Split-Path -Leaf $ResolvedLevelDir
if ($LevelName -ne "5x5x4") {
    throw "LevelDir must point to a directory named 5x5x4: $ResolvedLevelDir"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Cannot find the denoise environment Python: $PythonPath"
}
if ($Batch -le 0 -or $ExpectedScenes -le 0 -or $ExpectedFramesPerScene -lt 0) {
    throw "Batch/ExpectedScenes must be positive and ExpectedFramesPerScene cannot be negative"
}

# Refuse to train on a partial Level4 download.
$SceneDirs = New-Object System.Collections.ArrayList
foreach ($CandidateDir in Get-ChildItem -LiteralPath $ResolvedLevelDir -Directory) {
    if (
        $CandidateDir.Name -match '^\d+$' -and
        (Test-Path -LiteralPath (Join-Path $CandidateDir.FullName "npy") -PathType Container)
    ) {
        [void]$SceneDirs.Add($CandidateDir)
    }
}
if ($SceneDirs.Count -ne $ExpectedScenes) {
    throw "Incomplete Level4 scene count: current=$($SceneDirs.Count), expected=$ExpectedScenes"
}
$Incomplete = @()
foreach ($SceneDir in $SceneDirs) {
    $NpyDir = Join-Path $SceneDir.FullName "npy"
    $FrameFiles = if (Test-Path -LiteralPath $NpyDir -PathType Container) {
        @(Get-ChildItem -LiteralPath $NpyDir -File -Filter "*.npy")
    } else {
        @()
    }
    $FrameCount = $FrameFiles.Count
    if ($FrameCount -lt 2) {
        $Incomplete += "$($SceneDir.Name):only-$FrameCount-frames"
        continue
    }
    if ($ExpectedFramesPerScene -gt 0 -and $FrameCount -ne $ExpectedFramesPerScene) {
        $Incomplete += "$($SceneDir.Name):$FrameCount"
        continue
    }

    # Variable-length scenes must contain contiguous 0.npy..(N-1).npy indices.
    $Indices = [System.Collections.Generic.List[int]]::new()
    foreach ($FrameFile in $FrameFiles) {
        if ($FrameFile.BaseName -match '^\d+$') {
            $Indices.Add([Convert]::ToInt32($FrameFile.BaseName))
        } else {
            $Indices.Add(-1)
        }
    }
    $Indices.Sort()
    if ($Indices[0] -ne 0 -or $Indices[-1] -ne ($FrameCount - 1)) {
        $Incomplete += "$($SceneDir.Name):non-contiguous"
        continue
    }
    for ($Index = 0; $Index -lt $FrameCount; $Index++) {
        if ($Indices[$Index] -ne $Index) {
            $Incomplete += "$($SceneDir.Name):non-contiguous"
            break
        }
    }
}
if ($Incomplete.Count -gt 0) {
    $ExpectedText = if ($ExpectedFramesPerScene -gt 0) {
        "$ExpectedFramesPerScene frames per scene"
    } else {
        "contiguous variable-length sequences"
    }
    throw "Incomplete Level4 data (expected $ExpectedText): $($Incomplete -join ', ')"
}

$DataRoot = Split-Path -Parent $ResolvedLevelDir
$TotalFrames = 0
foreach ($SceneDir in $SceneDirs) {
    $TotalFrames += @(Get-ChildItem -LiteralPath (Join-Path $SceneDir.FullName "npy") -File -Filter "*.npy").Count
}
Write-Host "[INFO] data=$DataRoot level=$LevelName scenes=$($SceneDirs.Count) frames=$TotalFrames"
Write-Host "[INFO] seed=$Seed batch=$Batch gamma_cv=[0.025,0.075]"
if ($PreflightOnly) {
    Write-Host "[OK] Level4 preflight completed; training was not started"
    exit 0
}

$SaveDir = Join-Path $ProjectRoot "results\checkpoints\gammatune_E100_feature_w010_b${Batch}_s${Seed}"
$RunLogDir = Join-Path $ProjectRoot "results\logs\E100_gamma_feature010_b${Batch}_s${Seed}"
$RunLog = Join-Path $RunLogDir "gamma_feature_w010.log"
if ($ResumeCheckpoint) {
    $ResumeCheckpoint = (Resolve-Path -LiteralPath $ResumeCheckpoint).Path
    if (-not (Test-Path -LiteralPath $SaveDir -PathType Container)) {
        throw "Resume output directory does not exist: $SaveDir"
    }
} elseif (Test-Path -LiteralPath $SaveDir) {
    throw "Output directory already exists; refusing to mix histories: $SaveDir"
}
New-Item -ItemType Directory -Path $RunLogDir -Force | Out-Null

$GitCommit = (& git -C $ProjectRoot rev-parse --short HEAD 2>$null)
Write-Host "[INFO] python=$PythonPath git=$GitCommit"

$TrainArgs = @(
    "-u", (Join-Path $ProjectRoot "train_masked.py"),
    "--data_path", $DataRoot,
    "--data_subdir", "npy",
    "--strict_data_subdir", "1",
    "--levels", "4",
    "--intervals", "5", "7", "9",
    "--epochs", "100",
    "--crop_size", "512",
    "--batch_size", "$Batch",
    "--lr", "0.01",
    "--lr_final", "0.0005",
    "--warmup_pct", "0.1",
    "--rtv_weight", "0.01",
    "--weight_decay", "0.0001",
    "--train_fraction", "0.99",
    "--val_limit_batches", "20",
    "--intensity_transform", "log1p",
    "--corruption_mode", "gamma",
    "--mask_ratio", "0.25",
    "--mask_patch", "16",
    "--gamma_cv_min", "0.025",
    "--gamma_cv_max", "0.075",
    "--w_mask_pixel", "0",
    "--w_mask_feature", "0.10",
    "--mask_feature_scales", "encoder2", "encoder3",
    "--predictor_hidden_ratio", "1.0",
    "--ema_decay", "0.996",
    "--feature_warmup_frac", "0.1",
    "--freeze_masked_bn_stats", "1",
    "--deterministic_loader_rng", "1",
    "--grad_diag_every", "100",
    "--grad_diag_scales", "encoder2", "encoder3",
    "--data_parallel", "0",
    "--plot_loss_curve", "1",
    "--progress", "$Progress",
    "--seed", "$Seed",
    "--device", "cuda",
    "--save_dir", $SaveDir
)
if ($ResumeCheckpoint) {
    $TrainArgs += @("--resume_checkpoint", $ResumeCheckpoint)
    Write-Host "[INFO] resume=$ResumeCheckpoint progress=$Progress"
    Add-Content -LiteralPath $RunLog -Value (
        "`r`n[LAUNCH RESUME] $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') checkpoint=$ResumeCheckpoint"
    )
}

Push-Location $ProjectRoot
$PreviousErrorActionPreference = $ErrorActionPreference
$TrainingExitCode = -1
try {
    # Windows PowerShell 5 wraps native stderr (including normal tqdm output)
    # as NativeCommandError when ErrorActionPreference=Stop.
    $ErrorActionPreference = "Continue"
    & $PythonPath @TrainArgs 2>&1 | Tee-Object -FilePath $RunLog -Append
    $TrainingExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
    Pop-Location
}
if ($TrainingExitCode -ne 0) {
    throw "Gamma training failed with exit code $TrainingExitCode; log: $RunLog"
}

Write-Host "[OK] 100-epoch raw-domain local Gamma feature training completed: $SaveDir"
