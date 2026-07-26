param(
    [string]$LevelDir = "D:\Desktop\5x5x4",
    [string]$PythonPath = "D:\Anaconda\envs\denoise\python.exe",
    [int]$Batch = 8,
    [int]$Seed = 187,
    [Parameter(Mandatory = $true)]
    [int]$ExpectedScenes,
    [int]$ExpectedFramesPerScene = 1000
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
if ($Batch -le 0 -or $ExpectedScenes -le 0 -or $ExpectedFramesPerScene -le 0) {
    throw "Batch, ExpectedScenes, and ExpectedFramesPerScene must be positive"
}

# 正式训练前强制检查下载完整性，避免把下载中的局部数据误当成完整 Level4。
$SceneDirs = Get-ChildItem -LiteralPath $ResolvedLevelDir -Directory |
    Where-Object { $_.Name -match '^\d+$' } |
    Sort-Object { [int]($_.Name) }
$SceneDirs = @($SceneDirs)
if ($SceneDirs.Count -ne $ExpectedScenes) {
    throw "Incomplete Level4 scene count: current=$($SceneDirs.Count), expected=$ExpectedScenes"
}
$Incomplete = @()
foreach ($SceneDir in $SceneDirs) {
    $NpyDir = Join-Path $SceneDir.FullName "npy"
    $FrameCount = if (Test-Path -LiteralPath $NpyDir -PathType Container) {
        @(Get-ChildItem -LiteralPath $NpyDir -File -Filter "*.npy").Count
    } else {
        0
    }
    if ($FrameCount -ne $ExpectedFramesPerScene) {
        $Incomplete += "$($SceneDir.Name):$FrameCount"
    }
}
if ($Incomplete.Count -gt 0) {
    throw "Incomplete Level4 frame counts (expected $ExpectedFramesPerScene each): $($Incomplete -join ', ')"
}

$DataRoot = Split-Path -Parent $ResolvedLevelDir
$SaveDir = Join-Path $ProjectRoot "results\checkpoints\gammatune_E100_feature_w010_b${Batch}_s${Seed}"
$RunLogDir = Join-Path $ProjectRoot "results\logs\E100_gamma_feature010_b${Batch}_s${Seed}"
$RunLog = Join-Path $RunLogDir "gamma_feature_w010.log"
if (Test-Path -LiteralPath $SaveDir) {
    throw "Output directory already exists; refusing to mix histories: $SaveDir"
}
New-Item -ItemType Directory -Path $RunLogDir -Force | Out-Null

$GitCommit = (& git -C $ProjectRoot rev-parse --short HEAD 2>$null)
Write-Host "[INFO] data=$DataRoot level=$LevelName scenes=$($SceneDirs.Count)"
Write-Host "[INFO] seed=$Seed batch=$Batch gamma_cv=[0.025,0.075]"
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
    "--seed", "$Seed",
    "--device", "cuda",
    "--save_dir", $SaveDir
)

Push-Location $ProjectRoot
try {
    & $PythonPath @TrainArgs 2>&1 | Tee-Object -FilePath $RunLog
    if ($LASTEXITCODE -ne 0) {
        throw "Gamma training failed with exit code $LASTEXITCODE; log: $RunLog"
    }
} finally {
    Pop-Location
}

Write-Host "[OK] 100-epoch raw-domain local Gamma feature training completed: $SaveDir"
