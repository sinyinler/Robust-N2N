param(
    [string]$PythonExe = "D:\Anaconda\envs\denoise\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ValidationRoot = "D:\Desktop\数据集\SIDD\Validation"
$DataRoot = "D:\Desktop\数据集\SIDD\SIDD_Small_sRGB_Only"

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Invoke-SIDDArm {
    param(
        [string]$Config,
        [string]$TrainDir,
        [string]$EvalPrefix
    )

    if (Test-Path -LiteralPath $TrainDir) {
        throw "Training output already exists; refusing to mix trajectories: $TrainDir"
    }

    Invoke-CheckedPython @("train_sidd.py", "--config", $Config)
    $Checkpoint = Join-Path $TrainDir "best.pt"
    Invoke-CheckedPython @(
        "eval_sidd_blocks.py",
        "--noisy_mat", (Join-Path $ValidationRoot "ValidationNoisyBlocksSrgb.mat"),
        "--gt_mat", (Join-Path $ValidationRoot "ValidationGtBlocksSrgb.mat"),
        "--checkpoint", $Checkpoint,
        "--out_dir", "results\sidd\validation_$EvalPrefix"
    )
    Invoke-CheckedPython @(
        "eval_sidd.py",
        "--data_root", $DataRoot,
        "--checkpoint", $Checkpoint,
        "--scenes", "008",
        "--out_dir", "results\sidd\internal_test_scene008_$EvalPrefix"
    )
}

Set-Location -LiteralPath $ProjectRoot
Invoke-SIDDArm `
    -Config "configs\sidd_supervised_feature.json" `
    -TrainDir "results\sidd\supervised_feature_gaussian_s42" `
    -EvalPrefix "feature_gaussian_s42"

Invoke-SIDDArm `
    -Config "configs\sidd_supervised_feature_rtv.json" `
    -TrainDir "results\sidd\supervised_feature_gaussian_rtv1e4_s42" `
    -EvalPrefix "feature_gaussian_rtv1e4_s42"

Write-Output "SIDD feature ablation and all evaluations completed."
