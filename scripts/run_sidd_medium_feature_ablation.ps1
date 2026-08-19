param(
    [string]$PythonExe = "D:\Anaconda\envs\denoise\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReferenceConfig = Get-Content -LiteralPath `
    (Join-Path $ProjectRoot "configs\sidd_medium_feature.json") -Encoding UTF8 | ConvertFrom-Json
$DataRoot = $ReferenceConfig.data_root
$ValidationRoot = $ReferenceConfig.validation_root

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
        $history = Join-Path $TrainDir "history.jsonl"
        $checkpoint = Join-Path $TrainDir "best.pt"
        if (-not (Test-Path -LiteralPath $history) -or
            -not (Test-Path -LiteralPath $checkpoint) -or
            (Get-Content -LiteralPath $history).Count -ne 20) {
            throw "Incomplete training output exists; refusing ambiguous resume: $TrainDir"
        }
        Write-Output "Validated completed training; resume from evaluation: $TrainDir"
    }
    else {
        Invoke-CheckedPython @("train_sidd.py", "--config", $Config)
    }

    $checkpoint = Join-Path $TrainDir "best.pt"
    $blockDir = "results\sidd\validation_$EvalPrefix"
    if (-not (Test-Path -LiteralPath (Join-Path $blockDir "summary.json"))) {
        if (Test-Path -LiteralPath $blockDir) {
            throw "Incomplete block evaluation output exists: $blockDir"
        }
        Invoke-CheckedPython @(
            "eval_sidd_blocks.py",
            "--noisy_mat", (Join-Path $ValidationRoot "ValidationNoisyBlocksSrgb.mat"),
            "--gt_mat", (Join-Path $ValidationRoot "ValidationGtBlocksSrgb.mat"),
            "--checkpoint", $checkpoint,
            "--out_dir", $blockDir
        )
    }

    $sceneDir = "results\sidd\internal_test_scene008_$EvalPrefix"
    if (-not (Test-Path -LiteralPath (Join-Path $sceneDir "summary.json"))) {
        if (Test-Path -LiteralPath $sceneDir) {
            throw "Incomplete full-image evaluation output exists: $sceneDir"
        }
        Invoke-CheckedPython @(
            "eval_sidd.py", "--data_root", $DataRoot,
            "--checkpoint", $checkpoint, "--scenes", "008",
            "--out_dir", $sceneDir
        )
    }
}

Set-Location -LiteralPath $ProjectRoot
Invoke-SIDDArm `
    -Config "configs\sidd_medium_feature.json" `
    -TrainDir "results\sidd\medium_scene_split_feature_gaussian_s42" `
    -EvalPrefix "medium_feature_gaussian_s42"

Invoke-SIDDArm `
    -Config "configs\sidd_medium_feature_rtv.json" `
    -TrainDir "results\sidd\medium_scene_split_feature_gaussian_rtv1e4_s42" `
    -EvalPrefix "medium_feature_gaussian_rtv1e4_s42"

Write-Output "SIDD-Medium feature ablation and all evaluations completed."
