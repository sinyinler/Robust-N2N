param(
    [string]$PythonExe = "D:\Anaconda\envs\denoise\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReferenceConfig = Get-Content -LiteralPath (Join-Path $ProjectRoot "configs\sidd_supervised_feature.json") -Encoding UTF8 | ConvertFrom-Json
$DataRoot = $ReferenceConfig.data_root
$ValidationRoot = Join-Path (Split-Path -Parent $DataRoot) "Validation"

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
        $HistoryPath = Join-Path $TrainDir "history.jsonl"
        $Checkpoint = Join-Path $TrainDir "best.pt"
        if (-not (Test-Path -LiteralPath $HistoryPath) -or
            -not (Test-Path -LiteralPath $Checkpoint) -or
            (Get-Content -LiteralPath $HistoryPath).Count -ne 20) {
            throw "Incomplete training output exists; refusing to resume ambiguously: $TrainDir"
        }
        Write-Output "Validated completed training; resume from evaluation: $TrainDir"
    }
    else {
        Invoke-CheckedPython @("train_sidd.py", "--config", $Config)
    }

    $Checkpoint = Join-Path $TrainDir "best.pt"
    $BlockEvalDir = "results\sidd\validation_$EvalPrefix"
    if (Test-Path -LiteralPath (Join-Path $BlockEvalDir "summary.json")) {
        Write-Output "Validated completed block evaluation: $BlockEvalDir"
    }
    else {
        if (Test-Path -LiteralPath $BlockEvalDir) {
            throw "Incomplete block evaluation output exists: $BlockEvalDir"
        }
        Invoke-CheckedPython @(
            "eval_sidd_blocks.py",
            "--noisy_mat", (Join-Path $ValidationRoot "ValidationNoisyBlocksSrgb.mat"),
            "--gt_mat", (Join-Path $ValidationRoot "ValidationGtBlocksSrgb.mat"),
            "--checkpoint", $Checkpoint,
            "--out_dir", $BlockEvalDir
        )
    }

    $FullEvalDir = "results\sidd\internal_test_scene008_$EvalPrefix"
    if (Test-Path -LiteralPath (Join-Path $FullEvalDir "summary.json")) {
        Write-Output "Validated completed full-image evaluation: $FullEvalDir"
    }
    else {
        if (Test-Path -LiteralPath $FullEvalDir) {
            throw "Incomplete full-image evaluation output exists: $FullEvalDir"
        }
        Invoke-CheckedPython @(
            "eval_sidd.py",
            "--data_root", $DataRoot,
            "--checkpoint", $Checkpoint,
            "--scenes", "008",
            "--out_dir", $FullEvalDir
        )
    }
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
