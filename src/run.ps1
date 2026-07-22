# VAR 2026 — Digital Twin BTS: One-Click Pipeline
# Usage: .\run.ps1 [options]

param(
    [string[]]$Scenes = @(),
    [string]$Variant = "full_60k",
    [switch]$AllVariants,
    [switch]$TrainOnly,
    [switch]$RenderOnly,
    [switch]$SkipTrain,
    [switch]$DryRun
)

$args = @()
if ($Scenes.Count -gt 0) { $args += "--scenes"; $args += $Scenes }
if ($Variant -ne "full_60k") { $args += "--variant"; $args += $Variant }
if ($AllVariants) { $args += "--all-variants" }
if ($TrainOnly) { $args += "--train-only" }
if ($RenderOnly) { $args += "--render-only" }
if ($SkipTrain) { $args += "--skip-train" }
if ($DryRun) { $args += "--dry-run" }

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "VAR 2026 — DIGITAL TWIN BTS PIPELINE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python main.py @args
