"""VAR 2026 — Digital Twin BTS: Complete Local Pipeline
============================================================
One-click orchestrator: validate → train → render → eval → compact → tta → ensemble → post → package.

Fully leverages ALL gaussian-splatting baseline features:
  - eval mode, sh_degree=4, depth scheduling, densification tuning
  - exposure compensation, anti-aliasing, sparse Adam, white/random BG
  - competition metric evaluation (LPIPS+SSIM+PSNR → Score)
  - Gaussian-level compact merge + test-time adaptation
  - smart per-pixel ensemble + post-processing

Usage:
    python main.py                                # all scenes, full pipeline
    python main.py --scenes bonsai                 # single scene
    python main.py --variant full_60k              # single variant
    python main.py --all-variants                  # train ALL 10 variants
    python main.py --compact --tta                 # Gaussian merge + TTA
    python main.py --eval-only                     # evaluate metrics only
    python main.py --train-only                    # train only
    python main.py --dry-run                       # print plan only

Pipeline:
    Phase 1:   VALIDATE  → check data & 3DGS installation
    Phase 2:   TRAIN     → train 3DGS variants (--eval, sh=4, densify tuning)
    Phase 3:   RENDER    → render test poses from trained models
    Phase 3.2: EVAL      → compute competition score (LPIPS+SSIM+PSNR)
    Phase 3.5: COMPACT   → Gaussian-level merging (voxel-based)
    Phase 3.6: TTA       → Test-time adaptation delta layer
    Phase 3.7: PERCEPTUAL → LPIPS/DINO-v2 fine-tuning for LPIPS optimization (40% of score)
    Phase 4:   ENSEMBLE  → smart per-pixel confidence blending
    Phase 5:   POST      → edge-aware sharpen + color match
    Phase 6:   PACKAGE   → create submission_round1.zip
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Module-level flag for --data-dir propagation
DATA_DIR_ARG: list[str] = []

import config as _cfg
DATA_DIR = _cfg.DATA_DIR
GS_DIR = _cfg.GS_DIR
OUTPUT_DIR = _cfg.OUTPUT_DIR
SCENES = _cfg.SCENES
VARIANTS = _cfg.VARIANTS
SUBMISSION_NAME = _cfg.SUBMISSION_NAME
set_data_dir = _cfg.set_data_dir


# ═══════════════════════════════════════════════════════════════
#  Phase 1: Validate
# ═══════════════════════════════════════════════════════════════

def validate(scenes: list[str]) -> bool:
    """Check all scenes have required data."""
    print("=" * 60)
    print("PHASE 1: VALIDATE DATA")
    print("=" * 60)
    ok = True
    dd = _cfg.DATA_DIR  # luôn lấy từ config module (đã được set_data_dir cập nhật)
    for scene in scenes:
        d = dd / scene
        reqs = [
            (d / "train" / "images", "train images"),
            (d / "train" / "sparse" / "0" / "cameras.bin", "COLMAP cameras"),
            (d / "train" / "sparse" / "0" / "images.bin", "COLMAP images"),
            (d / "train" / "sparse" / "0" / "points3D.bin", "COLMAP points3D"),
        ]
        missing = [desc for path, desc in reqs if not path.exists()]
        test_csv = d / "test" / "test_poses.csv"
        if not test_csv.exists():
            test_csv = d / "test_poses.csv"
        n_poses = 0
        if test_csv.exists():
            with open(test_csv) as f:
                n_poses = len(list(csv.DictReader(f)))

        if missing:
            print(f"  \u274c {scene}: missing {missing}")
            ok = False
        else:
            n_imgs = len(list((d / "train" / "images").glob("*")))
            print(f"  \u2705 {scene}: {n_imgs} imgs, {n_poses} test poses")
    return ok


# ═══════════════════════════════════════════════════════════════
#  Phase 2: Train
# ═══════════════════════════════════════════════════════════════

def train_variants(scenes: list[str], variant_names: list[str], check: bool = False) -> dict:
    """Train specified variants on specified scenes."""
    results = {}
    for scene in scenes:
        results[scene] = {}
        for vname in variant_names:
            print(f"\n  -- {scene}/{vname} --")
            cmd = [sys.executable, str(ROOT / "train.py"),
                   "--scene", scene, "--variant", vname, "--gs-dir", str(GS_DIR),
                   *DATA_DIR_ARG]
            if check:
                cmd.append("--check")
            r = subprocess.run(
                cmd,
                capture_output=False, text=True,)
            results[scene][vname] = r.returncode == 0
    return results


# ═══════════════════════════════════════════════════════════════
#  Phase 3: Render
# ═══════════════════════════════════════════════════════════════

def render_variants(scenes: list[str], variant_names: list[str]) -> dict:
    """Render test poses for trained variants."""
    results = {}
    for scene in scenes:
        results[scene] = {}
        for vname in variant_names:
            print(f"\n  -- {scene}/{vname} --")
            r = subprocess.run(
                [sys.executable, str(ROOT / "render.py"),
                 "--scene", scene, "--variant", vname, "--gs-dir", str(GS_DIR),
                 *DATA_DIR_ARG],
                capture_output=False, text=True,
            )
            results[scene][vname] = r.returncode == 0
    return results


# ═══════════════════════════════════════════════════════════════
#  Phase 3.2: Evaluate Competition Metrics
# ═══════════════════════════════════════════════════════════════

def eval_scenes(scenes: list[str]) -> None:
    """Phase 3.2: Compute LPIPS/SSIM/PSNR + competition score."""
    for scene in scenes:
        subprocess.run(
            [sys.executable, str(ROOT / "eval.py"), "--scene", scene, "--all-variants",
             *DATA_DIR_ARG],
            capture_output=False,
        )


# ═══════════════════════════════════════════════════════════════
#  Phase 3.5-6: Compact → TTA → Ensemble → Post → Package
# ═══════════════════════════════════════════════════════════════

def perceptual_scenes(scenes: list[str], variant: str = "compact") -> None:
    """Phase 3.7: LPIPS/DINO-v2 perceptual fine-tuning."""
    for scene in scenes:
        subprocess.run(
            [sys.executable, str(ROOT / "perceptual_finetune.py"),
             "--scene", scene, "--variant", variant,
             *DATA_DIR_ARG],
            capture_output=False,
        )


def compact_scenes(scenes: list[str]) -> None:
    """Phase 3.5: Gaussian-level primitive merging."""
    for scene in scenes:
        subprocess.run(
            [sys.executable, str(ROOT / "compact.py"), "--scene", scene,
             *DATA_DIR_ARG],
            capture_output=False,
        )


def tta_scenes(scenes: list[str], model: str = "compact") -> None:
    """Phase 3.6: Test-time adaptation delta layer."""
    for scene in scenes:
        subprocess.run(
            [sys.executable, str(ROOT / "tta.py"),
             "--scene", scene, "--model", model,
             *DATA_DIR_ARG],
            capture_output=False,
        )


def ensemble_scenes(scenes: list[str]) -> None:
    for scene in scenes:
        subprocess.run(
            [sys.executable, str(ROOT / "ensemble.py"), "--scene", scene,
             *DATA_DIR_ARG],
            capture_output=False,
        )


def postprocess_scenes(scenes: list[str]) -> None:
    for scene in scenes:
        subprocess.run(
            [sys.executable, str(ROOT / "postprocess.py"), "--scene", scene,
             *DATA_DIR_ARG],
            capture_output=False,
        )


def package_scenes(scenes: list[str], source: str = "final") -> Path:
    r = subprocess.run(
        [sys.executable, str(ROOT / "package.py"),
         "--scenes", *scenes, "--source", source, "--output", SUBMISSION_NAME,
         *DATA_DIR_ARG],
        capture_output=True, text=True,
    )
    print(r.stdout)
    return ROOT / "submissions" / SUBMISSION_NAME


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="VAR 2026 Complete Local Pipeline")
    p.add_argument("--scenes", nargs="*", default=None, help=f"Default: all {len(SCENES)} scenes")
    p.add_argument("--variant", default="full_60k", help="Single variant name")
    p.add_argument("--variants", default=None, help="Comma-separated variant names")
    p.add_argument("--all-variants", action="store_true", help="Train ALL 10 variants")
    p.add_argument("--compact", action="store_true", help="Phase 3.5: Gaussian-level merging")
    p.add_argument("--tta", action="store_true", help="Phase 3.6: Test-time adaptation")
    p.add_argument("--tta-model", default="compact", help="Model for TTA")
    p.add_argument("--perceptual", action="store_true", help="Phase 3.7: LPIPS/DINO-v2 fine-tuning")
    p.add_argument("--perceptual-model", default=None, help="Model variant for perceptual fine-tuning (default: compact if --compact else full_60k)")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--eval-only", action="store_true", help="Evaluate metrics only")
    p.add_argument("--ensemble-only", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-ensemble", action="store_true")
    p.add_argument("--skip-post", action="store_true")
    p.add_argument("--data-dir", default=None, help="Override data directory (default: ../data)")
    p.add_argument("--check", action="store_true", help="Smoke test: 100 iters, no densify, no eval")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.data_dir:
        set_data_dir(args.data_dir)

    scenes = args.scenes if args.scenes else _cfg.SCENES

    if args.all_variants:
        variant_names = [v.name for v in VARIANTS]
    elif args.variants:
        variant_names = args.variants.split(",")
    else:
        variant_names = [args.variant]

    print(f"\n{'='*60}")
    print(f"VAR 2026 — DIGITAL TWIN BTS PIPELINE")
    print(f"{'='*60}")
    print(f"  Scenes:   {', '.join(scenes)}")
    print(f"  Variants: {', '.join(variant_names)}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"  Pipeline: validate → train → render → eval → compact → tta → ensemble → post → package")
    if args.compact:
        print(f"  + Gaussian-level compact merge (Phase 3.5)")
    if args.tta:
        print(f"  + Test-time adaptation (Phase 3.6)")

    global DATA_DIR_ARG
    DATA_DIR_ARG = ["--data-dir", str(_cfg.DATA_DIR)] if args.data_dir else []

    if args.dry_run:
        print(f"\n  [DRY RUN] Would execute but not run.")
        return

    t0 = time.time()

    # Phase 1: Validate
    if not validate(scenes):
        print("\n\u274c Data validation failed. Fix issues first.")
        return
    print()

    # Phase 2: Train
    if args.render_only or args.ensemble_only or args.eval_only:
        pass
    elif args.skip_train:
        print("  [SKIP] Training")
    else:
        print(f"\n{'='*60}")
        print(f"PHASE 2: TRAIN ({len(scenes)*len(variant_names)} jobs)")
        print(f"{'='*60}")
        train_variants(scenes, variant_names, args.check)

    if args.train_only:
        elapsed = (time.time() - t0) / 60
        print(f"\n\u2705 Training done in {elapsed:.0f} min")
        return

    # Phase 3: Render
    if args.ensemble_only or args.eval_only:
        pass
    elif args.skip_render:
        print("  [SKIP] Rendering")
    else:
        print(f"\n{'='*60}")
        print(f"PHASE 3: RENDER")
        print(f"{'='*60}")
        render_variants(scenes, variant_names)

    if args.render_only:
        elapsed = (time.time() - t0) / 60
        print(f"\n\u2705 Render done in {elapsed:.0f} min")
        return

    # Phase 3.2: Evaluate Competition Metrics
    if not args.skip_eval:
        print(f"\n{'='*60}")
        print(f"PHASE 3.2: EVALUATE COMPETITION METRICS")
        print(f"  Formula: Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm")
        print(f"{'='*60}")
        eval_scenes(scenes)

    if args.eval_only:
        elapsed = (time.time() - t0) / 60
        print(f"\n\u2705 Evaluation done in {elapsed:.0f} min")
        return

    # Phase 3.5: Compact Merge (Gaussian-level)
    if args.compact:
        print(f"\n{'='*60}")
        print(f"PHASE 3.5: GAUSSIAN-LEVEL COMPACT MERGE")
        print(f"{'='*60}")
        compact_scenes(scenes)

    # Phase 3.6: Test-Time Adaptation
    if args.tta:
        print(f"\n{'='*60}")
        print(f"PHASE 3.6: TEST-TIME ADAPTATION")
        print(f"{'='*60}")
        tta_scenes(scenes, args.tta_model)

    # Phase 3.7: Perceptual Fine-Tuning (LPIPS/DINOv2)
    if args.perceptual:
        print(f"\n{'='*60}")
        print(f"PHASE 3.7: PERCEPTUAL FINE-TUNE (LPIPS = 40% SCORE)")
        print(f"  Loss: LPIPS λ=1.0 + DINO λ=0.5 + L1 λ=0.2")
        print(f"  Goal: Directly optimize the LPIPS component of competition score")
        print(f"{'='*60}")
        # Determine which model to fine-tune:
        # --perceptual-model flag > --compact/--tta model > defaults
        perceptual_model = args.perceptual_model
        if perceptual_model is None:
            if args.tta:
                perceptual_model = args.tta_model
            elif args.compact:
                perceptual_model = "compact"
            else:
                perceptual_model = args.variant
        perceptual_scenes(scenes, perceptual_model)

    # Phase 4: Ensemble
    if not args.skip_ensemble:
        print(f"\n{'='*60}")
        print(f"PHASE 4: SMART ENSEMBLE")
        print(f"{'='*60}")
        ensemble_scenes(scenes)

    if args.ensemble_only:
        elapsed = (time.time() - t0) / 60
        print(f"\n\u2705 Ensemble done in {elapsed:.0f} min")
        return

    # Phase 5: Post-process
    if not args.skip_post:
        print(f"\n{'='*60}")
        print(f"PHASE 5: POST-PROCESS")
        print(f"{'='*60}")
        postprocess_scenes(scenes)

    # Phase 6: Package
    print(f"\n{'='*60}")
    print(f"PHASE 6: PACKAGE SUBMISSION")
    print(f"{'='*60}")
    zip_path = package_scenes(scenes, "final")

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"\U0001f3c6 PIPELINE COMPLETE! ({elapsed:.0f} min)")
    print(f"\U0001f4e6 {zip_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
