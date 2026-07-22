"""Evaluate 3DGS renders using competition metrics: LPIPS, SSIM, PSNR.

Uses gaussian-splatting/metrics.py for standard metrics, then computes
the VAR 2026 competition score:
    Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm

Usage:
    python eval.py --scene bonsai
    python eval.py --scene bonsai --variant full_60k
    python eval.py --scene bonsai --all-variants
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import (
    DATA_DIR, GS_DIR, OUTPUT_DIR, VARIANTS,
    PSNR_MAX, LPIPS_WEIGHT, SSIM_WEIGHT, PSNR_WEIGHT,
)


def compute_competition_score(lpips: float, ssim: float, psnr: float) -> float:
    """Compute VAR 2026 competition score."""
    psnr_norm = min(psnr / PSNR_MAX, 1.0)
    return LPIPS_WEIGHT * (1.0 - lpips) + SSIM_WEIGHT * ssim + PSNR_WEIGHT * psnr_norm


def evaluate_with_metrics_py(scene: str, variants: list[str] | None = None) -> dict:
    """Run gaussian-splatting/metrics.py on trained models and return scores.

    Uses the model_path's built-in test set (from --eval mode training).
    """
    if variants is None:
        variants = [v.name for v in VARIANTS if v.eval_mode and v.iters >= 7000]

    results = {}

    for vname in variants:
        model_dir = OUTPUT_DIR / "models" / scene / vname
        if not model_dir.exists():
            print(f"  [SKIP] {vname}: no model at {model_dir}")
            continue

        # Check if test renders exist
        test_dir = model_dir / "test"
        has_renders = False
        if test_dir.exists():
            for method_dir in test_dir.iterdir():
                if method_dir.is_dir() and (method_dir / "renders").exists():
                    has_renders = True
                    break

        if not has_renders:
            print(f"  [SKIP] {vname}: no test renders (train with --eval to enable)")
            continue

        # Run metrics.py (generates results.json)
        print(f"\n  ── {vname} ──")
        cmd = [sys.executable, "metrics.py", "-m", str(model_dir)]
        subprocess.run(
            cmd, cwd=str(GS_DIR), capture_output=True, text=True
        )

        # Read results.json for metrics
        ssim_val = None
        psnr_val = None
        lpips_val = None
        results_file = model_dir / "results.json"

        if results_file.exists():
            try:
                with open(results_file) as f:
                    res_json = json.load(f)
                # Get the first (and usually only) method's metrics
                for method, metrics in res_json.items():
                    ssim_val = metrics.get("SSIM", None)
                    psnr_val = metrics.get("PSNR", None)
                    lpips_val = metrics.get("LPIPS", None)
                    break
            except (json.JSONDecodeError, KeyError):
                pass

        if ssim_val is not None and psnr_val is not None and lpips_val is not None:
            score = compute_competition_score(lpips_val, ssim_val, psnr_val)
            results[vname] = {
                "LPIPS": lpips_val,
                "SSIM": ssim_val,
                "PSNR": psnr_val,
                "SCORE": score,
            }
            print(f"    LPIPS={lpips_val:.4f}  SSIM={ssim_val:.4f}  "
                  f"PSNR={psnr_val:.2f}  SCORE={score:.4f}")
        else:
            print(f"    [WARN] Could not parse metrics from {results_file}")

    return results


def evaluate_scene(scene: str, variants: list[str] | None = None) -> dict:
    """Full evaluation for one scene. Returns {variant: {LPIPS, SSIM, PSNR, SCORE}}."""
    print(f"\n{'='*60}")
    print(f"EVALUATE: {scene}")
    print(f"{'='*60}")

    if variants is None:
        variants = [v.name for v in VARIANTS if v.eval_mode and v.iters >= 7000]

    print(f"  Variants: {', '.join(variants)}")
    print(f"  Formula: Score = {LPIPS_WEIGHT}*(1-LPIPS) + {SSIM_WEIGHT}*SSIM + {PSNR_WEIGHT}*PSNR/{PSNR_MAX}")

    results = evaluate_with_metrics_py(scene, variants)

    # Summary
    if results:
        print(f"\n  {'─'*50}")
        print(f"  {'Variant':<20} {'LPIPS':>8} {'SSIM':>8} {'PSNR':>8} {'SCORE':>8}")
        print(f"  {'─'*50}")
        sorted_results = sorted(results.items(), key=lambda x: x[1]["SCORE"], reverse=True)
        for vname, m in sorted_results:
            print(f"  {vname:<20} {m['LPIPS']:>8.4f} {m['SSIM']:>8.4f} "
                  f"{m['PSNR']:>8.2f} {m['SCORE']:>8.4f}")

        best = sorted_results[0]
        print(f"\n  🏆 BEST: {best[0]} — Score={best[1]['SCORE']:.4f}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VAR 2026 Competition Metric Evaluation")
    p.add_argument("--scene", required=True)
    p.add_argument("--variant", default=None, help="Single variant to evaluate")
    p.add_argument("--all-variants", action="store_true", help="Evaluate all trained variants")
    args = p.parse_args()

    scene = args.scene

    if args.all_variants:
        variants = [v.name for v in VARIANTS if v.eval_mode and v.iters >= 7000]
    elif args.variant:
        variants = [args.variant]
    else:
        variants = None  # default: all eval-enabled

    evaluate_scene(scene, variants)
