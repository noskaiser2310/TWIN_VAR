"""Smart Per-Pixel Ensemble Blending.

Blends renders from multiple 3DGS variants using per-pixel confidence scoring.
For each pixel, selects the best variant based on 5 signals:
  1. Local contrast (alpha saturation proxy)
  2. Gradient agreement with other variants (depth consistency proxy)
  3. Color smoothness (low noise preference)
  4. Edge density (sharpness preference)
  5. Variant prior (pre-computed quality weight)

Usage:
    python ensemble.py --scene bonsai
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, OUTPUT_DIR, ENSEMBLE_VARIANTS, ENSEMBLE_FALLBACK, ENSEMBLE_PRIORS


def load_renders(scene: str, variant: str) -> dict[str, np.ndarray]:
    """Load all rendered PNGs for a variant as float32 arrays [0,1]."""
    render_dir = OUTPUT_DIR / "renders" / scene / variant
    if not render_dir.exists():
        return {}
    renders = {}
    for p in sorted(render_dir.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        renders[p.name] = img
    return renders


def gradient_mag(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude."""
    g = np.mean(img, axis=2) if img.ndim == 3 else img
    gy, gx = np.gradient(g)
    return np.sqrt(gx**2 + gy**2)


def local_std(img: np.ndarray, w: int = 7) -> np.ndarray:
    """Local std via box filter."""
    m = uniform_filter(img, w)
    ms = uniform_filter(img**2, w)
    return np.sqrt(np.maximum(ms - m**2, 0))


def blend_pixel(sources: dict[str, np.ndarray], priors: dict[str, float] | None = None) -> np.ndarray:
    """Per-pixel soft-voting blend from multiple renders."""
    if priors is None:
        priors = ENSEMBLE_PRIORS

    variants = list(sources.keys())
    if len(variants) == 1:
        return sources[variants[0]]

    h, w = list(sources.values())[0].shape[:2]
    confs = []

    for v in variants:
        img = sources[v]
        gray = np.mean(img, axis=2)
        # 1. Local contrast
        c1 = local_std(gray, 7)
        c1 = c1 / (c1.max() + 1e-8)
        # 2. Gradient agreement with others
        g_self = gradient_mag(img)
        other_g = np.stack([gradient_mag(sources[ov]) for ov in variants if ov != v], axis=-1)
        g_mean = other_g.mean(axis=-1)
        corr = (g_self - g_self.mean()) * (g_mean - g_mean.mean())
        corr /= (g_self.std() * g_mean.std() + 1e-8)
        c2 = np.clip(corr, 0, 1)
        # 3. Color smoothness
        c3 = 1.0 - local_std(img, 5)
        # 4. Edge density
        gn = gradient_mag(img)
        gn = gn / (gn.max() + 1e-8)
        edges = (gn > 0.1).astype(np.float32)
        c4 = gaussian_filter(edges, sigma=2.0)
        # 5. Prior
        p = priors.get(v, 0.5)
        c5 = np.full((h, w), p, dtype=np.float32)
        # Combine
        conf = 0.2 * c1 + 0.3 * c2 + 0.2 * c3 + 0.15 * c4 + 0.15 * c5
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        confs.append(conf)

    stacked = np.stack(confs, axis=-1)
    weights = np.exp(stacked * 2.0)
    weights /= weights.sum(axis=-1, keepdims=True) + 1e-8

    blended = np.zeros((h, w, 3), dtype=np.float32)
    for i, v in enumerate(variants):
        blended += sources[v] * weights[:, :, i:i+1]
    return np.clip(blended, 0, 1)


def ensemble_scene(scene: str, variants: list[str] | None = None) -> dict[str, np.ndarray]:
    """Run smart ensemble for an entire scene."""
    if variants is None:
        variants = ENSEMBLE_VARIANTS
    out_dir = OUTPUT_DIR / "ensemble" / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    test_csv = DATA_DIR / scene / "test" / "test_poses.csv"
    if not test_csv.exists():
        test_csv = DATA_DIR / scene / "test_poses.csv"
    with open(test_csv) as f:
        test_poses = list(csv.DictReader(f))

    # Load renders from all available variants
    all_renders: dict[str, dict[str, np.ndarray]] = {}
    for v in variants:
        r = load_renders(scene, v)
        if r:
            all_renders[v] = r

    available = list(all_renders.keys())
    print(f"[ENSEMBLE] {scene}: {len(available)} variants ({', '.join(available)})")
    if len(available) < 2:
        print("  Need >= 2 variants for ensemble. Using single variant.")
        # Just copy from best available
        best = available[0] if available else ENSEMBLE_FALLBACK[0]
        if best not in all_renders:
            for fb in ENSEMBLE_FALLBACK:
                if fb in all_renders:
                    best = fb
                    break
        rendered = {}
        for pose in test_poses:
            name = pose["image_name"]
            if name in all_renders.get(best, {}):
                img = all_renders[best][name]
                rendered[name] = img
                cv2.imwrite(str(out_dir / name),
                            cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        return rendered

    rendered = {}
    for i, pose in enumerate(test_poses):
        name = pose["image_name"]
        sources = {}
        for v in available:
            if name in all_renders[v]:
                sources[v] = all_renders[v][name]
        if not sources:
            # Fallback
            for fb in ENSEMBLE_FALLBACK:
                if fb in all_renders and name in all_renders[fb]:
                    sources = {fb: all_renders[fb][name]}
                    break
        if not sources:
            continue

        blended = blend_pixel(sources)
        rendered[name] = blended
        cv2.imwrite(str(out_dir / name),
                    cv2.cvtColor((blended*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(test_poses)}")

    print(f"  [OK] {scene}: {len(rendered)} images blended")
    return rendered


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--variants", default=None, help="comma-separated")
    args = p.parse_args()
    variants = args.variants.split(",") if args.variants else None
    ensemble_scene(args.scene, variants)
