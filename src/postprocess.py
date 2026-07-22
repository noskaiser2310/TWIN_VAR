"""Post-processing: sharpen + color correction for rendered images.

Usage:
    python postprocess.py --scene bonsai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, OUTPUT_DIR, SHARPEN_AMOUNT, SHARPEN_RADIUS, COLOR_MATCH


def compute_train_stats(scene: str) -> dict:
    """Compute mean/std of training images for color matching."""
    img_dir = DATA_DIR / scene / "train" / "images"
    if not img_dir.exists():
        img_dir = DATA_DIR / scene / "images"
    if not img_dir.exists():
        return {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}

    means, stds = [], []
    for p in sorted(img_dir.glob("*"))[:50]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        means.append(img.mean(axis=(0, 1)))
        stds.append(img.std(axis=(0, 1)))
    if not means:
        return {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}
    return {"mean": np.mean(means, axis=0).tolist(), "std": np.mean(stds, axis=0).tolist()}


def sharpen(img: np.ndarray, amount: float = SHARPEN_AMOUNT, radius: float = SHARPEN_RADIUS) -> np.ndarray:
    """Edge-aware unsharp mask."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius)
    detail = img - blurred
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100).astype(np.float32) / 255.0
    edges = cv2.GaussianBlur(edges, (5, 5), 1.0)
    edges = np.expand_dims(edges, axis=-1)
    return np.clip(img + amount * (0.3 + 0.7*edges) * detail, 0, 1)


def match_colors(img: np.ndarray, stats: dict) -> np.ndarray:
    """Match color distribution to training stats."""
    tmean = np.array(stats["mean"])
    tstd = np.array(stats["std"])
    result = img.copy()
    for c in range(3):
        ch = result[:, :, c]
        s = ch.std()
        if s > 0:
            ch = (ch - ch.mean()) * (tstd[c] / s) + tmean[c]
        result[:, :, c] = ch
    return np.clip(result, 0, 1)


def process_scene(scene: str, input_dir: Path | None = None) -> int:
    """Post-process all images for a scene."""
    if input_dir is None:
        input_dir = OUTPUT_DIR / "ensemble" / scene
    if not input_dir.exists():
        input_dir = OUTPUT_DIR / "renders" / scene
        if input_dir.exists():
            for d in sorted(input_dir.iterdir()):
                if d.is_dir():
                    input_dir = d
                    break

    out_dir = OUTPUT_DIR / "final" / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_train_stats(scene) if COLOR_MATCH else None

    count = 0
    for p in sorted(input_dir.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = sharpen(img)
        if stats and COLOR_MATCH:
            img = match_colors(img, stats)
        cv2.imwrite(str(out_dir / p.name),
                    cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        count += 1

    print(f"[POSTPROCESS] {scene}: {count} images enhanced")
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--input", default=None)
    args = p.parse_args()
    process_scene(args.scene, Path(args.input) if args.input else None)
