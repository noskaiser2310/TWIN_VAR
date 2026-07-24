"""Enhanced Per-Pixel Ensemble Blending — v2.0.

Three breakthroughs over simple averaging:
  1. Per-pixel VARIANCE weighting — high-disagreement pixels favor the anchor
  2. Score-weighted SOFTMAX temperature — per-scene optimal sharpness
  3. Protected-ANCHOR strategy — override only when ALL companions agree

Architecture:
    blend_pixel_enhanced(sources) → variance_mask → softmax(temp) → anchor_protect → output

Usage:
    python ensemble.py --scene bonsai
    python ensemble.py --scene HCM0421 --temperature 3.0
    python ensemble.py --scene bonsai --no-anchor
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as _cfg
from config import (
    OUTPUT_DIR,
    ENSEMBLE_VARIANTS, ENSEMBLE_FALLBACK, ENSEMBLE_PRIORS,
    ENSEMBLE_ANCHOR, ENSEMBLE_NUM_COMPANIONS, ENSEMBLE_AGREEMENT_THRESHOLD,
    ENSEMBLE_TEMPERATURE, ENSEMBLE_VARIANCE_WEIGHT,
    PER_SCENE_ENSEMBLE,
    set_data_dir,
)


# ═══════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION (reused from original ensemble)
# ═══════════════════════════════════════════════════════════════

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
        renders[p.name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return renders


def gradient_mag(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude."""
    g = np.mean(img, axis=2) if img.ndim == 3 else img
    gy, gx = np.gradient(g)
    return np.sqrt(gx ** 2 + gy ** 2)


def local_std(img: np.ndarray, w: int = 7) -> np.ndarray:
    """Local std via box filter."""
    m = uniform_filter(img, w)
    ms = uniform_filter(img ** 2, w)
    return np.sqrt(np.maximum(ms - m ** 2, 0))


# ═══════════════════════════════════════════════════════════════
#  CONFIDENCE SIGNALS — 5 signals per variant per pixel
# ═══════════════════════════════════════════════════════════════

def compute_confidence_signals(
    sources: dict[str, np.ndarray],
    priors: dict[str, float],
) -> list[np.ndarray]:
    """Compute 5 confidence signals for each variant.

    Returns:
        confs: list of [H, W, N] stacked (for softmax) + raw signals for tuning
    """
    variants = list(sources.keys())
    h, w = list(sources.values())[0].shape[:2]
    confs = []

    for v in variants:
        img = sources[v]
        gray = np.mean(img, axis=2)

        # 1. Local contrast (alpha saturation proxy)
        c1 = local_std(gray, 7)
        c1 = c1 / (c1.max() + 1e-8)

        # 2. Gradient agreement with other variants
        g_self = gradient_mag(img)
        other_g = np.stack(
            [gradient_mag(sources[ov]) for ov in variants if ov != v], axis=-1
        )
        g_mean = other_g.mean(axis=-1)
        corr = (g_self - g_self.mean()) * (g_mean - g_mean.mean())
        corr /= (g_self.std() * g_mean.std() + 1e-8)
        c2 = np.clip(corr, 0, 1)

        # 3. Color smoothness (inverse of local std)
        c3 = 1.0 - local_std(img, 5)

        # 4. Edge density (sharpness preference)
        gn = gradient_mag(img)
        gn = gn / (gn.max() + 1e-8)
        edges = (gn > 0.1).astype(np.float32)
        c4 = gaussian_filter(edges, sigma=2.0)

        # 5. Variant prior
        p = priors.get(v, 0.5)
        c5 = np.full((h, w), p, dtype=np.float32)

        # Combine: contrast 0.20 + agreement 0.25 + smoothness 0.15 + edge 0.15 + prior 0.25
        conf = 0.20 * c1 + 0.25 * c2 + 0.15 * c3 + 0.15 * c4 + 0.25 * c5
        conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
        confs.append(conf)

    return confs


# ═══════════════════════════════════════════════════════════════
#  CORE: Enhanced Ensemble Blending
# ═══════════════════════════════════════════════════════════════

def blend_pixel_enhanced(
    sources: dict[str, np.ndarray],
    priors: dict[str, float] | None = None,
    anchor: str | None = None,
    temperature: float | None = None,
    num_companions: int | None = None,
    agreement_threshold: float | None = None,
    variance_weight: float | None = None,
) -> np.ndarray:
    """Enhanced ensemble with variance weighting + softmax + protected anchor.

    Args:
        sources: {variant_name: [H, W, 3]} rendered images
        priors: per-variant quality weights (higher = better)
        anchor: variant to use as protected base (None = no anchor)
        temperature: softmax temperature (lower = sharper)
        num_companions: how many non-anchor variants must agree
        agreement_threshold: max per-pixel std for agreement
        variance_weight: how much to penalize high-variance regions

    Returns:
        blended: [H, W, 3] float32 in [0, 1]
    """
    if priors is None:
        priors = ENSEMBLE_PRIORS
    if anchor is None:
        anchor = ENSEMBLE_ANCHOR
    if temperature is None:
        temperature = ENSEMBLE_TEMPERATURE
    if num_companions is None:
        num_companions = ENSEMBLE_NUM_COMPANIONS
    if agreement_threshold is None:
        agreement_threshold = ENSEMBLE_AGREEMENT_THRESHOLD
    if variance_weight is None:
        variance_weight = ENSEMBLE_VARIANCE_WEIGHT

    variants = list(sources.keys())
    if len(variants) == 1:
        return sources[variants[0]]

    h, w = list(sources.values())[0].shape[:2]

    # ── Step 1: Per-pixel confidence signals ──
    confs = compute_confidence_signals(sources, priors)
    stacked_conf = np.stack(confs, axis=-1)  # [H, W, N]

    # ── Step 2: Per-pixel VARIANCE weighting ──
    # Where ensemble members disagree → prefer anchor over risky blend
    stacked_renders = np.stack([sources[v] for v in variants], axis=0)  # [N, H, W, 3]
    pixel_std = stacked_renders.std(axis=0).mean(axis=2)  # [H, W] mean RGB std
    # Variance mask: high-variance pixels get 0 weight → trust anchor
    variance_mask = 1.0 / (1.0 + 5.0 * pixel_std)  # [H, W] in [0.16, 1.0]

    # Apply variance penalty to all confidences equally
    # In high-variance regions, reduce ALL confidences → anchor wins
    for i in range(len(confs)):
        confs[i] *= (1.0 - variance_weight + variance_weight * variance_mask)

    # Re-stack after variance adjustment
    stacked_conf = np.stack(confs, axis=-1)

    # ── Step 3: Score-weighted SOFTMAX with temperature ──
    prior_weights = np.array([priors.get(v, 0.5) for v in variants])
    prior_weights = prior_weights / prior_weights.sum()

    # Softmax: exp(conf / T) * prior
    weights = np.exp(stacked_conf / temperature)  # [H, W, N]
    weights *= prior_weights[None, None, :]       # Apply per-variant prior
    weights /= weights.sum(axis=-1, keepdims=True) + 1e-8

    # ── Step 4: Protected-ANCHOR blending (VECTORIZED) ──
    if anchor is not None and anchor in variants:
        anchor_idx = variants.index(anchor)
        anchor_img = sources[anchor]

        # Clamp companions to avoid self-selection
        k_comp = min(num_companions, len(variants) - 1)
        if k_comp <= 0:
            return anchor_img

        # Vectorized companion selection: mask anchor → argsort → top-K
        conf_no_anchor = stacked_conf.copy()
        conf_no_anchor[:, :, anchor_idx] = -np.inf
        companion_mask = np.argsort(-conf_no_anchor, axis=-1)[:, :, :k_comp]  # [H, W, K]

        # Advanced indexing meshgrid for vectorized per-pixel selection
        y_idx = np.arange(h)[:, None, None]  # [H, 1, 1]
        x_idx = np.arange(w)[None, :, None]  # [1, W, 1]

        # Vectorized companion renders via advanced indexing
        all_renders = np.stack([sources[v] for v in variants], axis=2)  # [H, W, N, 3]
        companion_renders = all_renders[y_idx, x_idx, companion_mask]  # [H, W, K, 3]

        # Companion agreement: low std across companions → they agree
        companion_std = companion_renders.std(axis=2).mean(axis=2)  # [H, W]
        high_agreement = companion_std < agreement_threshold

        # Vectorized companion weights + blend
        companion_w = weights[y_idx, x_idx, companion_mask]  # [H, W, K]
        companion_w /= companion_w.sum(axis=-1, keepdims=True) + 1e-8
        companion_blend = np.sum(
            companion_renders * companion_w[:, :, :, None], axis=2
        )  # [H, W, 3]

        # Vectorized companion confidence (max of top-K)
        companion_conf = stacked_conf[y_idx, x_idx, companion_mask].max(axis=2)  # [H, W]
        anchor_conf = stacked_conf[:, :, anchor_idx]

        # Override: companions agree AND have meaningfully higher confidence
        override = high_agreement & (companion_conf > anchor_conf * 1.05)

        # Start with anchor, override where companions win
        blended = anchor_img.copy()
        blended[override] = companion_blend[override]
        return np.clip(blended, 0, 1)

    # ── Fallback: standard weighted blend (no anchor or anchor not available) ──
    blended = np.zeros((h, w, 3), dtype=np.float32)
    for i, v in enumerate(variants):
        blended += sources[v] * weights[:, :, i:i+1]
    return np.clip(blended, 0, 1)


# ═══════════════════════════════════════════════════════════════
#  SCENE-LEVEL ENSEMBLE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def ensemble_scene(
    scene: str,
    variants: list[str] | None = None,
    temperature: float | None = None,
    disable_anchor: bool = False,
) -> dict[str, np.ndarray]:
    """Run enhanced ensemble for an entire scene.

    Loads renders from all available variants, blends each test pose,
    and saves results to OUTPUT_DIR/ensemble/<scene>/.

    Args:
        scene: scene name (e.g., "bonsai", "HCM0421")
        variants: list of variant names (default: ENSEMBLE_VARIANTS)
        temperature: softmax temperature (None = auto from PER_SCENE_ENSEMBLE)
        disable_anchor: if True, skip protected-anchor blending
    """
    if variants is None:
        variants = ENSEMBLE_VARIANTS

    # Per-scene config
    scene_cfg = PER_SCENE_ENSEMBLE.get(scene, {})
    if temperature is None:
        temperature = scene_cfg.get("temperature", ENSEMBLE_TEMPERATURE)
    anchor = None if disable_anchor else scene_cfg.get("anchor", ENSEMBLE_ANCHOR)

    out_dir = OUTPUT_DIR / "ensemble" / scene
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read test poses
    test_csv = _cfg.DATA_DIR / scene / "test" / "test_poses.csv"
    if not test_csv.exists():
        test_csv = _cfg.DATA_DIR / scene / "test_poses.csv"
    with open(test_csv) as f:
        test_poses = list(csv.DictReader(f))

    # Load renders from all available variants
    all_renders: dict[str, dict[str, np.ndarray]] = {}
    for v in variants:
        r = load_renders(scene, v)
        if r:
            all_renders[v] = r

    available = list(all_renders.keys())
    print(f"\n[ENSEMBLE v2] {scene}")
    print(f"  Variants: {len(available)} ({', '.join(available)})")
    print(f"  Temperature: {temperature}")
    print(f"  Anchor: {anchor if not disable_anchor else 'DISABLED'}")

    if len(available) < 2:
        print("  Need >= 2 variants. Using single variant fallback.")
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
                cv2.imwrite(
                    str(out_dir / name),
                    cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                )
        return rendered

    # Blend each test pose
    rendered = {}
    n_poses = len(test_poses)

    for i, pose in enumerate(test_poses):
        name = pose["image_name"]

        # Gather sources for this pose
        sources = {}
        for v in available:
            if name in all_renders[v]:
                sources[v] = all_renders[v][name]

        if not sources:
            for fb in ENSEMBLE_FALLBACK:
                if fb in all_renders and name in all_renders[fb]:
                    sources = {fb: all_renders[fb][name]}
                    break
        if not sources:
            continue

        # Blend
        blended = blend_pixel_enhanced(
            sources,
            temperature=temperature,
            anchor=anchor if anchor in sources else None,
        )

        rendered[name] = blended
        cv2.imwrite(
            str(out_dir / name),
            cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_poses}")

    print(f"  [OK] {scene}: {len(rendered)}/{n_poses} images blended")
    return rendered


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Enhanced Ensemble Blending v2.0")
    p.add_argument("--scene", required=True, help="Scene name")
    p.add_argument("--variants", default=None, help="Comma-separated variant names")
    p.add_argument("--temperature", type=float, default=None,
                   help="Softmax temperature (default: per-scene auto)")
    p.add_argument("--no-anchor", action="store_true",
                   help="Disable protected-anchor blending")
    p.add_argument("--data-dir", default=None, help="Override data directory")
    args = p.parse_args()

    if args.data_dir:
        set_data_dir(args.data_dir)

    variants = args.variants.split(",") if args.variants else None
    ensemble_scene(
        args.scene,
        variants=variants,
        temperature=args.temperature,
        disable_anchor=args.no_anchor,
    )
