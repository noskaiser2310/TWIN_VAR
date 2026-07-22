"""Smart Per-Pixel Ensemble Blending Engine.

Replaces the simple fallback blending in package_submission.py with
per-pixel confidence-based selection from multiple 3DGS variants.

Based on research: Self-Ensembling Gaussian Splatting (SE-GS),
Gaussian Blending (Koo et al., 2024), and uncertainty-aware perturbation.

Key features:
- Per-pixel confidence scoring (alpha, depth consistency, color, edge)
- Soft voting with temperature scaling
- Scene-specific variant weighting
- Smart fallback when some variants missing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config_enhanced import (
    EnsembleConfig,
    OUTPUT_DIR,
    DATA_DIR,
)


class SmartEnsembleEngine:
    """Per-pixel confidence-based ensemble blending for 3DGS renders."""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()

    def blend_scene(
        self,
        scene: str,
        variant_renders: dict[str, dict[str, np.ndarray]],
        test_pose: dict,
    ) -> np.ndarray:
        """Blend multiple variant renders for one test pose.

        Args:
            scene: Scene name
            variant_renders: {variant_name: {image_name: np.ndarray(H,W,3)}}
            test_pose: Dict with image_name, width, height

        Returns:
            Blended image as np.ndarray (H, W, 3) in [0, 1]
        """
        img_name = test_pose["image_name"]
        h, w = int(float(test_pose["height"])), int(float(test_pose["width"]))

        # Collect available renders for this view
        renders: dict[str, np.ndarray] = {}
        for v in self.config.variants:
            if v in variant_renders and img_name in variant_renders[v]:
                renders[v] = variant_renders[v][img_name]

        if not renders:
            # Try fallback order
            for v in self.config.fallback_order:
                if v in variant_renders and img_name in variant_renders[v]:
                    return variant_renders[v][img_name]
            raise ValueError(f"No render found for {scene}/{img_name}")

        if len(renders) == 1:
            return list(renders.values())[0]

        # Compute per-pixel confidence maps
        conf_maps = self._compute_confidence_maps(renders, scene)

        # Soft voting blend
        blended = self._soft_voting_blend(renders, conf_maps, h, w)

        return blended

    def _compute_confidence_maps(
        self, renders: dict[str, np.ndarray], scene: str
    ) -> dict[str, np.ndarray]:
        """Compute per-pixel confidence map for each variant.

        Confidence = w1*alpha + w2*depth_consistency + w3*color_smoothness
                    + w4*edge_sharpness + w5*variant_prior
        """
        cf = self.config
        confs: dict[str, np.ndarray] = {}

        for variant, img in renders.items():
            h, w = img.shape[:2]

            # Signal 1: Local contrast (proxy for alpha saturation)
            gray = np.mean(img, axis=2)
            local_std = self._local_std(gray, window=7)
            alpha_conf = local_std / (local_std.max() + 1e-8)

            # Signal 2: Depth consistency (agreement with other variants)
            depth_conf = self._depth_consistency_confidence(variant, renders)

            # Signal 3: Color smoothness (low = noisy region)
            color_conf = 1.0 - self._local_color_std(img, window=5)

            # Signal 4: Edge sharpness (prefer sharp edges)
            edge_conf = self._edge_density(img)

            # Signal 5: Variant prior (learned quality weight)
            prior = cf.variant_priors.get(variant, 0.5)
            prior_map = np.full((h, w), prior, dtype=np.float32)

            # Combine
            confidence = (
                cf.alpha_weight * alpha_conf
                + cf.depth_consistency_weight * depth_conf
                + cf.color_consistency_weight * color_conf
                + cf.edge_sharpness_weight * edge_conf
                + cf.variant_prior_weight * prior_map
            )

            # Normalize
            conf_min = confidence.min()
            conf_max = confidence.max()
            if conf_max > conf_min:
                confidence = (confidence - conf_min) / (conf_max - conf_min)

            confs[variant] = confidence.astype(np.float32)

        return confs

    def _depth_consistency_confidence(
        self, variant: str, renders: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Estimate depth consistency by measuring agreement with other variants.

        Uses image gradient as a proxy for depth structure:
        - High agreement in gradient structure → high depth consistency
        """
        h, w = list(renders.values())[0].shape[:2]
        other_variants = [v for v in renders if v != variant]

        if not other_variants:
            return np.ones((h, w), dtype=np.float32)

        # Compute gradient magnitude for each variant
        current_grad = self._gradient_magnitude(renders[variant])
        other_grads = np.stack(
            [self._gradient_magnitude(renders[v]) for v in other_variants],
            axis=-1,
        )

        # Gradient correlation = depth consistency proxy
        other_mean = other_grads.mean(axis=-1)
        other_std = other_grads.std(axis=-1)

        # Normalized cross-correlation
        correlation = (current_grad - current_grad.mean()) * (
            other_mean - other_mean.mean()
        )
        correlation /= current_grad.std() * other_std + 1e-8
        correlation = np.clip(correlation, 0, 1)

        return correlation.astype(np.float32)

    def _soft_voting_blend(
        self,
        renders: dict[str, np.ndarray],
        confs: dict[str, np.ndarray],
        h: int, w: int,
    ) -> np.ndarray:
        """Soft voting: weight each pixel by exp(confidence * temperature)."""
        variant_list = list(renders.keys())
        n_variants = len(variant_list)

        if n_variants == 1:
            return renders[variant_list[0]]

        # Stack confidence maps
        stacked_confs = np.stack(
            [confs[v] for v in variant_list], axis=-1
        )  # (H, W, N)

        # Softmax with temperature
        temp = self.config.soft_voting_temperature
        exp_confs = np.exp(stacked_confs * temp)
        weights = exp_confs / (exp_confs.sum(axis=-1, keepdims=True) + 1e-8)

        # Weighted blend
        blended = np.zeros((h, w, 3), dtype=np.float32)
        for i, v in enumerate(variant_list):
            blended += renders[v] * weights[:, :, i : i + 1]

        return np.clip(blended, 0, 1)

    # ── Utility functions ──────────────────────────────────

    def _local_std(self, img: np.ndarray, window: int = 7) -> np.ndarray:
        """Compute local standard deviation using box filter."""
        from scipy.ndimage import uniform_filter

        mean = uniform_filter(img, window)
        mean_sq = uniform_filter(img**2, window)
        variance = mean_sq - mean**2
        return np.sqrt(np.maximum(variance, 0))

    def _local_color_std(self, img: np.ndarray, window: int = 5) -> np.ndarray:
        """Compute per-pixel local color standard deviation."""
        from scipy.ndimage import uniform_filter

        stds = []
        for c in range(3):
            channel = img[:, :, c]
            mean = uniform_filter(channel, window)
            mean_sq = uniform_filter(channel**2, window)
            variance = mean_sq - mean**2
            stds.append(np.sqrt(np.maximum(variance, 0)))
        return np.mean(stds, axis=0)  # Average across channels

    def _gradient_magnitude(self, img: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude (Sobel)."""
        gray = np.mean(img, axis=2) if img.ndim == 3 else img
        gy, gx = np.gradient(gray)
        return np.sqrt(gx**2 + gy**2)

    def _edge_density(self, img: np.ndarray) -> np.ndarray:
        """Compute local edge density map."""
        grad = self._gradient_magnitude(img)
        # Normalize and threshold to get edge map
        grad_norm = grad / (grad.max() + 1e-8)
        edge_map = (grad_norm > 0.1).astype(np.float32)

        # Gaussian blur to spread edges
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(edge_map, sigma=2.0)


def discover_variant_renders(scene: str, variant: str) -> dict[str, Path]:
    """Find rendered images for a scene from a specific variant."""
    kernel_output = OUTPUT_DIR / "kernel_outputs" / scene
    render_dir = kernel_output / "output" / scene / variant / "test_renders"

    if not render_dir.exists():
        for alt in kernel_output.glob(f"**/output/{scene}/{variant}/test_renders"):
            if alt.is_dir():
                render_dir = alt
                break

    if not render_dir.exists():
        return {}

    images = {}
    for png in sorted(render_dir.glob("*.png")):
        images[png.name] = png

    return images


def load_variant_renders_as_arrays(
    scene: str, variants: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Load all rendered images for variants as numpy arrays."""
    import cv2

    all_renders: dict[str, dict[str, np.ndarray]] = {}

    for variant in variants:
        renders = discover_variant_renders(scene, variant)
        if renders:
            all_renders[variant] = {}
            for img_name, img_path in renders.items():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                all_renders[variant][img_name] = img

    return all_renders


def ensemble_scene(
    scene: str,
    test_poses: list[dict],
    variants: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
    config: Optional[EnsembleConfig] = None,
) -> dict[str, np.ndarray]:
    """Run smart ensemble for an entire scene.

    Args:
        scene: Scene name
        test_poses: List of test pose dicts from test_poses.csv
        variants: List of variant names to use (default: from config)
        output_dir: Where to save blended images (default: OUTPUT_DIR/scene/ensemble/)
        config: Ensemble configuration

    Returns:
        {image_name: blended_numpy_array}
    """
    import cv2

    cfg = config or EnsembleConfig()
    variant_list = variants or cfg.variants

    if output_dir is None:
        output_dir = OUTPUT_DIR / "kernel_outputs" / scene / "ensemble"

    print(f"\n{'='*60}")
    print(f"SMART ENSEMBLE: {scene}")
    print(f"Variants: {', '.join(variant_list)}")
    print(f"{'='*60}")

    # Load all renders
    all_renders = load_variant_renders_as_arrays(scene, variant_list)
    available = [v for v in variant_list if v in all_renders and all_renders[v]]
    print(f"Available variants: {available}")

    if not available:
        print(f"  ❌ No renders found for {scene}!")
        return {}

    # Initialize ensemble engine
    engine = SmartEnsembleEngine(cfg)

    # Blend each test pose
    blended: dict[str, np.ndarray] = {}
    for i, pose in enumerate(test_poses):
        img_name = pose["image_name"]

        try:
            result = engine.blend_scene(scene, all_renders, pose)
            blended[img_name] = result

            # Save to disk
            out_path = output_dir / img_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result_bgr = cv2.cvtColor(
                (result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(str(out_path), result_bgr)

        except Exception as e:
            print(f"  ⚠️  {img_name}: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Rendered {i+1}/{len(test_poses)}")

    print(f"  ✅ {len(blended)}/{len(test_poses)} images blended")
    print(f"  Output: {output_dir}")

    return blended


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Smart Ensemble Engine")
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--variants", default=None, help="Comma-separated variant list")
    parser.add_argument(
        "--test-poses", default=None, help="Path to test_poses.csv"
    )
    args = parser.parse_args()

    scene = args.scene
    variants = args.variants.split(",") if args.variants else None

    # Load test poses
    test_csv = (
        Path(args.test_poses)
        if args.test_poses
        else DATA_DIR / scene / "test" / "test_poses.csv"
    )

    with open(test_csv) as f:
        test_poses = list(csv.DictReader(f))

    ensemble_scene(scene, test_poses, variants)
