"""Post-Processing Pipeline for 3DGS Rendered Images.

Enhances rendered Novel View Synthesis images with:
1. Edge-aware sharpening (unsharp mask)
2. Color distribution matching (match training images)
3. Sky region denoising (reduce floaters)

Expected LPIPS improvement: -0.005 to -0.01
Expected SSIM improvement: +0.005 to +0.01
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config_enhanced import PostProcessConfig, DATA_DIR


class PostProcessor:
    """Enhances rendered 3DGS images."""

    def __init__(
        self,
        config: Optional[PostProcessConfig] = None,
        train_stats: Optional[dict] = None,
    ):
        """
        Args:
            config: Post-processing configuration
            train_stats: Training image statistics {mean: [R,G,B], std: [R,G,B]}
                        Computed from scene training images for color matching.
        """
        self.config = config or PostProcessConfig()
        self.train_stats = train_stats

    def process(self, img: np.ndarray, sky_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply all post-processing steps to a rendered image.

        Args:
            img: Input image (H, W, 3) in [0, 1] float32
            sky_mask: Optional binary sky mask (H, W), 1=sky

        Returns:
            Processed image (H, W, 3) in [0, 1] float32
        """
        result = img.copy()

        # Step 1: Edge-aware sharpening
        if self.config.sharpen_enabled:
            result = self._edge_aware_sharpen(result)

        # Step 2: Color correction
        if self.config.color_correction_enabled and self.train_stats is not None:
            result = self._match_color_distribution(result)

        # Step 3: Sky denoising
        if self.config.sky_denoise_enabled and sky_mask is not None:
            result = self._denoise_sky(result, sky_mask)

        # Clip to valid range
        result = np.clip(result, 0.0, 1.0)

        return result

    def _edge_aware_sharpen(self, img: np.ndarray) -> np.ndarray:
        """Edge-aware unsharp mask sharpening.

        Formula: sharpened = img + amount * (img - blurred)
        Applies more sharpening at edges, less at flat regions.
        """
        amount = self.config.sharpen_amount
        radius = self.config.sharpen_radius

        # Gaussian blur
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=radius)

        # Detail layer (high-frequency)
        detail = img - blurred

        # Edge mask: weight sharpening by edge strength
        gray = cv2.cvtColor(
            (img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )
        edges = cv2.Canny(gray, 30, 100).astype(np.float32) / 255.0
        edges = cv2.GaussianBlur(edges, (5, 5), 1.0)
        edges = np.expand_dims(edges, axis=-1)

        # Adaptive sharpening: stronger at edges
        adaptive_amount = amount * (0.3 + 0.7 * edges)
        sharpened = img + adaptive_amount * detail

        return sharpened

    def _match_color_distribution(self, img: np.ndarray) -> np.ndarray:
        """Match color distribution to training image statistics.

        Uses mean/std matching per channel.
        """
        if self.train_stats is None:
            return img

        target_mean = np.array(self.train_stats["mean"])
        target_std = np.array(self.train_stats["std"])

        result = img.copy()
        for c in range(3):
            channel = result[:, :, c]
            src_mean = channel.mean()
            src_std = channel.std()

            if src_std > 0:
                channel = (channel - src_mean) * (target_std[c] / src_std)
                channel = channel + target_mean[c]
            result[:, :, c] = channel

        return result

    def _denoise_sky(self, img: np.ndarray, sky_mask: np.ndarray) -> np.ndarray:
        """Apply denoising to sky regions only.

        Sky regions in 3DGS often have floaters/artifacts.
        Gentle bilateral filter preserves edges at sky/object boundary.
        """
        strength = self.config.sky_denoise_strength

        # Bilateral filter on the whole image
        img_uint8 = (img * 255).astype(np.uint8)
        denoised = cv2.bilateralFilter(img_uint8, 9, strength * 10, strength * 10)
        denoised = denoised.astype(np.float32) / 255.0

        # Blend: use denoised for sky, keep original for non-sky
        mask_3ch = np.expand_dims(sky_mask.astype(np.float32), axis=-1)
        result = mask_3ch * denoised + (1.0 - mask_3ch) * img

        return result


def compute_train_stats(scene: str) -> dict:
    """Compute mean and std of training images for a scene.

    Used for color distribution matching in post-processing.
    """
    scene_dir = DATA_DIR / scene
    img_dir = scene_dir / "train" / "images"
    if not img_dir.exists():
        img_dir = scene_dir / "images"

    if not img_dir.exists():
        print(f"  [WARN] No training images for {scene}")
        return {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}

    means = []
    stds = []

    for img_path in sorted(img_dir.glob("*"))[:50]:  # Sample 50 images
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            means.append(img.mean(axis=(0, 1)))
            stds.append(img.std(axis=(0, 1)))
        except Exception:
            continue

    if not means:
        return {"mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}

    return {
        "mean": np.mean(means, axis=0).tolist(),
        "std": np.mean(stds, axis=0).tolist(),
    }


def post_process_scene(
    scene: str,
    input_dir: Path,
    output_dir: Path,
    sky_mask_dir: Optional[Path] = None,
    config: Optional[PostProcessConfig] = None,
) -> int:
    """Post-process all rendered images for a scene.

    Args:
        scene: Scene name
        input_dir: Directory containing rendered PNG images
        output_dir: Directory to save processed images
        sky_mask_dir: Optional directory with sky mask PNGs (same names as renders)
        config: Post-processing configuration

    Returns:
        Number of images processed
    """
    print(f"\n{'='*60}")
    print(f"POST-PROCESS: {scene}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Compute training stats
    train_stats = compute_train_stats(scene)
    print(f"  Train mean: {[f'{v:.3f}' for v in train_stats['mean']]}")
    print(f"  Train std:  {[f'{v:.3f}' for v in train_stats['std']]}")

    processor = PostProcessor(config, train_stats)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(input_dir.glob("*.png")):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load sky mask if available
        sky_mask = None
        if sky_mask_dir is not None:
            mask_path = sky_mask_dir / img_path.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    sky_mask = mask.astype(np.float32) / 255.0

        # Process
        result = processor.process(img, sky_mask)

        # Save
        out_path = output_dir / img_path.name
        result_bgr = cv2.cvtColor(
            (result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(str(out_path), result_bgr)

        count += 1
        if count % 20 == 0:
            print(f"  Processed {count} images...")

    print(f"  ✅ {count} images post-processed")
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post-Process Rendered Images")
    parser.add_argument("--scene", required=True, help="Scene name")
    parser.add_argument("--input", required=True, help="Input directory with PNGs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sky-masks", default=None, help="Sky masks directory")
    args = parser.parse_args()

    post_process_scene(
        scene=args.scene,
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        sky_mask_dir=Path(args.sky_masks) if args.sky_masks else None,
    )
