"""VAR 2026 — Enhanced Submission Packager with Smart Ensemble + Post-Processing.

Integrates:
- SmartEnsembleEngine: per-pixel confidence-based blending
- PostProcessor: edge-aware sharpening, color matching, sky denoising
- Fallback: if ensemble fails, uses best single variant
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config_enhanced import (
    DATA_DIR,
    SCENES,
    OUTPUT_DIR,
    SUBMISSION_NAME,
    SUBMISSION_DIR,
    EnsembleConfig,
    PostProcessConfig,
    get_ensemble_config,
    get_postprocess_config,
)
from pipeline.smart_ensemble import (
    SmartEnsembleEngine,
    load_variant_renders_as_arrays,
    discover_variant_renders,
)
from pipeline.post_process import PostProcessor, compute_train_stats


def read_test_poses(scene: str) -> list[dict]:
    """Read test_poses.csv for a scene."""
    scene_dir = DATA_DIR / scene
    csv_path = scene_dir / "test" / "test_poses.csv"
    if not csv_path.exists():
        csv_path = scene_dir / "test_poses.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"test_poses.csv not found for {scene}")
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def package_submission_enhanced(
    scenes: Optional[list[str]] = None,
    ensemble_cfg: Optional[EnsembleConfig] = None,
    post_cfg: Optional[PostProcessConfig] = None,
    output_name: Optional[str] = None,
    use_smart_ensemble: bool = True,
    use_post_process: bool = True,
) -> Path:
    """Create submission ZIP with smart ensemble + post-processing.

    For each scene:
    1. Load all variant renders
    2. Smart ensemble blend (per-pixel confidence selection)
    3. Post-process (sharpen, color match, sky denoise)
    4. Package into submission ZIP
    """
    if scenes is None:
        scenes = SCENES
    if ensemble_cfg is None:
        ensemble_cfg = get_ensemble_config()
    if post_cfg is None:
        post_cfg = get_postprocess_config()
    if output_name is None:
        output_name = SUBMISSION_NAME

    print("\n" + "=" * 60)
    print("ENHANCED SUBMISSION PACKAGER")
    print(f"Smart Ensemble: {'ON' if use_smart_ensemble else 'OFF'}")
    print(f"Post-Processing: {'ON' if use_post_process else 'OFF'}")
    print("=" * 60)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SUBMISSION_DIR / output_name

    engine = SmartEnsembleEngine(ensemble_cfg) if use_smart_ensemble else None
    report: dict[str, Any] = {"scenes": {}, "total_images": 0, "missing": []}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in scenes:
            print(f"\n── {scene} ──")
            test_poses = read_test_poses(scene)
            expected_count = len(test_poses)

            # Load all variant renders
            all_renders = load_variant_renders_as_arrays(
                scene, ensemble_cfg.variants
            )
            available = [v for v in ensemble_cfg.variants
                         if v in all_renders and all_renders[v]]
            print(f"  Available variants: {available} ({len(available)})")

            if not available:
                print(f"  ❌ No renders for {scene}, skipping!")
                report["scenes"][scene] = {
                    "expected": expected_count, "added": 0,
                    "missing": expected_count, "error": "No renders available",
                }
                report["missing"].append(scene)
                continue

            # Post-processor for this scene
            post_processor = None
            if use_post_process:
                train_stats = compute_train_stats(scene)
                post_processor = PostProcessor(post_cfg, train_stats)

            # Process each test pose
            added = 0
            scene_prefix = f"{scene}/"

            for i, pose in enumerate(test_poses):
                img_name = pose["image_name"]
                h = int(float(pose["height"]))
                w = int(float(pose["width"]))

                try:
                    if use_smart_ensemble and len(available) >= 2:
                        # Smart ensemble blend
                        result = engine.blend_scene(scene, all_renders, pose)
                    else:
                        # Single best variant
                        best_variant = available[0]
                        if img_name in all_renders[best_variant]:
                            result = all_renders[best_variant][img_name]
                        else:
                            # Try fallback
                            found = False
                            for fb in ensemble_cfg.fallback_order:
                                if (fb in all_renders and
                                        img_name in all_renders[fb]):
                                    result = all_renders[fb][img_name]
                                    found = True
                                    break
                            if not found:
                                raise ValueError(
                                    f"No render for {img_name} in any variant"
                                )

                    # Post-process
                    if post_processor is not None:
                        result = post_processor.process(result)

                    # Resize if needed
                    if result.shape[0] != h or result.shape[1] != w:
                        result = cv2.resize(result, (w, h))

                    # Save to ZIP
                    result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
                    result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)

                    # Write to temp file then add to ZIP
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        cv2.imwrite(tf.name, result_bgr)
                        arcname = scene_prefix + img_name
                        zf.write(tf.name, arcname)
                        Path(tf.name).unlink()

                    added += 1

                except Exception as e:
                    print(f"  ⚠️  {img_name}: {e}")
                    # Try fallback: copy raw render from best variant
                    try:
                        for fb in ensemble_cfg.fallback_order:
                            renders = discover_variant_renders(scene, fb)
                            if img_name in renders:
                                arcname = scene_prefix + img_name
                                zf.write(renders[img_name], arcname)
                                added += 1
                                break
                    except Exception as e2:
                        print(f"    Fallback also failed: {e2}")

                if (i + 1) % 20 == 0:
                    print(f"  Packaged {i+1}/{expected_count}")

            still_missing = expected_count - added
            report["scenes"][scene] = {
                "expected": expected_count,
                "added": added,
                "missing": still_missing,
                "variants_used": available,
            }
            report["total_images"] += added

            if still_missing > 0:
                report["missing"].append(scene)
                print(f"  ⚠️  Missing: {still_missing}/{expected_count}")
            else:
                print(f"  ✅ {added}/{expected_count} packaged")

    # Summary
    print(f"\n{'='*60}")
    print("SUBMISSION SUMMARY")
    print(f"{'='*60}")
    for scene, info in report["scenes"].items():
        status = "✅" if info["missing"] == 0 else "⚠️"
        print(f"  {status} {scene}: {info['added']}/{info['expected']}"
              f" (variants: {info.get('variants_used', [])})")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n  Total: {report['total_images']} images")
    print(f"  File: {zip_path} ({zip_size_mb:.1f} MB)")

    if report["missing"]:
        print(f"\n  ⚠️  WARNING: {len(report['missing'])} scenes have missing images!")
    else:
        print(f"\n  ✅ ALL IMAGES PRESENT — Ready to submit!")

    # Save report
    report_path = SUBMISSION_DIR / f"{output_name}.report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    return zip_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Enhanced Submission Packager with Smart Ensemble"
    )
    parser.add_argument("--scenes", nargs="*", help="Specific scenes")
    parser.add_argument("--output", default=None, help="Output ZIP name")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Disable smart ensemble")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable post-processing")
    args = parser.parse_args()

    zip_path = package_submission_enhanced(
        scenes=args.scenes,
        output_name=args.output,
        use_smart_ensemble=not args.no_ensemble,
        use_post_process=not args.no_postprocess,
    )
