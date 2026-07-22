"""Package rendered images into competition submission ZIP.

Reads rendered outputs from Kaggle kernels, renames images to match
image_name from test_poses.csv, and creates the final submission_round1.zip.

Output structure:
    submission_round1.zip
    ├── scene_001/
    │   ├── 0001.png
    │   └── ...
    └── ...
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (
    DATA_DIR,
    SCENES,
    OUTPUT_DIR,
    SUBMISSION_NAME,
    SUBMISSION_DIR,
)


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


def discover_rendered_images(scene: str, variant: str = "full_combo") -> dict[str, Path]:
    """Find rendered test images for a scene from a specific variant.

    Returns: {image_name: path_to_png}
    """
    kernel_output = OUTPUT_DIR / "kernel_outputs" / scene
    render_dir = kernel_output / "output" / scene / variant / "test_renders"

    if not render_dir.exists():
        # Try other locations
        for alt in kernel_output.glob(f"**/output/{scene}/{variant}/test_renders"):
            if alt.is_dir():
                render_dir = alt
                break

    if not render_dir.exists():
        print(f"  [WARN] No renders found for {scene}/{variant} at {render_dir}")
        return {}

    images = {}
    for png in render_dir.glob("*.png"):
        images[png.name] = png

    return images


def package_submission(
    scenes: list[str] | None = None,
    variant: str = "full_combo",
    fallback_variants: list[str] | None = None,
    output_name: str | None = None,
) -> Path:
    """Create submission ZIP file.

    For each scene, copies rendered images from the chosen variant,
    renaming them to match the required image_name format.
    Falls back to other variants if the primary one has missing images.

    Args:
        scenes: List of scene names (default: all in SCENES)
        variant: Primary variant to use for rendering
        fallback_variants: Fallback variants if primary has missing images
        output_name: Output ZIP filename (default: submission_round1.zip)

    Returns:
        Path to the created ZIP file
    """
    if scenes is None:
        scenes = SCENES

    if fallback_variants is None:
        fallback_variants = ["baseline", "depth", "exposure", "antialias", "fast"]

    if output_name is None:
        output_name = SUBMISSION_NAME

    print("\n" + "=" * 60)
    print("PACKAGE: SUBMISSION")
    print("=" * 60)

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SUBMISSION_DIR / output_name

    report: dict[str, Any] = {"scenes": {}, "total_images": 0, "missing": []}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in scenes:
            print(f"\n── {scene} ──")
            test_poses = read_test_poses(scene)
            expected = {row["image_name"] for row in test_poses}
            print(f"  Expected: {len(expected)} images")

            # Try primary variant
            renders = discover_rendered_images(scene, variant)
            missing = expected - set(renders.keys())

            # Fallback to other variants for missing images
            if missing and fallback_variants:
                print(f"  Missing from {variant}: {len(missing)} images, trying fallbacks...")
                for fb_variant in fallback_variants:
                    if fb_variant == variant:
                        continue
                    if not missing:
                        break
                    fb_renders = discover_rendered_images(scene, fb_variant)
                    fb_found = missing & set(fb_renders.keys())
                    if fb_found:
                        for img_name in fb_found:
                            renders[img_name] = fb_renders[img_name]
                        missing -= fb_found
                        print(f"    {fb_variant}: filled {len(fb_found)} images")

            # Add images to ZIP
            added = 0
            scene_prefix = f"{scene}/"

            for pose in test_poses:
                img_name = pose["image_name"]
                if img_name in renders:
                    arcname = scene_prefix + img_name
                    zf.write(renders[img_name], arcname)
                    added += 1

            still_missing = expected - set(renders.keys())
            report["scenes"][scene] = {
                "expected": len(expected),
                "added": added,
                "missing": len(still_missing),
                "missing_names": list(still_missing)[:10],  # Max 10 shown
            }
            report["total_images"] += added

            if still_missing:
                report["missing"].append(scene)
                print(f"  ⚠️  Still missing: {len(still_missing)} images")
            else:
                print(f"  ✅ All {added}/{len(expected)} images packaged")

    # Summary
    print(f"\n{'='*60}")
    print("SUBMISSION PACKAGE SUMMARY")
    print(f"{'='*60}")
    for scene, info in report["scenes"].items():
        status = "✅" if info["missing"] == 0 else "⚠️"
        print(f"  {status} {scene}: {info['added']}/{info['expected']}")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n  Total: {report['total_images']} images")
    print(f"  File: {zip_path} ({zip_size_mb:.1f} MB)")

    if report["missing"]:
        print(f"\n  ⚠️  WARNING: {len(report['missing'])} scenes have missing images!")
        print(f"  Affected: {', '.join(report['missing'])}")
    else:
        print(f"\n  ✅ ALL IMAGES PRESENT — Ready to submit!")

    # Save report
    report_path = SUBMISSION_DIR / f"{output_name}.report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    return zip_path


def validate_submission_zip(zip_path: Path) -> bool:
    """Validate a submission ZIP against competition requirements."""
    print(f"\n{'='*60}")
    print(f"VALIDATE: {zip_path.name}")
    print(f"{'='*60}")

    valid = True
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        # Check for scene directories
        scenes_in_zip = set()
        for name in names:
            parts = name.split("/")
            if len(parts) >= 1:
                scenes_in_zip.add(parts[0])

        print(f"  Scenes in ZIP: {sorted(scenes_in_zip)}")

        for scene in SCENES:
            if scene not in scenes_in_zip:
                print(f"  ❌ Missing scene: {scene}")
                valid = False
            else:
                test_poses = read_test_poses(scene)
                expected = {row["image_name"] for row in test_poses}
                scene_files = {n.split("/", 1)[1] for n in names if n.startswith(scene + "/")}
                missing = expected - scene_files
                if missing:
                    print(f"  ⚠️  {scene}: missing {len(missing)} images")
                    valid = False
                else:
                    print(f"  ✅ {scene}: {len(expected)} images")

    print(f"\n  Overall: {'✅ VALID' if valid else '❌ INVALID'}")
    return valid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--variant", default="full_combo")
    parser.add_argument("--output", default=None)
    parser.add_argument("--validate-only", type=str, default=None,
                        help="Validate an existing ZIP file")
    args = parser.parse_args()

    if args.validate_only:
        validate_submission_zip(Path(args.validate_only))
    else:
        zip_path = package_submission(
            scenes=args.scenes,
            variant=args.variant,
            output_name=args.output,
        )
        validate_submission_zip(zip_path)
