"""Upload scenes as Kaggle private datasets for GPU kernel training.

Each scene is uploaded as: {username}/var2026-{scene}
Dataset contains: images/, sparse/, depths/, depth_params.json, test_poses.csv

All datasets are PRIVATE by default (competition data!).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (
    DATA_DIR,
    KAGGLE_USERNAME,
    KAGGLE_DATASET_PREFIX,
    SCENES,
    OUTPUT_DIR,
)


def create_dataset_metadata(scene: str) -> dict:
    """Create dataset-metadata.json for Kaggle dataset."""
    return {
        "title": f"VAR2026-{scene}",
        "id": f"{KAGGLE_DATASET_PREFIX}-{scene.lower()}",
        "licenses": [{"name": "CC0-1.0"}],
        "is_private": True,
    }


def package_scene_for_kaggle(scene: str, output_dir: Path) -> Path:
    """Package scene data into Kaggle dataset format.

    Creates a directory with:
      - images/          (symlinked or copied training images)
      - sparse/0/        (COLMAP reconstruction)
      - depths/          (pre-computed depth maps, if available)
      - depth_params.json
      - test_poses.csv
      - dataset-metadata.json
    """
    scene_dir = DATA_DIR / scene
    pkg_dir = output_dir / scene
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Copy training images
    src_images = scene_dir / "train" / "images"
    if not src_images.exists():
        src_images = scene_dir / "images"

    if src_images.exists():
        dst_images = pkg_dir / "images"
        if not dst_images.exists():
            print(f"  Copying images: {src_images} → {dst_images}")
            _copy_or_symlink_dir(src_images, dst_images)

    # Copy COLMAP sparse reconstruction
    src_sparse = scene_dir / "train" / "sparse"
    if not src_sparse.exists():
        src_sparse = scene_dir / "sparse"
    if src_sparse.exists():
        dst_sparse = pkg_dir / "sparse"
        if not dst_sparse.exists():
            _copy_or_symlink_dir(src_sparse, dst_sparse)

    # Copy depth maps
    src_depths = scene_dir / "depths"
    if src_depths.exists() and list(src_depths.glob("*.png")):
        dst_depths = pkg_dir / "depths"
        if not dst_depths.exists():
            _copy_or_symlink_dir(src_depths, dst_depths)
        print(f"  Depths: {len(list(dst_depths.glob('*.png')))} maps")

    # Copy depth_params.json
    dp_json = scene_dir / "depth_params.json"
    if not dp_json.exists():
        dp_json = scene_dir / "train" / "depth_params.json"
    if dp_json.exists():
        _copy_file(dp_json, pkg_dir / "depth_params.json")

    # Copy test_poses.csv
    tp_csv = scene_dir / "test" / "test_poses.csv"
    if not tp_csv.exists():
        tp_csv = scene_dir / "test_poses.csv"
    if tp_csv.exists():
        _copy_file(tp_csv, pkg_dir / "test_poses.csv")

    # Write dataset-metadata.json
    metadata = create_dataset_metadata(scene)
    meta_path = pkg_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Summary
    n_images = len(list((pkg_dir / "images").glob("*"))) if (pkg_dir / "images").exists() else 0
    print(f"  Packaged: {scene} ({n_images} images)")

    return pkg_dir


def _copy_or_symlink_dir(src: Path, dst: Path):
    """Copy directory. Tries symlink first, falls back to copy."""
    import shutil as sh
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
    except (OSError, NotImplementedError):
        sh.copytree(str(src), str(dst))


def _copy_file(src: Path, dst: Path):
    """Copy a single file."""
    import shutil as sh
    sh.copy2(str(src), str(dst))


def upload_dataset(pkg_dir: Path) -> dict:
    """Upload a dataset to Kaggle using CLI."""
    scene = pkg_dir.name
    dataset_slug = f"{KAGGLE_DATASET_PREFIX}-{scene.lower()}"

    print(f"  [UPLOAD] {dataset_slug}...")

    result = subprocess.run(
        [
            "kaggle", "datasets", "create",
            "-p", str(pkg_dir),
            "--dir-mode", "zip",
        ],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode == 0:
        print(f"  [UPLOAD] {dataset_slug}: SUCCESS")
        return {"success": True, "slug": dataset_slug}
    else:
        # Check if dataset already exists - try version update
        stderr = result.stderr.lower()
        if "already exists" in stderr or "409" in stderr:
            print(f"  [UPLOAD] {dataset_slug}: already exists, creating new version...")
            result2 = subprocess.run(
                [
                    "kaggle", "datasets", "version",
                    "-p", str(pkg_dir),
                    "-m", "Updated scene data",
                    "--dir-mode", "zip",
                ],
                capture_output=True, text=True, timeout=600,
            )
            if result2.returncode == 0:
                print(f"  [UPLOAD] {dataset_slug}: VERSION UPDATE SUCCESS")
                return {"success": True, "slug": dataset_slug}
            else:
                print(f"  [UPLOAD] {dataset_slug}: VERSION FAILED: {result2.stderr[-300:]}")
                return {"success": False, "slug": dataset_slug, "error": result2.stderr}

        print(f"  [UPLOAD] {dataset_slug}: FAILED: {result.stderr[-300:]}")
        return {"success": False, "slug": dataset_slug, "error": result.stderr}


def upload_all_scenes(scenes: list[str] | None = None, dry_run: bool = False) -> dict:
    """Package and upload all scenes as Kaggle datasets."""
    if scenes is None:
        scenes = SCENES

    print("\n" + "=" * 60)
    print("PREPARE: KAGGLE DATASET UPLOAD")
    print("=" * 60)

    pkg_dir = OUTPUT_DIR / "kaggle_datasets"
    results = {}

    for scene in scenes:
        print(f"\n── {scene} ──")
        pkg = package_scene_for_kaggle(scene, pkg_dir)

        if not dry_run:
            result = upload_dataset(pkg)
            results[scene] = result
        else:
            print(f"  [DRY RUN] Would upload: {pkg}")
            results[scene] = {"success": True, "slug": f"{KAGGLE_DATASET_PREFIX}-{scene.lower()}", "dry_run": True}

    # Summary
    print(f"\n{'='*60}")
    print("DATASET UPLOAD SUMMARY")
    for scene, r in results.items():
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {scene}: {r['slug']}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    upload_all_scenes(args.scenes, args.dry_run)
