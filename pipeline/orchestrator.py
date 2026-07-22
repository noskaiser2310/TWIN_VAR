"""VAR 2026 Pipeline Orchestrator.

Coordinates the full auto pipeline:
  1. Validate local data
  2. Prepare Kaggle datasets (upload scenes)
  3. Create and push GPU kernels per scene
  4. Monitor kernel execution
  5. Download outputs
  6. Package submission

Reuses competition_hunter's Kaggle client infrastructure where possible.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add Viettel_Race_AI/De_1/ for pipeline imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (
    DATA_DIR,
    GS_DIR,
    KAGGLE_USERNAME,
    KAGGLE_DATASET_PREFIX,
    SCENES,
    OUTPUT_DIR,
    KERNEL_TIMEOUT_SECONDS,
)


def check_kaggle_cli() -> bool:
    """Verify Kaggle CLI is available and authenticated."""
    try:
        result = subprocess.run(
            ["kaggle", "config", "view"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            print("[ERROR] Kaggle CLI not configured. Run: kaggle config set")
            return False
        print(f"[OK] Kaggle CLI ready (user: {KAGGLE_USERNAME})")
        return True
    except FileNotFoundError:
        print("[ERROR] Kaggle CLI not found. Install: pip install kaggle")
        return False


def validate_scene_data(scene: str) -> dict[str, Any]:
    """Validate that a scene has all required data for 3DGS training."""
    scene_dir = DATA_DIR / scene
    report = {"scene": scene, "valid": True, "issues": []}

    checks = [
        (scene_dir / "train" / "images", "training images directory"),
        (scene_dir / "train" / "sparse" / "0" / "cameras.bin", "COLMAP cameras.bin"),
        (scene_dir / "train" / "sparse" / "0" / "images.bin", "COLMAP images.bin"),
        (scene_dir / "train" / "sparse" / "0" / "points3D.bin", "COLMAP points3D.bin"),
    ]

    # Also check flat structure (images/ at scene root)
    alt_checks = [
        (scene_dir / "images", "images directory (flat)"),
        (scene_dir / "sparse" / "0" / "cameras.bin", "sparse cameras.bin (flat)"),
    ]

    test_csv = scene_dir / "test" / "test_poses.csv"
    if not test_csv.exists():
        test_csv = scene_dir / "test_poses.csv"

    for path, desc in checks:
        if not path.exists():
            report["issues"].append(f"Missing: {desc} at {path}")
            report["valid"] = False

    if not test_csv.exists():
        report["issues"].append(f"Missing: test_poses.csv")
        report["valid"] = False
    else:
        import csv
        with open(test_csv) as f:
            reader = csv.DictReader(f)
            poses = list(reader)
        report["num_test_poses"] = len(poses)
        report["test_poses_file"] = str(test_csv)

    # Count training images
    for img_dir in [scene_dir / "train" / "images", scene_dir / "images"]:
        if img_dir.exists():
            report["num_train_images"] = len(list(img_dir.glob("*")))
            break

    return report


def validate_all_scenes() -> list[dict]:
    """Validate all scenes and return report."""
    print("\n" + "=" * 60)
    print("PHASE 1: DATA VALIDATION")
    print("=" * 60)

    reports = []
    for scene in SCENES:
        report = validate_scene_data(scene)
        status = "✅" if report["valid"] else "❌"
        print(f"  {status} {scene}: {report.get('num_train_images', '?')} images, "
              f"{report.get('num_test_poses', '?')} test poses")
        if report["issues"]:
            for issue in report["issues"]:
                print(f"      ⚠️  {issue}")
        reports.append(report)

    valid = all(r["valid"] for r in reports)
    print(f"\n  Overall: {'✅ All valid' if valid else '❌ Issues found'}")
    return reports


def create_kernel_metadata(scene: str, dataset_slug: str) -> dict:
    """Create kernel-metadata.json for one scene."""
    kernel_id = f"{KAGGLE_USERNAME}/var2026-train-{scene.lower()}"
    return {
        "id": kernel_id,
        "title": f"VAR2026-3DGS-{scene}",
        "code_file": "kernel_3dgs_train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [dataset_slug],
        "competition_sources": [],
        "kernel_sources": [],
    }


def build_kernel_bundle(scene: str, dataset_slug: str, bundle_dir: Path) -> Path:
    """Build a Kaggle kernel bundle for one scene."""
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Copy kernel script
    kernel_src = Path(__file__).resolve().parent / "kernel_3dgs_train.py"
    import shutil as sh
    sh.copy(str(kernel_src), str(bundle_dir / "kernel_3dgs_train.py"))

    # Write kernel-metadata.json
    metadata = create_kernel_metadata(scene, dataset_slug)
    meta_path = bundle_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Copy gaussian-splatting code (needed for CUDA extension build)
    gs_dst = bundle_dir / "gaussian-splatting"
    if not gs_dst.exists():
        # Copy only essential files (not SIBR_viewers, not .git)
        _copy_gs_code(GS_DIR, gs_dst)

    print(f"  [BUNDLE] {scene}: {bundle_dir}")
    return bundle_dir


def _copy_gs_code(src: Path, dst: Path, ignore_patterns: list[str] | None = None):
    """Copy gaussian-splatting code, skipping heavy files."""

    if ignore_patterns is None:
        ignore_patterns = [
            ".git", "SIBR_viewers", "submodules/.git",
            "output", "__pycache__", "*.pyc",
            "assets", "*.ipynb",
        ]

    def _ignore(path, names):
        ignored = set()
        for name in names:
            for pat in ignore_patterns:
                if pat.startswith("*"):
                    if name.endswith(pat[1:]):
                        ignored.add(name)
                elif name == pat:
                    ignored.add(name)
        return ignored

    sh.copytree(str(src), str(dst), ignore=_ignore, dirs_exist_ok=True)


def push_kernel(bundle_dir: Path) -> dict:
    """Push a kernel bundle to Kaggle."""
    print(f"  [PUSH] {bundle_dir.name}...")
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(bundle_dir)],
        capture_output=True, text=True, timeout=120,
    )
    return {
        "success": result.returncode == 0,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def poll_kernel(kernel_slug: str, timeout: int = KERNEL_TIMEOUT_SECONDS) -> dict:
    """Poll kernel status until complete/error/timeout."""
    print(f"  [POLL] {kernel_slug}...")
    t0 = time.time()
    last_status = ""

    while time.time() - t0 < timeout:
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_slug],
            capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout + result.stderr).lower()

        if "complete" in output:
            elapsed = time.time() - t0
            print(f"  [POLL] {kernel_slug}: COMPLETE ({elapsed/60:.0f} min)")
            return {"status": "complete", "elapsed": elapsed}
        elif "error" in output or "fail" in output:
            print(f"  [POLL] {kernel_slug}: ERROR")
            return {"status": "error", "output": output}
        elif "running" in output and last_status != "running":
            print(f"  [POLL] {kernel_slug}: running...")
        elif "queued" in output and last_status != "queued":
            print(f"  [POLL] {kernel_slug}: queued...")

        last_status = "running" if "running" in output else "queued"
        time.sleep(60)  # Check every minute

    print(f"  [POLL] {kernel_slug}: TIMEOUT")
    return {"status": "timeout", "elapsed": timeout}


def download_kernel_output(kernel_slug: str, output_dir: Path) -> bool:
    """Download kernel output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [DOWNLOAD] {kernel_slug} → {output_dir}")
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_slug, "-p", str(output_dir)],
        capture_output=True, text=True, timeout=300,
    )
    return result.returncode == 0


def run_full_pipeline(
    scenes: list[str] | None = None,
    skip_validation: bool = False,
    skip_push: bool = False,
    skip_poll: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run the complete auto pipeline.

    Args:
        scenes: List of scene names (default: all)
        skip_validation: Skip data validation
        skip_push: Skip kernel push (assume already pushed)
        skip_poll: Skip polling (manual monitoring)
        dry_run: Build bundles but don't push

    Returns:
        Pipeline execution report
    """
    import shutil as sh

    if scenes is None:
        scenes = SCENES

    report = {"scenes": {}, "submission_ready": False}

    # ── Phase 1: Validation ──
    if not skip_validation:
        validation_reports = validate_all_scenes()
        invalid = [r["scene"] for r in validation_reports if not r["valid"]]
        if invalid:
            print(f"\n[ABORT] Invalid scenes: {invalid}")
            return report
    else:
        print("[SKIP] Data validation")

    # ── Phase 2: Build kernel bundles ──
    print(f"\n{'='*60}")
    print("PHASE 2: BUILD KERNEL BUNDLES")
    print(f"{'='*60}")

    bundles_dir = OUTPUT_DIR / "bundles"
    bundle_map = {}

    for scene in scenes:
        dataset_slug = f"{KAGGLE_DATASET_PREFIX}-{scene.lower()}"
        bundle_dir = build_kernel_bundle(scene, dataset_slug, bundles_dir / scene)
        bundle_map[scene] = {
            "bundle_dir": str(bundle_dir),
            "dataset_slug": dataset_slug,
            "kernel_slug": f"{KAGGLE_USERNAME}/var2026-train-{scene.lower()}",
        }

    # ── Phase 3: Push kernels ──
    if not skip_push and not dry_run:
        print(f"\n{'='*60}")
        print("PHASE 3: PUSH KERNELS")
        print(f"{'='*60}")

        for scene, info in bundle_map.items():
            result = push_kernel(Path(info["bundle_dir"]))
            info["push_result"] = result
            if not result["success"]:
                print(f"  [ERROR] Push failed for {scene}: {result['stderr']}")

    elif dry_run:
        print(f"\n{'='*60}")
        print("PHASE 3: DRY RUN — bundles ready, not pushed")
        print(f"{'='*60}")
        for scene, info in bundle_map.items():
            print(f"  {scene}: {info['bundle_dir']}")

    # ── Phase 4: Poll + Download ──
    if not skip_poll and not dry_run:
        print(f"\n{'='*60}")
        print("PHASE 4: POLL & DOWNLOAD")
        print(f"{'='*60}")

        for scene, info in bundle_map.items():
            kernel_slug = info["kernel_slug"]
            poll_result = poll_kernel(kernel_slug)
            info["poll_result"] = poll_result

            if poll_result["status"] == "complete":
                output_dir = OUTPUT_DIR / "kernel_outputs" / scene
                download_ok = download_kernel_output(kernel_slug, output_dir)
                info["download_ok"] = download_ok
                info["output_dir"] = str(output_dir)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    for scene, info in bundle_map.items():
        push = info.get("push_result", {}).get("success", "skipped")
        poll = info.get("poll_result", {}).get("status", "skipped")
        dl = info.get("download_ok", "skipped")
        print(f"  {scene}: push={push}, poll={poll}, download={dl}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VAR 2026 Pipeline Orchestrator")
    parser.add_argument("--scenes", nargs="*", help="Specific scenes (default: all)")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--skip-poll", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate_all_scenes()
    else:
        run_full_pipeline(
            scenes=args.scenes,
            skip_validation=args.skip_validation,
            skip_push=args.skip_push,
            skip_poll=args.skip_poll,
            dry_run=args.dry_run,
        )
