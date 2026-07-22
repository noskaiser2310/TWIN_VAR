"""Prepare depth maps for all scenes using Depth-Anything v2.

Depth maps improve 3DGS reconstruction quality significantly,
especially for untextured surfaces (walls, roads, sky).
Uses Depth-Anything v2 ViT-L for best quality.

Can run on RTX 4060 8GB (ViT-L model fits comfortably).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import (
    DATA_DIR,
    SCENES,
    DEPTH_MODEL,
    DEPTH_ANYTHING_REPO,
    DEPTH_CHECKPOINT_URLS,
    OUTPUT_DIR,
)


def ensure_depth_anything_v2() -> Path:
    """Clone Depth-Anything v2 if not already present."""
    repo_dir = OUTPUT_DIR / "Depth-Anything-V2"

    if not repo_dir.exists():
        print(f"[SETUP] Cloning Depth-Anything v2...")
        subprocess.run(
            ["git", "clone", DEPTH_ANYTHING_REPO, str(repo_dir)],
            check=True, capture_output=True,
        )

    # Download checkpoint if not present
    ckpt_dir = repo_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"depth_anything_v2_{DEPTH_MODEL}.pth"

    if not ckpt_path.exists():
        print(f"[SETUP] Downloading Depth-Anything v2 {DEPTH_MODEL} checkpoint...")
        import urllib.request
        url = DEPTH_CHECKPOINT_URLS.get(DEPTH_MODEL, DEPTH_CHECKPOINT_URLS["vitl"])
        urllib.request.urlretrieve(url, str(ckpt_path))
        print(f"  Saved to {ckpt_path}")

    # Install requirements
    req_path = repo_dir / "requirements.txt"
    if req_path.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)],
            check=False,
        )

    return repo_dir


def generate_depths_for_scene(scene: str, depth_anything_dir: Path) -> bool:
    """Generate depth maps for one scene's training images."""
    scene_dir = DATA_DIR / scene

    # Find images directory
    img_dir = scene_dir / "train" / "images"
    if not img_dir.exists():
        img_dir = scene_dir / "images"
    if not img_dir.exists():
        print(f"  [SKIP] {scene}: no images directory found")
        return False

    # Output directory
    depth_out = scene_dir / "depths"
    if depth_out.exists():
        existing = list(depth_out.glob("*.png"))
        if len(existing) > 0:
            print(f"  [SKIP] {scene}: {len(existing)} depth maps already exist")
            return True

    depth_out.mkdir(parents=True, exist_ok=True)

    print(f"  [DEPTH] {scene}: generating depth maps for images in {img_dir}...")

    result = subprocess.run(
        [
            sys.executable,
            str(depth_anything_dir / "run.py"),
            "--encoder", DEPTH_MODEL,
            "--pred-only",
            "--grayscale",
            "--img-path", str(img_dir),
            "--outdir", str(depth_out),
        ],
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max
    )

    if result.returncode == 0:
        n_generated = len(list(depth_out.glob("*.png")))
        print(f"  [DEPTH] {scene}: {n_generated} depth maps generated")
        return True
    else:
        print(f"  [DEPTH] {scene}: FAILED — {result.stderr[-500:]}")
        return False


def generate_depth_params(scene: str) -> bool:
    """Create depth_params.json for a scene (scale alignment between COLMAP and mono depth)."""
    scene_dir = DATA_DIR / scene

    # Find sparse directory
    sparse_dir = scene_dir / "train" / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = scene_dir / "sparse" / "0"
    if not sparse_dir.exists():
        print(f"  [SKIP] {scene}: no COLMAP sparse directory")
        return False

    depth_dir = scene_dir / "depths"
    if not depth_dir.exists() or not list(depth_dir.glob("*.png")):
        print(f"  [SKIP] {scene}: no depth maps")
        return False

    # Use 3DGS's make_depth_scale.py
    gs_utils = DATA_DIR.parent / "gaussian-splatting" / "utils" / "make_depth_scale.py"
    if not gs_utils.exists():
        print(f"  [WARN] make_depth_scale.py not found at {gs_utils}")
        return False

    print(f"  [DEPTH_PARAMS] {scene}: computing depth scale alignment...")

    result = subprocess.run(
        [
            sys.executable, str(gs_utils),
            "--base_dir", str(sparse_dir.parent.parent),  # parent of sparse/
            "--depths_dir", str(depth_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode == 0:
        # Check if depth_params.json was created
        dp_json = scene_dir / "train" / "depth_params.json"
        if not dp_json.exists():
            dp_json = scene_dir / "depth_params.json"
        if dp_json.exists():
            print(f"  [DEPTH_PARAMS] {scene}: depth_params.json created")
            return True

    print(f"  [DEPTH_PARAMS] {scene}: completed (check output)")
    return True


def prepare_all_depths(scenes: list[str] | None = None) -> dict:
    """Generate depth maps for all scenes."""
    if scenes is None:
        scenes = SCENES

    print("\n" + "=" * 60)
    print("PREPARE: DEPTH MAPS (Depth-Anything v2)")
    print("=" * 60)

    # Ensure Depth-Anything v2 is available
    da_dir = ensure_depth_anything_v2()

    results = {}
    for scene in scenes:
        print(f"\n── {scene} ──")
        ok = generate_depths_for_scene(scene, da_dir)
        if ok:
            generate_depth_params(scene)
        results[scene] = ok

    # Summary
    print(f"\n{'='*60}")
    print("DEPTH MAPS SUMMARY")
    for scene, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {scene}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*", help="Specific scenes")
    args = parser.parse_args()
    prepare_all_depths(args.scenes)
