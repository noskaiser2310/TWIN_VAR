"""Kaggle kernel script: Multi-variant 3DGS training + rendering for one scene.

This script runs INSIDE a Kaggle kernel (T4 GPU 16GB).
It must be self-contained: install deps, train variants, render, save outputs.

Usage (inside Kaggle kernel):
    python kernel_3dgs_train.py --scene HCM0421 --data_dir /kaggle/input/var2026-HCM0421

Expected data_dir structure:
    /kaggle/input/var2026-{scene}/
    ├── images/          # Training images
    ├── sparse/0/        # COLMAP reconstruction
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    ├── depths/          # Pre-computed depth maps (optional)
    ├── depth_params.json
    └── test_poses.csv   # Test camera poses
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
#  Constants (mirror pipeline/config.py for kernel independence)
# ═══════════════════════════════════════════════════════════════

VARIANTS_CONFIG = [
    {"name": "baseline",    "args": "--iterations 30000 --data_device cpu"},
    {"name": "depth",       "args": "--iterations 30000 --data_device cpu -d depths/"},
    {"name": "exposure",    "args": "--iterations 30000 --data_device cpu --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp"},
    {"name": "antialias",   "args": "--iterations 30000 --data_device cpu --antialiasing"},
    {"name": "full_combo",  "args": "--iterations 30000 --data_device cpu --antialiasing --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp --optimizer_type sparse_adam -d depths/"},
    {"name": "fast",        "args": "--iterations 7000 --data_device cpu"},
]


def install_dependencies():
    """Install 3DGS dependencies on Kaggle kernel."""
    print("[INSTALL] Setting up 3D Gaussian Splatting environment...")

    # Kaggle already has PyTorch + CUDA. We need:
    pkgs = [
        "plyfile",
        "tqdm",
        "opencv-python-headless",
    ]
    for pkg in pkgs:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

    # Build 3DGS CUDA extensions
    gs_dir = Path("/kaggle/working/gaussian-splatting")
    if not gs_dir.exists():
        # The GS code is bundled with the kernel
        gs_dir = Path(__file__).resolve().parent / "gaussian-splatting"

    if gs_dir.exists():
        os.chdir(gs_dir)
        print(f"[INSTALL] Building CUDA extensions from {gs_dir}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "submodules/diff-gaussian-rasterization"],
            check=False, cwd=gs_dir,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "submodules/simple-knn"],
            check=False, cwd=gs_dir,
        )
        # Try to install fused-ssim and accelerated rasterizer
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "submodules/diff-gaussian-rasterization"],
            check=False, cwd=gs_dir,
            env={**os.environ, "GIT_CHECKOUT": "3dgs_accel"},
        )
    print("[INSTALL] Dependencies ready.")


def prepare_scene_data(data_dir: Path, scene: str, work_dir: Path) -> Path:
    """Copy scene data to working directory in COLMAP format expected by 3DGS."""
    scene_dir = work_dir / scene
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    src_images = data_dir / "images"
    dst_images = scene_dir / "images"
    if src_images.exists():
        if not dst_images.exists():
            shutil.copytree(str(src_images), str(dst_images))
    else:
        # Maybe images are in train/images/
        src_train = data_dir / "train" / "images"
        if src_train.exists():
            shutil.copytree(str(src_train), str(dst_images))

    # Copy sparse/
    src_sparse = data_dir / "sparse"
    if not src_sparse.exists():
        src_sparse = data_dir / "train" / "sparse"
    dst_sparse = scene_dir / "sparse"
    if src_sparse.exists() and not dst_sparse.exists():
        shutil.copytree(str(src_sparse), str(dst_sparse))

    # Copy depths if available
    src_depths = data_dir / "depths"
    dst_depths = scene_dir / "depths"
    if src_depths.exists() and not dst_depths.exists():
        shutil.copytree(str(src_depths), str(dst_depths))

    # Copy depth_params.json
    dp_json = data_dir / "depth_params.json"
    if dp_json.exists():
        shutil.copy(str(dp_json), str(scene_dir / "depth_params.json"))

    # Copy test_poses.csv
    tp_csv = data_dir / "test_poses.csv"
    if not tp_csv.exists():
        tp_csv = data_dir / "test" / "test_poses.csv"
    if tp_csv.exists():
        shutil.copy(str(tp_csv), str(scene_dir / "test_poses.csv"))

    print(f"[DATA] Scene prepared at {scene_dir}")
    print(f"  Images: {len(list(dst_images.glob('*')))} files")
    return scene_dir


def train_variant(scene_dir: Path, output_dir: Path, variant: dict, gs_dir: Path) -> bool:
    """Train one 3DGS variant. Returns True on success."""
    name = variant["name"]
    model_path = output_dir / name

    print(f"\n{'='*60}")
    print(f"[TRAIN] Variant: {name} ({variant['args']})")
    print(f"[TRAIN] Output: {model_path}")
    print(f"{'='*60}")

    cmd = (
        f"cd \"{gs_dir}\" && "
        f"python train.py "
        f'-s "{scene_dir}" '
        f'-m "{model_path}" '
        f"{variant['args']} "
        f"--quiet"
    )

    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    elapsed = time.time() - t0
    success = result.returncode == 0

    if success:
        print(f"[TRAIN] {name}: DONE in {elapsed/60:.1f} min")
    else:
        print(f"[TRAIN] {name}: FAILED after {elapsed/60:.1f} min")
        # Print last 20 lines of error
        lines = (result.stdout + result.stderr).splitlines()
        for line in lines[-20:]:
            print(f"  {line}")

    return success


def render_test_poses(
    scene_dir: Path,
    model_path: Path,
    gs_dir: Path,
    test_poses_csv: Path,
    output_img_dir: Path,
    variant_name: str,
) -> bool:
    """Render novel views from trained 3DGS model at test poses.

    Since 3DGS render.py works with its internal train/test split,
    we need a custom approach: modify the scene to treat test_poses as test cameras.
    """
    print(f"\n[RENDER] Rendering test poses for variant: {variant_name}")

    # Read test_poses.csv
    if not test_poses_csv.exists():
        print(f"[RENDER] ERROR: test_poses.csv not found at {test_poses_csv}")
        return False

    test_poses = []
    with open(test_poses_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_poses.append(row)

    print(f"[RENDER] {len(test_poses)} test poses to render")

    # Create custom render script that loads the model and renders at each pose
    render_script = output_img_dir.parent / f"render_{variant_name}.py"
    render_code = _build_custom_render_script(
        model_path=str(model_path),
        test_poses=test_poses,
        output_dir=str(output_img_dir),
        gs_dir=str(gs_dir),
    )
    render_script.write_text(render_code)

    # Run the render script
    result = subprocess.run(
        [sys.executable, str(render_script)],
        capture_output=True,
        text=True,
        cwd=str(gs_dir),
    )

    if result.returncode == 0:
        rendered = list(output_img_dir.glob("*.png"))
        print(f"[RENDER] {variant_name}: {len(rendered)} images generated")
        return True
    else:
        print(f"[RENDER] {variant_name}: FAILED")
        print(result.stderr[-1000:])
        return False


def _build_custom_render_script(
    model_path: str,
    test_poses: list[dict],
    output_dir: str,
    gs_dir: str,
) -> str:
    """Generate a Python script that loads a trained 3DGS model and renders at custom poses."""
    poses_json = json.dumps(test_poses)

    # Use triple-double-quotes inside the f-string to avoid conflict with outer triple-single-quotes
    return f'''"""Custom render script for VAR 2026 test poses."""
import json
import math
import os
import sys

sys.path.insert(0, "{gs_dir}")

import numpy as np
import torch
import torchvision
from pathlib import Path
from argparse import Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state

# Parse test poses
test_poses = json.loads("""{poses_json}""")

# Load model
safe_state(True)
dataset_args = Namespace(
    source_path="{model_path}",
    model_path="{model_path}",
    sh_degree=3,
    images="images",
    resolution=-1,
    white_background=False,
    data_device="cuda",
    eval=False,
    train_test_exp=False,
)
pipe_args = Namespace(
    convert_SHs_python=False,
    compute_cov3D_python=False,
    debug=False,
    antialiasing=False,
)

gaussians = GaussianModel(3)
scene = Scene(dataset_args, gaussians, load_iteration=-1, shuffle=False)

bg_color = [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

output_dir = Path("{output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Rendering {{len(test_poses)}} test poses...")

for i, pose in enumerate(test_poses):
    image_name = pose["image_name"]
    w, h = int(float(pose["width"])), int(float(pose["height"]))

    # Build camera
    from scene.cameras import MiniCam

    fx = float(pose["fx"]); fy = float(pose["fy"])
    cx = float(pose["cx"]); cy = float(pose["cy"])
    qw = float(pose["qw"]); qx = float(pose["qx"])
    qy = float(pose["qy"]); qz = float(pose["qz"])
    tx = float(pose["tx"]); ty = float(pose["ty"]); tz = float(pose["tz"])

    # Quaternion to rotation matrix
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ])
    T = np.array([tx, ty, tz])

    # Convert to 3DGS format: world-to-camera
    R_t = torch.tensor(R.T, dtype=torch.float32, device="cuda")
    T_t = torch.tensor(-R @ T, dtype=torch.float32, device="cuda")

    FoVx = 2 * math.atan(w / (2 * fx))
    FoVy = 2 * math.atan(h / (2 * fy))

    viewpoint = MiniCam(
        resolution=(w, h),
        colmap_id=i,
        R=R_t,
        T=T_t,
        FoVx=FoVx,
        FoVy=FoVy,
        depth_params=None,
        image=None,
        invdepthmap=None,
        depth_mask=None,
        depth_reliable=False,
    )

    with torch.no_grad():
        rendering = render(viewpoint, gaussians, pipe_args, background)["render"]
        rendering = torch.clamp(rendering, 0.0, 1.0)

    out_path = output_dir / image_name
    torchvision.utils.save_image(rendering, str(out_path))

    if (i + 1) % 10 == 0:
        print(f"  Rendered {{i+1}}/{{len(test_poses)}}")

print(f"Done! {{len(test_poses)}} images saved to {output_dir}")
'''


def main():
    parser = argparse.ArgumentParser(description="VAR 2026: 3DGS multi-variant training kernel")
    parser.add_argument("--scene", required=True, help="Scene name (e.g., HCM0421)")
    parser.add_argument("--data_dir", required=True, help="Path to input data (Kaggle dataset mount)")
    parser.add_argument("--gs_dir", default="/kaggle/working/gaussian-splatting",
                        help="Path to gaussian-splatting code")
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant names or 'all'")
    parser.add_argument("--skip_install", action="store_true",
                        help="Skip dependency installation")
    args = parser.parse_args()

    scene = args.scene
    data_dir = Path(args.data_dir)
    work_dir = Path("/kaggle/working")
    output_dir = work_dir / "output" / scene
    gs_dir = Path(args.gs_dir)

    print(f"{'='*60}")
    print(f"VAR 2026 3DGS KERNEL — Scene: {scene}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # 1. Install dependencies
    if not args.skip_install:
        install_dependencies()

    # 2. Prepare scene data in COLMAP format
    scene_dir = prepare_scene_data(data_dir, scene, work_dir / "scenes")
    test_poses_csv = scene_dir / "test_poses.csv"

    # 3. Select variants
    if args.variants == "all":
        variants = VARIANTS_CONFIG
    else:
        names = set(args.variants.split(","))
        variants = [v for v in VARIANTS_CONFIG if v["name"] in names]

    print(f"\n[PIPELINE] Training {len(variants)} variants: "
          f"{', '.join(v['name'] for v in variants)}")

    # 4. Train each variant
    results = {}
    for variant in variants:
        name = variant["name"]
        variant_output_dir = output_dir / name
        success = train_variant(scene_dir, variant_output_dir, variant, gs_dir)
        results[name] = {
            "success": success,
            "model_path": str(variant_output_dir),
        }

    # 5. Render test poses for each successful variant
    for variant in variants:
        name = variant["name"]
        if not results[name]["success"]:
            print(f"\n[RENDER] Skipping {name} (training failed)")
            continue

        render_dir = output_dir / name / "test_renders"
        success = render_test_poses(
            scene_dir=scene_dir,
            model_path=Path(results[name]["model_path"]),
            gs_dir=gs_dir,
            test_poses_csv=test_poses_csv,
            output_img_dir=render_dir,
            variant_name=name,
        )
        results[name]["render_success"] = success
        results[name]["render_dir"] = str(render_dir)

    # 6. Save results manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\n[MANIFEST] Saved to {manifest_path}")

    # 7. Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        train_status = "✅" if r["success"] else "❌"
        render_status = "✅" if r.get("render_success") else "❌"
        print(f"  {train_status} {name}: train={r['success']}, render={r.get('render_success', False)}")
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
