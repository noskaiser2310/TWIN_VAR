"""Render novel views at test poses from a trained 3DGS model.

Loads a trained 3DGS checkpoint and renders images at poses specified in test_poses.csv.
Handles Quaternion (WXYZ) → Rotation Matrix conversion as required.

Usage:
    python render.py --scene bonsai --variant full_60k
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, GS_DIR, OUTPUT_DIR


def get_test_poses_csv(scene: str) -> Path:
    """Find test_poses.csv for a scene."""
    for p in [
        DATA_DIR / scene / "test" / "test_poses.csv",
        DATA_DIR / scene / "test_poses.csv",
    ]:
        if p.exists():
            return p
    raise FileNotFoundError(f"test_poses.csv not found for {scene}")


def build_render_script(model_path: str, test_poses: list[dict], output_dir: str, gs_dir: str) -> str:
    """Generate a self-contained Python script that loads model + renders all poses."""
    poses_json = json.dumps(test_poses)

    return f'''"""Auto-generated render script for VAR 2026 test poses."""
import json, math, os, sys
sys.path.insert(0, "{gs_dir}")
import numpy as np, torch, torchvision
from pathlib import Path
from argparse import Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.general_utils import safe_state

test_poses = json.loads("""{poses_json}""")
safe_state(True)

ds = Namespace(source_path="{model_path}", model_path="{model_path}",
               sh_degree=3, images="images", resolution=-1,
               white_background=False, data_device="cuda",
               eval=False, train_test_exp=False)
pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                 debug=False, antialiasing=False)

gs = GaussianModel(3)
scene = Scene(ds, gs, load_iteration=-1, shuffle=False)
bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
out = Path("{output_dir}")
out.mkdir(parents=True, exist_ok=True)
print(f"Rendering {{len(test_poses)}} views...")

for i, p in enumerate(test_poses):
    w, h = int(float(p["width"])), int(float(p["height"]))
    fx, fy, cx, cy = float(p["fx"]), float(p["fy"]), float(p["cx"]), float(p["cy"])
    qw, qx, qy, qz = float(p["qw"]), float(p["qx"]), float(p["qy"]), float(p["qz"])
    tx, ty, tz = float(p["tx"]), float(p["ty"]), float(p["tz"])

    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ])
    T = np.array([tx, ty, tz])
    R_t = torch.tensor(R.T, dtype=torch.float32, device="cuda")
    T_t = torch.tensor(-R @ T, dtype=torch.float32, device="cuda")
    FoVx = 2 * math.atan(w / (2 * fx))
    FoVy = 2 * math.atan(h / (2 * fy))

    vp = MiniCam(resolution=(w, h), colmap_id=i, R=R_t, T=T_t,
                 FoVx=FoVx, FoVy=FoVy, depth_params=None,
                 image=None, invdepthmap=None, depth_mask=None)

    with torch.no_grad():
        rendering = render(vp, gs, pipe, bg)["render"]
    out_path = out / p["image_name"]
    torchvision.utils.save_image(rendering, str(out_path))
    if (i + 1) % 10 == 0:
        print(f"  {{i+1}}/{{len(test_poses)}}")

print(f"DONE: {{len(test_poses)}} images → {output_dir}")
'''


def render(scene: str, variant: str, gs_dir: Path | None = None) -> bool:
    """Render all test poses for a trained variant."""
    if gs_dir is None:
        gs_dir = GS_DIR

    model_path = OUTPUT_DIR / "models" / scene / variant
    pc_dir = model_path / "point_cloud"
    if not pc_dir.exists():
        print(f"  [SKIP] No trained model at {model_path}")
        return False
    has_checkpoint = any(
        (pc_dir / d).exists()
        for d in ["iteration_-1", "iteration_7000", "iteration_30000", "iteration_60000", "iteration_90000"]
    )
    if not has_checkpoint:
        print(f"  [SKIP] No checkpoint in {pc_dir}")
        return False

    test_csv = get_test_poses_csv(scene)
    with open(test_csv) as f:
        test_poses = list(csv.DictReader(f))

    output_dir = OUTPUT_DIR / "renders" / scene / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already rendered
    existing = list(output_dir.glob("*.png"))
    if len(existing) >= len(test_poses):
        print(f"  [SKIP] Already rendered {len(existing)} images")
        return True

    script_code = build_render_script(
        str(model_path), test_poses, str(output_dir), str(gs_dir)
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(script_code)
        script_path = tf.name

    try:
        print(f"  [RENDER] {scene}/{variant}: {len(test_poses)} poses...")
        r = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, cwd=str(gs_dir),
            timeout=3600,
        )
        if r.returncode == 0:
            rendered = list(output_dir.glob("*.png"))
            print(f"  [OK] {scene}/{variant}: {len(rendered)} images")
            return True
        else:
            print(f"  [FAIL] Render failed:")
            print(r.stderr[-1000:])
            return False
    finally:
        Path(script_path).unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--variant", default="full_60k")
    p.add_argument("--gs-dir", default=None)
    args = p.parse_args()
    success = render(args.scene, args.variant, Path(args.gs_dir) if args.gs_dir else None)
    sys.exit(0 if success else 1)
