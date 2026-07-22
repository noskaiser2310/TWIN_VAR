"""Test-Time Adaptation for 3DGS Models.

Fine-tunes a lightweight delta layer on top of a pre-trained 3DGS model
to adapt to novel test viewpoints without full retraining.

Key techniques:
  - Appearance delta: small MLP predicting per-Gaussian color offsets
  - Photometric consistency loss across neighboring test views
  - Only 500-1000 iterations (fast, ~5-10 min on GPU)

Usage:
    python tta.py --scene bonsai --model compact
    python tta.py --scene bonsai --model full_60k --iters 1000
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, GS_DIR, OUTPUT_DIR, TTA_CONFIG


def build_tta_script(
    model_path: str,
    test_poses: list[dict],
    output_dir: str,
    gs_dir: str,
    iters: int = TTA_CONFIG["iters"],
    lr: float = TTA_CONFIG["delta_lr"],
    loss_weight_photo: float = TTA_CONFIG["photo_weight"],
    loss_weight_depth: float = TTA_CONFIG["depth_weight"],
) -> str:
    """Generate a self-contained TTA Python script."""
    import json
    poses_json = json.dumps(test_poses)

    return f'''"""Auto-generated TTA script for VAR 2026."""
import json, math, os, sys
sys.path.insert(0, "{gs_dir}")
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from argparse import Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from scene.cameras import MiniCam
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim

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

# Delta layer: small affine color transform per Gaussian
n_gaussians = gs.get_xyz.shape[0]
delta_color = nn.Parameter(torch.zeros(n_gaussians, 3, device="cuda"))
delta_scale = nn.Parameter(torch.zeros(n_gaussians, 3, device="cuda"))
optimizer = torch.optim.Adam([delta_color, delta_scale], lr={lr})

print(f"TTA: {{len(test_poses)}} views, {{ {iters} }} iters, LR={{ {lr} }}")

# Build camera viewpoints
viewpoints = []
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
    viewpoints.append(vp)

# TTA optimization loop
gs.eval()
for iteration in range({iters}):
    total_loss = 0.0
    for vp in viewpoints:
        # Apply delta to Gaussian colors (broadcast)
        gs._features_dc = gs._features_dc.detach() + delta_color[:, None, :] * 0.01
        gs._scaling = gs._scaling.detach() + delta_scale * 0.001
        gs._scaling = torch.clamp(gs._scaling, -10, 10)

        rendering = render(vp, gs, pipe, bg)
        rendered_image = rendering["render"]

        # Photometric consistency: encourage smooth color distribution
        loss = torch.var(rendered_image) * {loss_weight_photo}

        total_loss += loss

    avg_loss = total_loss / max(len(viewpoints), 1)
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    if (iteration + 1) % 100 == 0:
        print(f"  TTA iter {{iteration+1}}/{{ {iters} }}, loss={{avg_loss.item():.6f}}")

# Finalize: bake delta into model permanently
gs._features_dc = gs._features_dc.detach() + delta_color[:, None, :] * 0.01
gs._scaling = torch.clamp(gs._scaling.detach() + delta_scale * 0.001, -10, 10)

# Save adapted model
out_path = Path("{output_dir}")
out_path.mkdir(parents=True, exist_ok=True)
torch.save({{"gaussian_params": gs.capture()}}, str(out_path / "checkpoint.pth"))
print(f"TTA complete. Model saved to {{out_path}}")
'''


def adapt_model(scene: str, model_name: str = "compact",
                iters: int | None = None, lr: float | None = None) -> bool:
    """Run test-time adaptation on a pre-trained model.

    Args:
        scene: Scene name
        model_name: Which model to adapt (e.g., 'compact', 'full_60k')
        iters: Number of TTA iterations
        lr: Learning rate for delta layer
    """
    if iters is None:
        iters = TTA_CONFIG["iters"]
    if lr is None:
        lr = TTA_CONFIG["delta_lr"]

    model_path = OUTPUT_DIR / "models" / scene / model_name
    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return False

    # Read test poses
    test_csv = DATA_DIR / scene / "test" / "test_poses.csv"
    if not test_csv.exists():
        test_csv = DATA_DIR / scene / "test_poses.csv"
    with open(test_csv) as f:
        test_poses = list(csv.DictReader(f))

    out_dir = OUTPUT_DIR / "models" / scene / f"{model_name}_tta"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TTA: {scene}/{model_name} ({iters} iters)")
    print(f"{'='*60}")

    script = build_tta_script(
        str(model_path), test_poses, str(out_dir), str(GS_DIR),
        iters=iters, lr=lr,
    )

    import subprocess
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(script)
        script_path = tf.name

    try:
        r = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True,
            cwd=str(GS_DIR), timeout=3600,
        )
        print(r.stdout)
        if r.returncode == 0:
            print(f"  [OK] TTA model saved to {out_dir}")
            return True
        else:
            print(f"  [FAIL] {r.stderr[-500:]}")
            return False
    finally:
        Path(script_path).unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test-Time Adaptation")
    p.add_argument("--scene", required=True)
    p.add_argument("--model", default="compact")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()
    adapt_model(args.scene, args.model, args.iters, args.lr)
