"""Perceptual Fine-Tuning for LPIPS Optimization.

The competition score is 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm,
meaning LPIPS has the HIGHEST weight (40%). Vanilla 3DGS trains with
L1+SSIM loss, NOT optimizing for LPIPS at all.

This module fine-tunes a trained 3DGS model with:
  1. LPIPS loss (via lpipsPyTorch) — directly optimizes the 40% metric
  2. DINO-v2 feature matching — perceptual consistency in feature space
  3. Reduced L1 weight — shift focus from pixel-perfect to perceptually good

Usage:
    python perceptual_finetune.py --scene bonsai --variant full_60k --iters 500
    python perceptual_finetune.py --scene bonsai --variant compact --iters 1000

Integration:
    python main.py --scenes bonsai --perceptual  # via main.py orchestrator
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, GS_DIR, OUTPUT_DIR, set_data_dir

# ── Default LPIPS fine-tuning params ──────────────────────────
PERCEPTUAL_ITERS = 500          # Extra iterations for LPIPS optimization
LPIPS_LAMBDA = 1.0              # LPIPS loss weight
DINO_LAMBDA = 0.5               # DINO-v2 feature loss weight
L1_LAMBDA = 0.2                 # Reduced L1 weight (was 0.8 in vanilla)
SSIM_LAMBDA = 0.3               # Reduced SSIM weight (was 0.2 in vanilla)


def build_finetune_script(
    model_path: str,
    gs_dir: str,
    output_dir: str,
    iters: int = PERCEPTUAL_ITERS,
    lpips_lambda: float = LPIPS_LAMBDA,
    dino_lambda: float = DINO_LAMBDA,
    l1_lambda_val: float = L1_LAMBDA,
    ssim_lambda_val: float = SSIM_LAMBDA,
) -> str:
    """Generate a self-contained perceptual fine-tuning script.

    Loads a trained 3DGS model and continues training with:
    - LPIPS loss (directly optimizes the 40% metric)
    - DINO-v2 feature matching loss (perceptual consistency)
    - Reduced L1+SSIM weights
    """
    return f'''"""Auto-generated perceptual fine-tuning script for VAR 2026."""
import json, math, os, sys
sys.path.insert(0, "{gs_dir}")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from argparse import Namespace
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from tqdm import tqdm

# DINO-v2 perceptual loss (optional, fallback to LPIPS-only if unavailable)
class DINOLoss(nn.Module):
    """Perceptual loss using DINOv2 features.

    NOTE: Requires internet for first download (torch.hub caches after).
    If unavailable, sets self.available=False and returns 0 loss.
    """
    def __init__(self):
        super().__init__()
        self.available = False
        try:
            import torch.hub as hub
            self.dino = hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda()
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad = False
            self.available = True
        except Exception as e:
            print(f"[WARN] DINOv2 not available: {e}")
            print("[INFO] Falling back to LPIPS-only perceptual loss")

    def forward(self, pred, target):
        if not self.available:
            return torch.tensor(0.0, device=pred.device)
        # pred/target: (1, 3, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1,3,1,1)
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        with torch.no_grad():
            feat_target = self.dino(target_norm)

        feat_pred = self.dino(pred_norm)
        loss = F.mse_loss(feat_pred, feat_target.detach())
        return loss

# LPIPS loss
from lpipsPyTorch import lpips

safe_state(True)

# Load model
model_dir = "{model_path}"
print(f"Loading model from {{model_dir}}...")

ds = Namespace(
    source_path=model_dir, model_path=model_dir,
    sh_degree=3, images="images", resolution=-1,
    white_background=False, data_device="cuda",
    eval=True, train_test_exp=False,
)
pipe = Namespace(
    convert_SHs_python=False, compute_cov3D_python=False,
    debug=False, antialiasing=False,
)

gs = GaussianModel(3)
scene = Scene(ds, gs, load_iteration=-1, shuffle=False)

bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
train_cams = scene.getTrainCameras()
n_cams = len(train_cams)

print(f"Loaded: {{gs.get_xyz.shape[0]}} Gaussians, {{n_cams}} train views")

# Setup optimizer (only fine-tune, much lower LR)
gs.training_setup(Namespace(
    model_path=model_dir,
    position_lr_init=0.000016,      # 10x lower than default
    position_lr_final=0.00000016,
    position_lr_delay_mult=0.01,
    position_lr_max_steps={iters},
    feature_lr=0.00025,
    opacity_lr=0.0025,
    scaling_lr=0.0005,
    rotation_lr=0.0001,
    exposure_lr_init=0.001,
    exposure_lr_final=0.0001,
    exposure_lr_delay_steps=0,
    exposure_lr_delay_mult=0.0,
    percent_dense=0.01,
    lambda_dssim=0.2,
    densification_interval=100,
    opacity_reset_interval=3000,
    densify_from_iter=0,                # 0 = disabled (densify_until_iter=0)
    densify_until_iter=0,               # 0 = completely disable densification
    densify_grad_threshold=0.0002,
    depth_l1_weight_init=0.0,        # No depth during fine-tune
    depth_l1_weight_final=0.0,
    random_background=False,
    optimizer_type="default",
    iterations={iters},
))

dino_loss = DINOLoss()
print("DINOv2 perceptual loss initialized")

out_dir = Path("{output_dir}")
out_dir.mkdir(parents=True, exist_ok=True)

# Fine-tuning loop
for iteration in tqdm(range(1, {iters} + 1), desc="Perceptual FT"):
    gs.update_learning_rate(iteration)

    # Pick random camera
    cam = train_cams[iteration % n_cams]
    cam.cuda()

    # Render
    render_pkg = render(cam, gs, pipe, bg)
    image = render_pkg["render"].unsqueeze(0)
    gt = cam.original_image.unsqueeze(0)

    # LPIPS loss (directly optimizes 40% of score)
    loss_lpips = lpips(image, gt, net_type="vgg").mean()

    # DINO-v2 feature matching (perceptual consistency, optional)
    loss_dino = dino_loss(image, gt)

    # Reduced L1 + SSIM (keep geometry stable)
    loss_l1 = l1_loss(image, gt)
    loss_ssim = 1.0 - ssim(image, gt)
    loss_photo = (1.0 - 0.2) * loss_l1 + 0.2 * loss_ssim

    # Total loss — LPIPS dominant
    total_loss = (
        {lpips_lambda} * loss_lpips +
        {dino_lambda} * loss_dino +
        {l1_lambda_val} * loss_photo
    )

    total_loss.backward()
    gs.optimizer.step()
    gs.optimizer.zero_grad()
    cam.cpu()

    if iteration % 100 == 0:
        print(f"  iter {{iteration}}: LPIPS={{loss_lpips.item():.4f}} "
              f"DINO={{loss_dino.item():.4f}} Photo={{loss_photo.item():.4f}}")

# Save fine-tuned model
gs.save_ply(str(out_dir / "point_cloud.ply"))
print(f"Fine-tuned model saved to {{out_dir / 'point_cloud.ply'}}")
print(f"Final: {{gs.get_xyz.shape[0]}} Gaussians")
'''


def finetune(
    scene: str,
    variant: str,
    gs_dir: Path | None = None,
    iters: int = PERCEPTUAL_ITERS,
) -> bool:
    """Run perceptual fine-tuning on a trained model.

    Args:
        scene: Scene name
        variant: Variant name (trained model to fine-tune)
        gs_dir: Path to 3DGS code
        iters: Number of fine-tuning iterations

    Returns:
        True if successful
    """
    if gs_dir is None:
        gs_dir = GS_DIR

    model_path = OUTPUT_DIR / "models" / scene / variant
    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return False

    out_dir = OUTPUT_DIR / "models" / scene / f"{variant}_perceptual"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PERCEPTUAL FINE-TUNE: {scene}/{variant}")
    print(f"  Iters: {iters}")
    print(f"  LPIPS λ={LPIPS_LAMBDA}, DINO λ={DINO_LAMBDA}, L1 λ={L1_LAMBDA}")
    print(f"{'='*60}")

    script = build_finetune_script(
        model_path=str(model_path),
        gs_dir=str(gs_dir),
        output_dir=str(out_dir),
        iters=iters,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(script)
        script_path = tf.name

    try:
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True,
            cwd=str(gs_dir), timeout=7200,
        )
        elapsed = time.time() - t0

        if r.returncode == 0:
            print(f"  [OK] {scene}/{variant}_perceptual: DONE in {elapsed/60:.1f} min")
            print(f"  Model: {out_dir / 'point_cloud.ply'}")
            return True
        else:
            print(f"  [FAIL] {scene}/{variant}_perceptual")
            for line in (r.stdout + r.stderr).splitlines()[-15:]:
                print(f"    {line}")
            return False
    finally:
        Path(script_path).unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Perceptual Fine-Tuning (LPIPS/DINOv2)")
    p.add_argument("--scene", required=True)
    p.add_argument("--variant", default="full_60k")
    p.add_argument("--iters", type=int, default=PERCEPTUAL_ITERS)
    p.add_argument("--gs-dir", default=None)
    p.add_argument("--data-dir", default=None, help="Override data directory")
    args = p.parse_args()

    if args.data_dir:
        set_data_dir(args.data_dir)

    success = finetune(
        args.scene, args.variant,
        Path(args.gs_dir) if args.gs_dir else None,
        args.iters,
    )
    sys.exit(0 if success else 1)
