"""
VAR 2026 — Digital Twin BTS: Complete Local Pipeline
=====================================================
Self-contained project. Fully leverages the gaussian-splatting baseline:
  - eval mode, sh_degree, depth scheduling, densification tuning
  - exposure compensation, anti-aliasing, sparse Adam, white/random BG
  - checkpoint resume, fused-ssim, competition metric evaluation

Architecture:
    main.py → train → render → eval → compact → tta → ensemble → post → package

Requirements: Python 3.10+, CUDA GPU (16GB+ VRAM), 3DGS submodules built
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
#  PATHS — edit these for your environment
# ═══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent                          # src/
GS_DIR = ROOT.parent / "gaussian-splatting"                     # 3DGS code
DATA_DIR = ROOT.parent / "data"                                 # scene data
OUTPUT_DIR = ROOT / "output"                                    # all outputs
SUBMISSION_DIR = ROOT / "submissions"                           # final ZIPs
SUBMISSION_NAME = "submission_round1.zip"

# ── Auto-detect scenes ────────────────────────────────────────
SCENES = sorted(
    d.name for d in DATA_DIR.iterdir()
    if d.is_dir() and (d / "train" / "images").exists()
)

# ═══════════════════════════════════════════════════════════════
#  TRAINING VARIANTS — fully leveraging gaussian-splatting baseline
# ═══════════════════════════════════════════════════════════════

@dataclass
class Variant:
    """One 3DGS training configuration. Maps directly to gaussian-splatting CLI args."""

    name: str

    # ── ModelParams ──────────────────────────────────────
    iters: int = 30_000
    sh_degree: int = 4                   # ↑4 for drone/outdoor (default 3)
    eval_mode: bool = True               # ← CRITICAL: separate test cameras for metrics
    white_bg: bool = False               # Outdoor=black, indoor=white
    resolution: int = 1
    data_device: str = "cuda"            # "cpu" saves VRAM on T4

    # ── Features ─────────────────────────────────────────
    depth: bool = False                  # -d <depths_dir>
    exposure: bool = False               # --exposure_lr_* --train_test_exp
    antialias: bool = False              # --antialiasing (PipelineParams)
    sparse_adam: bool = False            # --optimizer_type sparse_adam (2.7x faster)
    random_bg: bool = False              # --random_background
    start_checkpoint: str = ""           # Resume from this variant's checkpoint

    # ── OptimizationParams (drone-tuned) ─────────────────
    lambda_dssim: float = 0.3            # ↑0.3 for thin BTS structures (default 0.2)
    percent_dense: float = 0.02          # ↑0.02 for drone sparse captures (default 0.01)
    densify_until_iter: int = 25_000     # ↑25k for drone detail (default 15k)
    densify_grad_threshold: float = 0.00015  # ↓0.00015 for more densification
    depth_l1_weight_init: float = 1.0    # Depth regularization start weight
    depth_l1_weight_final: float = 0.05  # ↑0.05 keeps depth influence longer
    position_lr_init: float = 0.00016    # Standard (can override per-scene)
    feature_lr: float = 0.0025           # Standard
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001

    # ── Checkpoint config ───────────────────────────────
    checkpoint_iterations: list[int] = field(default_factory=list)  # Save .pth checkpoints
    quiet: bool = True                   # Suppress output

    # ── Extra CLI args ───────────────────────────────────
    extra: str = ""

    def args_list(self, src: str, model: str,
                  depth_dir: str = "",
                  base_model_path: str = "") -> list[str]:
        """Build list of CLI arguments for gaussian-splatting/train.py.

        Returns list[str] for safe subprocess.run (no shell injection).
        """
        a = [
            "-s", src,
            "-m", model,
            "--iterations", str(self.iters),
            "--data_device", self.data_device,
            "--sh_degree", str(self.sh_degree),
            "--densify_until_iter", str(self.densify_until_iter),
            "--densify_grad_threshold", str(self.densify_grad_threshold),
            "--percent_dense", str(self.percent_dense),
            "--lambda_dssim", str(self.lambda_dssim),
            "--depth_l1_weight_init", str(self.depth_l1_weight_init),
            "--depth_l1_weight_final", str(self.depth_l1_weight_final),
            "--position_lr_init", str(self.position_lr_init),
            "--feature_lr", str(self.feature_lr),
            "--opacity_lr", str(self.opacity_lr),
            "--scaling_lr", str(self.scaling_lr),
            "--rotation_lr", str(self.rotation_lr),
            "--quiet",
        ]

        if self.resolution > 1:
            a.extend(["-r", str(self.resolution)])
        if self.eval_mode:
            a.append("--eval")
        if self.white_bg:
            a.append("--white_background")
        if self.random_bg:
            a.append("--random_background")
        if self.depth and depth_dir:
            a.extend(["-d", depth_dir])
        if self.exposure:
            a.extend([
                "--exposure_lr_init", "0.01",
                "--exposure_lr_final", "0.001",
                "--exposure_lr_delay_steps", "5000",
                "--exposure_lr_delay_mult", "0.001",
                "--train_test_exp",
            ])
        if self.antialias:
            a.append("--antialiasing")
        if self.sparse_adam:
            a.extend(["--optimizer_type", "sparse_adam"])
        if self.checkpoint_iterations:
            a.append("--checkpoint_iterations")
            a.extend(str(c) for c in self.checkpoint_iterations)
        if self.quiet:
            a.append("--quiet")
        else:
            # Still suppress viewer for headless
            a.append("--disable_viewer")
        if self.start_checkpoint and base_model_path:
            chkpnt_dir = os.path.join(base_model_path, "point_cloud")
            candidate_iters = []
            if os.path.isdir(chkpnt_dir):
                for entry in os.listdir(chkpnt_dir):
                    if entry.startswith("iteration_"):
                        try:
                            candidate_iters.append(int(entry.split("_")[1]))
                        except ValueError:
                            pass
            if candidate_iters:
                load_iter = max(candidate_iters)
                a.extend(["--start_checkpoint",
                          str(Path(base_model_path) / f"chkpnt{load_iter}.pth")])
        if self.extra:
            a.extend(self.extra.split())
        return a


# ── 9 variants tối ưu cho drone BTS ──────────────────────────

VARIANTS: list[Variant] = [
    # Quick test
    Variant("fast", iters=7_000, eval_mode=False, quiet=False),

    # Anchor reference
    Variant("baseline", iters=30_000, sh_degree=3),

    # Single-feature variants
    Variant("depth",           iters=30_000, depth=True),
    Variant("exposure",        iters=30_000, exposure=True),
    Variant("antialias",       iters=30_000, antialias=True, sparse_adam=True),
    Variant("white_bg",        iters=30_000, white_bg=True),     # Indoor BTS

    # Combo variants (with checkpoint saving for resume)
    Variant("depth_expo",      iters=30_000, depth=True, exposure=True, sparse_adam=True),
    Variant("full",            iters=30_000, depth=True, exposure=True, antialias=True, sparse_adam=True),
    Variant("full_60k",        iters=60_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            checkpoint_iterations=[30000, 60000]),  # Save checkpoints for big variant resume
    Variant("big",             iters=90_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            start_checkpoint="full_60k"),  # Resumes from full_60k checkpoint
]

# ═══════════════════════════════════════════════════════════════
#  ENSEMBLE CONFIG
# ═══════════════════════════════════════════════════════════════

ENSEMBLE_VARIANTS = ["full_60k", "full", "depth_expo", "antialias", "exposure", "depth", "baseline"]
ENSEMBLE_FALLBACK = ["full_60k", "full", "depth_expo"]
ENSEMBLE_PRIORS = {
    "full_60k": 1.0, "full": 0.95, "depth_expo": 0.85, "antialias": 0.80,
    "exposure": 0.75, "depth": 0.70, "baseline": 0.60, "big": 1.05,
}

# ═══════════════════════════════════════════════════════════════
#  POST-PROCESS CONFIG
# ═══════════════════════════════════════════════════════════════

SHARPEN_AMOUNT = 1.2       # unsharp mask strength
SHARPEN_RADIUS = 2.0       # blur sigma
COLOR_MATCH = True          # match train color distribution

# ═══════════════════════════════════════════════════════════════
#  GAUSSIAN-LEVEL COMPACT MERGE
# ═══════════════════════════════════════════════════════════════

COMPACT_CONFIG = {
    "voxel_size": 0.005,
    "opacity_cull": 0.05,
}
COMPACT_VARIANTS = ["full_60k", "full", "depth_expo", "antialias"]

# ═══════════════════════════════════════════════════════════════
#  TEST-TIME ADAPTATION
# ═══════════════════════════════════════════════════════════════

TTA_CONFIG = {
    "iters": 1000,
    "delta_lr": 0.01,
    "photo_weight": 0.1,
    "depth_weight": 0.0,
}

# ═══════════════════════════════════════════════════════════════
#  COMPETITION METRIC EVALUATION
# ═══════════════════════════════════════════════════════════════

# VAR 2026 Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm
# PSNR normalization: assume max ~40 dB for 8-bit images
PSNR_MAX = 40.0
LPIPS_WEIGHT = 0.4
SSIM_WEIGHT = 0.3
PSNR_WEIGHT = 0.3

# ═══════════════════════════════════════════════════════════════
#  DEPTH CONFIG
# ═══════════════════════════════════════════════════════════════

DEPTH_MODEL = "vitl"
DEPTH_URLS = {
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
}
