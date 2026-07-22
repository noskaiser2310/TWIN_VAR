"""
VAR 2026 — Digital Twin BTS: Complete Local Pipeline
=====================================================
Self-contained project. Copy this folder anywhere, set paths in config.py, run main.py.

Architecture:
    main.py → train variants → render test poses → ensemble blend → postprocess → package

Requirements: Python 3.10+, CUDA GPU (16GB+ VRAM), 3DGS submodules built
"""

from __future__ import annotations

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
#  TRAINING VARIANTS
# ═══════════════════════════════════════════════════════════════

@dataclass
class Variant:
    name: str
    iters: int = 30_000
    depth: bool = False
    exposure: bool = False
    antialias: bool = False
    sparse_adam: bool = False
    random_bg: bool = False
    resolution: int = 1
    data_device: str = "cpu"
    extra: str = ""

    def args(self, src: str, model: str, depth_dir: str = "") -> str:
        a = [
            f'-s "{src}"',
            f'-m "{model}"',
            f"--iterations {self.iters}",
            f"--data_device {self.data_device}",
        ]
        if self.resolution > 1:
            a.append(f"-r {self.resolution}")
        if self.depth and depth_dir:
            a.append(f'-d "{depth_dir}"')
        if self.exposure:
            a.append("--exposure_lr_init 0.001 --exposure_lr_final 0.0001 "
                     "--exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 "
                     "--train_test_exp")
        if self.antialias:
            a.append("--antialiasing")
        if self.sparse_adam:
            a.append("--optimizer_type sparse_adam")
        if self.random_bg:
            a.append("--random_background")
        if self.extra:
            a.append(self.extra)
        return " ".join(a)


# 8 variants tối ưu
VARIANTS: list[Variant] = [
    Variant("baseline",     iters=30_000),
    Variant("depth",        iters=30_000, depth=True),
    Variant("exposure",     iters=30_000, exposure=True),
    Variant("antialias",    iters=30_000, antialias=True, sparse_adam=True),
    Variant("depth_expo",   iters=30_000, depth=True, exposure=True, sparse_adam=True),
    Variant("full",         iters=30_000, depth=True, exposure=True, antialias=True, sparse_adam=True),
    Variant("full_60k",     iters=60_000, depth=True, exposure=True, antialias=True, sparse_adam=True),
    Variant("big",          iters=90_000, depth=True, exposure=True, antialias=True, sparse_adam=True),
    Variant("fast",         iters=7_000),   # quick test
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
    "voxel_size": 0.005,        # Voxel grid size for spatial merging
    "opacity_cull": 0.05,       # Prune Gaussians below this opacity
}
COMPACT_VARIANTS = ["full_60k", "full", "depth_expo", "antialias"]

# ═══════════════════════════════════════════════════════════════
#  TEST-TIME ADAPTATION
# ═══════════════════════════════════════════════════════════════

TTA_CONFIG = {
    "iters": 1000,              # TTA optimization steps
    "delta_lr": 0.01,           # Learning rate for delta layer
    "photo_weight": 0.1,        # Photometric consistency weight
    "depth_weight": 0.0,        # Depth regularization (0 = off, no depth at test-time)
}

# ═══════════════════════════════════════════════════════════════
#  DEPTH CONFIG
# ═══════════════════════════════════════════════════════════════

DEPTH_MODEL = "vitl"
DEPTH_URLS = {
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
}
