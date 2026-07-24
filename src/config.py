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

import copy
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
#  PATHS — edit these for your environment
# ═══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent                          # src/
GS_DIR = ROOT / "_3dgs"                                         # 3DGS code (self-contained copy)
DATA_DIR: Path = ROOT.parent / "data"                           # scene data (overridable via set_data_dir)
OUTPUT_DIR = ROOT / "output"                                    # all outputs
SUBMISSION_DIR = ROOT / "submissions"                           # final ZIPs
SUBMISSION_NAME = "submission_round1.zip"

# ── Auto-detect scenes ────────────────────────────────────────
SCENES: list[str] = []


def _scan_scenes() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir() and (d / "train" / "images").exists()
    )


def set_data_dir(path: str | Path) -> None:
    global DATA_DIR, SCENES
    DATA_DIR = Path(path).resolve()
    SCENES = _scan_scenes()
    print(f"[DATA_DIR] → {DATA_DIR} ({len(SCENES)} scenes detected)")


SCENES = _scan_scenes()

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
    densify_method: str = "abs"          # "abs" (AbsGS 2024, +0.5-1dB) or "orig" (vanilla)
    depth: bool = False                  # -d <depths_dir>
    exposure: bool = False               # --exposure_lr_* --train_test_exp
    antialias: bool = False              # --antialiasing (PipelineParams)
    sparse_adam: bool = False            # --optimizer_type sparse_adam (2.7x faster)
    random_bg: bool = False              # --random_background
    multiscale: bool = False             # FreqDS: random resolution training [0.5-2.0x]
    multiscale_min: float = 0.5          # Min scale for multi-scale training
    multiscale_max: float = 2.0          # Max scale for multi-scale training
    sky_mask: bool = False               # Mask sky pixels in loss (outdoor drone)
    edge_guided: bool = False            # Edge-guided densification (thin structures)
    edge_boost: float = 0.5              # Boost multiplier for edge Gaussians
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
        if self.densify_method != "abs":
            a.extend(["--densify_method", self.densify_method])
        if self.multiscale:
            a.append("--multiscale")
            a.extend(["--multiscale_min", str(self.multiscale_min)])
            a.extend(["--multiscale_max", str(self.multiscale_max)])
        if self.sky_mask:
            a.append("--sky_mask")
        if self.edge_guided:
            a.append("--edge_guided")
            a.extend(["--edge_boost", str(self.edge_boost)])
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


# ── Per-scene tuning defaults ────────────────────────────
# Merged into Variant at train time via get_scene_variant().
# Strategy based on COLMAP point cloud density (points3D.bin size):
#   SPARSE scenes (low COLMAP density) → aggressive densification + strong depth
#   DENSE scenes (high COLMAP density) → lighter densification, trust COLMAP more
#   INDOOR scenes (bonsai, chair)       → white_bg, low SSIM, minimal densification

# Indoor — synthetic objects, white backgrounds
_INDOOR_DEFAULTS: dict = {
    "white_bg": True,
    "lambda_dssim": 0.2,
    "percent_dense": 0.01,
    "densify_until_iter": 15_000,
    "densify_grad_threshold": 0.0002,
    "depth_l1_weight_init": 1.0,
    "depth_l1_weight_final": 0.05,
    "sh_degree": 3,
}

# Chair — more complex geometry than bonsai (8.9MB COLMAP, 59 test poses, 205 imgs)
_CHAIR_DEFAULTS: dict = {
    "white_bg": True,
    "lambda_dssim": 0.25,                # Slightly higher SSIM for chair details
    "percent_dense": 0.015,              # ↑ Mild increase (denser COLMAP than bonsai)
    "densify_until_iter": 18_000,        # ↑ Slightly longer
    "densify_grad_threshold": 0.00018,   # ↓ Mildly lower
    "depth_l1_weight_init": 1.0,
    "depth_l1_weight_final": 0.05,
    "sh_degree": 3,
}

# Outdoor BTS with SPARSE COLMAP (HCM0421: 16.5MB, HCM0674: 15.2MB)
# → COLMAP struggles → compensate with stronger densification + depth.
#   Adjusted for T4 14.56 GiB (actual usable, not 16 GiB) to avoid OOM.
#   NOTE: sh_degree intentionally omitted (dataclass default=4) so variant-level
#   overrides like hcm_sh5_60k (sh_degree=5) are NOT silently overwritten.
_OUTDOOR_SPARSE_DEFAULTS: dict = {
    "white_bg": False,
    "lambda_dssim": 0.3,
    "percent_dense": 0.025,              # ↓ 0.035→0.025: giảm max Gaussians, tránh OOM trên T4
    "densify_until_iter": 22_000,        # ↓ 28k→22k: dừng densify sớm hơn, tiết kiệm VRAM
    "densify_grad_threshold": 0.00015,   # ↑ 0.0001→0.00015: ít densify hơn, kiểm soát số Gaussians
    "depth_l1_weight_init": 2.0,         # ↑↑ Strong depth (balanced: noisy drone depth)
    "depth_l1_weight_final": 0.12,       # ↑↑ Keep depth active
}

# Outdoor BTS with DENSE COLMAP (HCM0539: 22.1MB, HCM0644: 21.4MB, HCM0540: 20.2MB)
# → COLMAP already captured good structure → moderate tuning
#   Adjusted for T4 14.56 GiB: HCM0539 crashed OOM at iter 15,900 with 9.55 GiB allocated.
#   NOTE: sh_degree intentionally omitted (dataclass default=4) so variant-level
#   overrides like hcm_sh5_60k (sh_degree=5) are NOT silently overwritten.
_OUTDOOR_DENSE_DEFAULTS: dict = {
    "white_bg": False,
    "lambda_dssim": 0.3,
    "percent_dense": 0.02,               # ↓ 0.025→0.02: giảm max Gaussians, tránh OOM trên T4
    "densify_until_iter": 20_000,        # ↓ 25k→20k: dừng densify sớm hơn
    "densify_grad_threshold": 0.00015,   # ↑ 0.00012→0.00015: ít densify hơn
    "depth_l1_weight_init": 1.8,         # ↑ Moderate depth weight
    "depth_l1_weight_final": 0.08,       # ↑ Moderate
}

PER_SCENE_CONFIG: dict[str, dict] = {
    # ── Indoor ──
    "bonsai": _INDOOR_DEFAULTS.copy(),          # 248 imgs, 29 test, 6.1MB COLMAP
    "chair":  _CHAIR_DEFAULTS.copy(),           # 205 imgs, 59 test, 8.9MB COLMAP
    # ── Outdoor Sparse COLMAP (compensate aggressively) ──
    "HCM0421": _OUTDOOR_SPARSE_DEFAULTS.copy(), # 240 imgs, 16.5 MB
    "HCM0674": _OUTDOOR_SPARSE_DEFAULTS.copy(), # 240 imgs, 15.2 MB (sparsest)
    # ── Outdoor Dense COLMAP (moderate tuning) ──
    "HCM0539": _OUTDOOR_DENSE_DEFAULTS.copy(),  # 240 imgs, 22.1 MB (densest)
    "HCM0540": _OUTDOOR_DENSE_DEFAULTS.copy(),  # 240 imgs, 20.2 MB
    "HCM0644": _OUTDOOR_DENSE_DEFAULTS.copy(),  # 240 imgs, 21.4 MB
}


# Safer defaults for auto-detected unknown scenes (less aggressive than hand-tuned)
# NOTE: sh_degree intentionally omitted (dataclass default=4) so variant-level
# overrides like hcm_sh5_60k (sh_degree=5) are NOT silently overwritten.
_AUTO_OUTDOOR_DEFAULTS: dict = {
    "white_bg": False,
    "lambda_dssim": 0.25,
    "percent_dense": 0.025,
    "densify_until_iter": 25_000,
    "densify_grad_threshold": 0.00015,
    "depth_l1_weight_init": 1.5,
    "depth_l1_weight_final": 0.08,
}


def _analyze_scene(scene: str) -> dict:
    """Auto-detect scene characteristics from COLMAP data.

    Used when a scene is NOT in PER_SCENE_CONFIG. Uses per-image
    COLMAP density (MB/image) as a robust metric for classification.
    """
    scene_dir = DATA_DIR / scene
    if not scene_dir.exists():
        print(f"[WARN] Scene directory {scene_dir} not found — using auto outdoor defaults")
        return _AUTO_OUTDOOR_DEFAULTS.copy()

    # Detect image count
    img_dir = scene_dir / "train" / "images"
    if not img_dir.exists():
        img_dir = scene_dir / "images"
    n_images = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
    if n_images == 0:
        print(f"[WARN] Scene '{scene}' has 0 images — using auto outdoor defaults")
        return _AUTO_OUTDOOR_DEFAULTS.copy()

    # Detect COLMAP density (per image)
    points_paths = [
        scene_dir / "train" / "sparse" / "0" / "points3D.bin",
        scene_dir / "sparse" / "0" / "points3D.bin",
    ]
    colmap_mb = 0.0
    for pp in points_paths:
        if pp.exists():
            colmap_mb = pp.stat().st_size / (1024 * 1024)
            break

    density = colmap_mb / max(n_images, 1)  # MB per image

    # Classification by per-image COLMAP density (robust to image count)
    if density < 0.03:                      # < 0.03 MB/image → indoor/synthetic
        tier = "INDOOR"
        result = _INDOOR_DEFAULTS.copy()
    elif density < 0.08:                    # 0.03-0.08 → outdoor with moderate COLMAP
        tier = "OUTDOOR (auto)"
        result = _AUTO_OUTDOOR_DEFAULTS.copy()
    else:                                   # > 0.08 → outdoor with dense COLMAP
        tier = "OUTDOOR DENSE"
        result = _OUTDOOR_DENSE_DEFAULTS.copy()

    print(f"[AUTO-DETECT] Scene '{scene}': {tier} "
          f"({n_images} imgs, {colmap_mb:.1f}MB, {density:.3f} MB/img). "
          f"Add to PER_SCENE_CONFIG for optimal tuning.")
    return result


def get_scene_variant(variant: Variant, scene: str) -> Variant:
    """Create a scene-optimized copy of a Variant.

    Applies PER_SCENE_CONFIG overrides for known scenes,
    or auto-detects scene characteristics for unknown scenes.
    Returns a NEW Variant (does not mutate original).

    Usage:
        v = get_scene_variant(VARIANTS[0], "HCM0421")
        v.args_list(...)  # now has outdoor BTS densify params
    """
    overrides = PER_SCENE_CONFIG.get(scene)
    if overrides is None:
        # Unknown scene → auto-detect based on COLMAP + image count
        overrides = _analyze_scene(scene)

    # Deep-copy to avoid mutating the original Variant
    v = copy.deepcopy(variant)
    for key, value in overrides.items():
        if hasattr(v, key):
            setattr(v, key, value)
        else:
            warnings.warn(f"Unknown per-scene override key '{key}' — ignored")
    return v


# ── 10 variants tối ưu cho drone BTS ─────────────────────────

VARIANTS: list[Variant] = [
    # Quick test
    Variant("fast", iters=7_000, eval_mode=False, quiet=False),
    Variant("check", iters=100, eval_mode=False, quiet=False, densify_until_iter=0),

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

    # ── Multi-scale (FreqDS) variants — highest ROI for LPIPS/PSNR ──
    Variant("multiscale",      iters=30_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            multiscale=True),  # FreqDS: random resolution [0.5-2.0x]
    Variant("multiscale_60k",  iters=60_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            multiscale=True, checkpoint_iterations=[30000, 60000]),
    Variant("multiscale_sky",  iters=60_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            multiscale=True, sky_mask=True, checkpoint_iterations=[30000, 60000]),  # + sky masking

    # ── Edge-guided densification — thin BTS structures (antennas, cables) ──
    Variant("edge_60k",        iters=60_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            edge_guided=True, edge_boost=0.5, checkpoint_iterations=[30000, 60000]),
    Variant("multiscale_edge", iters=60_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            multiscale=True, sky_mask=True, edge_guided=True, edge_boost=0.5,
            checkpoint_iterations=[30000, 60000]),  # All features combined

    # ── Indoor-optimized variants (white_bg + reduced densification) ──
    Variant("indoor_60k",      iters=60_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            white_bg=True, checkpoint_iterations=[30000, 60000]),  # Indoor: full_60k + white_bg
    Variant("indoor_edge_60k", iters=60_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            white_bg=True, edge_guided=True, edge_boost=0.5,
            checkpoint_iterations=[30000, 60000]),  # Indoor: + edge for cables/structures
    Variant("indoor_90k",      iters=90_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            white_bg=True, multiscale=True,
            start_checkpoint="full_60k"),  # Indoor: resume từ full_60k + white_bg + multiscale (quality cao nhất)

    # ── Outdoor HCM-optimized variants (high quality, resume-based) ──
    Variant("hcm_90k",         iters=90_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            multiscale=True, edge_guided=True, edge_boost=0.5,
            start_checkpoint="full_60k"),  # Resume from full_60k + multiscale + edge
    Variant("hcm_edge_strong", iters=60_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            multiscale=True, edge_guided=True, edge_boost=0.8,
            checkpoint_iterations=[30000, 60000]),  # Stronger edge boost for thin structures

    # ── High SH degree variant (caution: ~36 coefs vs 25, may OOM on T4) ──
    Variant("hcm_sh5_60k",     iters=60_000, sh_degree=5, depth=True, exposure=True, antialias=True, sparse_adam=True,
            multiscale=True, checkpoint_iterations=[30000, 60000]),  # Higher SH for outdoor detail

    # ── Quick variants (15k iters, ~30 min, không cần chờ) ──
    Variant("quick_15k",       iters=15_000, depth=True, exposure=True, antialias=True, sparse_adam=True),
    Variant("quick_edge_15k",  iters=15_000, depth=True, exposure=True, antialias=True, sparse_adam=True,
            multiscale=True, edge_guided=True, edge_boost=0.5),  # Quick edge test
]

# ═══════════════════════════════════════════════════════════════
#  ENHANCED ENSEMBLE CONFIG (v2.0)
# ═══════════════════════════════════════════════════════════════

ENSEMBLE_VARIANTS = ["quick_edge_15k", "quick_15k", "hcm_90k", "hcm_edge_strong", "hcm_sh5_60k", "indoor_60k", "indoor_edge_60k", "multiscale_edge", "multiscale_sky", "edge_60k", "multiscale_60k", "multiscale", "full_60k", "big", "full", "depth_expo", "antialias", "exposure", "depth", "baseline"]
ENSEMBLE_FALLBACK = ["multiscale_edge", "multiscale_60k", "full_60k", "big", "full", "depth_expo"]
ENSEMBLE_PRIORS = {
    "quick_edge_15k": 0.85, "quick_15k": 0.80,  # Quick variants (lower weight vs 60k)
    "indoor_edge_60k": 1.20, "indoor_60k": 1.15,  # Indoor-optimized (very high quality)
    "hcm_90k": 1.18, "hcm_edge_strong": 1.15, "hcm_sh5_60k": 1.12,  # Outdoor HCM variants
    "multiscale_edge": 1.15, "multiscale_sky": 1.12, "edge_60k": 1.08,
    "multiscale_60k": 1.10, "multiscale": 1.05,
    "full_60k": 1.0, "big": 1.05,
    "full": 0.95, "depth_expo": 0.85, "antialias": 0.80,
    "exposure": 0.75, "depth": 0.70, "baseline": 0.60,
}

# Enhanced ensemble v2.0 — per-pixel variance + softmax + protected anchor
ENSEMBLE_ANCHOR = "multiscale_edge"       # Protected base variant
ENSEMBLE_NUM_COMPANIONS = 3               # Companions must agree to override anchor
ENSEMBLE_AGREEMENT_THRESHOLD = 0.015      # Max per-pixel RGB std for agreement
ENSEMBLE_TEMPERATURE = 2.0                # Default softmax temperature
ENSEMBLE_TEMPERATURE_SEARCH = [1.0, 2.0, 3.0, 4.0]
ENSEMBLE_VARIANCE_WEIGHT = 0.3            # Penalty for high-variance regions

# Per-scene ensemble overrides
PER_SCENE_ENSEMBLE = {
    # Indoor: indoor_60k là anchor mới (full_60k + white_bg = chất lượng cao nhất)
    "bonsai":  {"temperature": 3.0, "anchor": "indoor_60k"},
    "chair":   {"temperature": 2.5, "anchor": "indoor_60k"},
    # Outdoor sparse: multiscale_edge giữ nguyên (antenna detail)
    "HCM0421": {"temperature": 1.5, "anchor": "multiscale_edge"},
    "HCM0674": {"temperature": 1.5, "anchor": "multiscale_edge"},
    # Outdoor dense: hcm_edge_strong mới (edge_boost=0.8 cho cấu trúc mỏng)
    "HCM0539": {"temperature": 2.0, "anchor": "multiscale_edge"},
    "HCM0540": {"temperature": 2.0, "anchor": "multiscale_edge"},
    "HCM0644": {"temperature": 2.0, "anchor": "multiscale_edge"},
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
COMPACT_VARIANTS = ["full_60k", "indoor_60k", "indoor_edge_60k", "multiscale_edge", "hcm_edge_strong", "big", "full", "depth_expo", "antialias"]

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
