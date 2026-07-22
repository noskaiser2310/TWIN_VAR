"""Enhanced configuration for VAR 2026 Digital Twin BTS — Top 1 Strategy.

Extended from the original pipeline/config.py with:
- 8 optimized variants (Mip-Splatting, Scaffold-GS, Depth-guided, Sky-masked, etc.)
- Smart ensemble configuration
- Post-processing parameters
- Scene-specific tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # Viettel_Race_AI/De_1/
DATA_DIR = ROOT / "data"
GS_DIR = ROOT / "gaussian-splatting"
PIPELINE_DIR = ROOT / "pipeline"
OUTPUT_DIR = ROOT / "output"
SUBMISSION_NAME = "submission_round1.zip"

# ── Kaggle credentials ────────────────────────────────────────
KAGGLE_USERNAME = "snnguynvnk19hl"
KAGGLE_DATASET_PREFIX = f"{KAGGLE_USERNAME}/var2026"

# ── Scene list (auto-detected) ────────────────────────────────
SCENES = sorted(
    d.name for d in DATA_DIR.iterdir()
    if d.is_dir() and (d / "train" / "images").exists()
)

# ── Scene-specific tuning (override global params) ────────────
SCENE_SPECIFIC: dict[str, dict] = {
    # Example: adjust for scene complexity
    # "bonsai": {"iterations": 45000, "densify_until": 30000},
}


# ═══════════════════════════════════════════════════════════════
#  VARIANT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════

@dataclass
class VariantConfig:
    """Configuration for one 3DGS training variant."""

    name: str
    iterations: int = 60_000
    # Core 3DGS
    use_depth: bool = False
    use_exposure: bool = False
    use_antialiasing: bool = False
    use_sparse_adam: bool = False
    random_background: bool = False
    resolution: int = 1
    data_device: str = "cpu"
    # Advanced features
    use_mip_splatting: bool = False
    use_scaffold_gs: bool = False
    use_sky_mask: bool = False
    use_unbounded_reg: bool = False
    use_depth_init: bool = False
    use_multi_scale_depth: bool = False
    use_normal_consistency: bool = False
    # Appearance
    use_appearance_embeddings: bool = False
    # Test-time
    test_time_refine_iters: int = 0
    # Extra CLI args
    extra_args: str = ""

    def to_train_args(self, source_path: str, model_path: str, depth_path: str = "") -> str:
        """Build CLI arguments for train.py."""
        args = [
            f'-s "{source_path}"',
            f'-m "{model_path}"',
            f"--iterations {self.iterations}",
            f"--data_device {self.data_device}",
        ]
        if self.resolution > 1:
            args.append(f"-r {self.resolution}")
        if self.use_depth and depth_path:
            args.append(f'-d "{depth_path}"')
        if self.use_exposure:
            args.append(
                "--exposure_lr_init 0.001 --exposure_lr_final 0.0001 "
                "--exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 "
                "--train_test_exp"
            )
        if self.use_antialiasing:
            args.append("--antialiasing")
        if self.use_sparse_adam:
            args.append("--optimizer_type sparse_adam")
        if self.random_background:
            args.append("--random_background")
        if self.use_mip_splatting:
            args.append("--mip_splatting")
        if self.use_scaffold_gs:
            args.append("--scaffold_gs")
        if self.use_sky_mask:
            args.append("--sky_mask")
        if self.use_unbounded_reg:
            args.append("--unbounded_reg")
        if self.use_depth_init:
            args.append("--depth_init")
        if self.use_multi_scale_depth:
            args.append("--multi_scale_depth_loss")
        if self.use_normal_consistency:
            args.append("--normal_consistency_loss")
        if self.use_appearance_embeddings:
            args.append(
                "--appearance_embeddings "
                "--appearance_embedding_dim 64 "
                "--appearance_mlp_layers 2"
            )
        if self.extra_args:
            args.append(self.extra_args)
        return " ".join(args)


# ── 8 VARIANT DEFINITIONS ─────────────────────────────────────

VARIANTS: list[VariantConfig] = [
    # 1. Vanilla 3DGS — baseline with more iterations
    VariantConfig(
        name="vanilla_60k",
        iterations=60_000,
        data_device="cpu",
    ),
    # 2. Mip-Splatting — anti-aliasing + multi-scale (BEST perceptual quality)
    VariantConfig(
        name="mip_splatting",
        iterations=60_000,
        use_mip_splatting=True,
        use_sparse_adam=True,
        data_device="cpu",
    ),
    # 3. Depth-Guided — better geometry on texture-less surfaces
    VariantConfig(
        name="depth_guided",
        iterations=60_000,
        use_depth=True,
        use_depth_init=True,
        use_multi_scale_depth=True,
        use_normal_consistency=True,
        use_sparse_adam=True,
        data_device="cpu",
    ),
    # 4. Exposure + Anti-Aliasing — handles drone lighting variation
    VariantConfig(
        name="exposure_aa",
        iterations=60_000,
        use_exposure=True,
        use_antialiasing=True,
        use_appearance_embeddings=True,
        use_sparse_adam=True,
        data_device="cpu",
    ),
    # 5. Sky-Masked — no floaters in sky regions
    VariantConfig(
        name="sky_masked",
        iterations=60_000,
        use_sky_mask=True,
        use_unbounded_reg=True,
        use_antialiasing=True,
        use_sparse_adam=True,
        data_device="cpu",
    ),
    # 6. Scaffold-GS — anchor-based for structural accuracy
    VariantConfig(
        name="scaffold_gs",
        iterations=60_000,
        use_scaffold_gs=True,
        use_sparse_adam=True,
        data_device="cpu",
    ),
    # 7. Full Combo v2 — all compatible features
    VariantConfig(
        name="full_combo_v2",
        iterations=90_000,
        use_depth=True,
        use_depth_init=True,
        use_multi_scale_depth=True,
        use_exposure=True,
        use_antialiasing=True,
        use_mip_splatting=True,
        use_sky_mask=True,
        use_unbounded_reg=True,
        use_appearance_embeddings=True,
        use_sparse_adam=True,
        data_device="cpu",
    ),
    # 8. Test-Time Refined — full_combo_v2 + fine-tune on test poses
    VariantConfig(
        name="test_refined",
        iterations=90_000,
        use_depth=True,
        use_depth_init=True,
        use_multi_scale_depth=True,
        use_exposure=True,
        use_antialiasing=True,
        use_mip_splatting=True,
        use_sky_mask=True,
        use_unbounded_reg=True,
        use_appearance_embeddings=True,
        use_sparse_adam=True,
        test_time_refine_iters=500,
        data_device="cpu",
    ),
    # 9. Fast debug variant
    VariantConfig(
        name="fast",
        iterations=7_000,
        data_device="cpu",
    ),
]


# ═══════════════════════════════════════════════════════════════
#  SMART ENSEMBLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnsembleConfig:
    """Configuration for smart per-pixel ensemble blending."""

    # Which variants to use for ensemble
    variants: list[str] = field(default_factory=lambda: [
        "full_combo_v2",
        "mip_splatting",
        "depth_guided",
        "exposure_aa",
        "sky_masked",
        "scaffold_gs",
        "vanilla_60k",
    ])

    # Fallback order (if ensemble fails)
    fallback_order: list[str] = field(default_factory=lambda: [
        "full_combo_v2",
        "mip_splatting",
        "depth_guided",
    ])

    # Confidence weights
    alpha_weight: float = 0.25       # Alpha saturation confidence
    depth_consistency_weight: float = 0.30  # Depth agreement across variants
    color_consistency_weight: float = 0.20  # Local color smoothness
    edge_sharpness_weight: float = 0.15     # Preference for sharp edges
    variant_prior_weight: float = 0.10      # Learned per-variant quality

    # Soft voting temperature (higher = softer blending)
    soft_voting_temperature: float = 2.0

    # Per-variant quality priors (learned or estimated)
    variant_priors: dict[str, float] = field(default_factory=lambda: {
        "full_combo_v2": 1.0,
        "mip_splatting": 0.95,
        "depth_guided": 0.85,
        "exposure_aa": 0.80,
        "sky_masked": 0.75,
        "scaffold_gs": 0.70,
        "vanilla_60k": 0.60,
    })


# ═══════════════════════════════════════════════════════════════
#  POST-PROCESSING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

@dataclass
class PostProcessConfig:
    """Configuration for post-processing rendered images."""

    # Edge-aware sharpening
    sharpen_enabled: bool = True
    sharpen_amount: float = 1.3
    sharpen_radius: float = 3.0

    # Color correction (match training distribution)
    color_correction_enabled: bool = True

    # Denoising (for sky regions)
    sky_denoise_enabled: bool = True
    sky_denoise_strength: float = 5.0


# ═══════════════════════════════════════════════════════════════
#  DEPTH CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEPTH_MODEL = "vitl"  # vitl=best, vitb=faster, vits=fastest
DEPTH_ANYTHING_REPO = "https://github.com/DepthAnything/Depth-Anything-V2.git"
DEPTH_CHECKPOINT_URLS = {
    "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true",
    "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true",
    "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true",
}

# ═══════════════════════════════════════════════════════════════
#  SKY MASK CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SKY_MASK_MODEL = "sam2"  # SAM2 for segmentation
SKY_MASK_CHECKPOINT = ""  # Path to SAM2 checkpoint

# ═══════════════════════════════════════════════════════════════
#  KAGGLE KERNEL SETTINGS
# ═══════════════════════════════════════════════════════════════

KERNEL_GPU = True
KERNEL_INTERNET = True
KERNEL_TIMEOUT_SECONDS = 12 * 3600  # 12 hours max
KERNEL_LANGUAGE = "python"
KERNEL_TYPE = "script"

# ═══════════════════════════════════════════════════════════════
#  RENDERING CONFIGURATION
# ═══════════════════════════════════════════════════════════════

RENDER_SKIP_TRAIN = True
RENDER_ITERATION = -1  # Use final checkpoint

# ═══════════════════════════════════════════════════════════════
#  SUBMISSION CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SUBMISSION_DIR = ROOT / "submissions"


# ── Helper to get scene-specific config ───────────────────────

def get_scene_config(scene: str) -> dict:
    """Get scene-specific overrides if any, else empty dict."""
    return SCENE_SPECIFIC.get(scene, {})


def get_ensemble_config() -> EnsembleConfig:
    """Get ensemble configuration."""
    return EnsembleConfig()


def get_postprocess_config() -> PostProcessConfig:
    """Get post-processing configuration."""
    return PostProcessConfig()
