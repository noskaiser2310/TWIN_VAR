"""Configuration for VAR 2026 Digital Twin BTS auto pipeline."""

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

# ── Scene list (auto-detected from data/) ─────────────────────
SCENES = sorted(
    d.name for d in DATA_DIR.iterdir()
    if d.is_dir() and (d / "train" / "images").exists()
)

# ── Multi-variant training config ─────────────────────────────


@dataclass
class VariantConfig:
    """Configuration for one 3DGS training variant."""

    name: str
    iterations: int = 30_000
    # Depth regularization
    use_depth: bool = False
    # Exposure compensation
    use_exposure: bool = False
    # Anti-aliasing (Mip-Splatting EWA filter)
    use_antialiasing: bool = False
    # Sparse Adam optimizer (2.7x faster)
    use_sparse_adam: bool = False
    # Random background
    random_background: bool = False
    # Resolution scale (1 = full, 2 = half)
    resolution: int = 1
    # Data device (cpu saves VRAM)
    data_device: str = "cuda"
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
        if self.extra_args:
            args.append(self.extra_args)
        return " ".join(args)


# ── Variant definitions ───────────────────────────────────────
VARIANTS: list[VariantConfig] = [
    VariantConfig(
        name="baseline",
        iterations=30_000,
        data_device="cpu",  # Save VRAM for T4
    ),
    VariantConfig(
        name="depth",
        iterations=30_000,
        use_depth=True,
        data_device="cpu",
    ),
    VariantConfig(
        name="exposure",
        iterations=30_000,
        use_exposure=True,
        data_device="cpu",
    ),
    VariantConfig(
        name="antialias",
        iterations=30_000,
        use_antialiasing=True,
        data_device="cpu",
    ),
    VariantConfig(
        name="full_combo",
        iterations=30_000,
        use_depth=True,
        use_exposure=True,
        use_antialiasing=True,
        use_sparse_adam=True,
        data_device="cpu",
    ),
    VariantConfig(
        name="fast",
        iterations=7_000,
        data_device="cpu",
    ),
]

# ── Kaggle kernel settings ────────────────────────────────────
KERNEL_GPU = True
KERNEL_INTERNET = True  # Need internet to pip install deps
KERNEL_TIMEOUT_SECONDS = 9 * 3600  # 9 hours max
KERNEL_LANGUAGE = "python"
KERNEL_TYPE = "script"

# ── Depth-Anything v2 settings ────────────────────────────────
DEPTH_MODEL = "vitl"  # vitl = best quality, vitb = faster, vits = fastest
DEPTH_ANYTHING_REPO = "https://github.com/DepthAnything/Depth-Anything-V2.git"
DEPTH_CHECKPOINT_URL = (
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/"
    "resolve/main/depth_anything_v2_vitl.pth?download=true"
)

# ── Rendering config ──────────────────────────────────────────
RENDER_SKIP_TRAIN = True  # Only render test poses (not training set)
RENDER_ITERATION = 30_000  # Use final checkpoint

# ── Submission config ─────────────────────────────────────────
SUBMISSION_DIR = ROOT / "submissions"
