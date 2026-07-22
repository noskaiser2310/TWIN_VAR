# 🚀 SETUP.md — VAR 2026 Digital Twin BTS Pipeline

> **Version:** 2.3.0  
> **Last updated:** 2026-07-22  
> **Target OS:** Linux (Ubuntu 22.04+) / Windows (PowerShell)  
> **GPU:** NVIDIA RTX A4000 20GB / Tesla T4 16GB  
> **Time estimate:** 30 min setup + 1 min test run

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone & Directory Setup](#2-clone--directory-setup)
3. [Build 3DGS Submodules](#3-build-3dgs-submodules)
4. [Install Dependencies](#4-install-dependencies)
5. [Verify Installation](#5-verify-installation)
6. [Quick Test Run](#6-quick-test-run)
7. [Full Pipeline](#7-full-pipeline)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Check |
|-------------|---------|-------|
| Python | 3.10+ | `python --version` |
| CUDA Toolkit | 11.8+ | `nvcc --version` |
| NVIDIA Driver | 525+ | `nvidia-smi` |
| PyTorch | 2.1+ (CUDA) | `python -c "import torch; print(torch.cuda.is_available())"` |
| Git | 2.x | `git --version` |
| VRAM | 16GB+ | `nvidia-smi --query-gpu=memory.total --format=csv` |

### Install PyTorch (if missing)

```bash
# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Clone & Directory Setup

```bash
# Clone the repo
git clone <your-repo-url> var2026-bts
cd var2026-bts/De_1

# Verify structure
ls -la
# Expected: ARCHITECTURE.md  BASELINE_AUDIT.md  README.md  SETUP.md
#           STRATEGY_TOP1.md  data/  gaussian-splatting/  pipeline/  src/

# The data/ directory should contain 7 scenes:
ls data/
# Expected: bonsai/  chair/  HCM0421/  HCM0539/  HCM0540/  HCM0644/  HCM0674/
```

---

## 3. Build 3DGS Submodules

```bash
cd gaussian-splatting

# Initialize all submodules (already cloned, just need init)
git submodule update --init submodules/simple-knn
git submodule update --init submodules/diff-gaussian-rasterization
git submodule update --init submodules/fused-ssim

# Build & install (requires CUDA toolkit + PyTorch)
pip install -e submodules/simple-knn
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/fused-ssim

cd ..
```

> **Note:** `diff-gaussian-rasterization` uses the `dr_aa` branch which includes `--antialiasing` support.

### Verify submodules

```bash
python -c "from simple_knn._C import distCUDA2; print('✅ simple-knn OK')"
python -c "from diff_gaussian_rasterization import _C; print('✅ diff-gaussian OK')"
python -c "from fused_ssim import fused_ssim; print('✅ fused-ssim OK')"
```

---

## 4. Install Dependencies

```bash
# Core pipeline dependencies
pip install -r src/requirements.txt

# Additional (optional but recommended)
pip install rich scipy plyfile
```

**`src/requirements.txt` contents:**
```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
plyfile>=1.0.0
tqdm>=4.66.0
opencv-python-headless>=4.8.0
scipy>=1.10.0
Pillow>=9.0.0
```

---

## 5. Verify Installation

```bash
# 1. Check Python can import pipeline modules
cd src
python -c "from config import DATA_DIR, GS_DIR, SCENES, VARIANTS; print(f'✅ {len(SCENES)} scenes, {len(VARIANTS)} variants')"
cd ..

# 2. Check 3DGS train.py is accessible
python -c "import sys; sys.path.insert(0, 'gaussian-splatting'); from arguments import ModelParams; print('✅ 3DGS OK')"

# 3. Dry-run the pipeline
python src/main.py --dry-run
# Expected output: prints plan without executing
```

---

## 6. Quick Test Run

Test on a single scene with the fast variant (7,000 iterations, ~5-10 min):

```bash
# Power
python src/main.py --scenes bonsai --variant fast

# Or step-by-step
python src/train.py --scene bonsai --variant fast
python src/render.py --scene bonsai --variant fast
```

**Expected outputs:**
```
src/output/
├── workspaces/bonsai/           # Prepared data
├── models/bonsai/fast/          # Trained checkpoint
│   └── point_cloud/iteration_7000/point_cloud.ply
└── renders/bonsai/fast/         # Rendered test images
    ├── 00000.png
    ├── 00001.png
    └── ...
```

If you see `[OK] bonsai/fast: DONE in X min` — everything works!

---

## 7. Full Pipeline

### 7.1 Single Scene — Full Training

```bash
# Train, render, evaluate, ensemble, post-process, package
python src/main.py --scenes bonsai --variant full_60k --compact --tta
```

### 7.2 All Scenes — Competition Submission

```bash
# Train ALL 10 variants on ALL 7 scenes
python src/main.py --all-variants --compact --tta

# Output: src/submissions/submission_round1.zip
```

### 7.3 Partial Pipeline (skip phases)

```bash
# Skip training (models already exist)
python src/main.py --skip-train

# Train only (don't render yet)
python src/main.py --train-only

# Evaluate metrics only
python src/main.py --eval-only

# Render only (after training completes)
python src/main.py --render-only
```

### 7.4 Individual Module CLI

```bash
# Train one variant
python src/train.py --scene HCM0421 --variant full_60k

# Render test poses
python src/render.py --scene HCM0421 --variant full_60k

# Evaluate competition metrics
python src/eval.py --scene HCM0421 --all-variants

# Gaussian-level merging
python src/compact.py --scene HCM0421

# Test-time adaptation
python src/tta.py --scene HCM0421 --model compact

# Smart ensemble
python src/ensemble.py --scene HCM0421

# Post-process
python src/postprocess.py --scene HCM0421

# Package submission
python src/package.py --scenes HCM0421 --source final
```

---

## 8. Pipeline Phases Reference

| Phase | Module | Description | Time (per scene) |
|-------|--------|-------------|------------------|
| 1. Validate | `main.py` | Check data & COLMAP | < 1 sec |
| 2. Train | `train.py` | Train 3DGS variant | 5 min (fast) – 3h (big) |
| 3. Render | `render.py` | Render test poses | 1–5 min |
| 3.2 Eval | `eval.py` | LPIPS/SSIM/PSNR + Score | 1–2 min |
| 3.5 Compact | `compact.py` | Gaussian-level merge | 2–5 min |
| 3.6 TTA | `tta.py` | Test-time adaptation | 5–10 min |
| 4. Ensemble | `ensemble.py` | Per-pixel blending | 1–3 min |
| 5. Post | `postprocess.py` | Sharpen + color match | 1–2 min |
| 6. Package | `package.py` | Create submission.zip | < 1 sec |

---

## 8. Troubleshooting

### CUDA out of memory

```bash
# Set data_device to "cpu" in config.py Variant defaults
# Already default for some variants

# Or reduce resolution
python src/train.py --scene bonsai --variant full_60k --iters 7000
# (manually edit variant.resolution = 2 for half-res)
```

### "No module named 'torch'"

```bash
# Make sure PyTorch with CUDA is installed
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

### "submodule not initialized"

```bash
cd gaussian-splatting
git submodule update --init --recursive
cd ..
```

### "SparseAdam not available"

```bash
# Re-install diff-gaussian-rasterization
cd gaussian-splatting
pip uninstall diff_gaussian_rasterization -y
pip install -e submodules/diff-gaussian-rasterization
cd ..
```

### Permission denied (Linux)

```bash
chmod +x src/run.ps1  # If using PowerShell on Linux
# Or run directly: python src/main.py ...
```

### Check GPU memory usage

```bash
watch -n 1 nvidia-smi
```

---

## 📂 Final Project Structure (after running)

```
De_1/
├── SETUP.md                        # ← This file
├── ARCHITECTURE.md                 # System design
├── BASELINE_AUDIT.md               # Full baseline audit
├── STRATEGY_TOP1.md                # Competition strategy
├── README.md                       # Project overview
├── de.md                           # Problem statement
│
├── data/                           # Scene data (7 scenes)
├── gaussian-splatting/             # 3DGS + submodules
├── src/                            # Pipeline source
├── pipeline/                       # Reference pipeline
│
├── src/output/                     # All outputs
│   ├── workspaces/                 # Prepared data
│   ├── models/<scene>/<variant>/   # Trained checkpoints
│   ├── renders/<scene>/<variant>/  # Rendered PNGs
│   ├── ensemble/<scene>/           # Blended PNGs
│   ├── final/<scene>/              # Post-processed PNGs
│   └── submissions/                # submission_round1.zip
│
└── ARCHITECTURE.md, BASELINE_AUDIT.md, ...
```

---

> 🏆 **Ready to train!** Start with `python src/main.py --scenes bonsai --variant fast` to verify everything works.
