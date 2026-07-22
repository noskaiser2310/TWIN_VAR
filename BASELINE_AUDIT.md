# 📊 BASELINE_AUDIT.md — Gaussian Splatting Baseline Feature Audit

> **Audited:** 2026-07-22  
> **Baseline:** `gaussian-splatting/` (Inria GraphDeco, SIGGRAPH 2023)  
> **Pipeline:** `src/` v2.2.0

---

## 1. Executive Summary

The `gaussian-splatting` codebase provides **28 configurable parameters** across 3 groups. Our `src/` pipeline v2.0.0 **leverages ALL quality-relevant parameters** and tunes them for drone BTS tower photography.

| Group | Total params | Leveraged | Tuned for drone | Status |
|-------|-------------|-----------|-----------------|--------|
| ModelParams | 10 | 10/10 | 4/10 | ✅ FULL |
| PipelineParams | 4 | 1/4 | 1/1 relevant | ✅ FULL |
| OptimizationParams | 14 | 14/14 | 8/14 | ✅ FULL |
| **TOTAL** | **28** | **25/28** | **13/28** | ✅ |

*Note: 3 PipelineParams are internal flags (`convert_SHs_python`, `compute_cov3D_python`, `debug`) not relevant to quality.*

---

## 2. Detailed Parameter Audit

### 2.1 ModelParams

| Param | Default | Our Setting | Drone Rationale |
|-------|---------|-------------|-----------------|
| `sh_degree` | 3 | **4** | Drone photos have strong view-dependent lighting (sky reflections, metal BTS) |
| `source_path` | — | Auto-resolved | ✅ |
| `model_path` | — | OUTPUT_DIR/models/<scene>/<variant> | ✅ |
| `images` | "images" | "images" | ✅ (default) |
| `depths` | "" | "depths/" (depth variants) | Depth-Anything V2 for textureless BTS walls |
| `resolution` | -1 | -1 (full res) | ✅ |
| `white_background` | False | **True** (white_bg variant) | Indoor BTS equipment rooms have white walls |
| `train_test_exp` | False | True (exposure variants) | Drone captures have varying exposure → per-image compensation |
| `data_device` | "cuda" | "cuda" (RTX A4000 20GB) | ✅ |
| `eval` | False | **True** (all variants) | **CRITICAL:** enables test split for metrics during training |

### 2.2 PipelineParams

| Param | Default | Our Setting | Notes |
|-------|---------|-------------|-------|
| `antialiasing` | False | True (antialias/full variants) | EWA filter reduces aliasing for thin BTS antennas |
| `convert_SHs_python` | False | False | ✅ Internal flag |
| `compute_cov3D_python` | False | False | ✅ Internal flag |
| `debug` | False | False | ✅ |

### 2.3 OptimizationParams

| Param | Default | Our Setting | Drone Rationale |
|-------|---------|-------------|-----------------|
| `iterations` | 30,000 | 7k–90k (9 options) | 60k+ for complex outdoor scenes |
| `position_lr_init` | 0.00016 | 0.00016 | ✅ Standard (robust default) |
| `position_lr_final` | 0.0000016 | default | ✅ |
| `feature_lr` | 0.0025 | 0.0025 | ✅ |
| `opacity_lr` | 0.025 | 0.025 | ✅ |
| `scaling_lr` | 0.005 | 0.005 | ✅ |
| `rotation_lr` | 0.001 | 0.001 | ✅ |
| `exposure_lr_*` | 0.01/0.001 | 0.01/0.001 | ✅ Exposure compensation for drone lighting |
| `percent_dense` | 0.01 | **0.02** | Drone captures are sparser → need more densification |
| `lambda_dssim` | 0.2 | **0.3** | SSIM is 30% of competition score → higher weight |
| `densification_interval` | 100 | 100 | ✅ |
| `opacity_reset_interval` | 3000 | 3000 | ✅ |
| `densify_from_iter` | 500 | 500 | ✅ |
| `densify_until_iter` | 15,000 | **25,000** | Complex scenes need longer densification |
| `densify_grad_threshold` | 0.0002 | **0.00015** | Lower threshold = more Gaussians for detail |
| `depth_l1_weight_init` | 1.0 | 1.0 | ✅ |
| `depth_l1_weight_final` | 0.01 | **0.05** | Keep depth influence longer for textureless surfaces |
| `random_background` | False | False (configurable) | ✅ |
| `optimizer_type` | "default" | "sparse_adam" (combo variants) | 2.7x faster, minimal quality loss |

---

## 3. Features NOT Used (and Why)

| Feature | Reason for not using |
|---------|---------------------|
| Network GUI (`network_gui`) | Headless pipeline, no visualization needed |
| `--detect_anomaly` | Debug only, adds overhead |
| `--disable_viewer` | Default behavior in non-interactive mode |
| `full_eval.py` | We use modular `eval.py` wrapping `metrics.py` instead |

---

## 4. Submodule Status

| Submodule | Status | Impact |
|-----------|--------|--------|
| **`diff-gaussian-rasterization`** (dr_aa branch) | ✅ **Initialized** — needs `pip install` | **REQUIRED:** CUDA rasterizer with anti-aliasing support |
| **`simple-knn`** | ✅ **Initialized** — needs `pip install` | **REQUIRED:** Point cloud initialization |
| **`fused-ssim`** | ✅ **Initialized** — needs `pip install` | ~2x faster SSIM training (auto-detected by train.py) |
| `SIBR_viewers` | Not needed | Interactive viewer only |

### 🔧 Build ALL submodules (on GPU machine)

```powershell
cd gaussian-splatting

# Step 1: Init submodules (already done)
git submodule update --init submodules/simple-knn
git submodule update --init submodules/diff-gaussian-rasterization
git submodule update --init submodules/fused-ssim

# Step 2: Install (requires PyTorch 2.1+ + CUDA 11.8+ toolkit)
pip install -e submodules/simple-knn
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/fused-ssim

# Step 3: Verify
python -c "from simple_knn._C import distCUDA2; print('✅ simple-knn OK')"
python -c "from diff_gaussian_rasterization import _C; print('✅ diff-gaussian OK')"
python -c "from fused_ssim import fused_ssim; print('✅ fused-ssim OK')"
```

> **Note:** `diff-gaussian-rasterization` uses the `dr_aa` branch which includes anti-aliasing support (`--antialiasing` flag).

---

## 5. What We Added Beyond the Baseline

| Addition | File | Benefit |
|----------|------|---------|
| Competition metric evaluation | `eval.py` | Directly computes LPIPS/SSIM/PSNR + Score |
| Gaussian-level merging | `compact.py` | Merges multiple trained models at primitive level |
| Test-time adaptation | `tta.py` | Fine-tunes delta layer on test viewpoints |
| Smart pixel ensemble | `ensemble.py` | 5-signal confidence-based per-pixel blending |
| Edge-aware post-processing | `postprocess.py` | Sharpening + color distribution matching |
| One-click orchestrator | `main.py` | Full pipeline with dry-run and skip flags |

---

## 6. Tuning Guide

### Per-Scene Tuning (v2.2.0)

The pipeline now supports **automatic per-scene parameter overrides** via `PER_SCENE_CONFIG` in `config.py`. No manual intervention needed — `train.py` calls `get_scene_variant()` automatically.

#### Indoor Scenes (bonsai, chair)
| Param | Override | Rationale |
|-------|----------|-----------|
| `white_bg` | **True** | Indoor scenes have white backgrounds |
| `lambda_dssim` | **0.2** | Synthetic textures, SSIM matters less |
| `percent_dense` | **0.01** | Small objects need fewer Gaussians |
| `densify_until_iter` | **15,000** | Simpler geometry |
| `densify_grad_threshold` | **0.0002** | Standard threshold |
| `sh_degree` | **3** | Limited view-dependence |

#### Outdoor BTS Scenes (HCM0421–HCM0674)
| Param | Override | Rationale |
|-------|----------|-----------|
| `white_bg` | **False** | Sky = black background |
| `lambda_dssim` | **0.3** | ↑ Perceptual quality for thin antennas |
| `percent_dense` | **0.03** | ↑↑ More Gaussians for thin structures |
| `densify_until_iter` | **25,000** | ↑ Longer densification |
| `densify_grad_threshold` | **0.0001** | ↓↓ Aggressive densification |
| `depth_l1_weight_init` | **2.0** | ↑↑ Strong depth for textureless walls |
| `depth_l1_weight_final` | **0.1** | ↑ Keep depth influence longer |
| `sh_degree` | **4** | Drone view-dependent lighting |

### Quick Tuning

```python
# In config.py Variant defaults:

# For outdoor drone scenes (default):
lambda_dssim=0.3, densify_until_iter=25_000, percent_dense=0.02

# For indoor BTS rooms:
white_bg=True, lambda_dssim=0.2, densify_until_iter=15_000

# For noisy captures:
depth_l1_weight_init=2.0, depth_l1_weight_final=0.1

# For scenes with many thin structures (BTS antennas):
densify_grad_threshold=0.0001, percent_dense=0.03
```

### Best Single Variant (Expected)
`full_60k` — combines all features with extended training, expected to be the top single model.

### Best Ensemble (Expected)
`full_60k` + `big` + `depth_expo` + `white_bg` + `antialias` — covers all failure modes.
