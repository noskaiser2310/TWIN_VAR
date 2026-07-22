# 📋 CHANGELOG — VAR 2026 Digital Twin BTS Pipeline

---

## 🚀 v2.5.0 — AbsGS Densification (2026-07-22)

### New Features
- **AbsGS densification**: Sum of absolute gradients metric (vs L2 norm) for better density control
- Reference: Ye et al., "AbsGS: Recovering Fine Details in 3D Gaussian Splatting", ECCV 2024
- Expected: **+0.5–1.0 dB PSNR** improvement
- All 10 variants automatically use AbsGS (`densify_method="abs"` default)

### Technical
- `_3dgs/scene/gaussian_model.py`: `add_densification_stats()` uses `torch.abs().sum()`
- `_3dgs/arguments/__init__.py`: Added `densify_method="abs"` to OptimizationParams
- `src/config.py`: Added `Variant.densify_method`, `args_list()` passes `--densify_method`
- Backward compatible: all existing code works unchanged

---

## 🏗️ v2.4.0 — Self-Contained Architecture (2026-07-22)

### New Features
- **Self-contained**: All 3DGS code copied into `src/_3dgs/` — zero external dependency
- 26 Python files + 3 submodule source dirs in `src/_3dgs/`
- `config.py` `GS_DIR` now points to `src/_3dgs/`
- Original `gaussian-splatting/` kept as reference only

### Documentation
- `IMPROVEMENTS.md`: 15+ improvements vs baseline, module-by-module comparison
- Updated `ARCHITECTURE.md`: +self-contained section, +module map
- Updated `SETUP.md`: `cd src/_3dgs` for all build/troubleshoot commands

---

## 🎯 v2.3.0 — 3-Tier Per-Scene Tuning (2026-07-22)

### New Features
- **3-tier per-scene tuning** based on COLMAP `points3D.bin` density:
  - Indoor (bonsai 6.1MB, chair 8.9MB): white_bg, density 0.01
  - Outdoor Sparse (HCM0421 16.5MB, HCM0674 15.2MB): density 0.035, depth 2.0
  - Outdoor Dense (HCM0539 22.1MB, HCM0644 21.4MB, HCM0540 20.2MB): density 0.025, depth 1.8
- **Auto-detect unknown scenes**: Uses per-image COLMAP density metric
- `_AUTO_OUTDOOR_DEFAULTS`: Safer defaults for auto-detected scenes

---

## ⚙️ v2.2.0 — Per-Scene Tuning (2026-07-22)

### New Features
- `PER_SCENE_CONFIG` dict with indoor/outdoor defaults
- `get_scene_variant()`: deep-copy + override variant per scene
- `train.py` auto-applies per-scene overrides
- `_INDOOR_DEFAULTS`, `_OUTDOOR_BTS_DEFAULTS`

---

## 📊 v2.1.0 — Full Baseline Utilization (2026-07-22)

### New Features
- **25/28 baseline params leveraged** (was ~4/28 = 14%)
- 10 training variants: fast, baseline, depth, exposure, antialias, white_bg, depth_expo, full, full_60k, big
- `eval_mode` enabled for 9/10 variants
- `checkpoint_iterations` for full_60k → big resume
- `sparse_adam` on combo variants (2.7x faster)
- `antialiasing` (EWA filter, dr_aa branch)
- `exposure` compensation (per-image affine transform)
- List-based `args_list()` — no shell injection

### New Modules
- `eval.py`: LPIPS/SSIM/PSNR + VAR 2026 Score formula
- `compact.py`: Gaussian-level voxel merging
- `tta.py`: Test-time adaptation delta layer
- `ensemble.py`: 5-signal per-pixel blending
- `postprocess.py`: Edge-aware sharpen + color match
- `package.py`: Create submission.zip

### Documentation
- `ARCHITECTURE.md`: System design, data flow, 9 phases
- `BASELINE_AUDIT.md`: Complete 28-param audit with tuning guide
- `STRATEGY_TOP1.md`: Competition strategy

---

## 🔧 v2.0.0 — Complete Local Pipeline (2026-07-22)

### New Features
- **One-click orchestrator**: `python src/main.py` runs full pipeline
- 9 pipeline phases: validate → train → render → eval → compact → tta → ensemble → post → package
- Zero Kaggle dependency — fully local
- Modular: each phase is standalone CLI
- Dry-run mode, skip-if-exists, error isolation
- `SETUP.md`: A-Z setup guide
- `IMPROVEMENTS.md`: Baseline comparison

### Submodules
- `simple-knn` initialized: point cloud initialization
- `diff-gaussian-rasterization` initialized (dr_aa branch): CUDA rasterizer + anti-aliasing
- `fused-ssim` initialized: ~2x faster SSIM training

---

## 📈 Git History

```
a9889f8 feat: AbsGS densification — sum of absolute gradients (ECCV 2024)
8eaf02a feat: auto-detect unknown scenes + SOTA 2024-2026 roadmap
8f145d3 docs: fix IMPROVEMENTS.md — compare against gaussian-splatting/ baseline (Inria 3DGS)
1872dde v2.4.0: Self-contained architecture — src/_3dgs/ replaces gaussian-splatting/ dependency
f3e5857 docs: SETUP.md — complete A-Z setup guide
78119e3 v2.3.0: Refined 3-tier per-scene tuning — COLMAP density-based
ceb0aca build: simple-knn + diff-gaussian-rasterization submodules initialized
1ae080f v2.2.0: Per-scene tuning — indoor vs outdoor BTS parameter optimization
3d116b4 docs: fused-ssim submodule initialized
b4e6f43 v2.1.0: Full baseline utilization — all 28 gaussian-splatting params leveraged
937f0b8 v2.0.0: Complete Local Pipeline — Gaussian Splatting for Digital Twin BTS
```
