# 🏗️ ARCHITECTURE.md — VAR 2026 Digital Twin BTS Pipeline

> **Version:** 2.0.0  
> **Last updated:** 2026-07-22

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     src/main.py (Orchestrator)                    │
│                                                                  │
│  Phase 1   →  Phase 2   →  Phase 3   →  Phase 3.2  →  Phase 3.5 │
│  VALIDATE     TRAIN        RENDER       EVAL          COMPACT    │
│                                                                  │
│  Phase 3.6 →  Phase 4   →  Phase 5   →  Phase 6                  │
│  TTA          ENSEMBLE     POST         PACKAGE                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Map

| Module | Role | Wraps |
|--------|------|-------|
| `config.py` | All config: paths, 10 variants, ensemble, compact, TTA, eval | — |
| `main.py` | One-click orchestrator | All modules |
| `train.py` | Train 1 3DGS variant | `gaussian-splatting/train.py` |
| `render.py` | Render test poses | `gaussian-splatting/render.py` (via generated script) |
| `eval.py` | Compute LPIPS/SSIM/PSNR + competition score | `gaussian-splatting/metrics.py` |
| `compact.py` | Gaussian-level voxel-based merging | plyfile |
| `tta.py` | Test-time adaptation delta layer | PyTorch |
| `ensemble.py` | 5-signal per-pixel confidence blending | numpy, scipy, cv2 |
| `postprocess.py` | Edge-aware sharpen + color match | cv2, numpy |
| `package.py` | Create submission_round1.zip | zipfile |

---

## 3. Data Flow

```
data/<scene>/
  ├── train/images/          → train.py → OUTPUT_DIR/workspaces/<scene>/images/
  ├── train/sparse/0/        → train.py → OUTPUT_DIR/workspaces/<scene>/sparse/
  ├── depths/                → train.py → -d <depths>
  └── test/test_poses.csv    → render.py → Camera poses

OUTPUT_DIR/
  ├── workspaces/            # Prepared scene data
  ├── models/<scene>/        # Trained checkpoints
  │   ├── full_60k/point_cloud/iteration_60000/point_cloud.ply
  │   ├── compact/           # Merged model
  │   └── compact_tta/       # TTA-adapted model
  ├── renders/<scene>/       # Rendered PNGs per variant
  ├── ensemble/<scene>/      # Blended PNGs
  ├── final/<scene>/         # Post-processed PNGs
  └── submissions/           # submission_round1.zip
```

---

## 4. All 10 Training Variants

| # | Name | Iters | sh_deg | depth | exposure | antialias | sparse_adam | white_bg | eval | start_chkpnt |
|---|------|-------|--------|-------|----------|-----------|-------------|----------|------|--------------|
| 1 | `fast` | 7k | 4 | | | | | | | |
| 2 | `baseline` | 30k | 3 | | | | | | ✅ | |
| 3 | `depth` | 30k | 4 | ✅ | | | | | ✅ | |
| 4 | `exposure` | 30k | 4 | | ✅ | | | | ✅ | |
| 5 | `antialias` | 30k | 4 | | | ✅ | ✅ | | ✅ | |
| 6 | `white_bg` | 30k | 4 | | | | | ✅ | ✅ | |
| 7 | `depth_expo` | 30k | 4 | ✅ | ✅ | | ✅ | | ✅ | |
| 8 | `full` | 30k | 4 | ✅ | ✅ | ✅ | ✅ | | ✅ | |
| 9 | `full_60k` | 60k | 4 | ✅ | ✅ | ✅ | ✅ | | ✅ | |
| 10 | `big` | 90k | 4 | ✅ | ✅ | ✅ | ✅ | | ✅ | full_60k |

---

## 5. Baseline Feature Utilization

| Feature | gaussian-splatting param | Our config | Status |
|---------|-------------------------|------------|--------|
| SH degree | `--sh_degree` | 4 (drone outdoor) | ✅ Leveraged |
| Eval mode | `--eval` | All variants | ✅ Leveraged |
| Depth regularization | `-d`, `--depth_l1_weight_*` | depth variants | ✅ Leveraged |
| Exposure compensation | `--exposure_lr_*`, `--train_test_exp` | exposure variants | ✅ Leveraged |
| Anti-aliasing | `--antialiasing` | antialias variants | ✅ Leveraged |
| Sparse Adam | `--optimizer_type sparse_adam` | Combo variants | ✅ Leveraged |
| Random background | `--random_background` | Configurable | ✅ Available |
| White background | `--white_background` | white_bg variant | ✅ Leveraged |
| SSIM weight | `--lambda_dssim` | 0.3 (tuned) | ✅ Leveraged |
| Densification | `--densify_*`, `--percent_dense` | Tuned for drone | ✅ Leveraged |
| Checkpoint resume | `--start_checkpoint` | big variant | ✅ Leveraged |
| Fused SSIM | Auto-detected | Auto (if built) | ✅ Leveraged |
| Metrics evaluation | `metrics.py` | eval.py wraps it | ✅ Leveraged |

---

## 6. Competition Score Formula

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm

PSNR_norm = min(PSNR / 40.0, 1.0)
```

---

## 7. Ensemble Strategy: 5-Signal Per-Pixel Blending

For each pixel `(x, y)` across N variants:

```
confidence = 0.25 × AlphaSat
           + 0.30 × DepthConsistency
           + 0.20 × ColorSmoothness
           + 0.15 × EdgeSharpness
           + 0.10 × VariantPrior

weights = softmax(confidence × temperature)
pixel_final = Σ(variant_pixel × weight)
```

---

## 8. Post-Processing Pipeline

```
Rendered PNG → Edge-aware Unsharp Mask → Color Distribution Matching → Final PNG
```

- **Unsharp mask:** `sharpened = img + amount × (img − gaussian_blur(img, radius))`
- **Color matching:** Match mean/variance of each channel to training data distribution

---

## 9. Security & Robustness

- ✅ **No shell injection:** All subprocess calls use `list[str]` args (no `shell=True`)
- ✅ **Dry-run mode:** `--dry-run` prints plan without executing
- ✅ **Skip-if-exists:** render.py skips if PNGs already rendered
- ✅ **Fallback paths:** Multiple CSV/COLMAP location checks
- ✅ **Timeout:** Render scripts have 3600s timeout
- ✅ **Error isolation:** Per-variant failure doesn't kill pipeline
