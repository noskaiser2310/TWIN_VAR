# 🏆 VAR 2026 — Digital Twin BTS: Top-1 Pipeline

> **Cuộc thi:** VAI Race AI 2026 — Đề 1  
> **Bài toán:** Novel View Synthesis — Digital Twin cho Trạm BTS  
> **Strategy:** [docs/STRATEGY_TOP1.md](docs/STRATEGY_TOP1.md)
> **Setup:** [docs/SETUP.md](docs/SETUP.md)

---

## 🚀 Quick Start

```powershell
# 1. Setup
pip install -r src/requirements.txt

# 2. Test nhanh (1 scene, 1 variant nhẹ)
python src/main.py --scenes bonsai --variant fast

# 3. Full pipeline với compact merge + TTA
python src/main.py --scenes bonsai --all-variants --compact --tta

# 4. Dry-run xem kế hoạch
python src/main.py --dry-run
```

---

## 📁 Structure

```
De_1/
├── README.md                     # This file
├── docs/                         # Documentation
│   ├── STRATEGY_TOP1.md
│   ├── IMPROVEMENTS.md
│   ├── ARCHITECTURE.md
│   ├── BASELINE_AUDIT.md
│   ├── SETUP.md
│   └── CHANGELOG.md
├── de.md                         # Competition problem statement
├── pyproject.toml                # Package metadata
├── .gitignore
│
├── data/                         # Scene data (gitignored)
│   ├── bonsai/
│   ├── chair/
│   └── ...
│
├── gaussian-splatting/           # Reference (optional)
│
├── src/                          # ★ Pipeline + 3DGS (self-contained)
│   ├── _3dgs/                    # 3DGS source (train.py, render.py, ...)
│   │   └── submodules/           # C++ extensions
│   ├── __init__.py
│   ├── config.py                 # Paths, 9 variants, ensemble, compact, TTA
│   ├── main.py                   # One-click orchestrator
│   ├── train.py                  # Train 1 variant locally
│   ├── render.py                 # Render test poses
│   ├── compact.py                # Gaussian-level primitive merging
│   ├── tta.py                    # Test-time adaptation
│   ├── ensemble.py               # Smart per-pixel blending
│   ├── postprocess.py            # Sharpen + color match
│   ├── package.py                # Create submission.zip
│   ├── run.ps1                   # PowerShell launcher
│   ├── requirements.txt
│   └── README.md
│
├── pipeline/                     # Original pipeline (reference)
│   └── config.py, run_pipeline.py, ...
│
└── output/                       # Outputs (gitignored)
    ├── models/                   # Trained 3DGS checkpoints
    ├── renders/                  # Rendered test views
    ├── ensemble/                 # Blended outputs
    ├── final/                    # Post-processed final images
    └── submissions/              # Submission ZIPs
```

---

## 🧠 Pipeline Architecture

```
python src/main.py
  │
  ├── Phase 1:   VALIDATE   → check data & COLMAP structure
  ├── Phase 2:   TRAIN      → train 9 3DGS variants
  ├── Phase 3:   RENDER     → render test poses
  ├── Phase 3.5: COMPACT    → Gaussian-level voxel merging
  ├── Phase 3.6: TTA        → Test-time adaptation
  ├── Phase 4:   ENSEMBLE   → 5-signal per-pixel blending
  ├── Phase 5:   POST       → edge sharpen + color match
  └── Phase 6:   PACKAGE    → submission_round1.zip
```

### 9 Training Variants

| # | Variant | Iters | Features |
|---|---------|-------|----------|
| 1 | `baseline` | 30k | Vanilla 3DGS |
| 2 | `depth` | 30k | Depth-guided |
| 3 | `exposure` | 30k | Exposure compensation |
| 4 | `antialias` | 30k | Anti-aliasing + sparse Adam |
| 5 | `depth_expo` | 30k | Depth + exposure + sparse Adam |
| 6 | `full` | 30k | All features combined |
| 7 | `full_60k` | 60k | All features, extended training |
| 8 | `big` | 90k | All features, max training |
| 9 | `fast` | 7k | Quick test variant |

### Ensemble: 5-Signal Per-Pixel Blending

```
confidence = 0.25·AlphaSat + 0.30·DepthConsistency + 0.20·ColorSmoothness
           + 0.15·EdgeSharpness + 0.10·VariantPrior

weights = softmax(confidence · temperature)
pixel = Σ variant_pixel · weight
```

### Score Formula

```
Score = 0.4 · (1 − LPIPS) + 0.3 · SSIM + 0.3 · PSNR_norm
```

---

## 📊 Design Philosophy

- ✅ **No Kaggle dependency** — runs fully local on RTX A4000 20GB
- ✅ **Modular** — each phase is a standalone script
- ✅ **One-click** — `python src/main.py` runs everything
- ✅ **Production-grade** — error handling, dry-run mode, per-scene isolation
- ✅ **Research-backed** — Mip-Splatting, depth regularization, compact merge, TTA

---

> 🏆 **Good luck!**
