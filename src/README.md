# VAR 2026 — Digital Twin BTS: Complete Local Pipeline

One-click orchestrator để xây dựng **Digital Twin cho Trạm BTS** bằng 3D Gaussian Splatting.

## Quick Start

```powershell
# 1. Chạy test nhanh trên 1 scene
python main.py --scenes bonsai --variant fast

# 2. Train + render + ensemble + package — full pipeline
python main.py --scenes bonsai --variant full_60k

# 3. Full pipeline with Gaussian-level compact merge + TTA
python main.py --scenes bonsai --all-variants --compact --tta

# 4. Full pipeline, tất cả scenes
python main.py

# 5. Dry-run (xem sẽ làm gì)
python main.py --dry-run
```

## Pipeline

```
main.py
  ├── Phase 1:   VALIDATE  → check data exists
  ├── Phase 2:   TRAIN     → train 3DGS variants
  ├── Phase 3:   RENDER    → render test poses
  ├── Phase 3.5: COMPACT   → Gaussian-level merging (voxel-based)
  ├── Phase 3.6: TTA       → Test-time adaptation delta layer
  ├── Phase 4:   ENSEMBLE  → smart pixel blending (5-signal)
  ├── Phase 5:   POST      → sharpen + color match
  └── Phase 6:   PACKAGE   → submission.zip
```

## Cấu Trúc

```
src/
├── config.py         # All config: paths, 9 variants, ensemble, compact, TTA
├── train.py          # Train 1 3DGS variant locally
├── render.py         # Render test poses (Quaternion→Rotation conversion)
├── compact.py        # Gaussian-level primitive merging (voxel-based)
├── tta.py            # Test-time adaptation delta layer
├── ensemble.py       # Smart per-pixel confidence blending (5 signals)
├── postprocess.py    # Edge-aware sharpen + color distribution matching
├── package.py        # Create submission_round1.zip
├── main.py           # One-click orchestrator
├── run.ps1           # PowerShell launcher
├── requirements.txt
└── README.md
```

## Yêu Cầu

- Python 3.10+
- CUDA GPU (16GB+ VRAM recommended)
- Gaussian Splatting submodules đã build
- Dữ liệu scenes trong `../data/`

## Các Lệnh CLI

```powershell
# Train
python train.py --scene bonsai --variant full_60k
python train.py --scene bonsai --variant fast --iters 7000

# Render
python render.py --scene bonsai --variant full_60k

# Compact: Gaussian-level merge from multiple trained models
python compact.py --scene bonsai
python compact.py --scene bonsai --variants full_60k,depth_expo,antialias --voxel-size 0.003

# TTA: Test-time adaptation delta layer
python tta.py --scene bonsai --model compact
python tta.py --scene bonsai --model full_60k --iters 2000

# Ensemble (cần >=2 variants đã render)
python ensemble.py --scene bonsai

# Post-process
python postprocess.py --scene bonsai

# Package
python package.py --scenes bonsai chair --source final
```
