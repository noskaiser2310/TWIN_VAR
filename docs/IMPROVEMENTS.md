# 📊 IMPROVEMENTS.md — Cải thiện so với Baseline Gaussian Splatting

> **Baseline:** `gaussian-splatting/` (Inria GraphDeco, SIGGRAPH 2023 — [repo](https://github.com/graphdeco-inria/gaussian-splatting))  
> **Chúng tôi:** `src/` v2.4.0 — Pipeline tự chủ cho VAR 2026  
> **Tổng:** **+8 module mới**, **+6 pipeline phases**, **+15 tính năng**, **25/28 params leveraged**

---

## 1. So sánh Tổng Quan

| Tiêu chí | Baseline (`gaussian-splatting/`) | Chúng tôi (`src/`) |
|----------|----------------------------------|---------------------|
| **Loại** | Thư viện research (train, render, metrics) | **Pipeline tự động hoàn chỉnh** |
| **Số module** | 4 scripts (train, render, metrics, full_eval) | **13 modules** |
| **Training** | 1 config duy nhất, CLI thủ công | **10 variants**, auto-orchestration |
| **Params cấu hình** | CLI args (người dùng tự gõ) | **25/28 params được lập trình**, per-scene tuning |
| **Phụ thuộc** | Cần chạy từng bước thủ công | **One-click: `python main.py`** |
| **Evaluation** | metrics.py (SSIM/PSNR/LPIPS) | **eval.py + VAR Score formula** |
| **Ensemble** | ❌ Không có | ✅ **5-signal per-pixel blending** |
| **Post-process** | ❌ Không có | ✅ **Sharpen + color match** |
| **Gaussian merge** | ❌ Không có | ✅ **Voxel-based compact merge** |
| **TTA** | ❌ Không có | ✅ **Delta layer adaptation** |
| **Per-scene tuning** | ❌ Không có | ✅ **3-tier COLMAP density-based** |
| **Checkpoint resume** | Có sẵn nhưng thủ công | ✅ **Tự động resume big từ full_60k** |
| **Shell safety** | os.system() trong full_eval.py | ✅ **List-based subprocess.run** |
| **Tự chủ** | Phụ thuộc thư mục ngoài | ✅ **Đã copy vào `src/_3dgs/`** |

---

## 2. Baseline Có Gì — Chúng Tôi Thêm Gì

### 2.1 Baseline (`gaussian-splatting/`) — 4 scripts

| Script | Chức năng gốc |
|--------|--------------|
| `train.py` | Train 3DGS từ COLMAP sparse |
| `render.py` | Render trained model ra ảnh |
| `metrics.py` | Tính SSIM, PSNR, LPIPS |
| `full_eval.py` | Pipeline thô: train → render → metrics (dùng `os.system()`) |

**Hạn chế của baseline:**
- Mỗi lần train 1 model, phải gõ CLI thủ công
- Không có ensemble, không có post-process
- Không có evaluation pipeline cho competition
- Không có cơ chế tự động thử nhiều config
- `full_eval.py` dùng `os.system()` (shell injection risk)

### 2.2 Chúng tôi (`src/`) — 13 modules

| Module | Thêm so với baseline | Chức năng |
|--------|---------------------|-----------|
| **`config.py`** | ✅ MỚI | 10 variants + 25/28 params + PER_SCENE_CONFIG |
| **`main.py`** | ✅ MỚI | One-click 9-phase orchestrator |
| **`train.py`** | ✅ NÂNG CẤP | Wrap baseline train.py với args_list an toàn + checkpoint resume |
| **`render.py`** | ✅ MỚI | Render test poses từ Quaternion CSV |
| **`eval.py`** | ✅ MỚI | LPIPS/SSIM/PSNR + VAR 2026 Score |
| **`ensemble.py`** | ✅ MỚI | 5-signal per-pixel blending |
| **`compact.py`** | ✅ MỚI | Gaussian-level voxel merging |
| **`tta.py`** | ✅ MỚI | Test-time adaptation |
| **`postprocess.py`** | ✅ MỚI | Edge-aware sharpen + color match |
| **`package.py`** | ✅ MỚI | Tạo submission.zip |
| `_3dgs/` | ✅ MỚI | Self-contained copy của baseline |

---

## 3. Chi Tiết Cải Thiện Theo Từng Khía Cạnh

### 3.1 Training — Từ 1 config thủ công → 10 variants tự động

| Khía cạnh | Baseline | Chúng tôi |
|-----------|----------|-----------|
| Số config | 1 (gõ CLI) | **10 variants** (fast, baseline, depth, exposure, antialias, white_bg, depth_expo, full, full_60k, big) |
| `sh_degree` | Default 3 | **4** outdoor, **3** indoor |
| `eval_mode` | Không dùng | **Bật cho 9/10 variants** |
| `white_bg` | Không dùng | **white_bg variant + indoor override** |
| `lambda_dssim` | Default 0.2 | **0.2–0.3** tùy scene |
| `percent_dense` | Default 0.01 | **0.01–0.035** (3.5x) |
| `densify_until_iter` | Default 15k | **15k–28k** |
| `densify_grad_threshold` | Default 0.0002 | **0.0001–0.0002** |
| `depth_l1_weight_*` | Default 1.0/0.01 | **1.0–2.0 / 0.05–0.12** |
| `checkpoint_iterations` | `[]` (không lưu) | **[30000, 60000]** cho full_60k |
| `start_checkpoint` | Có sẵn nhưng thủ công | **Tự động resume (big ← full_60k)** |
| Sparse Adam | Không dùng | **2.7x faster** |
| Anti-aliasing | Không dùng | **EWA filter (dr_aa branch)** |
| Exposure compensation | Không dùng | **Per-image affine transform** |
| Fused SSIM | Có sẵn (auto) | **Đã build submodule** |

### 3.2 Rendering — Từ render tập test → render test poses cuộc thi

| Khía cạnh | Baseline | Chúng tôi |
|-----------|----------|-----------|
| Input | COLMAP test cameras | **CSV Quaternion poses** (VAR format) |
| Quaternion→Rotation | Không cần (COLMAP) | **Tự động convert WXYZ→Rotation matrix** |
| Skip-if-exists | Không có | **Bỏ qua nếu đã render** |
| MiniCam bug | Không liên quan | **Đã fix `depth_reliable=False`** |

### 3.3 Evaluation — Từ metrics.py → Competition pipeline

| Khía cạnh | Baseline | Chúng tôi |
|-----------|----------|-----------|
| Metrics | SSIM, PSNR, LPIPS (AlexNet) | SSIM, PSNR, **LPIPS (VGG)** + **VAR Score** |
| Score formula | Không có | **0.4×(1-LPIPS) + 0.3×SSIM + 0.3×PSNR/40** |
| Ranking | Không có | **Bảng xếp hạng variants** |
| Automation | Chạy thủ công | **Tự động trong pipeline (Phase 3.2)** |

### 3.4 Ensemble + Post-process — HOÀN TOÀN MỚI

Baseline không có bất kỳ cơ chế ensemble hay post-process nào.

| Module | Kỹ thuật |
|--------|----------|
| **ensemble.py** | 5 tín hiệu: AlphaSat + DepthConsistency + ColorSmoothness + EdgeSharpness + VariantPrior. Softmax-weighted per-pixel. |
| **compact.py** | Voxel-based Gaussian merging từ N models → 1 compact model. Opacity pruning. SVD rotation averaging. |
| **tta.py** | Test-time delta layer (color + scale offset). Photometric consistency loss. 500-1000 iterations. |
| **postprocess.py** | Unsharp mask sharpening + color distribution matching với train data. |

### 3.5 Per-Scene Tuning — HOÀN TOÀN MỚI

Baseline dùng chung 1 config cho mọi scene. Chúng tôi phân tích COLMAP density:

| Scene | COLMAP | Tier | Strategy |
|-------|--------|------|----------|
| bonsai | 6.1 MB | Indoor | white_bg, density 0.01 |
| chair | 8.9 MB | Indoor+ | density 0.015 (phức tạp hơn bonsai) |
| HCM0421 | 16.5 MB | Sparse | density 0.035, depth 2.0 |
| HCM0674 | 15.2 MB | Sparse | density 0.035, depth 2.0 |
| HCM0539 | 22.1 MB | Dense | density 0.025, depth 1.8 |
| HCM0540 | 20.2 MB | Dense | density 0.025, depth 1.8 |
| HCM0644 | 21.4 MB | Dense | density 0.025, depth 1.8 |

### 3.6 Pipeline Orchestration — HOÀN TOÀN MỚI

| Baseline | Chúng tôi |
|----------|-----------|
| 3 bước thủ công: `python train.py ... && python render.py ... && python metrics.py ...` | **1 lệnh: `python main.py`** |
| | 9 phases: validate → train → render → eval → compact → tta → ensemble → post → package |
| | `--dry-run`, `--skip-*`, `--train-only`, `--render-only`, `--eval-only` |
| | Báo thời gian mỗi phase, error isolation |

### 3.7 An toàn & Tự chủ

| Khía cạnh | Baseline | Chúng tôi |
|-----------|----------|-----------|
| Shell injection | `os.system()` trong full_eval.py | **List-based `subprocess.run`** |
| Path safety | String concatenation | **`list[str]` args, không shell=True** |
| Tự chủ | Phụ thuộc `gaussian-splatting/` bên ngoài | **Đã copy toàn bộ vào `src/_3dgs/`** |
| Dry-run | Không có | **`--dry-run`** |

---

## 4. Kiến Trúc

### Baseline (`gaussian-splatting/`)

```
train.py -s <data> -m <output> [args...]     ← 1 config, gõ tay
render.py -m <output>                         ← render test set
metrics.py -m <output>                        ← tính metrics
full_eval.py                                  ← pipeline thô (os.system)
```

### Chúng tôi (`src/`)

```
python src/main.py  ← ONE CLICK

  Phase 1:   VALIDATE   → check data & COLMAP
  Phase 2:   TRAIN      → 10 variants × 25/28 params × per-scene tuning
  Phase 3:   RENDER     → Quaternion→Rotation test poses
  Phase 3.2: EVAL       → LPIPS/SSIM/PSNR + VAR 2026 Score    ← MỚI
  Phase 3.5: COMPACT    → Gaussian-level merge                  ← MỚI
  Phase 3.6: TTA        → Test-time adaptation                  ← MỚI
  Phase 4:   ENSEMBLE   → 5-signal per-pixel blending           ← MỚI
  Phase 5:   POST       → sharpen + color match                 ← MỚI
  Phase 6:   PACKAGE    → submission_round1.zip                 ← MỚI

Tất cả chạy local, không phụ thuộc Kaggle, tự chủ trong src/_3dgs/
```

---

## 5. Tổng Kết Số Liệu

| Metric | Baseline | Chúng tôi | Cải thiện |
|--------|----------|-----------|-----------|
| Scripts | 4 | 13 | **+9** |
| Pipeline phases | 3 (thủ công) | 9 (tự động) | **+6** |
| Training variants | 1 | 10 | **+9** |
| Params leveraged | 4/28 (14%) | 25/28 (89%) | **+75%** |
| Ensemble methods | 0 | 3 (pixel, Gaussian, TTA) | **+3** |
| Per-scene tiers | 0 | 3 | **+3** |
| Shell injection risk | Có | Không | **Fixed** |
| Tự chủ | Không | Có (`_3dgs/`) | **Self-contained** |

---

> **Tóm lại:** Từ baseline 4-script research của Inria, chúng tôi đã xây dựng **pipeline 9-phase tự động hoàn toàn**, tận dụng **89% tham số** của 3DGS (vs 14% mặc định), với **3 phương pháp ensemble**, **3-tier per-scene tuning**, **post-processing**, **competition metric evaluation**, và kiến trúc **tự chủ hoàn toàn** không phụ thuộc thư mục ngoài.

---

## 6. SOTA 2024-2026 — Hướng Cải Thiện Tiếp Theo

Baseline 3DGS từ **SIGGRAPH 2023**. Đến 2026, có nhiều cải tiến **plug-and-play** có thể tích hợp:

### 6.1 Đã tích hợp

| Kỹ thuật | Paper | Tác động |
|----------|-------|----------|
| Depth regularization | Depth-Anything V2 (Yang, NeurIPS 2024) | +0.5-1.0 dB PSNR |
| Anti-aliasing (EWA filter) | Mip-Splatting (Yu, CVPR 2024) | +1-2 dB PSNR, giảm aliasing |
| Exposure compensation | 3DGS gốc (tích hợp sẵn) | Xử lý drone lighting variation |
| Sparse Adam | Taming-3DGS (2024) | 2.7x faster |
| Fused SSIM | Fused-SSIM (Goel, 2023) | ~2x faster training |

### 6.2 Có thể tích hợp thêm

| Kỹ thuật | Paper | Tác động dự kiến | Độ khó |
|----------|-------|-----------------|--------|
| **AbsGS** | AbsGS (Ye, ECCV 2024) | +0.5-1.0 dB — densification chính xác hơn, ít floaters | ⭐ Dễ (đổi gradient metric) |
| **PixelGS** | PixelGS (Zhang, 2024) | +0.3-0.8 dB — pixel-aware density | ⭐⭐ Vừa (đổi densify logic) |
| **LightGaussian** | LightGaussian (Fan, 2024) | Giảm 3-5x Gaussians, giữ chất lượng | ⭐⭐ Vừa (pruning pipeline) |
| **EAGLES** | EAGLES (2025) | Tối ưu PSNR/parameter | ⭐⭐⭐ Khó (cần sửa rasterizer) |
| **Neural Texture Splatting** | NTS (2025) | SH không đủ → latent texture cho specular | ⭐⭐⭐ Khó (cần thay đổi GaussianModel) |
| **DUSt3R/MASt3R init** | DUSt3R (Wang, CVPR 2024) | Khởi tạo không cần COLMAP | ⭐⭐ Vừa (thay đổi init pipeline) |
| **Background separation** | Street-Gaussians (2024) | Tách sky/foreground, giảm floaters | ⭐⭐ Vừa (thêm BG model) |
| **Aggressive early pruning** | CVPR 2025 winners | Loại bỏ low-opacity Gaussians sớm hơn | ⭐ Dễ (đổi opacity_reset_interval) |

### 6.3 Chiến lược tích hợp

**Ưu tiên cao nhất (dễ + tác động lớn):**
1. **AbsGS densification** — sửa gradient metric trong `_3dgs/train.py`, không cần thay đổi rasterizer
2. **Aggressive early pruning** — giảm `opacity_reset_interval` từ 3000 → 1500, prune sớm hơn
3. **Multi-scale depth priors** — dùng cả Depth-Anything V2 large + base để có multi-resolution depth

**Ưu tiên trung bình:**
4. **LightGaussian pruning** — sau train, prune model trước khi render/compact
5. **Background separation** — thêm sky mask cho outdoor HCM scenes

**Cần nghiên cứu thêm:**
6. Neural Texture Splatting — thay SH bằng neural texture (cần sửa GaussianModel)
7. DUSt3R init — bỏ COLMAP, khởi tạo từ ảnh trực tiếp
