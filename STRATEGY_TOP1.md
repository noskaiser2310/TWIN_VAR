# 🏆 Chiến Lược Top 1 — VAR 2026 Digital Twin BTS

> **Bài toán:** Novel View Synthesis — Digital Twin cho Trạm BTS  
> **Deadline:** 30/07/2026  
> **GPU:** RTX A4000 20GB (có thể dùng Kaggle T4 16GB cho training)  
> **Metric:** Score = 0.4×(1−LPIPS) + 0.3×SSIM + 0.3×PSNR_norm

---

## Mục Lục

1. [Phân Tích Hiện Trạng Pipeline](#1-phân-tích-hiện-trạng)
2. [Điểm Yếu & Cơ Hội Cải Thiện](#2-điểm-yếu--cơ-hội)
3. [Các Hướng Cải Tiến Cốt Lõi](#3-cải-tiến-cốt-lõi)
4. [Kiến Trúc Pipeline Mới (Top 1)](#4-kiến-trúc-pipeline-mới)
5. [Kế Hoạch Triển Khai 8 Ngày](#5-kế-hoạch-triển-khai)
6. [Chi Tiết Kỹ Thuật](#6-chi-tiết-kỹ-thuật)
7. [Appendix: Code Templates](#7-appendix)

---

## 1. Phân Tích Hiện Trạng Pipeline {#1-phân-tích-hiện-trạng}

### 1.1 Kiến trúc hiện tại

```
run_pipeline.py
  ├── Validate data (check COLMAP files)
  ├── Depth maps (Depth-Anything v2 ViT-L)
  ├── Upload Kaggle datasets
  ├── Build kernel bundles → Push Kaggle GPU
  ├── Train 6 variants per scene:
  │   ├── baseline      (30k iters, cpu data_device)
  │   ├── depth         (+ depth regularization)
  │   ├── exposure      (+ exposure compensation)
  │   ├── antialias     (+ EWA antialiasing)
  │   ├── full_combo    (depth + exposure + antialias + sparse_adam)
  │   └── fast          (7k iters, quick check)
  └── Package → fallback blend → submission.zip
```

### 1.2 Điểm mạnh

| Strengths | Chi tiết |
|-----------|----------|
| ✅ Auto pipeline hoàn chỉnh | Validate → Depth → Train → Render → Package |
| ✅ Multi-variant training | 6 variants, coverage tốt cơ bản |
| ✅ Kaggle GPU leverage | Tận dụng T4 16GB free |
| ✅ Depth-Anything v2 | Mono-depth regularization |
| ✅ Fallback blending | Dùng variant khác khi thiếu ảnh |
| ✅ Custom render script | Quaternion → rotation matrix, render tại test poses |

### 1.3 Điểm yếu & Cơ hội {#2-điểm-yếu--cơ-hội}

| # | Điểm yếu | Impact | Cơ hội cải thiện | Expected gain |
|---|----------|--------|-----------------|---------------|
| **1** | **Không có Mip-Splatting** — Vanilla 3DGS bị aliasing artifacts khi render ở resolution khác training | LPIPS ↑ 0.02-0.05 | Tích hợp EWA 3D filter + 2D Mip filter | **+0.03 Score** |
| **2** | **Không có Scaffold-GS** — Anchor-based giúp geometry tốt hơn nhiều | SSIM ↑ 0.01-0.02, PSNR ↑ 1-2dB | Thay thế một variant bằng Scaffold-GS | **+0.02-0.03 Score** |
| **3** | **Không có Sky/Large-scale handling** — Drone scenes có sky, unbounded | Floater artifacts ở sky region → LPIPS cao | Sky mask + unbounded scene regularization | **+0.01-0.02 Score** |
| **4** | **Depth regularization yếu** — Chỉ dùng depth trong 1 variant, chưa tối ưu | Geometry cue missing → blur trên texture-less surfaces | Depth-guided initialization + multi-scale depth loss | **+0.01-0.02 Score** |
| **5** | **Không có Appearance Modeling** — Mỗi scene có lighting variation giữa các ảnh drone | Color inconsistency → SSIM giảm | Appearance embeddings (AfME) per image | **+0.01 Score** |
| **6** | **Blending strategy đơn giản** — Chỉ fallback khi thiếu, không smart ensembling | Bỏ lỡ cơ hội chọn pixel tốt nhất từ mỗi model | Per-pixel confidence-based selection / uncertainty blending | **+0.02-0.03 Score** |
| **7** | **Không có post-processing** — Ảnh raw từ render | Thiếu sharpness, detail | Lightweight super-resolution / detail enhancement | **+0.01 Score** |
| **8** | **Không có test-time optimization** — Model cố định sau train | Không adapt được vào test poses cụ thể | Test-time pose-aware refinement | **+0.01-0.02 Score** |
| **9** | **Data device = cpu** — Chậm hơn, ít VRAM hơn cho Gaussians | Training chậm hơn → ít iterations hơn | `data_device=cuda` với memory optimization | **+0.01 Score** (more iters) |
| **10** | **30k iterations có thể chưa đủ** — Trên T4, 30k ~ 30-60 phút | Model chưa fully converged | Tăng lên 45k-60k với schedule tốt hơn | **+0.01-0.02 Score** |

**Tổng potential gain: +0.10 ~ +0.15 Score** — đủ để vượt từ middle pack lên top 1.

---

## 3. Các Hướng Cải Tiến Cốt Lõi {#3-cải-tiến-cốt-lõi}

### 3.1 TIER 1 — Must Have (Impact CAO, Effort THẤP) 🔥

#### A. Mip-Splatting Integration
```
Vanilla 3DGS:   Gaussian → Project → Alpha Blend (aliasing!)
Mip-Splatting:  Gaussian → 3D Smooth Filter → Project → 2D Mip Filter → Blend

Implementation:
  1. Thay thế gaussian_renderer bằng Mip-Splatting renderer
  2. Thêm 3D smoothing filter vào pre-processing Gaussians
  3. Thêm 2D Mip filter trong quá trình rendering
  4. Train với multiple resolution scales
  
Expected: PSNR +1-2dB, LPIPS -0.02
```

#### B. Smart Ensembling (Per-Pixel Selection)
```
Thay vì fallback đơn giản:
  for each pixel (x, y) in test image:
    for each variant model:
      render pixel → compute local confidence score
      confidence = combination of:
        - Depth consistency with neighbors
        - Alpha saturation (low alpha = uncertain)
        - Color consistency with nearby pixels
    select pixel from model with HIGHEST confidence

Code template: xem Appendix A
```

#### C. Sky Mask + Unbounded Regularization
```
Sky handling cho drone scenes:
  1. Dùng segmentation model (SAM2) detect sky regions trong training images
  2. Trong quá trình training: mask out sky loss (hoặc giảm weight)
  3. Add unbounded scene regularization: penalize Gaussians far from scene center
  4. Render background với solid color thay vì black

Expected: Loại bỏ floaters → LPIPS giảm rõ rệt
```

### 3.2 TIER 2 — Should Have (Impact TRUNG BÌNH, Effort TRUNG BÌNH)

#### D. Scaffold-GS / Octree-GS Variant
```
Scaffold-GS: dùng anchor points + view-dependent neural features
→ Geometry chính xác hơn nhiều so với vanilla 3DGS
→ Đặc biệt hiệu quả với texture-less surfaces (tường BTS, cột, kim loại)

Tích hợp như 1 variant bổ sung trong kernel training
```

#### E. Multi-Scale Depth Regularization
```
Cải thiện từ depth regularization hiện tại:
  1. Depth-guided initialization: dùng depth maps để init Gaussian positions
  2. Multi-scale depth loss: L1 + SSIM trên depth ở nhiều scale
  3. Normal consistency loss: đảm bảo surface normals smooth
  4. Edge-aware depth smoothing (bilateral filter trên depth)

Implementation: modify train.py để thêm depth loss terms
```

#### F. Appearance Embeddings (AfME)
```
Xử lý lighting variation giữa các drone ảnh:
  - Mỗi ảnh training → latent appearance vector
  - MLP: appearance_vector → per-Gaussian color offset
  - Train cùng với 3DGS, không cần data thêm
  
Implementation: thêm appearance MLP vào GaussianModel
```

### 3.3 TIER 3 — Nice to Have (Impact THẤP-VỪA, Effort CAO)

#### G. Test-Time Refinement
```
Sau khi train xong, fine-tune nhẹ trên test poses:
  1. Render test views từ trained model
  2. Dùng consistency loss giữa các test view lân cận
  3. Fine-tune 100-500 iterations
  4. Chỉ update Gaussian gần test viewpoints

→ Model adapt vào test views cụ thể
```

#### H. Post-Processing Enhancement
```
Lightweight enhancement trên rendered images:
  1. Edge-aware sharpening (unsharp mask)
  2. Color correction: match color distribution với training images
  3. Optional: lightweight super-resolution nếu test resolution > train

→ Cải thiện perceptual quality (LPIPS)
```

---

## 4. Kiến Trúc Pipeline Mới (Top 1) {#4-kiến-trúc-pipeline-mới}

```
┌──────────────────────────────────────────────────────────────┐
│                  VAR 2026 TOP 1 PIPELINE                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  DATA   │   │  DEPTH   │   │   SKY    │   │ APPEAR   │  │
│  │  VALID  │──▶│ ANYTHING │──▶│   MASK   │──▶│ EMBED    │  │
│  └─────────┘   │  V2-L    │   │  (SAM2)  │   │ PREP     │  │
│                └──────────┘   └──────────┘   └──────────┘  │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              MULTI-VARIANT TRAINING (Kaggle GPU)      │    │
│  │                                                       │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │    │
│  │  │ Vanilla  │ │   Mip-   │ │ Scaffold │ │ Depth-  │ │    │
│  │  │  3DGS    │ │ Splatting│ │   -GS    │ │ Guided  │ │    │
│  │  │ 60k iter │ │  60k     │ │  60k     │ │  60k    │ │    │
│  │  └──────────┘ └──────────┘ └──────────┘ └─────────┘ │    │
│  │                                                       │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │    │
│  │  │ Exposure │ │ Full     │ │ Test-Time│             │    │
│  │  │   +AA    │ │ Combo v2 │ │ Refined  │             │    │
│  │  │  60k     │ │  60-90k  │ │ +500iter │             │    │
│  │  └──────────┘ └──────────┘ └──────────┘             │    │
│  └──────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │            SMART ENSEMBLING ENGINE                     │    │
│  │                                                        │    │
│  │  Per-Pixel Confidence Selection:                       │    │
│  │  ├── Alpha saturation score                            │    │
│  │  ├── Depth consistency score                           │    │
│  │  ├── Color consistency score                           │    │
│  │  ├── Learned per-variant quality weight (per scene)    │    │
│  │  └── Soft voting với temperature                       │    │
│  └──────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              POST-PROCESSING                           │    │
│  │  ├── Edge-aware sharpening                             │    │
│  │  ├── Color distribution matching                       │    │
│  │  └── Sky region smoothing                              │    │
│  └──────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│                    submission.zip 🏆                           │
└──────────────────────────────────────────────────────────────┘
```

### 4.1 Variant Matrix (mới)

| # | Variant | Key Features | Expected PSNR | Training Time |
|---|---------|-------------|---------------|---------------|
| 1 | `vanilla_60k` | Baseline 3DGS, 60k iters | baseline | ~60 min |
| 2 | `mip_splatting` | Mip-Splatting renderer, anti-aliasing, multi-scale | +1.5 dB | ~75 min |
| 3 | `scaffold_gs` | Scaffold-GS anchor-based, view-adaptive | +1.0 dB | ~90 min |
| 4 | `depth_guided` | Depth-init + multi-scale depth loss + normal consistency | +0.8 dB | ~70 min |
| 5 | `exposure_aa` | Exposure compensation + EWA antialiasing | +0.5 dB | ~65 min |
| 6 | `full_combo_v2` | Mip + depth + exposure + scaffold components | +2.0 dB | ~120 min |
| 7 | `test_refined` | full_combo_v2 + 500 iter test-time refinement | +2.3 dB | ~130 min |

**Memory budget per variant (T4 16GB):**
- Mỗi variant cần ~8-12GB VRAM
- Với `data_device=cpu` → tiết kiệm ~2-4GB
- Có thể train tuần tự hoặc 2 variants song song nếu đủ VRAM

---

## 5. Kế Hoạch Triển Khai 8 Ngày {#5-kế-hoạch-triển-khai}

### Ngày 1-2: Setup & Quick Wins (TIER 1) 🔥

| Task | Người/Agent | Thời gian | Priority |
|------|-------------|-----------|----------|
| ✅ Pipeline hiện tại đã chạy được | Done | - | - |
| Tích hợp Mip-Splatting renderer vào 3DGS | Code | 4h | P0 |
| Tạo variant `mip_splatting` trong config | Code | 1h | P0 |
| Implement Smart Per-Pixel Ensembling | Code | 3h | P0 |
| Implement Sky Mask generation + training filter | Code | 2h | P0 |
| **Test nhanh trên scene bonsai** | Run | 2h | P0 |

**Milestone:** Baseline mới với Mip-Splatting + Smart Ensemble + Sky Mask chạy được

### Ngày 3-4: Depth & Geometry (TIER 1+2)

| Task | Người/Agent | Thời gian | Priority |
|------|-------------|-----------|----------|
| Depth-guided Gaussian initialization | Code | 2h | P1 |
| Multi-scale depth loss implementation | Code | 3h | P1 |
| Normal consistency loss | Code | 2h | P1 |
| Tạo variant `depth_guided` | Code | 1h | P1 |
| Appearance Embeddings (AfME) | Code | 3h | P1 |
| Tạo variant `exposure_aa` cải tiến | Code | 1h | P1 |
| **Test toàn bộ variants trên bonsai** | Run | 3h | P1 |

**Milestone:** 5 variants chạy được, smart ensemble hoạt động

### Ngày 5-6: Advanced Variants (TIER 2)

| Task | Người/Agent | Thời gian | Priority |
|------|-------------|-----------|----------|
| Scaffold-GS integration | Code | 4h | P2 |
| Tạo variant `scaffold_gs` | Code | 1h | P2 |
| Tạo variant `full_combo_v2` | Code | 1h | P2 |
| Test-time refinement implementation | Code | 2h | P2 |
| Post-processing pipeline | Code | 2h | P2 |
| **Test toàn bộ 7 variants trên bonsai + 1 scene BTS** | Run | 4h | P2 |

**Milestone:** Full 7 variants + test-time refinement + post-processing

### Ngày 7: Full Run & Fine-tuning

| Task | Người/Agent | Thời gian | Priority |
|------|-------------|-----------|----------|
| Chạy full pipeline trên TẤT CẢ scenes | Run | 8-12h | P0 |
| Per-scene quality analysis | Analysis | 2h | P0 |
| Điều chỉnh ensemble weights per scene | Code | 1h | P0 |
| **Re-run scenes có quality thấp** | Run | 4h | P1 |

**Milestone:** Full submission cho tất cả scenes

### Ngày 8: Polish & Submit

| Task | Người/Agent | Thời gian | Priority |
|------|-------------|-----------|----------|
| Validate submission (đúng format, đủ ảnh) | Validation | 1h | P0 |
| A/B test ensemble strategies trên validation set | Analysis | 2h | P1 |
| Tối ưu post-processing parameters | Tuning | 1h | P2 |
| **SUBMIT** | Submit | 0.5h | P0 |

---

## 6. Chi Tiết Kỹ Thuật {#6-chi-tiết-kỹ-thuật}

### 6.1 Mip-Splatting Integration

```python
# Thêm vào gaussian_renderer/__init__.py

def render_mip(viewpoint_camera, pc, pipe, bg_color, 
               scaling_modifier=1.0, override_color=None):
    """
    Mip-Splatting renderer with 3D + 2D filters.
    Based on: "Mip-Splatting: Alias-free 3D Gaussian Splatting" (CVPR 2024)
    """
    # 1. Compute 3D smoothing filter based on max screen-space sampling rate
    screen_size = viewpoint_camera.image_height * viewpoint_camera.image_width
    filter_3d_size = compute_3d_filter_size(pc, viewpoint_camera)
    
    # 2. Apply 3D smoothing to Gaussian covariances
    smoothed_scales = pc.get_scaling + filter_3d_size
    
    # 3. Project Gaussians
    screenspace_points = torch.zeros_like(pc.get_xyz, ...)
    
    # 4. Apply 2D Mip filter during rasterization
    # Tích hợp vào CUDA rasterizer hoặc sử dụng post-process
    rendered = rasterizer_with_mip_filter(
        means3D=pc.get_xyz,
        opacities=pc.get_opacity,
        scales=smoothed_scales,
        ...
    )
    
    return rendered
```

### 6.2 Smart Per-Pixel Ensembling

```python
def smart_ensemble(scene, variants, test_poses, base_output_dir):
    """Per-pixel smart selection from multiple 3DGS variants."""
    
    renders = {}
    for variant in variants:
        renders[variant] = load_renders(base_output_dir / variant)
    
    final_images = {}
    for pose in test_poses:
        img_name = pose['image_name']
        
        # Collect renders for this view
        candidates = {v: r[img_name] for v, r in renders.items()}
        
        # Build per-pixel confidence maps
        confidences = {}
        for variant, img in candidates.items():
            confidences[variant] = compute_confidence_map(
                img=img,
                variants=variants,
                renders=candidates,
            )
        
        # Select best pixel per location
        h, w = list(candidates.values())[0].shape[:2]
        final = np.zeros((h, w, 3))
        for y in range(h):
            for x in range(w):
                scores = {v: confidences[v][y, x] for v in variants}
                best_variant = max(scores, key=scores.get)
                final[y, x] = candidates[best_variant][y, x]
        
        final_images[img_name] = final
    
    return final_images


def compute_confidence_map(img, variant_name, variants, renders):
    """Compute per-pixel confidence based on multiple signals."""
    
    # Signal 1: Alpha saturation (high alpha = confident)
    alpha = compute_alpha_map(img)
    alpha_conf = alpha / alpha.max()
    
    # Signal 2: Depth consistency with other variants
    depth = compute_depth_map(img, variant_name)
    other_depths = [compute_depth_map(renders[v], v) for v in variants if v != variant_name]
    depth_std = np.std(other_depths, axis=0)
    depth_conf = 1.0 / (1.0 + depth_std)
    
    # Signal 3: Color consistency in local neighborhood
    color_std = compute_local_color_std(img, window=5)
    color_conf = 1.0 / (1.0 + color_std)
    
    # Signal 4: Edge detection (prefer sharp edges)
    edge = compute_edge_map(img)
    edge_conf = edge / edge.max()
    
    # Combine (weights can be tuned per scene)
    confidence = (
        0.3 * alpha_conf +
        0.3 * depth_conf +
        0.2 * color_conf +
        0.2 * edge_conf
    )
    
    return confidence
```

### 6.3 Sky Mask & Unbounded Regularization

```python
def generate_sky_mask(scene_images_dir, output_dir):
    """Generate sky masks using SAM2 for drone scene images."""
    from segment_anything import sam_model_registry, SamPredictor
    
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
    predictor = SamPredictor(sam)
    
    masks = {}
    for img_path in Path(scene_images_dir).glob("*.png"):
        image = cv2.imread(str(img_path))
        predictor.set_image(image)
        
        # Use top-center point as sky prompt
        h, w = image.shape[:2]
        sky_points = np.array([[w//2, 10], [w//4, 10], [3*w//4, 10]])
        sky_labels = np.array([1, 1, 1])
        
        mask, _, _ = predictor.predict(
            point_coords=sky_points,
            point_labels=sky_labels,
            multimask_output=False,
        )
        masks[img_path.stem] = mask[0]
    
    return masks


def unbounded_regularization_loss(gaussians, scene_center, scene_radius):
    """Penalize Gaussians far from the scene center."""
    positions = gaussians.get_xyz
    distances = torch.norm(positions - scene_center, dim=1)
    far_mask = distances > scene_radius * 1.5
    
    if far_mask.any():
        loss = (distances[far_mask] - scene_radius * 1.5).mean()
        return loss
    return torch.tensor(0.0, device=positions.device)
```

### 6.4 Post-Processing

```python
def post_process_rendered_image(img, train_images_stats=None):
    """Enhance rendered image quality."""
    
    # 1. Edge-aware sharpening
    img_sharp = cv2.addWeighted(
        img, 1.3,
        cv2.GaussianBlur(img, (0, 0), 3), -0.3,
        0
    )
    
    # 2. Color correction matching training distribution
    if train_images_stats is not None:
        for c in range(3):
            img_sharp[:, :, c] = (img_sharp[:, :, c] - img_sharp[:, :, c].mean())
            img_sharp[:, :, c] = (img_sharp[:, :, c] * 
                (train_images_stats['std'][c] / (img_sharp[:, :, c].std() + 1e-6)))
            img_sharp[:, :, c] = (img_sharp[:, :, c] + train_images_stats['mean'][c])
    
    # 3. Clip and convert
    img_sharp = np.clip(img_sharp, 0, 1)
    
    return img_sharp
```

### 6.5 Cấu hình Training Tối Ưu

```yaml
# Optimized training config
iterations: 60000  # Tăng từ 30000 lên 60000

# Learning rates
position_lr_init: 0.00016
position_lr_final: 0.0000016
position_lr_max_steps: 45000  # Đỉnh ở 75% training
scaling_lr: 0.005
rotation_lr: 0.001
opacity_lr: 0.05

# Densification
densification_interval: 100
opacity_reset_interval: 3000
densify_from_iter: 500
densify_until_iter: 45000  # Kéo dài densification
densify_grad_threshold: 0.0002

# Loss
lambda_dssim: 0.2
lambda_depth: 0.1  # Depth regularization weight
lambda_normal: 0.05  # Normal consistency weight

# Optimization
optimizer_type: sparse_adam  # 2.7x faster
data_device: cpu  # Tiết kiệm VRAM cho T4

# Multi-scale training (cho Mip-Splatting)
random_resolution: True
resolution_scale_range: [0.5, 2.0]
```

---

## 7. Appendix: Code Templates {#7-appendix}

### A. Per-pixel selection blending

```python
def blend_multi_variant(variant_renders, variant_confidences, test_pose):
    """
    variant_renders: {variant_name: {image_name: np.ndarray}}
    variant_confidences: {variant_name: np.ndarray} confidence map
    
    Returns blended image.
    """
    h, w = test_pose['height'], test_pose['width']
    img_name = test_pose['image_name']
    
    # Collect renders
    renders = {}
    confs = {}
    for v in variant_renders:
        if img_name in variant_renders[v]:
            renders[v] = variant_renders[v][img_name]
            confs[v] = variant_confidences[v]
    
    # Soft voting
    stacked_confs = np.stack(list(confs.values()), axis=-1)  # (H, W, N)
    weights = np.exp(stacked_confs * 2.0)  # Temperature scaling
    weights = weights / weights.sum(axis=-1, keepdims=True)
    
    # Blend
    blended = np.zeros((h, w, 3))
    for i, v in enumerate(renders):
        blended += renders[v] * weights[:, :, i:i+1]
    
    return blended
```

---

## Tổng Kết

| Chỉ số | Hiện tại | Sau cải tiến | Mục tiêu |
|--------|----------|-------------|----------|
| **Số variants** | 6 | 7+ | Cover nhiều strategy |
| **PSNR** | baseline | +1.5-2.0 dB | Top tier |
| **LPIPS** | baseline | -0.02-0.05 | Giảm perceptual artifacts |
| **SSIM** | baseline | +0.01-0.03 | Cấu trúc chính xác hơn |
| **Smart ensemble** | ❌ Fallback | ✅ Per-pixel selection | Chọn pixel tốt nhất |
| **Sky handling** | ❌ | ✅ SAM2 mask + regularization | Không floaters |
| **Post-processing** | ❌ | ✅ Sharpen + color correction | Ảnh sắc nét hơn |
| **Depth quality** | Basic | Multi-scale + normal consistency | Geometry chính xác |
| **Training iters** | 30k | 60k | Hội tụ tốt hơn |

> 🏆 **Target Score improvement: +0.10 to +0.15** — đủ để competitive cho top 1.
