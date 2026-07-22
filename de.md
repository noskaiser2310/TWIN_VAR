# VAR 2026 — Digital Twin cho Trạm BTS

> **Cuộc thi:** VAI Race AI 2026  
> **Bài toán:** Novel View Synthesis — Digital Twin cho hạ tầng viễn thông  
> **Baseline tham khảo:** [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

---

## Mục lục

1. [Tổng quan bài toán](#1-tổng-quan-bài-toán)
2. [Cấu trúc dữ liệu](#2-cấu-trúc-dữ-liệu)
3. [Thông tin dữ liệu](#3-thông-tin-dữ-liệu)
4. [Format test_poses.csv](#4-format-test_posescsv)
5. [Đầu vào bài toán](#5-đầu-vào-bài-toán)
6. [Đầu ra bài toán](#6-đầu-ra-bài-toán)
7. [Format submission](#7-format-submission)
8. [Metrics đánh giá](#8-metrics-đánh-giá)
9. [Hình thức thi](#9-hình-thức-thi)
10. [Quy định chống gian lận](#10-quy-định-chống-gian-lận-và-đảm-bảo-tính-công-bằng)
11. [Vòng 1 — Chi tiết](#11-vòng-1--chi-tiết)
12. [Baseline tham khảo](#12-baseline-thí-sinh-có-thể-tham-khảo)

---

## 1. Tổng quan bài toán

### Mục tiêu

Xây dựng hệ thống AI có khả năng:
1. **Tái dựng cấu trúc 3D ngầm định** của trạm BTS từ tập ảnh drone
2. **Sinh ảnh RGB** tại các góc nhìn chưa từng được chụp (novel view synthesis)

Đây là hướng tiếp cận hiện đại cho việc xây dựng **Digital Twin** — bản sao số 3D có độ chính xác cao của hạ tầng viễn thông — phục vụ:
- Giám sát
- Kiểm tra
- Bảo trì
- Quy hoạch lắp đặt thiết bị

### Quy mô mỗi scene

| Hạng mục | Số lượng |
|----------|----------|
| Ảnh training | 100 – 300 ảnh RGB |
| Góc nhìn mục tiêu (test poses) | 20 – 50 |

### Nguồn dữ liệu thu thập

- **Drone** bay quanh đối tượng
- **Camera cầm tay** (hand-held camera)

### Đối tượng trong scene

- Trạm BTS
- Công trình hạ tầng
- Các đối tượng thực tế khác

### Lĩnh vực liên quan

| Lĩnh vực | Mô tả |
|----------|-------|
| Computer Vision | Thị giác máy tính |
| 3D Vision | Thị giác 3D |
| Neural Rendering | Kết xuất hình ảnh bằng mạng neural |
| Novel View Synthesis | Sinh ảnh tại góc nhìn mới |
| Digital Twin | Bản sao số |

---

## 2. Cấu trúc dữ liệu

Mỗi scene dữ liệu có cấu trúc như sau:

```
<scene_name>/
├── train/
│   ├── images/              # Ảnh training (~80% tổng số ảnh)
│   └── sparse/0/            # Sparse reconstruction từ COLMAP
│       ├── cameras.bin      # Camera intrinsics
│       ├── images.bin       # Camera poses của ảnh training
│       └── points3D.bin     # Point cloud 3D
└── test/
    └── test_poses.csv       # Camera poses cho ảnh test (~20%)
```

### Mô tả các file

| File | Nội dung |
|------|----------|
| `images/` | Ảnh RGB training |
| `cameras.bin` | Camera intrinsics (focal length, principal point, distortion, v.v.) |
| `images.bin` | Camera poses cho từng ảnh training (rotation + translation) |
| `points3D.bin` | Point cloud 3D từ COLMAP sparse reconstruction |
| `test_poses.csv` | Danh sách camera poses mục tiêu cần sinh ảnh |

---

## 3. Thông tin dữ liệu

| Hạng mục | Giá trị |
|----------|--------|
| Train / Test split | ~80% train, ~20% test |
| Camera poses | Đã được dựng sẵn bằng COLMAP |
| Sparse reconstruction | Đã được cung cấp (cameras.bin, images.bin, points3D.bin) |

---

## 4. Format test_poses.csv

```csv
image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
```

### Giải thích các cột

| Cột | Ý nghĩa |
|-----|---------|
| `image_name` | Tên file ảnh đầu ra cần sinh (vd: `0001.png`) |
| `qw, qx, qy, qz` | **Quaternion rotation** theo format COLMAP (WXYZ) |
| `tx, ty, tz` | **Camera translation** (tọa độ world) |
| `fx, fy` | **Focal length** (pixel) |
| `cx, cy` | **Principal point** (pixel) |
| `width, height` | Kích thước ảnh cần sinh (pixel) |

---

## 5. Đầu vào bài toán

Thí sinh được cung cấp:

1. ✅ Tập ảnh train đa góc nhìn
2. ✅ Camera intrinsics (từ `cameras.bin`)
3. ✅ Camera poses (từ `images.bin`)
4. ✅ Sparse reconstruction từ COLMAP (`points3D.bin`)
5. ✅ Danh sách test poses (`test_poses.csv`)

---

## 6. Đầu ra bài toán

Thí sinh cần sinh **ảnh RGB** tương ứng với **toàn bộ test poses** được cung cấp.

### Yêu cầu chất lượng ảnh đầu ra

| Tiêu chí | Mô tả |
|----------|-------|
| Hình học | Đúng cấu trúc không gian 3D |
| Vị trí vật thể | Đúng vị trí các thiết bị, cấu kiện |
| Chất lượng | Hình ảnh chân thực, nhất quán giữa các góc nhìn |

---

## 7. Format submission

Submission là **file ZIP** chứa toàn bộ ảnh kết quả:

```
submission.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
└── ...
```

### Yêu cầu bắt buộc

| # | Yêu cầu |
|---|---------|
| 1 | **Đúng số lượng và tên scene** — khớp với dữ liệu test |
| 2 | **Đúng tên file ảnh** — theo cột `image_name` trong `test_poses.csv` |
| 3 | **Đúng kích thước ảnh** — theo `width`, `height` trong `test_poses.csv` |
| 4 | **Đúng số lượng ảnh mỗi scene** — không thiếu, không thừa |

> ⚠️ **Lưu ý:** Thiếu ảnh tại bất kỳ pose nào của bất kỳ scene nào sẽ ảnh hưởng đến kết quả.
> Nếu thiếu scene hoặc thừa scene so với groundtruth, **kết quả sẽ không được tính**.

---

## 8. Metrics đánh giá

Kết quả được đánh giá bằng cách so sánh ảnh sinh ra với ảnh ground-truth bằng **3 metrics**.

### 8.1 LPIPS — Learned Perceptual Image Patch Similarity

| Thuộc tính | Giá trị |
|------------|--------|
| Đo lường | Độ tương đồng cảm quan (perceptual similarity) |
| Cơ chế | So sánh đặc trưng deep learning giữa 2 ảnh |
| Chiều | **Càng thấp càng tốt** ↓ |
| Khoảng giá trị | [0, 1] |

> **Tham khảo:** Richard Zhang et al. *"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric."* CVPR 2018.  
> 🔗 [arXiv:1801.03924](https://arxiv.org/abs/1801.03924)

### 8.2 SSIM — Structural Similarity Index Measure

| Thuộc tính | Giá trị |
|------------|--------|
| Đo lường | Độ tương đồng về cấu trúc hình ảnh |
| Cơ chế | So sánh luminance, contrast, structure |
| Chiều | **Càng cao càng tốt** ↑ |
| Khoảng giá trị | [0, 1] (có thể âm trong một số trường hợp) |

> **Tham khảo:** Zhou Wang et al. *"Image quality assessment: from error visibility to structural similarity."* IEEE TIP, 2004.  
> 🔗 [doi:10.1109/TIP.2003.819861](https://doi.org/10.1109/TIP.2003.819861)

### 8.3 PSNR — Peak Signal-to-Noise Ratio

| Thuộc tính | Giá trị |
|------------|--------|
| Đo lường | Sai số mức pixel |
| Cơ chế | So sánh trực tiếp pixel giữa ảnh dự đoán và ground-truth |
| Chiều | **Càng cao càng tốt** ↑ |
| Đơn vị | dB (decibel) |

**Chuẩn hóa PSNR về [0, 1]:**

```
psnr_norm = torch.clamp(psnr_val / psnr_max, 0.0, 1.0)
```

Trong đó:
- `PSNR_max` là ngưỡng PSNR tối đa được lựa chọn trước
- `clamp` giới hạn giá trị trong khoảng [0, 1]

### 8.4 Công thức tính điểm cuối cùng

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
```

| Thành phần | Trọng số | Chiều |
|------------|----------|-------|
| (1 − LPIPS) | **40%** | ↑ Cao hơn = tốt hơn |
| SSIM | **30%** | ↑ Cao hơn = tốt hơn |
| PSNR_norm | **30%** | ↑ Cao hơn = tốt hơn |

> 📊 **Điểm trên bảng xếp hạng** là điểm **trung bình của toàn bộ các scene**.

---

## 9. Hình thức thi

- Dữ liệu và scene **hoàn toàn mới** được cung cấp cho mỗi vòng thi
- Cách thức tính điểm được **giữ nguyên** giữa các vòng

---

## 10. Quy định chống gian lận và đảm bảo tính công bằng

### 10.1 Cấm sử dụng dữ liệu ngoài

Thí sinh **chỉ được phép** sử dụng dữ liệu do Ban Tổ Chức cung cấp trong từng vòng thi.

**Nghiêm cấm:**

| Hành vi |
|---------|
| Sử dụng ảnh, video hoặc dữ liệu 3D bên ngoài có chứa cùng đối tượng hoặc cùng scene |
| Thu thập bổ sung dữ liệu thực địa hoặc từ Internet liên quan trực tiếp đến các scene |
| Sử dụng bất kỳ nguồn dữ liệu nào nhằm tái tạo hoặc suy luận ground-truth của tập test |

### 10.2 Cấm truy xuất hoặc suy đoán dữ liệu kiểm thử

**Nghiêm cấm:**

| Hành vi |
|---------|
| Truy cập trái phép vào dữ liệu ground-truth |
| Khai thác lỗ hổng hệ thống để thu thập thông tin về ảnh kiểm thử |

### 10.3 Yêu cầu khả năng tái lập kết quả

Ban Tổ Chức có quyền yêu cầu các đội đạt thứ hạng cao cung cấp:

| Tài liệu |
|----------|
| Mã nguồn huấn luyện và suy luận |
| File cấu hình (config) |
| Danh sách thư viện và phiên bản sử dụng |
| Checkpoint mô hình |
| Nhật ký huấn luyện (training logs) |

> ⚠️ Đội thi phải chứng minh rằng kết quả nộp bài có thể được **tái tạo** từ pipeline đã công bố.

### 10.4 Cấm chỉnh sửa thủ công ảnh đầu ra

Toàn bộ ảnh kết quả phải được sinh **tự động** bởi thuật toán hoặc mô hình AI.

**Nghiêm cấm:**

| Hành vi |
|---------|
| Chỉnh sửa thủ công từng ảnh bằng phần mềm đồ họa |
| Ghép ảnh, vẽ thêm hoặc xóa vật thể bằng thao tác thủ công |
| Can thiệp thủ công vào từng test pose |

> ⚠️ Ban Tổ Chức có quyền yêu cầu chứng minh quy trình sinh ảnh **hoàn toàn tự động**.

---

## 11. Vòng 1 — Chi tiết

### 11.1 Mô tả vòng thi

Đây là **vòng thi đầu tiên** của bài thi VAR 2026 - Digital Twin cho trạm BTS.

Ban tổ chức công bố **public set** và **private test #1** gồm các scenes khác nhau.

**Quy trình vòng 1:**

1. Thí sinh xây dựng pipeline và đánh giá trên **tập public set**
2. Sau khi công bố **tập private test #1**, thí sinh sử dụng các ảnh training của mỗi scene
3. Thực hiện sinh ảnh RGB tại các pose mục tiêu được yêu cầu trong file `test_pose.csv`

### 11.2 Dữ liệu vòng 1

| Hạng mục | Thông tin |
|----------|-----------|
| Số ảnh / scene | 150 – 300 ảnh RGB |
| Số poses mục tiêu / scene | 40 – 70 |
| Dung lượng | 200 – 300 MB |
| Cấu trúc | Giống mô tả trong [mục 2](#2-cấu-trúc-dữ-liệu) |

### 11.3 Yêu cầu submission vòng 1

Thí sinh nộp một file nén chứa toàn bộ ảnh sinh:

```
submission_round1.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   ├── 0001.png
│   └── ...
└── ...
```

**Yêu cầu:**

| # | Yêu cầu |
|---|---------|
| 1 | Kích thước ảnh đúng theo `width`, `height` trong `test_pose.csv` |
| 2 | Tên file theo `image_name` trong `test_pose.csv` |
| 3 | Đầy đủ: thiếu ảnh tại bất kỳ pose nào của bất kỳ scene nào sẽ ảnh hưởng đến kết quả |

### 11.4 Timeline vòng 1

| Mốc thời gian | Sự kiện |
|---------------|---------|
| **02/07/2026** | Công bố private test #1 — thí sinh tải dữ liệu |
| **30/07/2026** | ⛔ Deadline submission |

> ℹ️ Thí sinh có thể **submit nhiều lần** trong thời gian mở. Hệ thống ghi nhận bản submit **cuối cùng** trước deadline.

### 11.5 Lưu ý riêng cho vòng 1

| # | Lưu ý |
|---|-------|
| 🔍 | Đây là vòng làm quen với dữ liệu thực tế — **kiểm tra kỹ pipeline trên dữ liệu training public trước** khi chạy trên private test |
| 💻 | Hạ tầng huấn luyện do thí sinh **tự chuẩn bị**. Hãy ước lượng thời gian chạy để đảm bảo kịp deadline |
| 🖥️ | **Cấu hình tham khảo** cho mỗi job inference: |
| | • 1× RTX A4000 (20 GB VRAM) |
| | • 4–8 CPU cores |
| | • 16–32 GB RAM |
| 📧 | Mọi thắc mắc về dữ liệu hoặc submission liên hệ **kênh hỗ trợ chính thức** của ban tổ chức |

---

## 12. Baseline thí sinh có thể tham khảo

### Gaussian Splatting (3DGS)

| Thuộc tính | Chi tiết |
|------------|----------|
| **Repo** | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) |
| **Paper** | *3D Gaussian Splatting for Real-Time Radiance Field Rendering* (SIGGRAPH 2023) |
| **Phương pháp** | Tái dựng scene bằng tập hợp các 3D Gaussian primitives |
| **Ưu điểm** | Render real-time, chất lượng cao, huấn luyện nhanh (~30-60 phút/scene) |
| **Phù hợp** | Dữ liệu đầu vào có sẵn sparse reconstruction từ COLMAP |

> 📌 Đây là baseline được BTC gợi ý. Thí sinh có thể sử dụng hoặc phát triển phương pháp riêng.

---

## Phụ lục: Tổng quan các thư mục và file quan trọng

```
D:\Kaggle_agent_tool\Viettel_Race_AI\De_1\
├── de.md                          # ← File này — tổng hợp đề bài
├── data/                          # Dữ liệu public set
├── gaussian-splatting/            # Baseline 3DGS (clone từ GitHub)
└── VAI_NVS_DATA_ROUND2.zip        # Dữ liệu private test round 2 (nếu có)
```

---

> 🏁 **Chúc thí sinh thi tốt!**
