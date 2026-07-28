# 多摄像机 RTMPose 3D 人体姿态估计

## 项目概述

使用两台（或三台）正交放置的摄像机录制运动员视频，通过 RTMPose 逐帧提取各视角的 2D 人体关节坐标，利用 DLT 三角测量重建关节在三维空间中的真实位置。最终输出逐帧 TXT 坐标表格和 3D 骨架动画。

**技术栈**: Python · rtmlib + ONNX Runtime · YOLOX 人体检测 · RTMPose 姿态估计 · DLT 三角测量

---

## 项目结构

```
运动员身体数据采集/
├── config.py                  # 全局配置 (摄像机模式、检测参数、输出控制)
├── run_pipeline.py            # 主入口，串联 7 个步骤
│
├── camera_calibration.py      # 步骤1: 摄像机标定 (K/R/t/P 矩阵)
├── video_preprocess.py        # 步骤2: 视频帧提取 + 自动时间对齐
├── pose_detection.py          # 步骤3: RTMPose 2D 关键点检测
├── triangulation.py           # 步骤4&5: 跨视图匹配 + DLT 三角测量
├── visualization.py           # 步骤6&7: 后处理 + TXT表格 + 3D动画
│
├── inputs/                    # 原始视频 (camera_0.mp4, camera_1.mp4, ...)
├── checkpoints/               # 模型权重 (yolox_m.onnx, rtmpose_m_body.pth)
├── calib/                     # 标定结果 (multi_camera_calib.json)
├── output/                    # 所有输出
│   ├── frames/                #   预处理帧图像
│   ├── detection_vis/         #   2D 检测可视化
│   ├── camera_*_2d_keypoints.txt  # 各摄像机 2D 坐标
│   ├── all_2d_keypoints.json  #   汇总 2D JSON
│   ├── skeletons_3d.npz       #   原始 3D 数据
│   ├── skeletons_3d_smooth.txt    # 平滑后 3D 坐标表格 ★
│   └── person_*_3d_anim.mp4   #   3D 骨架动画
│
├── requirements.txt           # Python 依赖
├── install_deps.sh            # 一键安装脚本 (Mac/Linux 自适应)
└── multi_camera_3d_pose_pipeline.md  # 本文档
```

---

## 摄像机部署

### 世界坐标系

```
Z 轴向上, 单位: 米
原点位于运动员活动区域中心, 地面为 Z = 0
```

### 双机模式 (CAMERA_MODE = "dual")

```
        Camera 0 (正面)
        (0, -5, 1.5) ───→ 看向 (0, 0, 1.0)
              │                    ← 运动员髋部高度
              │
    ┌─────────●─────────┐
    │   运动员活动区域    │
    │   (0, 0, 0)       │
    └───────────────────┘
              ▲
              │
        Camera 1 (侧面)
        (5, 0, 1.5) ───→ 看向 (0, 0, 1.0)

两摄像机相距 10m (各距中心 5m), 正交 90°
```

### 三机模式 (CAMERA_MODE = "triple")

在双机基础上增加顶部摄像机 `Camera 2 (0, 0, 5) → (0, 0, 0)`，垂直向下。

---

## 工作流程

整个流水线分 7 步，可通过命令行分步或一键运行。

### 数据流

```
inputs/camera_*.mp4
      │
      ▼ [步骤1] 摄像机标定 ──→ calib/multi_camera_calib.json
      │                       (K内参 / R旋转 / t平移 / P投影)
      │
      ▼ [步骤2] 视频预处理 ──→ output/frames/camera_*/frame_*.jpg
      │                       自动运动峰值对齐 + 帧率统一
      │
      ▼ [步骤3] 2D 关键点检测 ──→ output/camera_*_2d_keypoints.txt
      │                         YOLOX 检测人体 → RTMPose 17关节
      │
      ▼ [步骤4-5] 3D 重建 ──→ output/skeletons_3d.npz
      │                      单人直配 / 多人极线匹配 → DLT三角测量
      │
      ▼ [步骤6-7] 输出 ──→ output/skeletons_3d_smooth.txt
                          output/person_0_3d_anim.mp4
```

### 各步骤详解

#### 步骤 1: 摄像机标定

根据 `config.py` 中预设的摄像机物理位置和镜头参数，自动计算：

- **内参 K**: 由焦距 (14mm) / 传感器宽度 (36mm 全画幅) / 图像分辨率 (3840×2160) 计算
- **外参 R, t**: 由摄像机位置和朝向构建右手坐标系
- **投影矩阵 P**: `P = K × [R | t]`，3D 世界坐标 → 2D 像素坐标的桥梁

```python
fx = (焦距 / 传感器宽度) × 图像宽度     # 14/36 × 3840 = 1493.33
cx = 图像宽度 / 2                       # 1920
cy = 图像高度 / 2                       # 1080
```

#### 步骤 2: 视频预处理

- 读取所有视频的基本信息 (分辨率、帧率、时长)
- **自动时间对齐**: 默认使用运动峰值检测 — 在每台摄像机的视频中找帧间变化最大的时刻，对齐到同一时间点
- 不同帧率自动统一到最低帧率
- 视频时长不一致时，短视频结束后填充黑帧

支持 4 种对齐方法 (`SYNC_METHOD`):

| 方法 | 原理 | 适用场景 |
|------|------|---------|
| `motion` | 检测帧间像素变化峰值 | 运动员有明显动作 |
| `brightness` | 检测画面亮度突变 | 有闪光/灯光变化 |
| `timestamp` | 读取视频内嵌时间戳 | 设备时钟准确 |
| `manual` | 手动指定帧偏移量 | 已知具体偏移 |

#### 步骤 3: RTMPose 2D 关键点检测

**不使用 MMPose/mmdet**，改用 `rtmlib` + ONNX Runtime，兼容 macOS / Linux / Windows。

两阶段推理：

```
原始帧 (3840×2160)
      │
      ▼
┌──────────────┐
│  YOLOX-m     │  ONNX 推理, 输入 640×640
│  人体检测     │  输出: [x1, y1, x2, y2] × N 人
└──────┬───────┘
       │ 裁剪人物区域
       ▼
┌──────────────┐
│  RTMPose-m   │  ONNX 推理, 输入 (256, 384)
│  姿态估计     │  SimCC 分类: 每个像素 → 关节概率 → argmax
│              │  输出: 17 个 (x, y, confidence)
└──────────────┘
```

**COCO 17 关键点**:

```
 0: 鼻子        1: 左眼      2: 右眼
 3: 左耳        4: 右耳
 5: 左肩        6: 右肩
 7: 左肘        8: 右肘
 9: 左腕       10: 右腕
11: 左髋       12: 右髋
13: 左膝       14: 右膝
15: 左踝       16: 右踝
```

**输出**:
- `output/camera_0_2d_keypoints.txt` — 正面摄像机每帧每关节像素坐标
- `output/camera_1_2d_keypoints.txt` — 侧面摄像机每帧每关节像素坐标
- `output/all_2d_keypoints.json` — 汇总数据供 3D 重建使用

#### 步骤 4: 跨视图匹配

**单人场景** (所有摄像机都只有 1 人): 直接配对，跳过极线匹配。

**多人场景**: 以 Camera 0 为基准，用基础矩阵 F 计算 Sampson 极线距离，找最小匹配代价的人：

```
F = K₂⁻ᵀ · [t_rel]× · R_rel · K₁⁻¹

d = Sampson距离(kpt_cam0, kpt_cam1, F)
  = (p₂ · F · p₁)² / (极线参数)
```

#### 步骤 5: DLT 三角测量

核心数学：从 N 台摄像机的 2D 坐标反推 3D 坐标。

```
已知: 3D点 X 在摄像机 i 的投影为 (uᵢ, vᵢ)
满足: (uᵢ, vᵢ, 1)ᵀ ∼ Pᵢ · X    (∼ 表示差一个尺度因子)

交叉乘消去未知尺度:
  uᵢ × (Pᵢ[2]·X) = Pᵢ[0]·X
  vᵢ × (Pᵢ[2]·X) = Pᵢ[1]·X

整理为齐次线性方程组:
  A · X = 0    A: 2N × 4 矩阵

SVD 分解 A, 最小奇异值对应的右奇异向量 → X
```

双摄像机 = 4 个方程，刚好求解。三摄像机 = 6 个方程，冗余提高鲁棒性。

#### 步骤 6: 后处理

- **缺失值插值**: 线性插值填充因遮挡/低置信度产生的 NaN 关节
- **Savitzky-Golay 时序平滑**: 窗口 11 帧，多项式阶数 3，消除逐帧抖动

#### 步骤 7: 结果输出

**TXT 详细表格** (`skeletons_3d_smooth.txt`):

```
 frame | person | joint_id |      joint_name |       x(m) |       y(m) |       z(m)
-------|--------|----------|----------------|----------|----------|----------
     0 |      0 |        0 |            nose |    -1.8618 |    -2.6792 |     4.5168
     0 |      0 |        5 |   left_shoulder |    -1.2216 |    -3.8210 |     1.5893
     ...
```

**TXT 简洁表格** (`skeletons_3d_summary.txt`): 每行一帧，所有关节 XYZ 平铺。

**3D 骨架动画** (`person_*_3d_anim.mp4`): Matplotlib 3D 渲染，视角自动旋转。

---

## 快速开始

### 1. 安装

```bash
cd 运动员身体数据采集
bash install_deps.sh        # 自动适配 Mac (ONNX/CPU) / Linux (CUDA)

# 手动安装 (如果脚本失败)
source venv/bin/activate
pip install openmim
mim install mmengine mmcv
pip install rtmlib opencv-python numpy scipy matplotlib tqdm
```

### 2. 准备视频

```
inputs/
├── camera_0.mp4    ← 正面
├── camera_1.mp4    ← 侧面
└── camera_2.mp4    ← 顶部 (三机模式才需要)
```

### 3. 配置

编辑 `config.py`:

```python
CAMERA_MODE = "dual"          # "dual" = 两机, "triple" = 三机
SYNC_METHOD = "motion"        # 自动时间对齐方法
KPT_CONF_THRESHOLD = 0.15     # 关键点置信度 (超广角建议 0.15)
```

### 4. 运行

```bash
source venv/bin/activate

# 方式一: 一键完整流水线
python run_pipeline.py --max-frames 100    # 先试 100 帧
python run_pipeline.py                     # 全量

# 方式二: 分步运行
python run_pipeline.py --mode calibrate    # 仅标定
python run_pipeline.py --mode extract2d    # 仅 2D 提取
python run_pipeline.py --mode reconstruct3d # 仅 3D 重建

# 选项
--no-anim            # 不渲染动画
--skip-smooth        # 跳过平滑
--image-width 3840 --image-height 2160 --focal-length 14   # 自定义摄像机参数
```

---

## 配置参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CAMERA_MODE` | `"dual"` | `"dual"` / `"triple"` |
| `SYNC_METHOD` | `"motion"` | `"motion"` / `"brightness"` / `"timestamp"` / `"manual"` |
| `KPT_CONF_THRESHOLD` | `0.15` | 关节置信度阈值，超广角设低些 |
| `DET_CONF_THRESHOLD` | `0.30` | YOLOX 人体检测阈值 |
| `POSE_MODEL_INPUT_SIZE` | `(256, 384)` | RTMPose 输入尺寸，越大越准但越慢 |
| `TRIANG_METHOD` | `"dlt"` | `"dlt"` / `"midpoint"` |
| `RANSAC_ITERATIONS` | `0` | RANSAC 去噪，双机模式建议关闭 |
| `MIN_VIEWS` | `2` | 最少可见视角数 |
| `SG_WINDOW` | `11` | 平滑窗口 (奇数) |

---

## 常见问题

### 检测到的人很少或没有

- 降低 `DET_CONF_THRESHOLD` (YOLOX 门槛) 和 `KPT_CONF_THRESHOLD` (关键点门槛)
- 增大 `POSE_MODEL_INPUT_SIZE`，如 `(384, 512)`，保留更多细节
- 检查视频中人物是否太小 (14mm 超广角距离 5m 建议用更高输入分辨率)

### 3D 坐标全是 NaN

- 确认标定文件存在: `ls calib/multi_camera_calib.json`
- 确认图像分辨率与标定一致: 4K → `calibrate --image-width 3840 --image-height 2160`
- 检查视频重叠窗口: 双机都有人物的时间范围决定有效帧数

### 匹配到 2 个人 (实际只有 1 个)

- 单人场景会自动直配，偶尔 YOLOX 误检会产生多人帧 (<2%)
- 不影响主要结果，可忽略

### 3D 坐标偏移或尺度不对

- 检查焦距参数: `--focal-length` 应与实际镜头一致
- 检查传感器宽度: 全画幅 = 36mm, APS-C = 24mm
- 绝对尺度受焦距和传感器参数影响，关节间**相对位置**更可靠

### Mac 上速度太慢

- ONNX Runtime 在 Mac 上使用 CPU 推理 (MPS 对姿态模型支持不完整)
- 建议用 `--max-frames 100` 快速验证
- 完整推理部署到 Linux CUDA 服务器

---

## 输出文件清单

| 文件 | 格式 | 内容 |
|------|------|------|
| `skeletons_3d_smooth.txt` | TXT 表格 | 平滑后逐帧逐关节 XYZ 坐标 |
| `skeletons_3d_summary.txt` | TXT 表格 | 简洁版，每行一帧全关节 |
| `person_0_3d_anim.mp4` | MP4 视频 | 3D 骨架旋转动画 |
| `skeletons_3d.npz` | NumPy | 原始 3D 数据 (T×17×3) |
| `camera_0_2d_keypoints.txt` | TXT 表格 | 正面摄像机 2D 像素坐标 |
| `camera_1_2d_keypoints.txt` | TXT 表格 | 侧面摄像机 2D 像素坐标 |
| `multi_camera_calib.json` | JSON | 标定参数 (K/R/t/P) |

---

## 关键公式速查

```
投影:   x = P · X     (P: 3×4, X: 4×1, x: 3×1 齐次)
内参:   K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
外参:   t = -R · camera_position
三角化: SVD(A) → 最小奇异值右奇异向量 → 3D 坐标
极线:   l₁₂ = F · p₁    (Camera 1 画面中对应的极线)
```
