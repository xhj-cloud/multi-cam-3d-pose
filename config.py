"""
全局配置文件
根据实际场景修改以下参数

设备说明:
  - MacBook Pro (本机): 自动使用 MPS (Apple Silicon) 或 CPU
  - Linux 服务器:      自动使用 CUDA
  - 也可手动指定:      DEVICE = "cuda:0" / "mps" / "cpu"

摄像机模式:
  - CAMERA_MODE = "dual"   → 两台正交摄像机 (正面 + 侧面)
  - CAMERA_MODE = "triple" → 三台正交摄像机 (正面 + 侧面 + 顶部)
"""

import os
import torch
import platform

# ============================================================
#  项目路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "inputs")         # 视频文件存放目录
CALIB_DIR = os.path.join(PROJECT_ROOT, "calib")         # 标定结果输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")       # 所有输出目录

# ============================================================
#  设备自动检测 (Mac / Linux CUDA 自适应)
# ============================================================

def _detect_device():
    """自动选择最优推理设备: CUDA > MPS > CPU"""
    env_device = os.environ.get("MVP3D_DEVICE", None)
    if env_device:
        return env_device

    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

DEVICE = _detect_device()
IS_CUDA = DEVICE.startswith("cuda")
IS_MPS = (DEVICE == "mps")
IS_CPU = (DEVICE == "cpu")

SYSTEM = platform.system()
IS_MAC = (SYSTEM == "Darwin")

# ============================================================
#  摄像机模式 — 修改 CAMERA_MODE 即可切换双机 / 三机
# ============================================================
# "dual" = 两台 (正面 + 侧面)
# "triple" = 三台 (正面 + 侧面 + 顶部)
# 也可通过环境变量覆盖: export MVP3D_CAMERA_MODE=dual
CAMERA_MODE = os.environ.get("MVP3D_CAMERA_MODE", "dual")

# ============================================================
#  摄像机配置 (由 CAMERA_MODE 自动派生)
# ============================================================

# ---- 摄像机物理位置 (世界坐标系: Z轴向上, 单位: 米) ----
_CAMERA_PRESETS = {

    "dual": {
        "num": 2,
        "names": [
            "Camera_0_正面(Front)",
            "Camera_1_侧面(Side)",
        ],
        "positions": [
            [ 0.0, -5.0, 1.5],    # 正面: Y轴负方向 5m, 高 1.5m
            [ 5.0,  0.0, 1.5],    # 侧面: X轴正方向 5m, 高 1.5m
        ],
        "lookats": [
            [0.0, 0.0, 1.0],      # 正面看向原点上方 1m
            [0.0, 0.0, 1.0],      # 侧面看向原点上方 1m
        ],
        "sync_offsets": [0, 0],
    },

    "triple": {
        "num": 3,
        "names": [
            "Camera_0_正面(Front)",
            "Camera_1_侧面(Side)",
            "Camera_2_顶部(Top)",
        ],
        "positions": [
            [ 0.0, -5.0, 1.5],    # 正面
            [ 5.0,  0.0, 1.5],    # 侧面
            [ 0.0,  0.0, 5.0],    # 顶部: 正上方 5m
        ],
        "lookats": [
            [0.0, 0.0, 1.0],      # 正面看向髋部高度
            [0.0, 0.0, 1.0],      # 侧面看向髋部高度
            [0.0, 0.0, 0.0],      # 顶部看向地面原点
        ],
        "sync_offsets": [0, 0, 0],
    },
}

# 校验模式
if CAMERA_MODE not in _CAMERA_PRESETS:
    raise ValueError(f"CAMERA_MODE 必须为 'dual' 或 'triple', 当前: {CAMERA_MODE}")

_preset = _CAMERA_PRESETS[CAMERA_MODE]

NUM_CAMERAS               = _preset["num"]
CAMERA_NAMES              = _preset["names"]
ORTHOGONAL_CAM_POSITIONS  = _preset["positions"]
ORTHOGONAL_CAM_LOOKATS    = _preset["lookats"]
SYNC_FRAME_OFFSETS        = _preset["sync_offsets"]

# 视频文件路径 (按摄像机编号排列)
VIDEO_PATHS = [
    os.path.join(DATA_DIR, f"camera_{i}.mp4")
    for i in range(NUM_CAMERAS)
]

# ============================================================
#  标定配置
# ============================================================

CHESSBOARD_SIZE = (9, 6)
CHESSBOARD_SQUARE_SIZE = 0.030  # 30mm

CALIB_IMAGE_DIRS = [
    os.path.join(CALIB_DIR, "calib_images", f"camera_{i}")
    for i in range(NUM_CAMERAS)
]

INTRINSICS_FILE = os.path.join(CALIB_DIR, "intrinsics.npz")
EXTRINSICS_FILE = os.path.join(CALIB_DIR, "extrinsics.npz")
CALIB_JSON_FILE = os.path.join(CALIB_DIR, "multi_camera_calib.json")

# ============================================================
#  视频预处理配置
# ============================================================

# 时间同步方法:
#   "motion"     → 自动检测运动峰值对齐 (推荐: 运动员有明显动作)
#   "brightness" → 自动检测亮度突变对齐 (推荐: 开始录制时有闪光/拍手)
#   "timestamp"  → 基于视频时间戳对齐 (最精确, 但取决于录制设备)
#   "auto"       → 自动依次尝试 timestamp → motion → brightness
#   "manual"     → 使用下方 SYNC_FRAME_OFFSETS 手动指定
SYNC_METHOD = "motion"

TARGET_FPS = None         # None = 自动取最慢摄像机的帧率
FRAME_FORMAT = "jpg"
FRAME_QUALITY = 95

# ============================================================
#  RTMPose 模型配置
# ============================================================
# 模型权重由 pose_detection.py 直接从 checkpoints/ 加载:
#   - checkpoints/yolox_m.onnx      (YOLOX 人体检测, ONNX)
#   - checkpoints/rtmpose_m_body.pth (RTMPose 姿态估计, PyTorch)
# 推理使用 rtmlib + ONNX Runtime, 无需 MMPose/mmdet

# CUDA 专用参数 (仅在 IS_CUDA=True 时生效)
CUDA_BENCHMARK = True          # torch.backends.cudnn.benchmark
CUDA_MEMORY_FRACTION = 0.85    # GPU 显存使用比例上限
BATCH_SIZE = 1                 # 推理批大小 (CUDA 下可适当增大)
USE_AMP = IS_CUDA              # 混合精度推理 (仅 CUDA 支持)

# 检测置信度阈值 (YOLOX 人体检测, 低于此值的检测结果会被丢弃)
DET_CONF_THRESHOLD = 0.3

# 关键点置信度阈值 (RTMPose 关节检测, 低于此值置零)
# 14mm 超广角下人物较小, 降低阈值以保留更多关节
KPT_CONF_THRESHOLD = 0.15

# RTMPose 模型输入尺寸 (宽, 高)
# 默认 (192, 256), 超广角远距离建议 (256, 384) 或 (384, 512)
POSE_MODEL_INPUT_SIZE = (256, 384)

# 检测框收紧系数 (0~1, 越大边框越紧贴人物)
# YOLOX 在超广角下容易给过大框, 收紧可提升 RTMPose 精度
BBOX_TIGHTEN_RATIO = 0.0

# 跳帧检测: 每隔 N 帧做一次完整人体检测, 中间帧用上一帧的框
SKIP_FRAME_INTERVAL = 1  # 1 = 不跳帧

# ============================================================
#  三角测量与 3D 重建
# ============================================================
# 三角测量方法: "dlt" = DLT, "midpoint" = 中点法 (更快)
TRIANG_METHOD = "dlt"

# RANSAC 迭代次数 (0 = 不启用 RANSAC, 直接 DLT)
RANSAC_ITERATIONS = 0

# RANSAC 重投影误差阈值 (像素)
# 双机模式下 RANSAC 意义不大(只有2个视角), 直接用 DLT
RANSAC_THRESHOLD = 200.0

# 最少可见视角数 (低于此值的关节标记为 NaN)
MIN_VIEWS = 2

# 关键点数量 (COCO 17 关节)
NUM_JOINTS = 17

# ============================================================
#  后处理
# ============================================================
# Savitzky-Golay 滤波窗口 (奇数, 0 = 不滤波)
SG_WINDOW = 11
SG_ORDER = 3

# 骨骼长度约束 (0 = 不约束)
BONE_LENGTH_WEIGHT = 0.0

# ============================================================
#  输出配置
# ============================================================
SAVE_FRAMES = True
SAVE_DET_VIS = True

# TXT 表格精度
TXT_PRECISION = 4

# 3D 动画配置
ANIM_FPS = 30
ANIM_DPI = 100
ANIM_ROTATE = True        # 是否旋转视角
ANIM_ELEVATION = 20       # 视角仰角
ANIM_AZIMUTH_SPEED = 1.0  # 旋转速度 (度/秒)

# ============================================================
#  自动创建必要目录
# ============================================================
for _d in [DATA_DIR, CALIB_DIR, OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)
