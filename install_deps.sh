#!/bin/bash
# ============================================================
# 多摄像机 RTMPose 3D 人体姿态估计 — 一键安装脚本
#
# 自动适配:
#   macOS (Apple Silicon / Intel) -> MPS / CPU 模式
#   Linux (NVIDIA GPU)            -> CUDA 模式
#
# 用法:
#   bash install_deps.sh                   # 自动检测平台
#   bash install_deps.sh --cuda 12.4       # Linux 上指定 CUDA 版本
#   bash install_deps.sh --device cpu      # 强制 CPU 模式
# ============================================================
set -e

# ------ 平台检测 ------
OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" = "Darwin" ]; then
    PLATFORM="macos"
    echo "检测到 macOS ($ARCH)"
elif [ "$OS" = "Linux" ]; then
    PLATFORM="linux"
    echo "检测到 Linux ($ARCH)"
else
    echo "未知系统: $OS, 按 Linux 处理"
    PLATFORM="linux"
fi

# ------ 默认配置 ------
DEVICE_MODE="auto"
CUDA_VERSION=""
PYTORCH_CUDA=""

# ------ 解析参数 ------
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            case $2 in
                11.8) CUDA_VERSION="118"; PYTORCH_CUDA="cu118" ;;
                12.1) CUDA_VERSION="121"; PYTORCH_CUDA="cu121" ;;
                12.4) CUDA_VERSION="124"; PYTORCH_CUDA="cu124" ;;
                *) echo "不支持的 CUDA 版本: $2 (可选: 11.8, 12.1, 12.4)"; exit 1 ;;
            esac
            DEVICE_MODE="cuda"
            shift 2
            ;;
        --device)
            DEVICE_MODE="$2"
            shift 2
            ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ------ 自动判断 CUDA / CPU ------
if [ "$DEVICE_MODE" = "auto" ]; then
    if [ "$PLATFORM" = "macos" ]; then
        DEVICE_MODE="mps"
        echo "自动选择: MPS/CPU 模式 (macOS 无 CUDA)"
    elif command -v nvidia-smi &>/dev/null; then
        DEVICE_MODE="cuda"
        if [ -z "$CUDA_VERSION" ]; then
            CUDA_VER_STR=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9.]+" || echo "")
            if [ -n "$CUDA_VER_STR" ]; then
                CUDA_MAJOR=$(echo "$CUDA_VER_STR" | cut -d. -f1)
                CUDA_MINOR=$(echo "$CUDA_VER_STR" | cut -d. -f2)
                CUDA_VERSION="${CUDA_MAJOR}${CUDA_MINOR}"
                PYTORCH_CUDA="cu${CUDA_VERSION}"
            else
                CUDA_VERSION="124"
                PYTORCH_CUDA="cu124"
            fi
        fi
        echo "自动选择: CUDA $PYTORCH_CUDA (检测到 NVIDIA GPU)"
    else
        DEVICE_MODE="cpu"
        echo "自动选择: CPU 模式 (未检测到 NVIDIA GPU)"
    fi
fi

echo "=============================================="
echo "  平台: $PLATFORM  |  模式: $DEVICE_MODE"
if [ "$DEVICE_MODE" = "cuda" ]; then
    echo "  PyTorch: $PYTORCH_CUDA  |  mmcv: cu$CUDA_VERSION"
fi
echo "=============================================="

# ================================================================
#  安装步骤
# ================================================================

# --- [1/5] PyTorch ---
echo ""
echo "[1/5] 安装 PyTorch..."
if [ "$DEVICE_MODE" = "cuda" ]; then
    pip install torch>=2.0.0 torchvision>=0.15.0 \
        --index-url "https://download.pytorch.org/whl/$PYTORCH_CUDA"
elif [ "$PLATFORM" = "macos" ]; then
    pip install torch>=2.0.0 torchvision>=0.15.0
else
    pip install torch>=2.0.0 torchvision>=0.15.0 \
        --index-url "https://download.pytorch.org/whl/cpu"
fi

# --- [2/5] 基础包 ---
echo ""
echo "[2/5] 安装基础 Python 包..."
pip install \
    opencv-python>=4.8.0 \
    opencv-contrib-python>=4.8.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    matplotlib>=3.7.0 \
    tqdm>=4.65.0 \
    PyYAML>=6.0 \
    Pillow>=10.0.0

# --- [3/5] openmim ---
echo ""
echo "[3/5] 安装 openmim..."
pip install openmim>=0.3.9

# --- [4/5] OpenMMLab ---
echo ""
echo "[4/5] 安装 mmengine, mmcv, mmdet, mmpose..."
mim install "mmengine>=0.10.0"

if [ "$DEVICE_MODE" = "cuda" ]; then
    mim install "mmcv>=2.0.0" --force-cuda "cu${CUDA_VERSION}"
else
    mim install "mmcv>=2.0.0"
fi

mim install "mmdet>=3.0.0"
mim install "mmpose>=1.0.0"

# --- [5/5] 下载模型 ---
echo ""
echo "[5/5] 下载 RTMPose 模型权重..."
mkdir -p checkpoints
if [ ! -f "checkpoints/rtmdet_m.pth" ]; then
    echo "  下载 RTMDet-m 检测模型..."
    wget -q --show-progress \
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth" \
        -O checkpoints/rtmdet_m.pth || \
    curl -L -o checkpoints/rtmdet_m.pth \
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
else
    echo "  [跳过] RTMDet-m 已存在"
fi

if [ ! -f "checkpoints/rtmpose_m_body.pth" ]; then
    echo "  下载 RTMPose-m 身体模型..."
    wget -q --show-progress \
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth" \
        -O checkpoints/rtmpose_m_body.pth || \
    curl -L -o checkpoints/rtmpose_m_body.pth \
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
else
    echo "  [跳过] RTMPose-m 身体模型已存在"
fi

# ------ 完成 ------
echo ""
echo "=============================================="
echo "  安装完成!"
echo "=============================================="
echo ""
echo "验证安装:"
echo '  python -c "import torch; print(\"PyTorch\", torch.__version__); print(\"CUDA\", torch.cuda.is_available()); print(\"MPS\", torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False)"'
echo '  python -c "import mmpose; print(\"MMPose\", mmpose.__version__)"'
echo '  python -c "import cv2;  print(\"OpenCV\", cv2.__version__)"'
echo ""
echo "运行流水线:"
echo "  python run_pipeline.py --help"
echo ""
