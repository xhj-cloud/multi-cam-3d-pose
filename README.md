# multi-cam-3d-pose

**Multi-camera 3D human pose estimation pipeline** — 由 2~3 台正交摄像机重建 17 关节 3D 人体姿态（YOLOX + RTMPose + DLT 三角测量）。

多摄像机 RTMPose 3D 人体姿态估计：跨相机 3D 人体姿态重建流水线。

## 技术路线

| 阶段 | 方法 |
| --- | --- |
| 人体检测 | YOLOX |
| 2D 关节提取 | RTMPose（17 关节），rtmlib + ONNX Runtime 跨平台推理 |
| 多视角 3D 重建 | DLT 三角测量 |
| 时序处理 | 自动运动峰值时间对齐 + Savitzky-Golay 平滑 |
| 输出 | TXT 关节坐标表 + 3D 骨架动画 |

## 目录结构

```
camera_calibration.py            # 相机标定
calib/                           # 标定数据
config.py                        # 相机与流水线配置
video_preprocess.py              # 视频预处理
pose_detection.py                # 2D 姿态检测（YOLOX + RTMPose）
triangulation.py                 # 多视角 DLT 三角测量
visualization.py                 # 3D 骨架可视化
run_pipeline.py                  # 流水线入口
install_deps.sh                  # 系统依赖安装脚本
requirements.txt                 # Python 依赖
multi_camera_3d_pose_pipeline.md # 完整设计文档
```

## 快速开始

```bash
# 1. 安装依赖
./install_deps.sh                # 或 pip install -r requirements.txt

# 2. 准备标定数据（calib/ 目录）并配置相机（config.py）

# 3. 运行流水线
python run_pipeline.py
```

支持双机 / 三机正交摄像机部署。完整流程与参数说明见 [multi_camera_3d_pose_pipeline.md](multi_camera_3d_pose_pipeline.md)。

## License

[MIT](LICENSE) © 2026 [xhj-cloud](https://github.com/xhj-cloud)
