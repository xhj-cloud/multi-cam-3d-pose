#!/usr/bin/env python3
"""
run_pipeline.py
多摄像机 RTMPose 3D 人体姿态估计 — 完整流水线

用法:
  python run_pipeline.py                         # 完整流水线 (标定→预处理→2D→3D→导出)
  python run_pipeline.py --mode extract2d        # 仅 2D 关键点提取 (输出各摄像机独立 TXT)
  python run_pipeline.py --mode reconstruct3d    # 仅 3D 重建 (需已有 2D 结果)
  python run_pipeline.py --mode calibrate        # 仅摄像机标定
  python run_pipeline.py --no-anim               # 不生成动画 (仅 TXT 表格)
  python run_pipeline.py --max-frames 100        # 仅处理前 100 帧
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


# ================================================================
#  步骤 1: 摄像机标定
# ================================================================

def _build_camera_layout_string():
    """根据 CAMERA_MODE 动态生成摄像机布设信息"""
    layout_modes = {
        "dual":  "两台正交摄像机 (正面 + 侧面)",
        "triple": "三台正交摄像机 (正面 + 侧面 + 顶部)",
    }
    lines = [
        f"  摄像机模式: {CAMERA_MODE} ({layout_modes.get(CAMERA_MODE, '')})",
        f"  世界坐标系: Z轴向上, 单位: 米",
        f"  ┌{'─' * 45}┐",
    ]
    for i in range(NUM_CAMERAS):
        pos = ORTHOGONAL_CAM_POSITIONS[i]
        look = ORTHOGONAL_CAM_LOOKATS[i]
        name = CAMERA_NAMES[i]
        lines.append(f"  │ {name:<43} │")
        lines.append(f"  │   位置: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]{' ' * (29 - len(str(pos)))}│")
        lines.append(f"  │   看向: [{look[0]:.1f}, {look[1]:.1f}, {look[2]:.1f}]{' ' * (29 - len(str(look)))}│")
        if i < NUM_CAMERAS - 1:
            lines.append(f"  │{' ' * 45}│")
    lines.append(f"  └{'─' * 45}┘")
    return "\n".join(lines)


def step_calibration(args):
    """使用预设的摄像机位置进行标定"""
    from camera_calibration import run_calibration

    print("\n" + "=" * 60)
    print("步骤 1/7: 摄像机标定")
    print("=" * 60)
    print(_build_camera_layout_string())

    intrinsics, extrinsics, calib_data = run_calibration(
        mode="manual",
        camera_positions=ORTHOGONAL_CAM_POSITIONS,
        camera_lookats=ORTHOGONAL_CAM_LOOKATS,
        image_width=args.image_width,
        image_height=args.image_height,
        focal_length_mm=args.focal_length,
        sensor_width_mm=args.sensor_width,
    )

    return intrinsics, extrinsics, calib_data


# ================================================================
#  步骤 2: 视频预处理
# ================================================================

def step_preprocess():
    """提取帧并同步"""
    from video_preprocess import run_preprocess

    print("\n" + "=" * 60)
    print("步骤 2/7: 视频预处理与帧提取")
    print("=" * 60)

    return run_preprocess()


# ================================================================
#  步骤 3: RTMPose 2D 关键点检测
# ================================================================

def step_pose_detection(frame_dirs, total_frames):
    """RTMPose 逐帧 2D 关键点检测"""
    from pose_detection import process_all_cameras

    print("\n" + "=" * 60)
    print("步骤 3/7: RTMPose 2D 人体关键点检测")
    print("=" * 60)

    return process_all_cameras(frame_dirs, total_frames)


# ================================================================
#  步骤 4 & 5: 3D 三角测量重建
# ================================================================

def step_triangulation(all_keypoints, calib_data):
    """跨视图匹配 + 3D 三角测量"""
    from triangulation import reconstruct_3d

    print("\n" + "=" * 60)
    print("步骤 4&5/7: 跨视图匹配 + 3D 三角测量重建")
    print("=" * 60)

    return reconstruct_3d(all_keypoints, calib_data)


# ================================================================
#  步骤 6 & 7: 后处理 + 导出
# ================================================================

def step_export(animations, args):
    """后处理平滑 + TXT 表格 + 3D 动画"""
    from visualization import (
        postprocess_and_export,
        export_txt_table,
        export_txt_summary,
        create_3d_animation,
        print_statistics,
        smooth_sequence,
        interpolate_missing,
    )

    print("\n" + "=" * 60)
    print("步骤 6&7/7: 后处理 + 导出结果")
    print("=" * 60)

    # 后处理
    if args.skip_smooth:
        print("\n跳过平滑")
        animations_smooth = {
            pid: interpolate_missing(seq) for pid, seq in animations.items()
        }
    else:
        animations_smooth = {
            pid: smooth_sequence(seq, SG_WINDOW, SG_ORDER)
            for pid, seq in animations.items()
        }

    # 导出 TXT 表格
    print("\n导出 TXT 表格...")
    export_txt_table(animations_smooth,
                     os.path.join(OUTPUT_DIR, "skeletons_3d_smooth.txt"))
    export_txt_summary(animations_smooth,
                       os.path.join(OUTPUT_DIR, "skeletons_3d_summary.txt"))

    # 导出原始数据 (未平滑)
    if args.skip_smooth:
        export_txt_table(animations,
                         os.path.join(OUTPUT_DIR, "skeletons_3d_raw.txt"))

    # 3D 动画
    if not args.no_anim:
        print("\n生成 3D 骨架动画...")
        create_3d_animation(
            animations_smooth, OUTPUT_DIR,
            fps=ANIM_FPS, dpi=ANIM_DPI,
            rotate=ANIM_ROTATE, elevation=ANIM_ELEVATION,
            azimuth_speed=ANIM_AZIMUTH_SPEED,
        )

    # 统计
    print_statistics(animations_smooth)

    return animations_smooth


# ================================================================
#  完整流水线
# ================================================================

def run_full_pipeline(args):
    """执行完整的 1-7 步流水线"""
    calib_data = None
    frame_dirs = None
    total_frames = None

    # ---- 步骤 1: 标定 ----
    if not args.skip_calib:
        _, _, calib_data = step_calibration(args)
    else:
        # 加载已有标定
        print("\n跳过标定, 加载已有数据...")
        from triangulation import load_calib
        _, _, _, _, calib_data = load_calib(CALIB_JSON_FILE)

    # ---- 步骤 2: 视频预处理 ----
    if not args.skip_preprocess:
        frame_dirs, total_frames, fps, meta = step_preprocess()
    else:
        frames_root = Path(OUTPUT_DIR) / "frames"
        if frames_root.exists():
            frame_dirs = sorted(frames_root.glob("camera_*"))
            if frame_dirs:
                sample_dir = frame_dirs[0]
                total_frames = len(list(sample_dir.glob("frame_*.jpg")))
                fps = 30  # 默认
                print(f"\n加载已有帧: {total_frames} 帧")
            else:
                print("[错误] 帧目录为空")
                return
        else:
            print("[错误] 未找到预提取帧, 请先运行预处理")
            return

    # ---- 步骤 3: 2D 检测 ----
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)
    if not args.skip_detect:
        all_keypoints = step_pose_detection(frame_dirs, total_frames)
    else:
        kpts_path = os.path.join(OUTPUT_DIR, "all_2d_keypoints.json")
        if os.path.exists(kpts_path):
            print("\n加载已有 2D 关键点...")
            with open(kpts_path) as f:
                all_keypoints = json.load(f)
            print(f"  加载 {len(all_keypoints)} 帧")
        else:
            print("[错误] 未找到 2D 关键点文件")
            return

    # ---- 步骤 4 & 5: 3D 重建 ----
    animations = step_triangulation(all_keypoints, calib_data)

    if not animations:
        print("\n[错误] 3D 重建失败: 未检测到有效人物")
        return

    # ---- 步骤 6 & 7: 导出 ----
    step_export(animations, args)

    # ---- 完成 ----
    print("\n" + "=" * 60)
    print("  流水线执行完毕!")
    print("=" * 60)
    print(f"\n输出文件:")
    output_files = [
        ("TXT 详细表格", os.path.join(OUTPUT_DIR, "skeletons_3d_smooth.txt")),
        ("TXT 简洁表格", os.path.join(OUTPUT_DIR, "skeletons_3d_summary.txt")),
        ("3D 骨架 NPZ",  os.path.join(OUTPUT_DIR, "skeletons_3d.npz")),
    ]
    for name, path in output_files:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"  {name}: {path} ({size:.1f} KB)")
    # 动画文件
    for f in sorted(Path(OUTPUT_DIR).glob("person_*_3d_anim.mp4")):
        size = os.path.getsize(f) / 1024 / 1024
        print(f"  3D 动画: {f} ({size:.1f} MB)")

    print(f"\n标定文件: {CALIB_JSON_FILE}")
    print(f"2D 关键点: {os.path.join(OUTPUT_DIR, 'all_2d_keypoints.json')}")


# ================================================================
#  命令行入口
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="三台正交摄像机 RTMPose 3D 人体姿态估计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_pipeline.py                          完整流水线
  python run_pipeline.py --mode calibrate         仅摄像机标定
  python run_pipeline.py --mode detect            2D检测 + 3D重建
  python run_pipeline.py --no-anim                不渲染动画
  python run_pipeline.py --skip-smooth            跳过平滑
  python run_pipeline.py --max-frames 100         仅处理前100帧
  python run_pipeline.py --image-width 3840 --image-height 2160  4K视频
        """,
    )

    parser.add_argument("--mode",
                        choices=["calibrate", "extract2d", "reconstruct3d", "full"],
                        default="full", help="运行模式")
    parser.add_argument("--image-width", type=int, default=3840, help="图像宽度")
    parser.add_argument("--image-height", type=int, default=2160, help="图像高度")
    parser.add_argument("--focal-length", type=float, default=14.0, help="焦距 (mm)")
    parser.add_argument("--sensor-width", type=float, default=36.0, help="传感器宽度 (mm)")

    # 跳过标志
    parser.add_argument("--skip-calib", action="store_true", help="跳过标定")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过视频预处理")
    parser.add_argument("--skip-detect", action="store_true", help="跳过 2D 检测")
    parser.add_argument("--skip-smooth", action="store_true", help="跳过平滑")
    parser.add_argument("--no-anim", action="store_true", help="不生成动画")

    parser.add_argument("--max-frames", type=int, default=None, help="最大处理帧数")

    args = parser.parse_args()

    print("=" * 60)
    print("  多摄像机 RTMPose 3D 人体姿态估计")
    print("=" * 60)
    print(f"  摄像机: {NUM_CAMERAS} 台 — {CAMERA_MODE} 模式")
    print(f"          {' | '.join(CAMERA_NAMES)}")
    print(f"  设备:   {DEVICE}")
    print(f"  模式:   {args.mode}")
    print("=" * 60)

    if args.mode == "calibrate":
        step_calibration(args)

    elif args.mode == "extract2d":
        # ------ 仅 2D 提取: RTMPose 对每台摄像机分别提取关节坐标 ------
        frames_root = Path(OUTPUT_DIR) / "frames"
        total_frames = 0
        if frames_root.exists():
            frame_dirs = sorted(frames_root.glob("camera_*"))
            if frame_dirs:
                total_frames = len(list(frame_dirs[0].glob("frame_*.jpg")))

        if total_frames == 0:
            print("未找到预提取帧, 自动运行预处理...")
            frame_dirs, total_frames, fps, meta = step_preprocess()
        if args.max_frames:
            total_frames = min(total_frames, args.max_frames)

        all_keypoints = step_pose_detection(frame_dirs, total_frames)
        print("\n各摄像机 2D 关键点已导出到 output/ 目录:")
        for cam_id in range(NUM_CAMERAS):
            print(f"  output/camera_{cam_id}_2d_keypoints.txt")

    elif args.mode == "reconstruct3d":
        # ------ 仅 3D 重建: 从已有 2D 结果计算 3D 坐标 ------
        kpts_path = os.path.join(OUTPUT_DIR, "all_2d_keypoints.json")
        if not os.path.exists(kpts_path):
            print(f"[错误] 未找到 2D 关键点文件: {kpts_path}")
            print("请先运行: python run_pipeline.py --mode extract2d")
            return

        from triangulation import load_calib
        _, _, _, _, calib_data = load_calib(CALIB_JSON_FILE)

        print(f"\n加载 2D 关键点: {kpts_path}")
        with open(kpts_path) as f:
            all_keypoints = json.load(f)

        animations = step_triangulation(all_keypoints, calib_data)
        if animations:
            step_export(animations, args)

    elif args.mode == "full":
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
