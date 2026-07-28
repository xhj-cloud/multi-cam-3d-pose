"""
步骤 6 & 7: 后处理 + TXT 表格输出 + 3D 骨架动画

输出:
  1. skeletons_3d.txt       每帧每个关节的 XYZ 坐标表格
  2. skeletons_3d_smooth.txt 平滑后的坐标表格
  3. person_*_3d_anim.mp4   3D 骨架旋转动画
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 无 GUI 后端, 支持服务器渲染
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from config import (
    OUTPUT_DIR, NUM_JOINTS,
    SG_WINDOW, SG_ORDER, BONE_LENGTH_WEIGHT,
    TXT_PRECISION, ANIM_FPS, ANIM_DPI, ANIM_ROTATE,
    ANIM_ELEVATION, ANIM_AZIMUTH_SPEED,
)


# ================================================================
#  COCO 骨架连接定义
# ================================================================

COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 头部
    (5, 6),                                  # 肩膀
    (5, 7), (7, 9), (6, 8), (8, 10),       # 手臂
    (5, 11), (6, 12), (11, 12),             # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16), # 腿部
]

BONE_NAMES = [
    "nose->Leye", "nose->Reye", "Leye->Lear", "Reye->Rear",
    "Lsho->Rsho",
    "Lsho->Lelb", "Lelb->Lwri", "Rsho->Relb", "Relb->Rwri",
    "Lsho->Lhip", "Rsho->Rhip", "Lhip->Rhip",
    "Lhip->Lkne", "Lkne->Lank", "Rhip->Rkne", "Rkne->Rank",
]

SKELETON_COLORS = [
    "#00FF00", "#00FF00", "#00FF00", "#00FF00",  # 头 绿
    "#FF0000",                                      # 肩 红
    "#FF8000", "#FF8000", "#0080FF", "#0080FF",    # 臂 橙/蓝
    "#FFFF00", "#FFFF00", "#FFFF00",                # 躯干 黄
    "#FF00FF", "#FF00FF", "#800080", "#800080",    # 腿 紫
]


# ================================================================
#  第一部分: 后处理
# ================================================================

def interpolate_missing(seq):
    """
    线性插值填充 NaN 关节坐标

    参数:
        seq: (T, 17, 3)
    返回:
        filled: (T, 17, 3)
    """
    T = seq.shape[0]
    filled = seq.copy()

    for j in range(NUM_JOINTS):
        for d in range(3):
            series = seq[:, j, d]
            valid = ~np.isnan(series)

            if valid.sum() < 2:
                filled[:, j, d] = 0.0
                continue

            idx = np.arange(T)
            f = interp1d(
                idx[valid], series[valid],
                kind="linear", fill_value="extrapolate"
            )
            filled[:, j, d] = f(idx)

    return filled


def smooth_sequence(seq, window=11, order=3):
    """
    Savitzky-Golay 时序平滑

    参数:
        seq:    (T, 17, 3)
        window: 窗口长度 (奇数)
        order:  多项式阶数
    """
    if window <= 0:
        return seq

    T = seq.shape[0]
    if T < window:
        print(f"  [跳过平滑] 帧数 {T} < 窗口 {window}")
        return seq

    # 先填 NaN
    filled = interpolate_missing(seq)
    smoothed = np.zeros_like(filled)

    for j in range(NUM_JOINTS):
        for d in range(3):
            smoothed[:, j, d] = savgol_filter(filled[:, j, d], window, order)

    return smoothed


# ================================================================
#  第二部分: TXT 表格输出
# ================================================================

def export_txt_table(animations, output_path=None, precision=4):
    """
    导出逐帧 3D 关节坐标 TXT 表格

    格式:
    frame | person | joint_id | joint_name | x(m) | y(m) | z(m)
    ----- | ------ | -------- | ---------- | ---- | ---- | ----
       0  |    0   |    0     |   nose     | 0.12 | 0.34 | 1.56
       ...
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "skeletons_3d.txt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fmt = f"{{:>{precision + 6}.{precision}f}}"

    with open(output_path, "w", encoding="utf-8") as f:
        # 表头
        header = (
            f"{'frame':>7s} | {'person':>6s} | {'joint_id':>8s} | "
            f"{'joint_name':>15s} | {'x(m)':>{precision+6}s} | "
            f"{'y(m)':>{precision+6}s} | {'z(m)':>{precision+6}s}"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for person_id, seq in animations.items():
            T = seq.shape[0]
            for t in range(T):
                for j in range(NUM_JOINTS):
                    x, y, z = seq[t, j]
                    name = COCO_KEYPOINT_NAMES[j]

                    if np.isnan(x):
                        x_str = f"{'NaN':>{precision+6}s}"
                        y_str = f"{'NaN':>{precision+6}s}"
                        z_str = f"{'NaN':>{precision+6}s}"
                    else:
                        x_str = fmt.format(x)
                        y_str = fmt.format(y)
                        z_str = fmt.format(z)

                    line = (
                        f"{t:7d} | {person_id:6d} | {j:8d} | "
                        f"{name:>15s} | {x_str} | {y_str} | {z_str}"
                    )
                    f.write(line + "\n")

            # 人与人之间空一行
            f.write("\n")

    file_size = os.path.getsize(output_path)
    print(f"TXT 表格已保存: {output_path} ({file_size / 1024:.1f} KB)")

    return output_path


def export_txt_summary(animations, output_path=None, precision=4):
    """
    导出简洁版 TXT: 每行一帧, 所有关节坐标平铺

    格式:
    frame | person | x_nose y_nose z_nose | x_leye y_leye z_leye | ...
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "skeletons_3d_summary.txt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fmt = f"{{:.{precision}f}}"

    with open(output_path, "w", encoding="utf-8") as f:
        # 表头
        header = "frame | person"
        for name in COCO_KEYPOINT_NAMES:
            header += f" | x_{name} y_{name} z_{name}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for person_id, seq in animations.items():
            T = seq.shape[0]
            for t in range(T):
                line = f"{t:5d} | {person_id:6d}"
                for j in range(NUM_JOINTS):
                    x, y, z = seq[t, j]
                    if np.isnan(x):
                        line += " | NaN NaN NaN"
                    else:
                        line += f" | {fmt.format(x)} {fmt.format(y)} {fmt.format(z)}"
                f.write(line + "\n")
            f.write("\n")

    print(f"简洁表格已保存: {output_path}")
    return output_path


# ================================================================
#  第三部分: 3D 骨架动画
# ================================================================

def create_3d_animation(animations, output_dir=None, fps=30, dpi=100,
                        rotate=True, elevation=20, azimuth_speed=1.0):
    """
    生成 3D 骨架旋转动画 MP4

    参数:
        animations:    {person_id: (T, 17, 3)}
        output_dir:    输出目录
        fps:           帧率
        dpi:           分辨率
        rotate:        是否旋转视角
        elevation:     初始仰角
        azimuth_speed: 旋转速度 (度/秒)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for person_id, seq in animations.items():
        T = seq.shape[0]
        if T == 0:
            continue

        # 平滑
        seq_display = smooth_sequence(seq, SG_WINDOW, SG_ORDER)

        # 计算坐标范围 (使用非 NaN 值)
        valid = ~np.isnan(seq_display).any(axis=2)
        if valid.sum() == 0:
            continue

        all_pts = seq_display[valid]
        center = np.mean(all_pts, axis=0)
        max_range = np.max(np.ptp(all_pts, axis=0)) / 1.8

        # 创建图形
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        def update(frame_idx):
            ax.clear()

            skeleton = seq_display[frame_idx]

            # 绘制骨架
            for bone_idx, (p_idx, c_idx) in enumerate(COCO_SKELETON):
                if np.isnan(skeleton[p_idx]).any() or np.isnan(skeleton[c_idx]).any():
                    continue
                ax.plot(
                    [skeleton[p_idx, 0], skeleton[c_idx, 0]],
                    [skeleton[p_idx, 1], skeleton[c_idx, 1]],
                    [skeleton[p_idx, 2], skeleton[c_idx, 2]],
                    color=SKELETON_COLORS[bone_idx],
                    linewidth=3,
                )

            # 绘制关节点
            valid_mask = ~np.isnan(skeleton).any(axis=1)
            if valid_mask.sum() > 0:
                ax.scatter(
                    skeleton[valid_mask, 0],
                    skeleton[valid_mask, 1],
                    skeleton[valid_mask, 2],
                    c="white", s=40, edgecolors="black", linewidths=0.5,
                )

            # 坐标轴
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title(
                f"Person {person_id}  |  Frame {frame_idx}/{T}  |  "
                f"Time {frame_idx/fps:.2f}s",
                fontsize=12,
            )

            # 旋转视角
            if rotate:
                azimuth = frame_idx * azimuth_speed / fps
                ax.view_init(elev=elevation, azim=azimuth)
            else:
                ax.view_init(elev=elevation, azim=-90)

        # 渲染动画
        output_path = os.path.join(output_dir, f"person_{person_id}_3d_anim.mp4")

        print(f"渲染 Person {person_id} 动画 ({T} 帧)...")
        ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps)

        writer = FFMpegWriter(fps=fps, metadata={"title": f"Person {person_id} 3D Skeleton"})
        ani.save(output_path, writer=writer, dpi=dpi)
        plt.close(fig)

        file_size = os.path.getsize(output_path)
        print(f"  动画已保存: {output_path} ({file_size / 1024 / 1024:.1f} MB)")

    return output_dir


def create_multi_view_animation(animations, output_dir=None, fps=30):
    """
    生成多视角对比动画: 正面 + 侧面 + 俯视 三视图
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for person_id, seq in animations.items():
        T = seq.shape[0]
        if T == 0:
            continue

        seq_display = smooth_sequence(seq, SG_WINDOW, SG_ORDER)
        valid = ~np.isnan(seq_display).any(axis=2)
        if valid.sum() == 0:
            continue

        all_pts = seq_display[valid]
        center = np.mean(all_pts, axis=0)
        max_range = np.max(np.ptp(all_pts, axis=0)) / 1.8

        fig = plt.figure(figsize=(18, 6))
        views = [
            (131, "正面 (Front)", 90, 0),
            (132, "侧面 (Side)", 0, 0),
            (133, "俯视 (Top)", 90, -90),
        ]

        def update(frame_idx):
            skeleton = seq_display[frame_idx]
            for subplot_idx, title, elev, azim in views:
                ax = fig.add_subplot(subplot_idx, projection="3d")
                ax.clear()

                for bone_idx, (p_idx, c_idx) in enumerate(COCO_SKELETON):
                    if np.isnan(skeleton[p_idx]).any() or np.isnan(skeleton[c_idx]).any():
                        continue
                    ax.plot(
                        [skeleton[p_idx, 0], skeleton[c_idx, 0]],
                        [skeleton[p_idx, 1], skeleton[c_idx, 1]],
                        [skeleton[p_idx, 2], skeleton[c_idx, 2]],
                        color=SKELETON_COLORS[bone_idx], linewidth=2,
                    )

                valid_mask = ~np.isnan(skeleton).any(axis=1)
                if valid_mask.sum() > 0:
                    ax.scatter(
                        skeleton[valid_mask, 0],
                        skeleton[valid_mask, 1],
                        skeleton[valid_mask, 2],
                        c="white", s=20, edgecolors="black", linewidths=0.3,
                    )

                ax.set_xlim(center[0] - max_range, center[0] + max_range)
                ax.set_ylim(center[1] - max_range, center[1] + max_range)
                ax.set_zlim(center[2] - max_range, center[2] + max_range)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(title, fontsize=10)
                ax.view_init(elev=elev, azim=azim)

        output_path = os.path.join(output_dir, f"person_{person_id}_3d_multiview.mp4")
        print(f"渲染 Person {person_id} 三视图动画...")
        ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps)
        writer = FFMpegWriter(fps=fps)
        ani.save(output_path, writer=writer, dpi=100)
        plt.close(fig)

        print(f"  动画已保存: {output_path}")

    return output_dir


# ================================================================
#  第四部分: 统计报告
# ================================================================

def print_statistics(animations):
    """打印重建质量统计"""
    print("\n" + "=" * 60)
    print("重建质量统计")
    print("=" * 60)

    for person_id, seq in animations.items():
        T = seq.shape[0]
        print(f"\n--- Person {person_id} ---")
        print(f"  总帧数: {T}")

        nan_count = np.isnan(seq).sum()
        total = seq.size
        print(f"  NaN 比例: {nan_count}/{total} = {nan_count/total*100:.1f}%")

        # 每个关节的 NaN 比例
        print("  各关节完整率:")
        for j, name in enumerate(COCO_KEYPOINT_NAMES):
            joint_nan = np.isnan(seq[:, j]).any(axis=1).sum()
            rate = (T - joint_nan) / T * 100
            bar = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
            print(f"    {name:>15s}: {bar} {rate:5.1f}%")

        # 关节点范围
        non_nan = seq[~np.isnan(seq).any(axis=2)]
        if len(non_nan) > 0:
            p_min = non_nan.min(axis=0)
            p_max = non_nan.max(axis=0)
            print(f"\n  整体范围 (m):")
            print(f"    X: [{p_min[0]:.3f}, {p_max[0]:.3f}]  范围: {p_max[0]-p_min[0]:.3f}")
            print(f"    Y: [{p_min[1]:.3f}, {p_max[1]:.3f}]  范围: {p_max[1]-p_min[1]:.3f}")
            print(f"    Z: [{p_min[2]:.3f}, {p_max[2]:.3f}]  范围: {p_max[2]-p_min[2]:.3f}")


# ================================================================
#  第五部分: 一站式后处理 + 导出
# ================================================================

def postprocess_and_export(animations, output_dir=None, do_smooth=True):
    """
    一站式: 后处理 + TXT 导出 + 动画渲染 + 统计

    参数:
        animations: {person_id: (T, 17, 3)}
        output_dir: 输出目录
        do_smooth:  是否进行平滑处理
    返回:
        animations_smooth: 平滑后的动画数据
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("后处理与结果导出")
    print("=" * 60)

    # 1. 平滑
    animations_smooth = {}
    if do_smooth and SG_WINDOW > 0:
        print(f"\n[1/5] Savitzky-Golay 平滑 (窗口={SG_WINDOW}, 阶数={SG_ORDER})...")
        for pid, seq in animations.items():
            animations_smooth[pid] = smooth_sequence(seq, SG_WINDOW, SG_ORDER)
    else:
        print("\n[1/5] 跳过平滑")
        for pid, seq in animations.items():
            animations_smooth[pid] = interpolate_missing(seq)

    # 2. TXT 详细表格 (原始数据)
    print("\n[2/5] 导出原始 TXT 表格...")
    export_txt_table(animations, os.path.join(output_dir, "skeletons_3d.txt"))

    # 3. TXT 详细表格 (平滑后)
    if do_smooth:
        print("\n[3/5] 导出平滑后 TXT 表格...")
        export_txt_table(
            animations_smooth,
            os.path.join(output_dir, "skeletons_3d_smooth.txt")
        )

        # 简洁版
        export_txt_summary(
            animations_smooth,
            os.path.join(output_dir, "skeletons_3d_smooth_summary.txt")
        )

    # 4. 3D 动画
    print("\n[4/5] 生成 3D 骨架动画...")
    create_3d_animation(
        animations_smooth, output_dir,
        fps=ANIM_FPS, dpi=ANIM_DPI,
        rotate=ANIM_ROTATE, elevation=ANIM_ELEVATION,
        azimuth_speed=ANIM_AZIMUTH_SPEED,
    )

    # 可选: 三视图
    # create_multi_view_animation(animations_smooth, output_dir, fps=ANIM_FPS)

    # 5. 统计
    print("\n[5/5] 质量统计...")
    print_statistics(animations)

    return animations_smooth


if __name__ == "__main__":
    # 测试: 从 npz 加载并导出
    npz_path = os.path.join(OUTPUT_DIR, "skeletons_3d.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        animations = data["animations"].item()
        postprocess_and_export(animations)
    else:
        print(f"未找到 3D 骨架文件: {npz_path}")
