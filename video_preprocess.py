"""
步骤 2: 视频预处理与帧同步
 - 自动检测多机位时间偏移 (运动峰值 / 亮度突变 / 时间戳)
 - 不同帧率自动统一
 - 按实际时间对齐提取同步帧
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from config import (
    VIDEO_PATHS, NUM_CAMERAS, CAMERA_NAMES,
    SYNC_METHOD, SYNC_FRAME_OFFSETS, TARGET_FPS,
    FRAME_FORMAT, FRAME_QUALITY, OUTPUT_DIR, SAVE_FRAMES,
)


# ================================================================
#  第一部分: 视频信息读取 & 一致性检查
# ================================================================

def get_video_info(video_paths):
    """读取所有视频的基本信息"""
    info_list = []
    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        info_list.append({
            "id": i,
            "name": CAMERA_NAMES[i],
            "path": path,
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_sec": duration,
        })
        cap.release()

        print(f"  {CAMERA_NAMES[i]}: {width}×{height}, {fps:.2f} FPS, "
              f"{total_frames} 帧, {duration:.1f}s")

    return info_list


def check_video_consistency(info_list):
    """检查多摄像机视频参数一致性"""
    issues = []

    fps_set = set(info["fps"] for info in info_list)
    if len(fps_set) > 1:
        issues.append(f"帧率不一致: {fps_set} — 将自动统一到最低帧率")

    res_set = set((info["width"], info["height"]) for info in info_list)
    if len(res_set) > 1:
        issues.append(f"分辨率不一致: {res_set}")

    max_dur = max(info["duration_sec"] for info in info_list)
    min_dur = min(info["duration_sec"] for info in info_list)
    if max_dur - min_dur > 1.0:
        issues.append(f"视频时长差异较大: {min_dur:.1f}s ~ {max_dur:.1f}s")

    if issues:
        print("\n  [⚠ 视频参数不一致]")
        for iss in issues:
            print(f"    - {iss}")
    else:
        print("  [✓] 所有视频参数一致")

    return issues


# ================================================================
#  第二部分: 自动时间对齐 (核心新增)
# ================================================================

def _sample_frames(cap, max_samples=300):
    """
    均匀采样视频帧用于同步分析 (避免遍历全部帧)
    返回: [(frame_idx, timestamp_sec, gray_image), ...]
    """
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    if total <= max_samples:
        indices = list(range(total))
    else:
        step = total // max_samples
        indices = list(range(0, total, step))[:max_samples]

    samples = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ts = idx / fps
        samples.append((idx, ts, gray))

    return samples, fps


def detect_sync_by_motion(video_paths):
    """
    方法一: 运动峰值对齐
    原理: 同一运动事件在各摄像机中产生的帧间变化峰值对应同一时刻
    适用: 运动员进入画面、起跳等剧烈动作
    """
    print("\n--- 方法: 运动峰值检测 ---")

    motion_curves = []
    fps_list = []

    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        samples, fps = _sample_frames(cap, max_samples=300)
        fps_list.append(fps)

        # 计算帧间运动量 (相邻采样帧的像素差)
        motions = []
        timestamps = []
        for k in range(1, len(samples)):
            _, t_prev, gray_prev = samples[k - 1]
            _, t_curr, gray_curr = samples[k]
            diff = cv2.absdiff(gray_prev, gray_curr)
            motion = np.mean(diff)  # 平均像素变化
            motions.append(motion)
            timestamps.append(t_curr)

        motion_curves.append({"timestamps": timestamps, "motions": motions})
        cap.release()
        print(f"  {CAMERA_NAMES[i]}: {len(motions)} 个运动采样点")

    # 找到每台摄像机运动峰值的时刻
    peak_times = []
    for curve in motion_curves:
        motions = np.array(curve["motions"])
        timestamps = np.array(curve["timestamps"])
        if len(motions) < 2:
            peak_times.append(0.0)
            continue
        peak_idx = np.argmax(motions)
        peak_times.append(timestamps[peak_idx])

    # 以 Camera 0 的峰值为基准, 计算偏移
    ref_time = peak_times[0]
    offsets_sec = []
    for i, t in enumerate(peak_times):
        offset = t - ref_time
        offsets_sec.append(offset)
        print(f"  {CAMERA_NAMES[i]}: 运动峰值 @ {t:.2f}s, 相对偏移 = {offset:+.3f}s")

    fps_common = min(fps_list) if TARGET_FPS is None else TARGET_FPS
    offsets_frames = [int(round(off * fps_common)) for off in offsets_sec]

    return offsets_frames, offsets_sec, fps_common, "motion"


def detect_sync_by_brightness(video_paths):
    """
    方法二: 亮度突变对齐
    原理: 用闪光灯/拍手/灯光变化作为同步信号
    适用: 录制开始时有明显的亮度变化
    """
    print("\n--- 方法: 亮度突变检测 ---")

    peak_times = []
    fps_list = []

    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        samples, fps = _sample_frames(cap, max_samples=300)
        fps_list.append(fps)

        brightnesses = []
        timestamps = []
        for _, ts, gray in samples:
            brightnesses.append(np.mean(gray))
            timestamps.append(ts)

        # 找亮度变化最大的瞬间 (一阶差分的绝对值最大)
        if len(brightnesses) >= 2:
            diffs = np.abs(np.diff(brightnesses))
            peak_idx = np.argmax(diffs)
            peak_times.append(timestamps[peak_idx + 1])
        else:
            peak_times.append(0.0)

        cap.release()
        print(f"  {CAMERA_NAMES[i]}: 亮度突变 @ {peak_times[-1]:.2f}s")

    ref_time = peak_times[0]
    offsets_sec = [t - ref_time for t in peak_times]
    fps_common = min(fps_list) if TARGET_FPS is None else TARGET_FPS
    offsets_frames = [int(round(off * fps_common)) for off in offsets_sec]

    return offsets_frames, offsets_sec, fps_common, "brightness"


def detect_sync_by_timestamp(video_paths, sample_interval=30):
    """
    方法三: 基于视频时间戳对齐
    每 N 帧读取一次 cv2.CAP_PROP_POS_MSEC, 取最早公共起始时间
    最精确但依赖视频文件内嵌时间戳的准确性
    """
    print("\n--- 方法: 时间戳对齐 ---")

    caps = [cv2.VideoCapture(p) for p in video_paths]
    fps_list = []

    # 记录每台摄像机的时间戳序列
    time_series = []
    for i, cap in enumerate(caps):
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_list.append(fps)

        timestamps = []
        frame_idx = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamps.append((frame_idx, t_ms / 1000.0))
            frame_idx += 1

        time_series.append(timestamps)
        print(f"  {CAMERA_NAMES[i]}: {len(timestamps)} 时间戳采样点, "
              f"范围 {timestamps[0][1]:.1f}s ~ {timestamps[-1][1]:.1f}s")

    for cap in caps:
        cap.release()

    # 找到所有摄像机时间轴的最大公共起点
    start_times = [(ts[0][0], ts[0][1]) for ts in time_series]  # (frame, second)
    # 取最晚开始的作为对齐起点
    latest_start_sec = max(ts[0][1] for ts in time_series)

    offsets_sec = []
    for ts in time_series:
        # 该摄像机需要跳过多少秒才能对齐到 latest_start
        skip_sec = latest_start_sec - ts[0][1]
        offsets_sec.append(skip_sec)

    fps_common = min(fps_list) if TARGET_FPS is None else TARGET_FPS
    offsets_frames = [int(round(off * fps_common)) for off in offsets_sec]

    print(f"  对齐起点: {latest_start_sec:.1f}s (最晚开始的摄像机时间)")
    for i in range(NUM_CAMERAS):
        print(f"  {CAMERA_NAMES[i]}: 偏移 = {offsets_sec[i]:.2f}s = {offsets_frames[i]} 帧")

    return offsets_frames, offsets_sec, fps_common, "timestamp"


def auto_detect_sync(video_paths, method="motion"):
    """
    自动检测多机位视频的同步偏移

    返回:
        offsets_frames: 每台摄像机需跳过的帧数 (正数=跳过, 负数=等待)
        offsets_sec:    每台摄像机的时间偏移 (秒)
        fps_common:     统一后的帧率
        method_used:    实际使用的方法
    """
    print("\n" + "─" * 50)
    print("自动时间对齐")
    print("─" * 50)

    if method == "motion":
        return detect_sync_by_motion(video_paths)
    elif method == "brightness":
        return detect_sync_by_brightness(video_paths)
    elif method == "timestamp":
        return detect_sync_by_timestamp(video_paths)
    elif method == "auto":
        # 自动选择: 优先时间戳 → 运动峰值 → 亮度突变
        for m in ["timestamp", "motion", "brightness"]:
            try:
                result = auto_detect_sync(video_paths, method=m)
                return result
            except Exception as e:
                print(f"  {m} 失败: {e}")
        raise RuntimeError("所有自动同步方法均失败, 请使用 manual 模式指定 SYNC_FRAME_OFFSETS")
    else:
        raise ValueError(f"未知同步方法: {method}")


# ================================================================
#  第三部分: 帧提取 (支持时间对齐 & 不同帧率)
# ================================================================

def extract_frames(video_paths, output_root, sync_offsets=None,
                   target_fps=None, save_images=True):
    """
    从视频中提取时间对齐的帧

    参数:
        video_paths:   视频路径列表
        output_root:   输出根目录
        sync_offsets:  每台摄像机的时间偏移 (帧数), None = 不做同步
        target_fps:    统一帧率, None = 自动取最慢 FPS
        save_images:   是否保存帧图像
    返回:
        frame_dirs:    帧目录列表
        total_frames:  对齐后总帧数
        fps_used:      实际帧率
    """
    if sync_offsets is None:
        sync_offsets = [0] * len(video_paths)

    output_root = Path(output_root) / "frames"
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("视频帧提取 (时间对齐模式)")
    print("=" * 60)

    # 获取各视频 FPS
    caps_orig = [cv2.VideoCapture(p) for p in video_paths]
    fps_list = [cap.get(cv2.CAP_PROP_FPS) for cap in caps_orig]
    for cap in caps_orig:
        cap.release()

    # 确定统一帧率
    if target_fps is None:
        fps_used = min(fps_list)
        if len(set(fps_list)) > 1:
            print(f"\n  帧率不一致: {fps_list}, 统一为 {fps_used} FPS")
    else:
        fps_used = target_fps

    # ---- 计算每台摄像机的有效时间窗口 ----
    # 将所有偏移归一化: 正 = 跳过, 负 = 等待
    # 以最"早"的摄像机为基准
    min_offset = min(sync_offsets)
    normalized_offsets = [off - min_offset for off in sync_offsets]
    # normalized_offsets 中, 最小值 = 0, 表示该摄像机最先开始

    # ---- 重新打开视频 & 定位 ----
    caps = []
    frame_dirs = []

    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        caps.append(cap)

        cam_dir = output_root / f"camera_{i}"
        if save_images:
            cam_dir.mkdir(parents=True, exist_ok=True)
        frame_dirs.append(cam_dir)

        # 定位到对齐起点
        skip_frames = sync_offsets[i]
        if skip_frames < 0:
            # 该摄像机开始得晚 → 前 -skip_frames 帧输出黑帧
            print(f"  {CAMERA_NAMES[i]}: 前 {-skip_frames} 帧填充黑帧 "
                  f"(晚 {-skip_frames/fps_used:.2f}s 开始)")
        elif skip_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
            print(f"  {CAMERA_NAMES[i]}: 跳过前 {skip_frames} 帧 "
                  f"(差 {skip_frames/fps_used:.2f}s)")

    # ---- 逐帧提取 ----
    frame_idx = 0
    max_empty = max(0, -min(sync_offsets))  # 需要填充黑帧的数量
    black_frames_remaining = [max(0, -off) for off in sync_offsets]

    while True:
        all_done = True
        frames = []

        for i, cap in enumerate(caps):
            if black_frames_remaining[i] > 0:
                # 输出黑帧
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                black_frames_remaining[i] -= 1
                all_done = False
            else:
                ret, frame = cap.read()
                if not ret:
                    # 该视频已播完, 输出黑帧
                    if frames and i < len(frames):
                        h, w = frames[-1].shape[:2] if len(frames) > 0 else (1080, 1920)
                    else:
                        h, w = 1080, 1920
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    all_done = False

            frames.append(frame)

        if all_done and frame_idx > 0:
            break

        # 保存帧
        if save_images:
            for i, frame in enumerate(frames):
                fname = frame_dirs[i] / f"frame_{frame_idx:06d}.{FRAME_FORMAT}"
                if FRAME_FORMAT.lower() in ("jpg", "jpeg"):
                    cv2.imwrite(str(fname), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
                else:
                    cv2.imwrite(str(fname), frame)

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  已提取 {frame_idx} 帧...")

    for cap in caps:
        cap.release()

    print(f"\n  提取完成: 共 {frame_idx} 帧")
    print(f"  帧率: {fps_used:.2f} FPS  |  时长: {frame_idx / fps_used:.1f}s")
    print(f"  帧目录: {output_root}")

    return frame_dirs, frame_idx, fps_used


# ================================================================
#  第四部分: 加载函数
# ================================================================

def load_synced_frames(frame_dirs, frame_idx):
    """加载所有摄像机在指定帧索引的同步帧"""
    frames = []
    for d in frame_dirs:
        fname = d / f"frame_{frame_idx:06d}.{FRAME_FORMAT}"
        frame = cv2.imread(str(fname))
        if frame is None:
            if frames:
                h, w = frames[0].shape[:2]
            else:
                h, w = 1080, 1920
            frame = np.zeros((h, w, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


# ================================================================
#  第五部分: 主函数
# ================================================================

def run_preprocess():
    """
    完整视频预处理流程
    """
    print("\n" + "=" * 60)
    print("视频预处理")
    print("=" * 60)

    # 1. 读取视频信息
    print("\n[1/4] 读取视频信息...")
    info_list = get_video_info(VIDEO_PATHS)
    issues = check_video_consistency(info_list)

    # 2. 确定同步偏移
    print(f"\n[2/4] 时间对齐 (方法: {SYNC_METHOD})...")

    if SYNC_METHOD == "manual":
        offsets_frames = SYNC_FRAME_OFFSETS
        offsets_sec = [f / (info_list[0]["fps"] or 30) for f in offsets_frames]
        fps_used = min(info["fps"] for info in info_list) if TARGET_FPS is None else TARGET_FPS
        print(f"  手动偏移: {offsets_frames} 帧")
    else:
        offsets_frames, offsets_sec, fps_used, method_used = auto_detect_sync(
            VIDEO_PATHS, method=SYNC_METHOD
        )
        print(f"\n  检测到偏移 (帧): {offsets_frames}")
        print(f"  检测到偏移 (秒): {[f'{s:.3f}' for s in offsets_sec]}")

    # 3. 提取帧
    print(f"\n[3/4] 提取时间对齐帧 (FPS={fps_used:.1f})...")
    frame_dirs, total_frames, fps = extract_frames(
        VIDEO_PATHS,
        output_root=OUTPUT_DIR,
        sync_offsets=offsets_frames,
        target_fps=fps_used,
        save_images=SAVE_FRAMES,
    )

    # 4. 保存元信息
    print("\n[4/4] 保存元信息...")
    meta = {
        "num_cameras": NUM_CAMERAS,
        "total_frames": total_frames,
        "fps": fps,
        "sync_method": SYNC_METHOD,
        "sync_offsets_frames": offsets_frames,
        "sync_offsets_seconds": offsets_sec,
        "video_info": [
            {k: v for k, v in info.items() if k != "path"}
            for info in info_list
        ],
    }
    meta_path = os.path.join(OUTPUT_DIR, "preprocess_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  元信息已保存: {meta_path}")

    return frame_dirs, total_frames, fps, meta


if __name__ == "__main__":
    run_preprocess()
