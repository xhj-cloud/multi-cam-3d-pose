"""
步骤 3: RTMPose 2D 人体关键点检测
 - 人体检测 (YOLOX) -> 姿态估计 (RTMPose)
 - 支持单人和多人场景
 - 跳帧检测优化
 - 结果可视化
 - 使用 rtmlib + ONNX Runtime (兼容 macOS, 无需 mmcv/mmdet/mmpose)
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from config import (
    KPT_CONF_THRESHOLD, DET_CONF_THRESHOLD,
    POSE_MODEL_INPUT_SIZE, BBOX_TIGHTEN_RATIO,
    SKIP_FRAME_INTERVAL, NUM_CAMERAS, CAMERA_NAMES,
    OUTPUT_DIR, SAVE_DET_VIS,
)


# ================================================================
#  COCO 17 关键点定义及骨架连接
# ================================================================

COCO_KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 头部
    (5, 6),                                  # 肩膀
    (5, 7), (7, 9), (6, 8), (8, 10),       # 手臂
    (5, 11), (6, 12), (11, 12),             # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16), # 腿部
]

COCO_SKELETON_COLORS = [
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),  # 头 绿
    (255, 0, 0),                                            # 肩 蓝
    (255, 128, 0), (255, 128, 0), (0, 128, 255), (0, 128, 255),  # 臂 橙/青
    (255, 255, 0), (255, 255, 0), (255, 255, 0),            # 躯干 青
    (255, 0, 255), (255, 0, 255), (128, 0, 128), (128, 0, 128),  # 腿 紫
]

LIMB_COLORS_HEX = [
    "#00FF00", "#00FF00", "#00FF00", "#00FF00",
    "#FF0000",
    "#FF8000", "#FF8000", "#0080FF", "#0080FF",
    "#FFFF00", "#FFFF00", "#FFFF00",
    "#FF00FF", "#FF00FF", "#800080", "#800080",
]


# ================================================================
#  运行时初始化
# ================================================================

def _resolve_device():
    """
    解析推理设备
    对于 rtmlib + ONNX Runtime, 统一使用 CPU 模式
    (ONNX Runtime 在 macOS 上通过 CoreML 加速, 性能已足够)
    """
    print("  推理后端: ONNX Runtime (CPU)")
    return "cpu"


# ================================================================
#  模型初始化 (使用 rtmlib, 不依赖 mmcv/mmdet/mmpose)
# ================================================================

def init_models():
    """
    初始化人体检测器和姿态估计器

    使用 rtmlib 加载 YOLOX (检测) + RTMPose (姿态估计) ONNX 模型
    通过 ONNX Runtime 推理, 兼容 macOS/Linux/Windows

    返回:
        detector:   YOLOX 人体检测模型
        pose_model: RTMPose 姿态估计模型
    """
    from rtmlib import YOLOX, RTMPose

    device = _resolve_device()

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')

    det_path = os.path.join(checkpoint_dir, 'yolox_m.onnx')
    pose_path = os.path.join(checkpoint_dir, 'rtmpose_m_body.pth')

    print(f"\n初始化模型 (设备: {device})...")
    print(f"  检测器:  {det_path}")
    print(f"  姿态估计: {pose_path}")

    detector = YOLOX(
        det_path,
        model_input_size=(640, 640),
        backend='onnxruntime',
        device=device,
    )

    pose_model = RTMPose(
        pose_path,
        model_input_size=POSE_MODEL_INPUT_SIZE,
        to_openpose=False,
        backend='onnxruntime',
        device=device,
    )

    print(f"  模型已加载 (rtmlib + ONNX Runtime)")
    return detector, pose_model


# ================================================================
#  单帧推理
# ================================================================

def detect_pose_single(detector, pose_model, frame):
    """
    对单个图像帧进行姿态估计 (单人场景, 全图推理)

    rtmlib 实现: 直接使用整张图像作为检测框
    """
    h, w = frame.shape[:2]
    bbox = np.array([[0, 0, w, h]], dtype=np.float32)

    keypoints, scores = pose_model(frame, bboxes=bbox)

    if len(keypoints) == 0:
        return [], [], []

    kpts = np.array(keypoints[0], dtype=np.float32)
    sc = np.array(scores[0], dtype=np.float32)

    return [kpts], [sc], [bbox[0]]


def detect_pose_multi(detector, pose_model, frame, prev_bboxes=None):
    """
    多人场景: 先做人体检测, 再对每个人做姿态估计

    参数:
        detector:    YOLOX 人体检测器 (rtmlib)
        pose_model:  RTMPose 姿态估计器 (rtmlib)
        frame:       BGR 图像 (H, W, 3)
        prev_bboxes: 上一帧的检测框 (用于跳帧模式), None 表示重新检测
    返回:
        person_keypoints: [(17, 2), ...]  每个人的关键点坐标
        person_scores:    [(17,), ...]    每个人的置信度
        bboxes:           [(4,), ...]     检测框 [x1, y1, x2, y2]
    """
    # Step 1: 人体检测 (YOLOX)
    if prev_bboxes is None or len(prev_bboxes) == 0:
        det_result = detector(frame)
        # YOLOX 返回 (N, 4): [x1, y1, x2, y2] (不含置信度)
        if not isinstance(det_result, np.ndarray) or det_result.shape[0] == 0:
            return [], [], []
        bboxes_raw = np.array(det_result, dtype=np.float32)

        # 收紧检测框: 四周各收 BBOX_TIGHTEN_RATIO 比例
        if BBOX_TIGHTEN_RATIO > 0:
            bboxes = []
            h, w = frame.shape[:2]
            for b in bboxes_raw:
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                shrink_w = bw * BBOX_TIGHTEN_RATIO
                shrink_h = bh * BBOX_TIGHTEN_RATIO
                b_new = np.array([
                    max(0, b[0] + shrink_w),
                    max(0, b[1] + shrink_h),
                    min(w, b[2] - shrink_w),
                    min(h, b[3] - shrink_h),
                ], dtype=np.float32)
                bboxes.append(b_new)
            bboxes = np.array(bboxes)
        else:
            bboxes = bboxes_raw
    else:
        bboxes = np.array(prev_bboxes, dtype=np.float32)

    # Step 2: 姿态估计 (RTMPose)
    keypoints, scores = pose_model(frame, bboxes=bboxes)

    person_keypoints = []
    person_scores = []
    valid_bboxes = []

    for i in range(len(keypoints)):
        kpts = np.array(keypoints[i], dtype=np.float32)   # (17, 2)
        sc = np.array(scores[i], dtype=np.float32)         # (17,)

        # 过滤低置信度关键点
        kpts[sc < KPT_CONF_THRESHOLD] = 0.0

        # 过滤检测框内没有有效关键点的人
        if np.sum(sc >= KPT_CONF_THRESHOLD) < 3:
            continue

        person_keypoints.append(kpts)
        person_scores.append(sc)
        valid_bboxes.append(bboxes[i] if i < len(bboxes) else np.array([0, 0, 0, 0], dtype=np.float32))

    return person_keypoints, person_scores, valid_bboxes


# ================================================================
#  可视化
# ================================================================

def draw_skeleton(frame, keypoints, scores, bbox=None, thickness=2):
    """
    在图像上绘制骨架和关键点
    """
    vis = frame.copy()

    # 画骨架连接
    for bone_idx, (p_idx, c_idx) in enumerate(COCO_SKELETON):
        if (scores[p_idx] < KPT_CONF_THRESHOLD or
                scores[c_idx] < KPT_CONF_THRESHOLD):
            continue
        p1 = tuple(keypoints[p_idx].astype(int))
        p2 = tuple(keypoints[c_idx].astype(int))
        color = COCO_SKELETON_COLORS[bone_idx]
        cv2.line(vis, p1, p2, color, thickness)

    # 画关键点
    for i, (kp, sc) in enumerate(zip(keypoints, scores)):
        if sc < KPT_CONF_THRESHOLD or np.all(kp == 0):
            continue
        cv2.circle(vis, tuple(kp.astype(int)), 4, (0, 255, 255), -1)

    # 画检测框
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return vis


# ================================================================
#  批量处理流水线
# ================================================================

def process_all_cameras(frame_dirs, total_frames, calib_data=None):
    """
    对所有摄像机、所有帧进行 2D 关键点检测

    参数:
        frame_dirs:   各摄像机帧目录列表 [Path, ...]
        total_frames: 总帧数
        calib_data:   标定数据 (可选, 用于可视化时标注摄像机名)
    返回:
        all_keypoints: {frame_idx: {cam_id: {"keypoints": [...], "scores": [...], "bboxes": [...]}}}
    """
    print("\n" + "=" * 60)
    print("RTMPose 2D 关键点检测")
    print("=" * 60)

    # 初始化模型
    detector, pose_model = init_models()

    # 输出目录
    vis_dir = Path(OUTPUT_DIR) / "detection_vis"
    if SAVE_DET_VIS:
        for i in range(NUM_CAMERAS):
            (vis_dir / f"camera_{i}").mkdir(parents=True, exist_ok=True)

    all_keypoints = {}

    # 跳帧检测状态
    prev_bboxes_per_cam = [None] * NUM_CAMERAS

    print(f"\n处理 {total_frames} 帧 × {NUM_CAMERAS} 台摄像机...")

    for frame_idx in tqdm(range(total_frames), desc="2D 检测"):
        all_keypoints[str(frame_idx)] = {}   # 统一使用字符串 key（与 JSON/Triangulation 一致）

        for cam_id in range(NUM_CAMERAS):
            # 加载帧
            fname = frame_dirs[cam_id] / f"frame_{frame_idx:06d}.jpg"
            frame = cv2.imread(str(fname))
            if frame is None:
                # 填充空结果
                all_keypoints[str(frame_idx)][cam_id] = {
                    "keypoints": [], "scores": [], "bboxes": []
                }
                continue

            # 跳帧逻辑
            use_prev_bbox = (
                SKIP_FRAME_INTERVAL > 1 and
                frame_idx % SKIP_FRAME_INTERVAL != 0 and
                prev_bboxes_per_cam[cam_id] is not None
            )

            # 推理
            kpts, scores, bboxes = detect_pose_multi(
                detector, pose_model, frame,
                prev_bboxes=prev_bboxes_per_cam[cam_id] if use_prev_bbox else None,
            )

            # 更新跳帧缓存
            if not use_prev_bbox:
                prev_bboxes_per_cam[cam_id] = bboxes if bboxes else None

            # 转成可序列化的列表
            all_keypoints[str(frame_idx)][cam_id] = {
                "keypoints": [k.tolist() for k in kpts],
                "scores": [s.tolist() for s in scores],
                "bboxes": [b.tolist() for b in bboxes],
                "num_persons": len(kpts),
            }

            # 可视化 (每 30 帧保存一张, 节省空间)
            if SAVE_DET_VIS and frame_idx % 30 == 0:
                vis_frame = frame.copy()
                for k, s, b in zip(kpts, scores, bboxes):
                    vis_frame = draw_skeleton(vis_frame, k, s, b)
                # 标注摄像机名
                cv2.putText(vis_frame,
                            f"{CAMERA_NAMES[cam_id]} | Frame {frame_idx} | {len(kpts)} person(s)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out_path = vis_dir / f"camera_{cam_id}" / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), vis_frame)

    # ---- 导出每台摄像机独立的 2D 关键点 TXT ----
    print("\n导出各摄像机 2D 关键点坐标...")
    for cam_id in range(NUM_CAMERAS):
        txt_path = os.path.join(OUTPUT_DIR, f"camera_{cam_id}_2d_keypoints.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Camera {cam_id} ({CAMERA_NAMES[cam_id]}) — 2D 关键点坐标 (像素)\n")
            f.write(f"# 格式: frame | person | joint_id | joint_name | x(px) | y(px) | confidence\n")
            f.write("-" * 90 + "\n")
            for frame_idx_str, frame_data in sorted(all_keypoints.items(),
                                                     key=lambda x: int(x[0])):
                # cam_id 按 int 存储, 兼容 str
                if cam_id in frame_data:
                    data = frame_data[cam_id]
                elif str(cam_id) in frame_data:
                    data = frame_data[str(cam_id)]
                else:
                    continue
                for p_idx, (kpts, scores) in enumerate(
                    zip(data["keypoints"], data["scores"])
                ):
                    for j_idx in range(len(kpts)):
                        x, y = kpts[j_idx]
                        conf = scores[j_idx]
                        name = COCO_KEYPOINT_NAMES[j_idx] if j_idx < len(COCO_KEYPOINT_NAMES) else f"joint_{j_idx}"
                        f.write(f"{int(frame_idx_str):6d} | {p_idx:3d} | {j_idx:2d} | "
                                f"{name:>15s} | {x:8.1f} | {y:8.1f} | {conf:.4f}\n")
        fsize_kb = os.path.getsize(txt_path) / 1024
        print(f"  {txt_path} ({fsize_kb:.1f} KB)")

    # ---- 保存 JSON (供 3D 重建使用) ----
    result_path = os.path.join(OUTPUT_DIR, "all_2d_keypoints.json")
    with open(result_path, "w") as f:
        json.dump(all_keypoints, f, ensure_ascii=False)

    # 统计信息
    total_detections = sum(
        data[cam_id]["num_persons"]
        for frame_idx, data in all_keypoints.items()
        for cam_id in range(NUM_CAMERAS)
    )
    print(f"\n检测完成:")
    print(f"  总帧数: {total_frames}")
    print(f"  总检测人次: {total_detections}")
    print(f"  平均每帧每人: {total_detections / (total_frames * NUM_CAMERAS):.2f}")
    print(f"  JSON 结果: {result_path}")
    if SAVE_DET_VIS:
        print(f"  可视化: {vis_dir}")

    return all_keypoints


# ================================================================
#  直接对视频列表处理 (无需预提取帧)
# ================================================================

def process_videos_direct(video_paths, calib_data=None, max_frames=None):
    """
    直接对视频文件进行 2D 检测 (跳过预提取帧步骤)
    适合磁盘空间有限或不需要保留中间帧的场景

    参数:
        video_paths: 视频文件路径列表
        max_frames:  最大处理帧数 (None = 全部)
    """
    detector, pose_model = init_models()

    caps = [cv2.VideoCapture(p) for p in video_paths]
    for cap in caps:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频")

    all_keypoints = {}
    frame_idx = 0
    prev_bboxes_per_cam = [None] * len(caps)

    while True:
        if max_frames and frame_idx >= max_frames:
            break

        frames_ok = True
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            frames.append(frame)
            if not ret:
                frames_ok = False
                break

        if not frames_ok:
            break

        all_keypoints[str(frame_idx)] = {}
        for cam_id, frame in enumerate(frames):
            use_prev = (
                SKIP_FRAME_INTERVAL > 1 and
                frame_idx % SKIP_FRAME_INTERVAL != 0 and
                prev_bboxes_per_cam[cam_id] is not None
            )

            kpts, scores, bboxes = detect_pose_multi(
                detector, pose_model, frame,
                prev_bboxes=prev_bboxes_per_cam[cam_id] if use_prev else None,
            )

            if not use_prev:
                prev_bboxes_per_cam[cam_id] = bboxes if bboxes else None

            all_keypoints[str(frame_idx)][cam_id] = {
                "keypoints": [k.tolist() for k in kpts],
                "scores": [s.tolist() for s in scores],
                "bboxes": [b.tolist() for b in bboxes],
                "num_persons": len(kpts),
            }

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  已处理 {frame_idx} 帧")

    for cap in caps:
        cap.release()

    return all_keypoints, frame_idx


if __name__ == "__main__":
    # 测试: 直接处理视频
    from config import VIDEO_PATHS
    kpts, n_frames = process_videos_direct(VIDEO_PATHS, max_frames=10)
    print(f"\n测试完成: {n_frames} 帧")
