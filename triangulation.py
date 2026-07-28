"""
步骤 4 & 5: 跨视图匹配 + 3D 三角测量重建

输入: 各摄像机 2D 关键点 + 标定参数
输出: 逐帧 3D 骨架坐标
"""

import os
import json
import numpy as np
from tqdm import tqdm

from config import (
    NUM_CAMERAS, CAMERA_NAMES,
    TRIANG_METHOD, RANSAC_ITERATIONS, RANSAC_THRESHOLD, MIN_VIEWS,
    OUTPUT_DIR,
)

# ---- 复用 pose_detection 的关键点名称 ----
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_JOINTS = len(COCO_KEYPOINT_NAMES)  # 17


# ================================================================
#  第一部分: 投影矩阵构建
# ================================================================

def load_calib(calib_json_path=None):
    """
    加载标定数据, 构建投影矩阵列表

    返回:
        P_list:    [(3,4), ...] 每台摄像机的投影矩阵
        K_list:    [(3,3), ...] 内参矩阵
        R_list:    [(3,3), ...] 旋转矩阵
        t_list:    [(3,1), ...] 平移向量
        calib:     dict 原始标定数据
    """
    import os
    if calib_json_path is None:
        calib_json_path = os.path.join(OUTPUT_DIR, "..", "calib", "multi_camera_calib.json")
        # 也尝试相对路径
        if not os.path.exists(calib_json_path):
            from config import CALIB_JSON_FILE
            calib_json_path = CALIB_JSON_FILE

    if not os.path.exists(calib_json_path):
        raise FileNotFoundError(
            f"标定文件不存在: {calib_json_path}\n"
            f"请先运行: python run_pipeline.py --mode calibrate"
        )

    with open(calib_json_path) as f:
        calib = json.load(f)

    P_list = []
    K_list, R_list, t_list = [], [], []

    for cam in calib["cameras"]:
        P = np.array(cam["P"], dtype=np.float64)    # (3,4)
        K = np.array(cam["K"], dtype=np.float64)    # (3,3)
        R = np.array(cam["R"], dtype=np.float64)    # (3,3)
        t = np.array(cam["t"], dtype=np.float64).reshape(3, 1)  # (3,1)
        P_list.append(P)
        K_list.append(K)
        R_list.append(R)
        t_list.append(t)

    return P_list, K_list, R_list, t_list, calib


# ================================================================
#  第二部分: DLT 三角测量
# ================================================================

def triangulate_dlt(kpts_2d, P_list, confidences=None):
    """
    DLT (Direct Linear Transformation) 三角测量

    从 N 个视角的 2D 坐标重建单个 3D 点

    参数:
        kpts_2d:     (N, 2) N 个视角的 2D 坐标
        P_list:      [P_0, P_1, ...] 投影矩阵
        confidences: (N,) 各视角置信度权重 (可选)
    返回:
        X: (3,) 3D 坐标
    """
    N = len(kpts_2d)
    A = np.zeros((2 * N, 4), dtype=np.float64)

    for i in range(N):
        x, y = kpts_2d[i]
        if x == 0 and y == 0:   # 无效点
            continue
        P = P_list[i]
        w = confidences[i] if confidences is not None else 1.0
        A[2 * i]     = w * (x * P[2] - P[0])
        A[2 * i + 1] = w * (y * P[2] - P[1])

    # SVD 求解 Ax = 0
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]          # 最小奇异值对应的右奇异向量
    X = X[:3] / X[3]    # 齐次坐标归一化

    return X


def triangulate_midpoint(kpts_2d, P_list, confidences=None):
    """
    中点三角测量 (速度快, 精度略低于 DLT)

    分别计算每对摄像机的 3D 点, 取加权平均
    """
    N = len(kpts_2d)
    points_3d = []
    weights = []

    for i in range(N):
        for j in range(i + 1, N):
            x1, y1 = kpts_2d[i]
            x2, y2 = kpts_2d[j]
            if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
                continue

            # 用 DLT 对这对视角做三角测量
            pt = triangulate_dlt(
                np.array([[x1, y1], [x2, y2]]),
                [P_list[i], P_list[j]]
            )

            # 权重: 两个视角置信度的乘积
            w = 1.0
            if confidences is not None:
                w = confidences[i] * confidences[j]
            points_3d.append(pt)
            weights.append(w)

    if not points_3d:
        return np.array([np.nan, np.nan, np.nan])

    points_3d = np.array(points_3d)
    weights = np.array(weights)

    # 加权平均
    if weights.sum() > 0:
        return np.average(points_3d, axis=0, weights=weights)
    return points_3d.mean(axis=0)


# ================================================================
#  第三部分: RANSAC 鲁棒三角测量
# ================================================================

def triangulate_ransac(kpts_2d, P_list, confidences=None,
                       n_iter=100, threshold=30.0):
    """
    带 RANSAC 的鲁棒三角测量, 自动剔除离群视角

    返回:
        X: (3,) 最优 3D 点
        inlier_mask: (N,) bool 内点标记
        n_inliers:   内点数量
    """
    N = len(kpts_2d)

    # 过滤有效视角
    valid_idx = [i for i in range(N)
                 if not (kpts_2d[i][0] == 0 and kpts_2d[i][1] == 0)]
    if len(valid_idx) < 2:
        return np.array([np.nan, np.nan, np.nan]), None, 0

    valid_kpts = kpts_2d[valid_idx]
    valid_P = [P_list[i] for i in valid_idx]

    best_inliers = 0
    best_point = None
    best_mask = None

    for _ in range(n_iter):
        if len(valid_idx) < 2:
            break

        # 随机选 2 个视角
        idxs = np.random.choice(len(valid_idx), size=2, replace=False)

        # 估算 3D 点
        candidate = triangulate_dlt(
            valid_kpts[idxs],
            [valid_P[i] for i in idxs]
        )

        # 检查所有视角的重投影误差
        inlier_mask = np.zeros(N, dtype=bool)
        for i in range(N):
            if kpts_2d[i][0] == 0 and kpts_2d[i][1] == 0:
                continue
            if confidences is not None and confidences[i] < 0.2:
                continue

            proj = P_list[i] @ np.append(candidate, 1.0)
            proj = proj[:2] / proj[2]
            error = np.linalg.norm(proj - kpts_2d[i])
            if error < threshold:
                inlier_mask[i] = True

        n = inlier_mask.sum()
        if n > best_inliers and n >= MIN_VIEWS:
            best_inliers = n
            best_mask = inlier_mask
            # 用所有内点重算
            best_point = triangulate_dlt(
                kpts_2d[best_mask],
                [P_list[i] for i in range(N) if best_mask[i]],
                confidences[best_mask] if confidences is not None else None,
            )

    return best_point, best_mask, best_inliers


# ================================================================
#  第四部分: 跨视图多人匹配
# ================================================================

def compute_fundamental_matrix(K1, R1, t1, K2, R2, t2):
    """计算两个摄像机之间的基础矩阵 F"""
    R_rel = R2 @ R1.T
    t_rel = t2.reshape(3, 1) - R_rel @ t1.reshape(3, 1)

    tx = np.array([
        [0, -t_rel[2, 0], t_rel[1, 0]],
        [t_rel[2, 0], 0, -t_rel[0, 0]],
        [-t_rel[1, 0], t_rel[0, 0], 0]
    ])

    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    F = K2_inv.T @ tx @ R_rel @ K1_inv
    return F


def epipolar_distance(kpt1, kpt2, F):
    """两点之间的 Sampson 极线距离"""
    p1 = np.array([kpt1[0], kpt1[1], 1.0])
    p2 = np.array([kpt2[0], kpt2[1], 1.0])

    l2 = F @ p1
    l1 = F.T @ p2

    num = (p2 @ l2) ** 2
    denom = l1[0]**2 + l1[1]**2 + l2[0]**2 + l2[1]**2

    if denom < 1e-10:
        return 1e9
    return num / denom


def match_persons(cam_kpts, K_list, R_list, t_list):
    """
    跨视图多人匹配: 以 Camera 0 为基准, 在其他视角中找到对应的人

    策略:
      - 所有摄像机都只有 1 人 → 直接配对 (单人场景)
      - 有多人 → 使用极线几何距离匹配
    """
    ref_persons = cam_kpts.get(0, [])
    if not ref_persons:
        return []

    # ---- 单人场景: 直接配对 ----
    all_single = all(
        len(cam_kpts.get(c, [])) == 1
        for c in range(NUM_CAMERAS)
        if c in cam_kpts
    )
    if all_single and len(ref_persons) == 1:
        match = [0]  # camera 0 person 0
        for cam_id in range(1, NUM_CAMERAS):
            match.append(0 if (cam_id in cam_kpts and cam_kpts[cam_id]) else -1)
        return [match]

    # ---- 多人场景: 极线匹配 ----
    F_matrices = {}
    for i in range(NUM_CAMERAS):
        for j in range(i + 1, NUM_CAMERAS):
            if i >= len(K_list) or j >= len(K_list):
                continue
            F_matrices[(i, j)] = compute_fundamental_matrix(
                K_list[i], R_list[i], t_list[i],
                K_list[j], R_list[j], t_list[j]
            )

    matches = []
    for p_idx, ref_kpts in enumerate(ref_persons):
        match = [p_idx]

        for cam_id in range(1, NUM_CAMERAS):
            if cam_id not in cam_kpts or not cam_kpts[cam_id]:
                match.append(-1)
                continue

            other_persons = cam_kpts[cam_id]
            best_idx = -1
            best_cost = float('inf')

            pair_key = (0, cam_id) if (0, cam_id) in F_matrices else (cam_id, 0)
            F = F_matrices[pair_key]

            for o_idx, other_kpts in enumerate(other_persons):
                total_cost = 0
                valid = 0
                for j in range(NUM_JOINTS):
                    if np.all(ref_kpts[j] == 0) or np.all(other_kpts[j] == 0):
                        continue
                    d = epipolar_distance(ref_kpts[j], other_kpts[j], F)
                    if d < 2000:  # 放宽阈值
                        total_cost += d
                        valid += 1

                if valid >= 2:  # 降低要求
                    avg_cost = total_cost / valid
                    if avg_cost < best_cost:
                        best_cost = avg_cost
                        best_idx = o_idx

            match.append(best_idx)
        matches.append(match)

    return matches


# ================================================================
#  第五部分: 完整骨架重建
# ================================================================

def triangulate_skeleton(kpts_per_cam, P_list, conf_per_cam=None):
    """
    对单个骨架 (17 个关节) 进行三角测量

    参数:
        kpts_per_cam:  (N_cam, 17, 2) 各摄像机中某人的关键点
        P_list:        [P_0, ...]
        conf_per_cam:  (N_cam, 17) 置信度 (可选)
    返回:
        skeleton_3d: (17, 3)
        reproj_errors: (17,) 每个关节的重投影误差
    """
    N_cam = kpts_per_cam.shape[0]
    skeleton_3d = np.full((NUM_JOINTS, 3), np.nan)
    reproj_errors = np.full(NUM_JOINTS, np.nan)

    # 过滤: 找出至少 MIN_VIEWS 个有效视角的关节
    for j in range(NUM_JOINTS):
        kpts_2d = kpts_per_cam[:, j, :]                       # (N_cam, 2)
        confs = conf_per_cam[:, j] if conf_per_cam is not None else None

        valid_mask = np.array([
            not (kpts_2d[i][0] == 0 and kpts_2d[i][1] == 0)
            for i in range(N_cam)
        ])

        if valid_mask.sum() < MIN_VIEWS:
            continue

        valid_kpts = kpts_2d[valid_mask]
        valid_P = [P_list[i] for i in range(N_cam) if valid_mask[i]]
        valid_conf = confs[valid_mask] if confs is not None else None

        if RANSAC_ITERATIONS > 0 and len(valid_kpts) > 2:
            pt, _, _ = triangulate_ransac(
                valid_kpts, valid_P, valid_conf,
                n_iter=RANSAC_ITERATIONS, threshold=RANSAC_THRESHOLD
            )
        elif TRIANG_METHOD == "midpoint":
            pt = triangulate_midpoint(valid_kpts, valid_P, valid_conf)
        else:
            pt = triangulate_dlt(valid_kpts, valid_P, valid_conf)

        skeleton_3d[j] = pt

        # 计算重投影误差
        errors = []
        for i in range(N_cam):
            if not valid_mask[i]:
                continue
            proj = P_list[i] @ np.append(pt, 1.0)
            proj = proj[:2] / proj[2]
            errors.append(np.linalg.norm(proj - kpts_2d[i]))
        if errors:
            reproj_errors[j] = np.mean(errors)

    return skeleton_3d, reproj_errors


# ================================================================
#  第六部分: 逐帧重建主函数
# ================================================================

def reconstruct_3d(all_keypoints, calib_data):
    """
    对所有帧、所有人进行 3D 重建

    参数:
        all_keypoints: {frame_idx: {cam_id: {"keypoints": [[...]], "scores": [[...]], ...}}}
        calib_data:    标定数据 dict (含 cameras[].P 等)
    返回:
        animations_3d: {person_id: [(T, 17, 3), ...]}
                      注意: 简化处理, 每帧的 person_id 可能不对齐,
                      实际使用时需要做时序追踪
    """
    print("\n" + "=" * 60)
    print("3D 三角测量重建")
    print("=" * 60)

    # 构建投影矩阵
    P_list = []
    K_list, R_list, t_list = [], [], []
    for cam in calib_data["cameras"]:
        P_list.append(np.array(cam["P"], dtype=np.float64))
        K_list.append(np.array(cam["K"], dtype=np.float64))
        R_list.append(np.array(cam["R"], dtype=np.float64))
        t_list.append(np.array(cam["t"], dtype=np.float64))

    num_frames = len(all_keypoints)
    print(f"共 {num_frames} 帧, {NUM_CAMERAS} 台摄像机")
    print(f"三角测量方法: {TRIANG_METHOD}, RANSAC: {RANSAC_ITERATIONS} 次")

    # 按帧重建
    all_skeletons = {}   # {frame_idx: [skeleton_3d_0, skeleton_3d_1, ...]}

    for frame_idx in tqdm(range(num_frames), desc="3D 重建"):
        frame_data = all_keypoints[str(frame_idx)]

        # 收集各摄像机的人
        cam_kpts = {}
        cam_confs = {}
        for cam_id in range(NUM_CAMERAS):
            cam_id_str = str(cam_id)
            if cam_id_str not in frame_data:
                continue
            data = frame_data[cam_id_str]
            kpts_list = [np.array(k, dtype=np.float64) for k in data.get("keypoints", [])]
            scores_list = [np.array(s, dtype=np.float64) for s in data.get("scores", [])]
            cam_kpts[cam_id] = kpts_list
            cam_confs[cam_id] = scores_list

        if not cam_kpts.get(0):
            all_skeletons[frame_idx] = []
            continue

        # 跨视图匹配
        matches = match_persons(cam_kpts, K_list, R_list, t_list)

        # 对每个匹配的人做三角测量
        frame_skeletons = []
        for match in matches:
            kpts_per_cam = np.zeros((NUM_CAMERAS, NUM_JOINTS, 2), dtype=np.float64)
            conf_per_cam = np.zeros((NUM_CAMERAS, NUM_JOINTS), dtype=np.float64)

            for cam_id, person_idx in enumerate(match):
                if person_idx >= 0 and cam_id in cam_kpts:
                    k = cam_kpts[cam_id][person_idx]
                    kpts_per_cam[cam_id] = k
                    if cam_id in cam_confs:
                        conf_per_cam[cam_id] = cam_confs[cam_id][person_idx]

            skeleton_3d, reproj = triangulate_skeleton(kpts_per_cam, P_list, conf_per_cam)
            frame_skeletons.append(skeleton_3d)

        all_skeletons[frame_idx] = frame_skeletons

    # 按人物整理时序序列
    # 简化: 假设每帧人数一致, 直接按索引对齐
    num_persons = max(
        len(skels) for skels in all_skeletons.values()
    ) if all_skeletons else 0

    animations = {}
    for pid in range(num_persons):
        seq = []
        for frame_idx in range(num_frames):
            skels = all_skeletons.get(frame_idx, [])
            if pid < len(skels):
                seq.append(skels[pid])
            else:
                seq.append(np.full((NUM_JOINTS, 3), np.nan))
        animations[pid] = np.array(seq)   # (T, 17, 3)

    # 保存
    save_path = os.path.join(OUTPUT_DIR, "skeletons_3d.npz")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez_compressed(save_path, animations=animations)

    print(f"\n重建完成: {num_persons} 人, {num_frames} 帧")
    for pid, seq in animations.items():
        valid = np.sum(~np.isnan(seq).any(axis=(1, 2)))
        print(f"  Person {pid}: {valid}/{num_frames} 帧有效")
    print(f"  结果已保存: {save_path}")

    return animations


def reconstruct_from_json(keypoints_json_path, calib_json_path=None):
    """
    从保存的 JSON 文件加载 2D 关键点并重建 3D
    (跳过 2D 检测步骤, 直接从已有结果重建)
    """
    import os

    print(f"加载 2D 关键点: {keypoints_json_path}")
    with open(keypoints_json_path) as f:
        all_keypoints = json.load(f)

    calib_data = None
    if calib_json_path and os.path.exists(calib_json_path):
        with open(calib_json_path) as f:
            calib_data = json.load(f)
    else:
        _, _, _, _, calib_data = load_calib()

    return reconstruct_3d(all_keypoints, calib_data)


if __name__ == "__main__":
    # 测试: 从已有结果重建
    import os
    kpts_path = os.path.join(OUTPUT_DIR, "all_2d_keypoints.json")
    if os.path.exists(kpts_path):
        animations = reconstruct_from_json(kpts_path)
        print("测试完成")
    else:
        print(f"未找到 2D 关键点文件: {kpts_path}")
        print("请先运行 2D 检测步骤")
