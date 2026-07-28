"""
步骤 1: 多摄像机标定
 - 单摄像机内参标定 (棋盘格)
 - 多摄像机外参标定 (以 Camera 0 为世界原点)
 - 支持手动输入摄像机物理位置
"""

import os
import json
import glob
import numpy as np
import cv2

from config import (
    CALIB_DIR, CALIB_IMAGE_DIRS, INTRINSICS_FILE, EXTRINSICS_FILE,
    CALIB_JSON_FILE, CHESSBOARD_SIZE, CHESSBOARD_SQUARE_SIZE,
    NUM_CAMERAS, CAMERA_NAMES,
)


# ================================================================
#  第一部分: 单摄像机内参标定
# ================================================================

def calibrate_single_camera(image_dir, chess_size=CHESSBOARD_SIZE,
                            square_size=CHESSBOARD_SQUARE_SIZE):
    """
    使用棋盘格照片标定单台摄像机内参

    参数:
        image_dir:   棋盘格照片所在目录
        chess_size:  内角点数 (列, 行)
        square_size: 方格物理边长 (米)
    返回:
        K:    内参矩阵 (3×3)
        dist: 畸变系数 (k1,k2,p1,p2[,k3,...])
        rms:  重投影均方根误差
        img_size: 图像尺寸 (宽, 高)
    """
    # 3D 世界坐标点 (z=0 平面)
    objp = np.zeros((chess_size[0] * chess_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []   # 每组照片对应的 3D 点
    img_points = []   # 每组照片对应的 2D 角点
    used_images = []

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_files = []
    for pat in patterns:
        image_files.extend(glob.glob(os.path.join(image_dir, pat)))
        image_files.extend(glob.glob(os.path.join(image_dir, pat.upper())))
    image_files = sorted(set(image_files))

    if not image_files:
        raise FileNotFoundError(f"在 {image_dir} 中未找到标定图片")

    print(f"  找到 {len(image_files)} 张标定图片")

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)
        if not ret:
            print(f"  [警告] 未检测到棋盘格: {os.path.basename(fname)}")
            continue

        # 亚像素精度优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points.append(corners)
        used_images.append(fname)

    if len(obj_points) < 3:
        raise RuntimeError(f"有效标定图片不足 ({len(obj_points)} 张), 至少需要 3 张")

    img_size = (img.shape[1], img.shape[0])

    # 执行标定
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    print(f"  有效图片: {len(used_images)}/{len(image_files)}")
    print(f"  重投影误差 RMS: {rms:.4f} 像素")
    print(f"  内参矩阵:\n{K}")

    return K, dist, rms, img_size


def calibrate_all_intrinsics():
    """
    对每台摄像机逐一进行内参标定
    """
    print("\n" + "=" * 60)
    print("单摄像机内参标定")
    print("=" * 60)

    intrinsics = {}
    for i in range(NUM_CAMERAS):
        print(f"\n--- {CAMERA_NAMES[i]} ---")
        image_dir = CALIB_IMAGE_DIRS[i]
        if not os.path.isdir(image_dir):
            print(f"  [跳过] 标定图片目录不存在: {image_dir}")
            continue
        try:
            K, dist, rms, img_size = calibrate_single_camera(image_dir)
            intrinsics[i] = {
                "K": K,
                "dist": dist,
                "rms": rms,
                "image_size": img_size,
            }
        except Exception as e:
            print(f"  [错误] {e}")

    # 保存内参
    np.savez(INTRINSICS_FILE, intrinsics=intrinsics)
    print(f"\n内参已保存到: {INTRINSICS_FILE}")
    return intrinsics


# ================================================================
#  第二部分: 多摄像机外参标定
# ================================================================

def calibrate_extrinsics_from_images(intrinsics,
                                     shared_calib_dir,
                                     chess_size=CHESSBOARD_SIZE,
                                     square_size=CHESSBOARD_SQUARE_SIZE):
    """
    使用同一标定板在所有摄像机中的照片计算外参
    要求: 所有摄像机同时拍摄同一块标定板

    参数:
        intrinsics:       {cam_id: {"K": ..., "dist": ...}}
        shared_calib_dir: 存放了所有摄像机同步拍摄的标定板照片
                          文件名格式建议: camera_0.jpg, camera_1.jpg, ...
        chess_size:       棋盘格内角点
        square_size:      方格边长 (米)
    返回:
        extrinsics: {cam_id: {"R": (3,3), "t": (3,1)}}
    """
    print("\n" + "=" * 60)
    print("多摄像机外参标定 (同步标定板)")
    print("=" * 60)

    objp = np.zeros((chess_size[0] * chess_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 加载每台摄像机的标定板照片
    all_img_points = {}
    for cam_id in range(NUM_CAMERAS):
        img_path = os.path.join(shared_calib_dir, f"camera_{cam_id}.jpg")
        if not os.path.exists(img_path):
            # 尝试其他格式
            for ext in [".png", ".bmp"]:
                alt = os.path.join(shared_calib_dir, f"camera_{cam_id}{ext}")
                if os.path.exists(alt):
                    img_path = alt
                    break

        if not os.path.exists(img_path):
            print(f"  [警告] Camera {cam_id} 的同步标定照片不存在: {img_path}")
            continue

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)

        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            all_img_points[cam_id] = corners
            print(f"  Camera {cam_id}: 检测到 {len(corners)} 个角点")
        else:
            print(f"  [警告] Camera {cam_id}: 未检测到棋盘格角点")

    if len(all_img_points) < 2:
        raise RuntimeError("至少需要 2 台摄像机检测到棋盘格才能标定外参")

    # 以 Camera 0 为世界坐标系原点, 用 solvePnP 求解其外参
    extrinsics = {}

    # 先对 Camera 0 求解 -> R=单位阵, t=0 是世界原点, 但标定板不在原点
    # 让 Camera 0 的相机坐标系 = 世界坐标系
    # solvePnP 解的是世界坐标 -> 相机坐标的旋转平移
    cam0_K = intrinsics[0]["K"]
    cam0_corners = all_img_points[0]
    ret0, rvec0, tvec0 = cv2.solvePnP(objp, cam0_corners, cam0_K, None)
    R0, _ = cv2.Rodrigues(rvec0)
    t0 = tvec0.reshape(3, 1)

    extrinsics[0] = {"R": R0, "t": t0}

    # 对其他摄像机, 在同一世界坐标系下求解
    for cam_id in range(1, NUM_CAMERAS):
        if cam_id not in all_img_points or cam_id not in intrinsics:
            continue
        K = intrinsics[cam_id]["K"]
        corners = all_img_points[cam_id]
        ret, rvec, tvec = cv2.solvePnP(objp, corners, K, None)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        extrinsics[cam_id] = {"R": R, "t": t}
        print(f"  Camera {cam_id}: R=\n{R}\n  t=\n{t}")

    np.savez(EXTRINSICS_FILE, extrinsics=extrinsics)
    print(f"\n外参已保存到: {EXTRINSICS_FILE}")
    return extrinsics


# ================================================================
#  第三部分: 手动设定摄像机位置 (无需标定板)
# ================================================================

def manual_camera_setup(camera_positions, camera_lookats,
                        image_width, image_height,
                        focal_length_mm=35.0, sensor_width_mm=36.0):
    """
    根据物理位置手动构建摄像机参数

    参数:
        camera_positions: [(x,y,z), ...]  每台摄像机在世界坐标系中的位置 (米)
        camera_lookats:   [(x,y,z), ...]  每台摄像机的观察目标点 (米)
        image_width:      图像宽度 (像素)
        image_height:     图像高度 (像素)
        focal_length_mm:  镜头焦距 (毫米)
        sensor_width_mm:  传感器宽度 (毫米, 全画幅=36, APS-C≈24)
    返回:
        intrinsics, extrinsics, 可直接用于后续步骤
    """
    print("\n" + "=" * 60)
    print("手动设置摄像机参数")
    print("=" * 60)

    # 内参: 基于焦距和传感器尺寸估算
    fx = (focal_length_mm / sensor_width_mm) * image_width
    fy = fx  # 假设方形像素
    cx = image_width / 2
    cy = image_height / 2

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ], dtype=np.float64)

    dist = np.zeros((5,), dtype=np.float64)  # 假设无畸变

    intrinsics = {}
    extrinsics = {}

    for cam_id in range(NUM_CAMERAS):
        # 内参 (所有摄像机用同样的估算内参)
        intrinsics[cam_id] = {
            "K": K,
            "dist": dist,
            "rms": 0.0,
            "image_size": (image_width, image_height),
        }

        # 外参: 从摄像机位置和朝向计算
        pos = np.array(camera_positions[cam_id], dtype=np.float64)
        lookat = np.array(camera_lookats[cam_id], dtype=np.float64)

        # 构建相机坐标系:
        #   Z 轴: 从相机指向目标 (视线方向)
        #   Y 轴: 世界 Z 轴 (向上)
        #   X 轴: Z × Y (右手系)
        z_axis = lookat - pos
        z_axis = z_axis / np.linalg.norm(z_axis)

        world_up = np.array([0, 0, 1], dtype=np.float64)
        x_axis = np.cross(z_axis, world_up)
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            # 摄像机垂直朝下/上的退化情况
            world_up = np.array([0, 1, 0], dtype=np.float64)
            x_axis = np.cross(z_axis, world_up)
            x_norm = np.linalg.norm(x_axis)
        x_axis = x_axis / x_norm

        y_axis = np.cross(x_axis, z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # 旋转矩阵: 世界坐标系 -> 相机坐标系
        R_cam = np.vstack([x_axis, y_axis, -z_axis])  # 3×3, 注意 Z 方向

        # 实际上 cv2 的风格是: 世界坐标 -> 相机坐标
        # X_cam = R @ X_world + t
        # R 是 3×3 旋转, t = -R @ camera_position
        R = R_cam
        t = -R @ pos.reshape(3, 1)

        extrinsics[cam_id] = {"R": R, "t": t}

        print(f"\n  {CAMERA_NAMES[cam_id]}:")
        print(f"    位置: {pos}")
        print(f"    朝向: {lookat}")
        print(f"    R:\n{R}")
        print(f"    t:\n{t.ravel()}")

    # 保存
    np.savez(INTRINSICS_FILE, intrinsics=intrinsics)
    np.savez(EXTRINSICS_FILE, extrinsics=extrinsics)
    print(f"\n参数已保存到: {INTRINSICS_FILE}, {EXTRINSICS_FILE}")

    return intrinsics, extrinsics


# ================================================================
#  第四部分: 构建投影矩阵 & 导出
# ================================================================

def build_projection_matrices(intrinsics, extrinsics):
    """
    构建每台摄像机的投影矩阵 P = K × [R | t]
    """
    P_list = []
    K_list = []
    for cam_id in range(NUM_CAMERAS):
        K = intrinsics[cam_id]["K"]
        R = extrinsics[cam_id]["R"]
        t = extrinsics[cam_id]["t"]
        Rt = np.hstack([R, t.reshape(3, 1)])
        P = K @ Rt
        P_list.append(P)
        K_list.append(K)
    return P_list, K_list


def export_calib_json(intrinsics, extrinsics, output_path=CALIB_JSON_FILE):
    """
    将所有标定数据导出为单个 JSON 文件 (方便后续读取)
    """
    P_list, K_list = build_projection_matrices(intrinsics, extrinsics)

    calib_data = {
        "num_cameras": NUM_CAMERAS,
        "cameras": []
    }

    for cam_id in range(NUM_CAMERAS):
        K = intrinsics[cam_id]["K"]
        R = extrinsics[cam_id]["R"]
        t = extrinsics[cam_id]["t"]
        P = P_list[cam_id]

        calib_data["cameras"].append({
            "id": cam_id,
            "name": CAMERA_NAMES[cam_id],
            "K": K.tolist(),
            "dist": intrinsics[cam_id]["dist"].tolist(),
            "R": R.tolist(),
            "t": t.reshape(-1).tolist(),
            "P": P.tolist(),
            "image_size": intrinsics[cam_id].get("image_size", [1920, 1080]),
        })

    with open(output_path, "w") as f:
        json.dump(calib_data, f, indent=2, ensure_ascii=False)

    print(f"\n标定数据已导出: {output_path}")
    return calib_data


# ================================================================
#  第五部分: 主入口
# ================================================================

def run_calibration(mode="manual", **kwargs):
    """
    运行摄像机标定流水线

    参数:
        mode: "chessboard" = 棋盘格标定
              "manual"     = 手动指定摄像机位置
        kwargs: 传递给对应方法的参数
    返回:
        intrinsics, extrinsics, calib_data
    """
    os.makedirs(CALIB_DIR, exist_ok=True)

    if mode == "chessboard":
        intrinsics = calibrate_all_intrinsics()
        extrinsics = calibrate_extrinsics_from_images(
            intrinsics, kwargs.get("shared_calib_dir")
        )

    elif mode == "manual":
        intrinsics, extrinsics = manual_camera_setup(
            camera_positions=kwargs["camera_positions"],
            camera_lookats=kwargs["camera_lookats"],
            image_width=kwargs.get("image_width", 1920),
            image_height=kwargs.get("image_height", 1080),
            focal_length_mm=kwargs.get("focal_length_mm", 35.0),
            sensor_width_mm=kwargs.get("sensor_width_mm", 36.0),
        )

    else:
        raise ValueError(f"未知标定模式: {mode}")

    # 导出 JSON
    calib_data = export_calib_json(intrinsics, extrinsics)
    # 也保存为 .npz (方便 Python 加载)
    np.savez(os.path.join(CALIB_DIR, "calib_all.npz"),
             intrinsics=intrinsics, extrinsics=extrinsics)

    return intrinsics, extrinsics, calib_data


if __name__ == "__main__":
    # 默认演示: 使用棋盘格标定
    # 确保 calib/calib_images/camera_0/ 等目录下有标定照片
    run_calibration(mode="chessboard")
