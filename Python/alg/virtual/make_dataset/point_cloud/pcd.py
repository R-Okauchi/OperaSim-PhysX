import imageio.v3 as iio
import numpy as np
import open3d as o3d
import os
import re

def read_exr_to_numpy(file_path):
    """
    EXRファイルをNumPy配列として読み込む
    :param file_path: EXRファイルのパス
    :return: NumPy配列 (深度マップ)
    """
    depth_data = iio.imread(file_path, plugin="EXR-FI")
    if len(depth_data.shape) == 3:  # R, G, B の場合、1チャンネル目を使用
        depth_data = depth_data[..., 0]
    return depth_data

def read_rgb_to_numpy(file_path):
    """
    RGB画像をNumPy配列として読み込む
    :param file_path: RGB画像のパス
    :return: NumPy配列 (RGB画像)
    """
    rgb_data = iio.imread(file_path)
    return rgb_data

def parse_camera_params(internal_file, external_file):
    """
    カメラの内部・外部パラメータを読み込む
    :param internal_file: 内部パラメータファイルのパス
    :param external_file: 外部パラメータファイルのパス
    :return: (intrinsics, extrinsics)
    """
    # 内部パラメータを読み込み
    with open(internal_file, "r") as f:
        lines = f.readlines()
    fov = float(re.search(r"Field of View: ([\d.]+)", lines[0]).group(1))
    aspect_ratio = float(re.search(r"Aspect Ratio: ([\d.]+)", lines[1]).group(1))
    near_clip_plane = float(re.search(r"Near Clip Plane: ([\d.]+)", lines[2]).group(1))
    far_clip_plane = float(re.search(r"Far Clip Plane: ([\d.]+)", lines[3]).group(1))
    # cx = 960  # 画像幅の中心 (例: 1920x1080 の場合)
    cx = 1103/2
    # cy = 540  # 画像高さの中心 (例: 1920x1080 の場合)
    cy = 679/2

    fx = fy = 1.0 / np.tan(np.deg2rad(fov) / 2.0)

    intrinsics = (fx, fy, cx, cy, aspect_ratio, near_clip_plane, far_clip_plane)
    # 外部パラメータを読み込み
    with open(external_file, "r") as f:
        lines = f.readlines()
    position = eval(re.search(r"Position: \((.+)\)", lines[0]).group(1))
    rotation = eval(re.search(r"Rotation: \((.+)\)", lines[1]).group(1))
    
    # rontation = [x, y, z, w] として、[w, x, y, z] に変換
    rotation = [rotation[3], rotation[0], rotation[1], rotation[2]]
    # 回転（クォータニオン）から回転行列に変換
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(rotation)
    extrinsics[:3, 3] = position

    return intrinsics, extrinsics

def depth_to_point_cloud_with_color(depth_map, rgb_map, intrinsics, extrinsics):
    """
    深度マップとRGB画像を使用して点群を生成
    :param depth_map: NumPy配列 (深度マップ)
    :param rgb_map: NumPy配列 (RGB画像)
    :param intrinsics: 内部パラメータ (fx, fy, cx, cy)
    :param extrinsics: 外部パラメータ (4x4変換行列)
    :return: 点群 (Nx3 の NumPy配列) と色 (Nx3 の NumPy配列)
    """
    h, w = depth_map.shape
    fx, fy, cx, cy, aspect, near_clip, far_clip = intrinsics

    # 画像座標系 (ピクセル) を生成
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    z_raw = depth_map.flatten()
    print(f"min depth: {np.min(z_raw)}, max depth: {np.max(z_raw)}")

    # マスク: 深度が0の画素を除外
    valid_mask = (z_raw > 0)
    z_raw = z_raw[valid_mask]

    xx = xx.flatten()[valid_mask]
    yy = yy.flatten()[valid_mask]
    z = near_clip + z_raw * (far_clip - near_clip)
    x = (xx.flatten() - cx) * z / fx
    y = (yy.flatten() - cy) * z / fy
    points_camera = np.vstack((x, y, z)).T
    
    # 反転
    points_camera[:, 1] = -points_camera[:, 1]
    
    

    # 外部パラメータでワールド座標系に変換˚
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3] * 47
    points_world = (R @ points_camera.T + t.reshape(-1, 1)).T

    # 色を取得
    colors = rgb_map.reshape(-1, 3)[valid_mask] / 255.0

    return points_world, colors

def create_combined_point_cloud(depth_dir, params_dir, voxel_size=0.05):
    """
    複数のカメラから得られた点群を統合し、ダウンサンプル
    :param depth_dir: 深度画像(EXR)が保存されているディレクトリ
    :param params_dir: カメラパラメータが保存されているディレクトリ
    :param voxel_size: ダウンサンプリング時のボクセルサイズ (単位: メートル)
    :return: 統合されたダウンサンプル済み点群
    """
    point_clouds = []
    pattern = re.compile(r"Camera_\d+_\d+_depth\.exr")

    for file_name in os.listdir(depth_dir):
        if pattern.match(file_name):
            print(f"Processing {file_name}...")
            # 深度マップを読み込む
            depth_map = read_exr_to_numpy(os.path.join(depth_dir, file_name))
            print()

            # 対応するRGB画像を読み込む
            rgb_file = file_name.replace("_depth.exr", ".png")
            rgb_map = read_rgb_to_numpy(os.path.join(depth_dir, rgb_file))
            print(rgb_map.shape)

            # 内部・外部パラメータを取得
            cam_name = file_name.replace("_depth.exr", "")
            internal_file = os.path.join(params_dir, f"{cam_name}_internal.txt")
            external_file = os.path.join(params_dir, f"{cam_name}_external.txt")
            intrinsics, extrinsics = parse_camera_params(internal_file, external_file)

            # 点群を生成
            points, colors = depth_to_point_cloud_with_color(depth_map, rgb_map, intrinsics, extrinsics)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            point_clouds.append(pcd)

    # すべての点群を統合
    combined_pcd = point_clouds[0]
    # バウンディングボックスを計算
    bbox = combined_pcd.get_axis_aligned_bounding_box()
    print(f"Bounding box: {bbox}")
    for pcd in point_clouds[1:]:
        combined_pcd += pcd
    print(f"Combined point cloud has {len(combined_pcd.points)} points.")
    # ダウンサンプリング
    downsampled_pcd = combined_pcd.voxel_down_sample(voxel_size)
    print(f"Downsampled point cloud has {len(downsampled_pcd.points)} points.")

    return downsampled_pcd

# メイン処理
if __name__ == "__main__":
    depth_dir = "../data_images/iteration_0/"  # EXRファイルのディレクトリ
    params_dir = "../data_images/iteration_0/"  # 内部・外部パラメータが保存されたディレクトリ
    voxel_size = 0.1  # ダウンサンプル時のボクセルサイズ

    # 統合された点群を作成
    pcd = create_combined_point_cloud(depth_dir, params_dir, voxel_size)

    # 点群を保存 & 可視化
    o3d.io.write_point_cloud("combined_point_cloud_downsampled.ply", pcd)
    o3d.visualization.draw_geometries([pcd])
    
    