import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PIL import Image
import imageio.v3 as iio
# import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def load_internal_parameters(file_path):
    """
    Read camera internal parameters from a text file.
    """
    intrinsics = {}
    with open(file_path, "r") as file:
        for line in file:
            if "Field of View" in line:
                fov_y = float(line.split(":")[1].strip())
                intrinsics["fov_y"] = fov_y
            elif "Aspect Ratio" in line:
                aspect_ratio = float(line.split(":")[1].strip())
                intrinsics["aspect_ratio"] = aspect_ratio
            elif "Near Clip Plane" in line:
                intrinsics["near_clip"] = float(line.split(":")[1].strip())
            elif "Far Clip Plane" in line:
                intrinsics["far_clip"] = float(line.split(":")[1].strip())
    return intrinsics


def load_external_parameters(file_path):
    """
    Read camera external parameters (position and rotation) from a text file.
    """
    position = None
    rotation = None
    with open(file_path, "r") as file:
        for line in file:
            if "Position" in line:
                position = np.array(
                    [float(val) for val in line.split(":")[1].strip().replace("(", "").replace(")", "").split(",")]
                )
            elif "Rotation" in line:
                quaternion = [
                    float(val) for val in line.split(":")[1].strip().replace("(", "").replace(")", "").split(",")
                ]
                rotation = R.from_quat(quaternion).as_matrix()
    return {"position": position, "rotation": rotation}


def calculate_focal_length(intrinsics, width, height):
    """
    Calculate focal lengths (fx, fy) and principal points (cx, cy) from FOV and image size.
    """
    fov_y_rad = np.radians(intrinsics["fov_y"])
    aspect_ratio = intrinsics["aspect_ratio"]

    fy = height / (2 * np.tan(fov_y_rad / 2))
    fx = fy * aspect_ratio
    cx = width / 2
    cy = height / 2
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def load_depth_image(depth_path):
    """
    Load depth image from an EXR file using imageio.
    """
    depth = iio.imread(depth_path)
    return depth[:, :, 0]


def load_color_image(color_path):
    """
    Load the color image from a PNG file.
    """
    return np.array(Image.open(color_path))


def generate_point_cloud(depth, color, intrinsics, extrinsics):
    """
    Generate a point cloud from depth and color images.

    Parameters:
        depth (ndarray): Depth image (2D array).
        color (ndarray): Color image (2D array).
        intrinsics (dict): Camera intrinsic parameters.
        extrinsics (dict): Camera extrinsic parameters.

    Returns:
        o3d.geometry.PointCloud: The generated point cloud.
    """
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    R = extrinsics["rotation"]
    T = extrinsics["position"]

    # Create a grid of pixel coordinates
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    x = (i - cx) - (depth - 0.005702) * 60.0 * (i - cx) / (1 - 0.005702)
    y = (j - cy) - (depth - 0.005702) * 60.0 * (j - cy) / (1 - 0.005702)
    z = depth * 38500

    # Stack into 3D points in camera space
    points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Apply mask to filter out invalid depths
    mask = depth > 0
    points_camera = points_camera[mask.flatten()]
    colors = color.reshape(-1, 3)[mask.flatten()] / 255.0  # Normalize to [0, 1]
    
    points_camera[:, 1] = -points_camera[:, 1]
    points_camera = points_camera / 15

    # Transform points to world coordinates
    points_world = (R @ points_camera.T).T + T

    return points_world, colors, mask


def find_camera_data(directory):
    """
    Automatically find camera data (depth, color, internal, external) in a directory.
    Assumes files are named in a consistent format.
    """
    camera_data = []

    # Group files by camera identifier (e.g., Camera_0_0)
    files = os.listdir(directory)
    grouped_files = {}

    for file in files:
        if not file.startswith("Camera_") or not os.path.isfile(os.path.join(directory, file)):
            continue
        camera_id = "_".join(file.split("_")[:3])  # Extract Camera identifier (e.g., Camera_0_0)
        camera_id = camera_id.split(".")[0]  # Remove file extension
        if camera_id not in grouped_files:
            grouped_files[camera_id] = []
        grouped_files[camera_id].append(file)
    # Organize files into camera data
    for camera_id, file_list in grouped_files.items():
        depth_image = next((f for f in file_list if "depth" in f and "noTerrain" not in f), None)
        color_image = next((f for f in file_list if f.endswith(".png") and "noTerrain" not in f), None)
        internal_params = next((f for f in file_list if "internal" in f and "noTerrain" not in f), None)
        external_params = next((f for f in file_list if "external" in f and "noTerrain" not in f), None)
        zx120_map = next((f for f in file_list if "zx120" in f and "annotation" in f), None)
        ic120_map = next((f for f in file_list if "ic120" in f and "annotation" in f), None)
        d37pxi24_map = next((f for f in file_list if "d37pxi24" in f and "annotation" in f), None)
        if depth_image and color_image and internal_params and external_params:
            camera_data.append({
                "depth_image": os.path.join(directory, depth_image),
                "color_image": os.path.join(directory, color_image),
                "internal_params": os.path.join(directory, internal_params),
                "external_params": os.path.join(directory, external_params),
                "zx120_map": os.path.join(directory, zx120_map),
                "ic120_map": os.path.join(directory, ic120_map),
                "d37pxi24_map": os.path.join(directory, d37pxi24_map),
            })

    return camera_data

def process_camera_data(camera):
    # Load parameters
    intrinsics = load_internal_parameters(camera["internal_params"])
    extrinsics = load_external_parameters(camera["external_params"])

    # Load depth and color images
    depth = load_depth_image(camera["depth_image"])
    color = load_color_image(camera["color_image"])

    # Automatically determine image dimensions
    image_height, image_width = depth.shape
    intrinsics.update(calculate_focal_length(intrinsics, image_width, image_height))

    # Annotation
    annotation_map = np.load(camera["zx120_map"]) + np.load(camera["ic120_map"]) + np.load(camera["d37pxi24_map"])

    # Generate point cloud for this camera
    points, colors, mask = generate_point_cloud(depth, color, intrinsics, extrinsics)
    return points, colors, annotation_map[mask].flatten()

def generate_combined_point_cloud(camera_data):
    """
    Generate a combined point cloud from multiple cameras.

    Parameters:
        camera_data (list): List of dictionaries with paths for depth, color, and parameters for each camera.

    Returns:
        o3d.geometry.PointCloud: Combined point cloud.
    """
    all_points = []
    all_colors = []
    all_annotations = []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_camera_data, camera_data))

    for points, colors, annotations in results:
        all_points.append(points)
        all_colors.append(colors)
        all_annotations.append(annotations)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    all_annotations = np.concatenate(all_annotations, axis=0)
    return all_points, all_colors, all_annotations


def downsample_point_cloud(points, colors, annotations=None, voxel_size=0.05):
    """
    NumPyを使用して点群をボクセルグリッドでダウンサンプリングする。
    Args:
        points (ndarray): 点群の座標データ (N, 3)。
        colors (ndarray): 点群の色データ (N, 3)。
        voxel_size (float): ボクセルグリッドのサイズ。

    Returns:
        tuple: ダウンサンプリングされた点群 (points, colors)。
    """
    # ボクセルのインデックスを計算
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    # インデックスを1Dキーに変換
    keys = np.dot(voxel_indices, np.array([1, 10000, 1000000]))
    # 一意のボクセルキーを取得し、インデックスを計算
    unique_keys, unique_indices = np.unique(keys, return_index=True)
    # 一意なインデックスを基に点群と色を抽出
    downsampled_points = points[unique_indices]
    downsampled_colors = colors[unique_indices]
    if annotations is not None:
        downsampled_annotations = annotations[unique_indices]
        return downsampled_points, downsampled_colors, downsampled_annotations

    return downsampled_points, downsampled_colors

def fade_point_cloud(points, colors, annotations, no_fade_distance=1.0, fade_start=1.5, fade_end=3.0):
    """
    点群の周縁部を疎にし、フェードアウトさせる。
    
    Args:
        points (ndarray): 点群の座標データ (N, 3)。
        colors (ndarray): 点群の色データ (N, 3)。
        voxel_size (float): ボクセルグリッドサイズ。
        no_fade_distance (float): フェード処理をかけない距離の閾値。
        fade_start (float): フェードアウトが始まる距離。
        fade_end (float): フェードアウトが終了する距離。

    Returns:
        ndarray, ndarray: フェードアウト後の点群の座標と色。
    """
    # 点群の中心を計算
    center = np.mean(points, axis=0)
    
    # 点群の各点から中心までの距離を計算
    distances = np.linalg.norm(points - center, axis=1)
    
    # フェード対象外の点を選別
    no_fade_mask = distances <= no_fade_distance
    
    # フェードアウトの比率を計算 (0: 完全表示, 1: フェードアウト)
    fade_ratio = np.zeros_like(distances)  # 初期値は0 (フェードなし)
    fade_zone_mask = (distances > no_fade_distance) & (distances <= fade_end)
    fade_ratio[fade_zone_mask] = np.clip(
        (distances[fade_zone_mask] - fade_start) / (fade_end - fade_start), 0, 1
    )
    
    # サンプリング確率をフェード比率で調整 (フェード比率が高いほどサンプリングされにくい)
    sampling_prob = np.ones_like(distances)  # 初期値は1 (完全表示)
    sampling_prob[fade_zone_mask] = 1 - fade_ratio[fade_zone_mask]
    sampled_indices = np.random.rand(len(points)) < sampling_prob
    
    # フェード対象外の点はそのまま保持
    final_mask = sampled_indices | no_fade_mask
    
    final_mask &= distances <= fade_end
    
    # サンプリング後の点群と色
    faded_points = points[final_mask]
    faded_colors = colors[final_mask]
    faded_annotations = annotations[final_mask]
    
    # フェードアウト色 (透明化などをシミュレーション)
    # faded_colors[~no_fade_mask[final_mask]] *= (1 - fade_ratio[final_mask][~no_fade_mask[final_mask]][:, None])
    
    return faded_points, faded_colors, faded_annotations


def reconstruct_point_cloud(pcd_name, base_directory):
    # Automatically find camera data
    camera_data = find_camera_data(base_directory)
    if not camera_data:
        print("No valid camera data found in the directory!")
        return

    # Generate combined point cloud
    all_points, all_colors, all_annotations = generate_combined_point_cloud(camera_data)
    all_points, all_colors, all_annotations = fade_point_cloud(
        all_points, all_colors, all_annotations, no_fade_distance=25.0, fade_start=25, fade_end=50
    )
    all_points, all_colors, all_annotations = downsample_point_cloud(
        all_points, all_colors, all_annotations, voxel_size=0.1
    )
    # 反転
    all_points[:, 1] = -all_points[:, 1]
    all_points[:, 2] = -all_points[:, 2]
    unique_annotations = np.unique(all_annotations)
    set_annotations_val = set(unique_annotations)
    print(set_annotations_val)
    annotation_colors = {value: np.random.rand(3) for value in unique_annotations}
    colored_annotations = np.array([annotation_colors[val] for val in all_annotations])

    # Add annotations to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    os.mkdir("../dataset_1", exist_ok=True)

    # Save point cloud
    output_pcd_path = os.path.join("../dataset_1", pcd_name)

    o3d.io.write_point_cloud(output_pcd_path, pcd)
    # print(f"Saved reconstructed point cloud with annotations to {output_pcd_path}")

    # Save the annotations
    annotation_path = os.path.join(
        "../dataset_1", f"{pcd_name.split('.')[0]}_annotations.npy"
    )
    np.save(annotation_path, all_annotations)

    # # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    annotated_pcd = o3d.geometry.PointCloud()
    annotated_pcd.points = o3d.utility.Vector3dVector(all_points)
    annotated_pcd.colors = o3d.utility.Vector3dVector(colored_annotations)
    o3d.visualization.draw_geometries([annotated_pcd])


if __name__ == "__main__":
    reconstruct_point_cloud("reconstructed_point_cloud_with_annotations.ply", "../data_images/iteration_0")

