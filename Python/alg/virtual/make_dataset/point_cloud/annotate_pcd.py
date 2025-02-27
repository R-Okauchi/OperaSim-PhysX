import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PIL import Image
import imageio.v3 as iio

# import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from perlin_noise import PerlinNoise
import random
from scipy.interpolate import RegularGridInterpolator


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
                    [
                        float(val)
                        for val in line.split(":")[1]
                        .strip()
                        .replace("(", "")
                        .replace(")", "")
                        .split(",")
                    ]
                )
            elif "Rotation" in line:
                quaternion = [
                    float(val)
                    for val in line.split(":")[1]
                    .strip()
                    .replace("(", "")
                    .replace(")", "")
                    .split(",")
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
    fx, fy, cx, cy = (
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
    )
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
        if not file.startswith("Camera_") or not os.path.isfile(
            os.path.join(directory, file)
        ):
            continue
        camera_id = "_".join(
            file.split("_")[:3]
        )  # Extract Camera identifier (e.g., Camera_0_0)
        camera_id = camera_id.split(".")[0]  # Remove file extension
        if camera_id not in grouped_files:
            grouped_files[camera_id] = []
        grouped_files[camera_id].append(file)
    # Organize files into camera data
    for camera_id, file_list in grouped_files.items():
        depth_image = next(
            (f for f in file_list if "depth" in f and "noTerrain" not in f), None
        )
        color_image = next(
            (f for f in file_list if f.endswith(".png") and "noTerrain" not in f), None
        )
        internal_params = next(
            (f for f in file_list if "internal" in f and "noTerrain" not in f), None
        )
        external_params = next(
            (f for f in file_list if "external" in f and "noTerrain" not in f), None
        )
        zx120_map = next(
            (f for f in file_list if "zx120" in f and "annotation" in f), None
        )
        ic120_map = next(
            (f for f in file_list if "ic120" in f and "annotation" in f), None
        )
        d37pxi24_map = next(
            (f for f in file_list if "d37pxi24" in f and "annotation" in f), None
        )
        zx200_map = next(
            (f for f in file_list if "zx200" in f and "annotation" in f), None
        )
        c30r_map = next(
            (f for f in file_list if "c30r" in f and "annotation" in f), None
        )
        scaffold_map = next(
            (f for f in file_list if "scaffold" in f and "annotation" in f), None
        )
        cut_cone_map = next(
            (f for f in file_list if "cut_cone" in f and "annotation" in f), None
        )
        if depth_image and color_image and internal_params and external_params:
            camera_data.append(
                {
                    "depth_image": os.path.join(directory, depth_image),
                    "color_image": os.path.join(directory, color_image),
                    "internal_params": os.path.join(directory, internal_params),
                    "external_params": os.path.join(directory, external_params),
                    "zx120_map": os.path.join(directory, zx120_map),
                    "ic120_map": os.path.join(directory, ic120_map),
                    "d37pxi24_map": os.path.join(directory, d37pxi24_map),
                    "zx200_map": os.path.join(directory, zx200_map),
                    "c30r_map": os.path.join(directory, c30r_map),
                    "scaffold_map": os.path.join(directory, scaffold_map),
                    "cut_cone_map": os.path.join(directory, cut_cone_map),
                }
            )

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
    # annotation_map = np.load(camera["zx120_map"]) + np.load(camera["ic120_map"]) + np.load(camera["d37pxi24_map"])
    annotation_map = (
        np.load(camera["zx120_map"])
        * np.load(camera["ic120_map"])
        * np.load(camera["d37pxi24_map"])
        * np.load(camera["zx200_map"])
        * np.load(camera["c30r_map"])
        * np.load(camera["scaffold_map"])
        * np.load(camera["cut_cone_map"])
    )

    valid_annotations = {1, 2, 3, 5, 7, 11, 13, 17}
    unique_annotations = np.unique(annotation_map)
    # print(unique_annotations)
    if not set(unique_annotations).issubset(valid_annotations):
        raise ValueError(
            f"Invalid annotations found: {set(unique_annotations) - valid_annotations}"
        )
    remap_dict = {1: 0, 2: 1, 3: 2, 5: 3, 7: 4, 11: 5, 13: 6, 17: 7}
    annotation_map = np.vectorize(remap_dict.get)(annotation_map)

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


def fade_point_cloud(
    points,
    colors,
    annotations,
    center=None,
    half_side=40.0,
    amplitude=10.0,
    scale=0.05,
    fade_range=5.0,
    seed=None,
    grid_size=256,
):
    if seed is None:
        seed = random.randint(0, 9999999)
    noise_fn = PerlinNoise(octaves=1, seed=seed)

    if center is None:
        cx = np.mean(points[:, 0])
        cz = np.mean(points[:, 2])
    else:
        cx, cz = center

    x = points[:, 0]
    z = points[:, 2]

    # グリッド範囲を設定 (正方形の範囲 + 余裕)
    margin = amplitude * 2
    min_x = cx - (half_side + margin)
    max_x = cx + (half_side + margin)
    min_z = cz - (half_side + margin)
    max_z = cz + (half_side + margin)

    gx = np.linspace(min_x, max_x, grid_size)
    gz = np.linspace(min_z, max_z, grid_size)

    # 4つのオフセットを使い分ける
    off_left = random.uniform(0, 1e6)
    off_right = random.uniform(0, 1e6)
    off_near = random.uniform(0, 1e6)
    off_far = random.uniform(0, 1e6)

    def make_noise_map(mode="left", offset=0.0):
        # 2Dのnoiseマップを作成
        noise_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        for j in range(grid_size):
            for i in range(grid_size):
                if mode in ["left", "right"]:
                    # zをメイン入力
                    val = noise_fn([gz[j] * scale, offset])
                else:
                    # near, far は xをメイン入力
                    val = noise_fn([gx[i] * scale, offset])
                val_01 = (val + 1.0) * 0.5
                noise_map[j, i] = val_01
        return noise_map

    # 左辺/右辺/手前辺/奥辺のノイズマップ
    nm_left = make_noise_map("left", off_left)
    nm_right = make_noise_map("right", off_right)
    nm_near = make_noise_map("near", off_near)
    nm_far = make_noise_map("far", off_far)

    # 補間器作成
    # left/right -> (z, x)の順で補間するが、今回は x を固定的に使う or 無視するので
    # ここでは z 軸を第1, x軸を第2にしておき、(z, x) 順で呼び出す
    interp_left = RegularGridInterpolator((gz, gx), nm_left)
    interp_right = RegularGridInterpolator((gz, gx), nm_right)
    # near/far は x軸を第1, z軸を第2
    nm_near_T = nm_near.T
    nm_far_T = nm_far.T
    interp_near = RegularGridInterpolator((gx, gz), nm_near_T)
    interp_far = RegularGridInterpolator((gx, gz), nm_far_T)

    # 左辺/右辺のうねり
    # (z, cx) で呼び出し、x軸は中心付近の適当な値を与えて補間
    # 本来は x 方向も細かく補間すればもっと精度は出ますが、例として簡単化
    left_vals = interp_left(np.stack([z, np.full_like(z, cx)], axis=-1))
    right_vals = interp_right(np.stack([z, np.full_like(z, cx)], axis=-1))
    bl = half_side + (left_vals * 2 - 1) * amplitude
    br = half_side + (right_vals * 2 - 1) * amplitude

    # 手前/奥は (x, cz)
    near_vals = interp_near(np.stack([x, np.full_like(x, cz)], axis=-1))
    far_vals = interp_far(np.stack([x, np.full_like(x, cz)], axis=-1))
    bn = half_side + (near_vals * 2 - 1) * amplitude
    bf = half_side + (far_vals * 2 - 1) * amplitude

    dl = x - (cx - bl)
    dr = (cx + br) - x
    dn = z - (cz - bn)
    df = (cz + bf) - z
    md = np.min([dl, dr, dn, df], axis=0)

    inside = md >= 0
    outside = md < -fade_range
    fade_zone = ~(inside | outside)

    ratio = np.zeros_like(md)
    ratio[fade_zone] = -md[fade_zone] / fade_range
    keep_prob = 1.0 - ratio
    fade_mask = np.random.rand(len(points)) < keep_prob

    mask = inside | (fade_zone & fade_mask)
    return points[mask], colors[mask], annotations[mask]


def reconstruct_point_cloud(pcd_name, base_directory, output_directory):
    # Automatically find camera data
    camera_data = find_camera_data(base_directory)
    if not camera_data:
        print("No valid camera data found in the directory!")
        return

    # Generate combined point cloud
    all_points, all_colors, all_annotations = generate_combined_point_cloud(camera_data)
    all_points, all_colors, all_annotations = downsample_point_cloud(
        all_points, all_colors, all_annotations, voxel_size=0.15
    )
    # all_points, all_colors, all_annotations = fade_point_cloud(
    #     all_points, all_colors, all_annotations
    # )
    # print(f"Reconstructed point cloud has {len(all_points)} points.")

    if len(all_points) < 150000:
        raise ValueError("Point cloud is too small. Please try again.")
    # 反転
    all_points[:, 1] = -all_points[:, 1]
    all_points[:, 2] = -all_points[:, 2]
    unique_annotations = np.unique(all_annotations)
    set_annotations_val = set(unique_annotations)
    print(set_annotations_val)
    # annotation_colors = {value: np.random.rand(3) for value in unique_annotations}
    annotation_colors = {
                0: np.array([0.5, 0.5, 0.5]),
                1: np.array([1, 0, 0]),
                2: np.array([0, 1, 0]),
                3: np.array([0, 0, 1]),
                4: np.array([0.5, 0.5, 0]),
                5: np.array([1, 0, 1]),
                6: np.array([0, 1, 1]),
                7: np.array([1, 1, 0]),
            }
    colored_annotations = np.array([annotation_colors[val] for val in all_annotations])

    # Add annotations to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Save point cloud
    output_pcd_path = os.path.join(output_directory, pcd_name)

    o3d.io.write_point_cloud(output_pcd_path, pcd)
    # print(f"Saved reconstructed point cloud with annotations to {output_pcd_path}")

    # Save the annotations
    annotation_path = os.path.join(
        output_directory, f"{pcd_name.split('.')[0]}_annotations.npy"
    )
    np.save(annotation_path, all_annotations)

    # # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])
    annotated_pcd = o3d.geometry.PointCloud()
    annotated_pcd.points = o3d.utility.Vector3dVector(all_points)
    annotated_pcd.colors = o3d.utility.Vector3dVector(colored_annotations)
    o3d.visualization.draw_geometries([annotated_pcd])


if __name__ == "__main__":
    reconstruct_point_cloud(
        "reconstructed_point_cloud_with_annotations.ply",
        "../data_images/iteration_0",
        "../dataset",
    )
