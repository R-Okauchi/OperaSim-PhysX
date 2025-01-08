import json
import os

import numpy as np
import open3d as o3d
from PIL import Image


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


def reconstruct_point_cloud(
    omniverse_dir, annotation_dir_1, annotation_dir_2, annotation_dir_3, pcd_name
):
    all_points = []
    all_colors = []
    all_annotations = []

    # Load extrinsic parameters
    with open(os.path.join(omniverse_dir, "extrinsic_params.json"), "r") as f:
        extrinsic_params = json.load(f)

    # Iterate over each camera directory
    camera_dirs = [
        d
        for d in os.listdir(omniverse_dir)
        if os.path.isdir(os.path.join(omniverse_dir, d))
    ]
    camera_dirs = sorted(camera_dirs)
    ant_dirs_1 = [
        d
        for d in os.listdir(annotation_dir_1)
        if os.path.isdir(os.path.join(annotation_dir_1, d))
    ]
    ant_dirs_1 = sorted(ant_dirs_1)

    ant_dirs_2 = [
        d
        for d in os.listdir(annotation_dir_2)
        if os.path.isdir(os.path.join(annotation_dir_2, d))
    ]
    ant_dirs_2 = sorted(ant_dirs_2)

    ant_dirs_3 = [
        d
        for d in os.listdir(annotation_dir_3)
        if os.path.isdir(os.path.join(annotation_dir_3, d))
    ]
    ant_dirs_3 = sorted(ant_dirs_3)

    for extrinsic_param, camera_dir, ant_dir_1, ant_dir_2, ant_dir_3 in zip(
        extrinsic_params.values(), camera_dirs, ant_dirs_1, ant_dirs_2, ant_dirs_3
    ):
        camera_path = os.path.join(omniverse_dir, camera_dir)
        rgb_path = os.path.join(camera_path, "rgb/rgb_0000.png")
        depth_from_image_plane_path = os.path.join(
            camera_path, "distance_to_image_plane/distance_to_image_plane_0000.npy"
        )
        camera_params_path = os.path.join(
            camera_path, "camera_params/camera_params_0000.json"
        )
        ant_path_1 = os.path.join(annotation_dir_1, ant_dir_1)
        annotation_map_path_1 = os.path.join(ant_path_1, "annotations.npy")
        ant_path_2 = os.path.join(annotation_dir_2, ant_dir_2)
        annotation_map_path_2 = os.path.join(ant_path_2, "annotations.npy")
        ant_path_3 = os.path.join(annotation_dir_3, ant_dir_3)
        annotation_map_path_3 = os.path.join(ant_path_3, "annotations.npy")

        # Load RGB image
        rgb = np.array(Image.open(rgb_path)) / 255.0
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        # Load depth and annotation map
        depth_from_image_plane = np.load(depth_from_image_plane_path)
        annotation_map_1 = np.load(annotation_map_path_1)
        annotation_map_2 = np.load(annotation_map_path_2)
        annotation_map_3 = np.load(annotation_map_path_3)

        whole_annotation_map = annotation_map_1 + annotation_map_2 + annotation_map_3

        # 非 inf 部分のマスク作成
        non_inf_mask = ~np.isinf(depth_from_image_plane)
        if not np.any(non_inf_mask):
            print(f"Camera {camera_dir} has no valid depth data. Skipping...")
            continue

        # フィルタリング前に座標を作成
        image_height, image_width = depth_from_image_plane.shape
        u_coords, v_coords = np.meshgrid(
            np.arange(image_width), np.arange(image_height)
        )

        # フィルタリング
        depth_from_image_plane = depth_from_image_plane[non_inf_mask]
        u_coords = u_coords[non_inf_mask]
        v_coords = v_coords[non_inf_mask]
        rgb = rgb.reshape(-1, 3)[non_inf_mask.flatten()]
        annotations = whole_annotation_map[non_inf_mask]

        # Load camera parameters
        with open(camera_params_path, "r") as f:
            camera_params = json.load(f)

        # Compute intrinsic parameters
        fx = (
            camera_params["cameraFocalLength"] / camera_params["cameraAperture"][0]
        ) * camera_params["renderProductResolution"][0]
        fy = (
            camera_params["cameraFocalLength"] / camera_params["cameraAperture"][0]
        ) * camera_params["renderProductResolution"][1]
        cx, cy = (
            camera_params["renderProductResolution"][0] / 2.0,
            camera_params["renderProductResolution"][1] / 2.0,
        )

        # Compute camera-to-world transform
        camera_view_transform = np.array(camera_params["cameraViewTransform"]).reshape(
            4, 4
        )
        camera_to_world = np.linalg.inv(camera_view_transform)

        # Compute 3D points in camera space
        x_c = (u_coords - cx) * depth_from_image_plane / fx
        y_c = (v_coords - cy) * depth_from_image_plane / fy
        z_c = depth_from_image_plane
        points_camera = np.stack([x_c, y_c, z_c, np.ones_like(z_c)], axis=1)

        # Transform to world coordinates
        points_world = (camera_to_world @ points_camera.T).T[:, :3]
        points_world[:, 1] = -points_world[:, 1]
        points_world[:, 2] = -points_world[:, 2]
        points_world = (
            points_world / camera_params["metersPerSceneUnit"]
            + extrinsic_param["position"]
        )

        # Append to global arrays
        all_points.append(points_world)
        all_colors.append(rgb)
        all_annotations.append(annotations)
        print(f"Processed {camera_dir}")

    # Concatenate all points, colors, and annotations
    if not all_points:
        raise ValueError("No valid points reconstructed from the dataset.")
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    all_annotations = np.concatenate(all_annotations, axis=0)

    # Downsample points
    all_points, all_colors, all_annotations = downsample_point_cloud(
        all_points, all_colors, all_annotations, voxel_size=100
    )

    # Color points based on annotations
    unique_annotations = np.unique(all_annotations)
    set_annotations_val = set(unique_annotations)
    print(set_annotations_val)
    annotation_colors = {value: np.random.rand(3) for value in unique_annotations}
    colored_annotations = np.array([annotation_colors[val] for val in all_annotations])
    print(colored_annotations)

    # Add annotations to point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    # Save point cloud
    output_pcd_path = os.path.join("dataset", pcd_name)

    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"Saved reconstructed point cloud with annotations to {output_pcd_path}")

    # Save the annotations
    annotation_path = os.path.join(
        "dataset", f"{pcd_name.split('.')[0]}_annotations.npy"
    )
    np.save(annotation_path, all_annotations)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    annotated_pcd = o3d.geometry.PointCloud()
    annotated_pcd.points = o3d.utility.Vector3dVector(all_points)
    annotated_pcd.colors = o3d.utility.Vector3dVector(colored_annotations)
    o3d.visualization.draw_geometries([annotated_pcd])


# Usage
if __name__ == "__main__":
    omniverse_dir = "world"
    annotation_dir_1 = "object_1"
    annotation_dir_2 = "object_2"
    annotation_dir_3 = "object_3"
    pcd_name = "reconstructed_point_cloud_with_annotations.ply"
    reconstruct_point_cloud(
        omniverse_dir, annotation_dir_1, annotation_dir_2, annotation_dir_3, pcd_name
    )
