import numpy as np
import math
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PIL import Image
import imageio.v3 as iio
import matplotlib.pyplot as plt


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
                # 括弧を削除して数値に変換
                position = np.array(
                    [float(val) for val in line.split(":")[1].strip().replace("(", "").replace(")", "").split(",")]
                )
            elif "Rotation" in line:
                # Quaternionの括弧を削除して数値に変換
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
    # x = (i - cx)
    # y = (j - cy)
    # z = depth * 100000
    print(depth)
    x = (i - cx) - (depth - 0.005702) * 100 * (i - cx) / (1 - 0.005702)
    y = (j - cy) - (depth - 0.005702) * 100 * (j - cy) / (1 - 0.005702)
    z = depth * 100000
    # x = np.where((i - cx) / fx > 0, (i - cx) / fx +depth * math.tan(fov_x_rad / 2) * 100, (i - cx) / fx - depth * math.tan(fov_x_rad / 2) * 100)
    # y = np.where((j - cy) / fy > 0, (j - cy) / fy +depth * math.tan(fov_y_rad / 2) * 100, (j - cy) / fy - depth * math.tan(fov_y_rad / 2) * 100)
    # z = depth * 100


    # Stack into 3D points in camera space
    points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Apply mask to filter out invalid depths
    mask = depth > 0
    points_camera = points_camera[mask.flatten()]
    colors = color.reshape(-1, 3)[mask.flatten()] / 255.0  # Normalize to [0, 1]

    # Transform points to world coordinates
    points_world = (R @ points_camera.T).T + T

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def main():
    # Paths to data
    color_image_path = "../data_images/iteration_0/Camera_2_2.png"
    depth_image_path = "../data_images/iteration_0/Camera_2_2_depth.exr"
    internal_params_path = "../data_images/iteration_0/Camera_2_2_internal.txt"
    external_params_path = "../data_images/iteration_0/Camera_2_2_external.txt"

    # Load images
    depth = load_depth_image(depth_image_path)
    plt.imshow(depth, cmap="gray")
    plt.colorbar()
    plt.show()
    color = load_color_image(color_image_path)

    # Image dimensions
    image_height, image_width = color.shape[:2]

    # Load camera parameters
    intrinsics = load_internal_parameters(internal_params_path)
    extrinsics = load_external_parameters(external_params_path)

    # Calculate focal length and principal point
    intrinsics.update(calculate_focal_length(intrinsics, image_width, image_height))

    # Generate point cloud
    pcd = generate_point_cloud(depth, color, intrinsics, extrinsics)
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"Bounding box: {bbox}")
    # Save and visualize the point cloud
    o3d.io.write_point_cloud("output_point_cloud.ply", pcd)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
