import open3d as o3d
import numpy as np

def save_and_load_point_cloud(pcd, output_pcd_path):
    # 点群を保存
    o3d.io.write_point_cloud(output_pcd_path, pcd)
    print(f"Saved point cloud to {output_pcd_path}")

    # 点群を読み込み
    loaded_pcd = o3d.io.read_point_cloud(output_pcd_path)
    print(f"Loaded point cloud from {output_pcd_path}")

    return loaded_pcd

if __name__ == "__main__":
    # 例としてランダムな点群を生成
    points = np.random.rand(1000, 3)
    colors = np.random.rand(1000, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    output_pcd_path = "../dataset/test_point_cloud.ply"
    loaded_pcd = save_and_load_point_cloud(pcd, output_pcd_path)

    # 保存前と保存後の点群を表示
    print("Original Point Cloud:")
    o3d.visualization.draw_geometries([pcd])

    print("Loaded Point Cloud:")
    o3d.visualization.draw_geometries([loaded_pcd])