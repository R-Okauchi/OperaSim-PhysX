import open3d as o3d
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def find_nearest_annotation(pcd_tree, point, annotations):
    print(f"Finding nearest annotation for point {point}")
    [k, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
    return annotations[idx[0]]

def view_point_cloud(pcd_path, annotation_path=None, voxel_size=0.05):
    # 点群を読み込む
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # ボクセルダウンサンプリング
    sampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    if annotation_path:
        # アノテーションを読み込む
        annotations = np.load(annotation_path)
        
        # 元の点群とダウンサンプリングされた点群の対応を見つける
        sampled_points = np.asarray(sampled_pcd.points)
        print(f"Original point cloud has {len(pcd.points)} points.")
        print(f"Downsampled point cloud has {len(sampled_points)} points.")
        
        # 最近傍点を探してアノテーションを対応付け
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        with ThreadPoolExecutor() as executor:
            sampled_annotations = list(executor.map(lambda point: find_nearest_annotation(pcd_tree, point, annotations), sampled_points))
        
        # カラーマッピング
        annotation_colors = {
            1: np.array([1, 0, 0]),
            2: np.array([0, 1, 0]),
            3: np.array([0, 0, 1])
        }
        colored_annotations = np.array([annotation_colors.get(val, np.array([0.5, 0.5, 0.5])) for val in sampled_annotations])
        sampled_pcd.colors = o3d.utility.Vector3dVector(colored_annotations)
    
    # 点群を表示
    o3d.visualization.draw_geometries([sampled_pcd])

if __name__ == "__main__":
    pcd_path = "../dataset/pcd_911.ply"
    annotation_path = "../dataset/pcd_911_annotations.npy"
    view_point_cloud(pcd_path, annotation_path, voxel_size=0.1)