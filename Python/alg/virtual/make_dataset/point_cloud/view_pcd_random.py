import open3d as o3d
import numpy as np

def view_point_cloud(pcd_path, annotation_path=None, n_samples=250000):
    # 点群を読み込む
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    # ランダムサンプリング
    points = np.asarray(pcd.points)
    if len(points) > n_samples:
        indices = np.random.choice(len(points), n_samples, replace=False)
        sampled_points = points[indices]
        
        # 新しい点群オブジェクトを作成
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        
        if annotation_path:
            # アノテーションを読み込む
            annotations = np.load(annotation_path)
            sampled_annotations = annotations[indices]
            annotation_colors = {
                1: np.array([1, 0, 0]),
                2: np.array([0, 1, 0]),
                3: np.array([0, 0, 1])
            }
            colored_annotations = np.array([annotation_colors.get(val, np.array([0.5, 0.5, 0.5])) for val in sampled_annotations])
            sampled_pcd.colors = o3d.utility.Vector3dVector(colored_annotations)
    else:
        sampled_pcd = pcd
        if annotation_path:
            annotations = np.load(annotation_path)
            annotation_colors = {
                1: np.array([1, 0, 0]),
                2: np.array([0, 1, 0]),
                3: np.array([0, 0, 1])
            }
            colored_annotations = np.array([annotation_colors.get(val, np.array([0.5, 0.5, 0.5])) for val in annotations])
            sampled_pcd.colors = o3d.utility.Vector3dVector(colored_annotations)
    
    # 点群を表示
    o3d.visualization.draw_geometries([sampled_pcd])

if __name__ == "__main__":
    pcd_path = "../dataset/pcd_911.ply"
    annotation_path = "../dataset/pcd_911_annotations.npy"
    view_point_cloud(pcd_path, annotation_path)