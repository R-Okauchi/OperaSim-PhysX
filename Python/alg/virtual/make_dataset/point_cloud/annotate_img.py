import glob
import os

import imageio
import numpy as np
from PIL import Image

def annotate_non_inf_pixels(depth_array, rgb_array, annotation_value=1):
    """
    深度に値があるピクセルをオブジェクトとしてアノテーションする。
    ここでは 0.99以上を背景扱い (== Far clip 付近) とみなす例。
    """
    # 1. 基本的に \infty, nan, 0以下 は使わない
    invalid_mask = np.isinf(depth_array) | np.isnan(depth_array) | (depth_array <= 0)

    # 2. 1.0 に近いもの (0.99以上) も背景扱い
    far_threshold = 0.99
    invalid_mask |= (depth_array >= far_threshold)

    non_inf_mask = ~invalid_mask

    # 3. アノテーションマップを作成
    annotation_map = np.zeros((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8)
    annotation_map[non_inf_mask] = annotation_value

    # 4. オーバーレイ用画像
    annotated_image = rgb_array.copy()
    if annotation_value == 1:
        annotated_image[non_inf_mask] = [255, 0, 0]  # 赤
    elif annotation_value == 2:
        annotated_image[non_inf_mask] = [0, 255, 0]
    elif annotation_value == 3:
        annotated_image[non_inf_mask] = [0, 0, 255]

    return annotation_map, annotated_image


def annotate_single_prefab_images(images_dir):
    """
    指定フォルダ配下の "Camera_*_noTerrain_only_*.png" 画像と
    同名の "Camera_*_noTerrain_only_*_depth.exr" 深度をペアで読んで、
    アノテーションマップとアノテーション画像を作成・保存するサンプル関数。

    Args:
        root_dir (str): ルートフォルダ (例: "data_images")
        annotation_value (int): オブジェクトに割り当てたいラベルIDなど (デフォルト=1)
    """
    images_path = os.path.join(images_dir)

    # "noTerrain_only_*" を含む PNG を探索
    png_files = glob.glob(os.path.join(images_path, "*_noTerrain_only_*.png"))
    if not png_files:
        print(f"[{images_dir}] 該当するPNGが見つかりませんでした。")

    for png_file in png_files:
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        depth_file = os.path.join(images_path, f"{base_name}_depth.exr")
        if not os.path.exists(depth_file):
            # print(f"対応するEXRファイルが見つかりません: {depth_file}")
            continue

        # --- 1. RGB画像の読み込み (PIL) ---
        rgb_image = Image.open(png_file).convert("RGB")
        rgb_array = np.array(rgb_image)
        # print(f"RGB画像を読み込みました: {rgb_array.shape}")

        # --- 2. EXR深度の読み込み (imageio) ---
        #     imageio で読み込むと shape = (H, W) or (H, W, C) になる場合がある
        depth_data = imageio.imread(depth_file, format="EXR-FI")
        depth_set = set(depth_data.flatten())
        if len(depth_set) == 1:
            print(f"深度データが一定値のためスキップ: {depth_set}")
            raise ValueError("深度データが一定値のためスキップ")

        # depth_data が (H, W, 3) の場合は、チャネルを1つにまとめる等の処理が必要。
        # Unity の出力方法によって異なるので、必要に応じて下記を修正。
        # 例: とりあえずRチャンネルだけ使う:
        if depth_data.ndim == 3 and depth_data.shape[2] > 1:
            depth_data = depth_data[:, :, 0]

        # --- 3. アノテーション作成 ---
        if "zx120" in base_name:
            annotation_value = 1
        elif "ic120" in base_name:
            annotation_value = 2
        elif "d37pxi24" in base_name:
            annotation_value = 3
        
        annotation_map, annotated_image = annotate_non_inf_pixels(
            depth_data, rgb_array, annotation_value
        )

        # --- 4. 保存 ---
        # アノテーションマップは Numpy の .npy で保存
        annotation_map_path = os.path.join(
            images_path, f"{base_name}_annotation.npy"
        )
        np.save(annotation_map_path, annotation_map)

        # アノテーションを可視化した画像を PNG として保存
        annotated_image_path = os.path.join(
            images_path, f"{base_name}_annotated.png"
        )
        Image.fromarray(annotated_image).save(annotated_image_path)
        
        # print(f"アノテーションを作成しました: {annotation_map_path}, {annotated_image_path}")
        # break


# スクリプトを直接実行する場合
if __name__ == "__main__":
    # Unity で出力したルートフォルダ (例: "data_images") に合わせて指定
    root_dir = "../data_images/iteration_0"
    annotate_single_prefab_images(root_dir)
