import json
import os

import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom

# 定数設定
FIELD_SIZE = [50000, 50000, 0]
SCENARIO = "omniverse://localhost/Projects/test.usd"
COORDINATE_DIR = "/Users/ryout/Documents/main_desk/omniverse_driver/world"
OUTPUT_DIR = "/Users/ryout/Documents/main_desk/omniverse_driver/object_1"

EXCLUDE_PRIMS = [
    "/OmniverseKit_Persp",
    "/OmniverseKit_Front",
    "/OmniverseKit_Top",
    "/OmniverseKit_Right",
    "/Render",
    "/OmniKit_Viewport_LightRig",
]


# USDステージを開く
def open_stage(context, scenario):
    context.open_stage(scenario)
    return context.get_stage()


# メッシュ探索
def find_meshes_recursively(prim, exclude_list):
    meshes = []
    if prim.GetPath().pathString in exclude_list:
        return meshes
    if UsdGeom.Mesh(prim):
        meshes.append(prim)
    for child in prim.GetChildren():
        meshes.extend(find_meshes_recursively(child, exclude_list))
    return meshes


def get_all_valid_mesh_prims(stage, exclude_list):
    pseudo_root = stage.GetPseudoRoot()
    all_mesh_prims = []
    for prim in pseudo_root.GetChildren():
        all_mesh_prims.extend(find_meshes_recursively(prim, exclude_list))
    return all_mesh_prims


def get_top_level_prims(mesh_prims):
    """
    メッシュの親ノード（トップレベルのXform）を取得する。
    親ノードが1種類だけになるまで親をたどる。
    """
    current_prims = set(mesh_prims)
    while len(current_prims) > 1:
        next_prims = set()
        for prim in current_prims:
            parent = prim.GetParent()
            if parent:
                next_prims.add(parent)
        if not next_prims:
            break
        current_prims = next_prims
    if not current_prims:
        raise ValueError("トップレベルのPrimが見つかりません")
    return next(iter(current_prims))


# オブジェクト配置
def place_objects_on_field(prim):
    """
    フィールド上にオブジェクトをランダムに配置します。
    """
    # jsonファイルから座標を読み込む
    with open(os.path.join(COORDINATE_DIR, "object_coordinates.json"), "r") as f:
        coordinates = json.load(f)

    x, y, z, angle = coordinates[0]

    # PrimをXformに変換
    xform = UsdGeom.Xformable(prim)
    if not xform:
        print(
            f"警告: Prim {prim.GetPath()} は Xformable ではありません。スキップします。"
        )

    # 既存のXformOpをクリア
    xform.ClearXformOpOrder()

    # 変換操作を追加（順序に注意）
    translate_op = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate)
    translate_op.Set(Gf.Vec3d(x, y, z))

    rotate_op = xform.AddXformOp(UsdGeom.XformOp.TypeRotateZ)
    rotate_op.Set(angle)


# カメラ配置と画像保存
def setup_cameras_and_save_images(stage, field_size, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with rep.new_layer():
        cameras = []
        render_products = []
        extrinsic_params = {}
        grid_size = 4
        camera_height = 30000
        x_spacing = field_size[0] / grid_size
        y_spacing = field_size[1] / grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                x = -field_size[0] / 2 + x_spacing * (i + 0.5)
                y = -field_size[1] / 2 + y_spacing * (j + 0.5)
                z = camera_height
                position = [x, y, z]
                look_at = [x, y, 0]
                camera_name = f"MyTestCamera_{i}_{j}"
                extrinsic_params[camera_name] = {
                    "look_at": look_at,
                    "position": position,
                    "scale": [1, 1, 1],
                }
                camera = rep.create.camera(
                    position=position,
                    look_at=look_at,
                    name=camera_name,
                    look_at_up_axis=[0, 0, 1],
                )
                cameras.append(camera)
                path_pattern = f".*/MyTestCamera_{i}_{j}"
                get_camera = rep.get.camera(path_pattern)
                render_product = rep.create.render_product(
                    get_camera, resolution=(1080, 1080)
                )
                render_products.append(render_product)

        with rep.trigger.on_frame(num_frames=1, rt_subframes=64):
            pass

        output_usd_path = os.path.join(output_dir, "final_stage.usd")
        stage.GetRootLayer().Export(output_usd_path)
        with open(os.path.join(output_dir, "extrinsic_params.json"), "w") as f:
            json.dump(extrinsic_params, f, indent=4)

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=output_dir,
            rgb=True,
            distance_to_image_plane=True,
            distance_to_camera=True,
            camera_params=True,
        )
        writer.attach(render_products)
        rep.orchestrator.run()


# メイン関数
def main():
    context = omni.usd.get_context()
    stage = open_stage(context, SCENARIO)
    mesh_prims = get_all_valid_mesh_prims(stage, EXCLUDE_PRIMS)
    if not mesh_prims:
        raise ValueError("除外ノード以外に有効なメッシュPrimが見つかりません")
    top_level_prim = get_top_level_prims(mesh_prims)
    place_objects_on_field(top_level_prim)
    setup_cameras_and_save_images(stage, FIELD_SIZE, OUTPUT_DIR)


# if __name__ == "__main__":
#     print("start")
main()
