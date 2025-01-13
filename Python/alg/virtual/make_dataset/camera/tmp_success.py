import json
import os
import random

import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

# フィールドのサイズとパラメータ
FIELD_SIZE = [50000, 50000, 0]  # フィールドの幅、奥行き、高さ
FIELD_NAME = "/Field"
MATERIAL_PATH = "/FieldMaterial"
TEXTURE_PATH = "omniverse://localhost/Materials/Textures/soil_albedo.png"  # 土のアルベドテクスチャパス
SCENARIO = "omniverse://localhost/Projects/test.usd"
output_dir = "/Users/ryout/Documents/main_desk/omniverse_driver/field_output"

# USDステージを開く
context = omni.usd.get_context()
context.open_stage(SCENARIO)
stage = context.get_stage()

# 除外するノードの名前リスト
EXCLUDE_PRIMS = [
    "/OmniverseKit_Persp",
    "/OmniverseKit_Front",
    "/OmniverseKit_Top",
    "/OmniverseKit_Right",
    "/Render",
    "/OmniKit_Viewport_LightRig",
]


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


mesh_prims = get_all_valid_mesh_prims(stage, EXCLUDE_PRIMS)
if not mesh_prims:
    raise ValueError("除外ノード以外に有効なメッシュPrimが見つかりません")


# フィールド作成
def create_field(stage, field_size, field_name):
    field_prim = UsdGeom.Mesh.Define(stage, Sdf.Path(field_name))
    vertices = [
        Gf.Vec3f(-field_size[0] / 2, -field_size[1] / 2, 0),
        Gf.Vec3f(field_size[0] / 2, -field_size[1] / 2, 0),
        Gf.Vec3f(field_size[0] / 2, field_size[1] / 2, 0),
        Gf.Vec3f(-field_size[0] / 2, field_size[1] / 2, 0),
    ]
    indices = [0, 1, 2, 2, 3, 0]
    field_prim.GetPointsAttr().Set(vertices)
    field_prim.GetFaceVertexIndicesAttr().Set(indices)
    field_prim.GetFaceVertexCountsAttr().Set([3, 3])
    return field_prim


field = create_field(stage, FIELD_SIZE, FIELD_NAME)


# マテリアル作成
def create_material(stage, material_path, texture_path):
    material_prim = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(material_path + "/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")

    # アルベドテクスチャの設定
    texture = UsdShade.Shader.Define(stage, Sdf.Path(material_path + "/Albedo"))
    texture.CreateIdAttr("UsdUVTexture")
    texture_input = texture.CreateInput("file", Sdf.ValueTypeNames.Asset)
    texture_input.Set(texture_path)
    texture_output = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    # アルベドをシェーダに接続
    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3)
    diffuse_input.ConnectToSource(texture_output)

    # シェーダーの surface 出力を取得して接続
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material_prim.CreateSurfaceOutput().ConnectToSource(shader_output)

    return material_prim


material = create_material(stage, MATERIAL_PATH, TEXTURE_PATH)
UsdShade.MaterialBindingAPI(field).Bind(material)


def get_top_level_prims(mesh_prims):
    """
    メッシュの親ノード（トップレベルのXform）を取得する。
    親ノードが1種類だけになるまで親をたどる。
    """
    current_prims = set(mesh_prims)  # 初期はメッシュPrims
    while len(current_prims) > 1:  # 親が1種類になるまでループ
        next_prims = set()
        for prim in current_prims:
            parent = prim.GetParent()
            if parent:
                next_prims.add(parent)
        if not next_prims:  # 親が見つからない場合は終了
            break
        current_prims = next_prims
    if not current_prims:
        raise ValueError("トップレベルのPrimが見つかりません")
    return next(iter(current_prims))


# オブジェクト配置
def place_objects_on_field(stage, field_size, prim):
    """
    フィールド上にオブジェクトをランダムに配置します。
    """
    # ランダム位置の計算
    x = (random.random() - 0.5) * field_size[0]
    y = (random.random() - 0.5) * field_size[1]
    z = field_size[2]

    # ランダムな回転角度（度単位）
    angle = random.uniform(0, 360)

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


top_level_prim = get_top_level_prims(mesh_prims)

place_objects_on_field(stage, FIELD_SIZE, top_level_prim)


# バウンディングボックス取得
def get_combined_bounds(mesh_prims):
    min_bound = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
    max_bound = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))

    for prim in mesh_prims:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), includedPurposes=["default", "render", "proxy"]
        )
        bbox = bbox_cache.ComputeWorldBound(prim)
        min_bound = Gf.Vec3d(
            min(min_bound[0], bbox.GetRange().GetMin()[0]),
            min(min_bound[1], bbox.GetRange().GetMin()[1]),
            min(min_bound[2], bbox.GetRange().GetMin()[2]),
        )
        max_bound = Gf.Vec3d(
            max(max_bound[0], bbox.GetRange().GetMax()[0]),
            max(max_bound[1], bbox.GetRange().GetMax()[1]),
            max(max_bound[2], bbox.GetRange().GetMax()[2]),
        )
    return min_bound, max_bound


min_bound, max_bound = get_combined_bounds(mesh_prims)
center = (min_bound + max_bound) / 2
extent = max_bound - min_bound
radius = max(extent) * 1.5
center_list = [center[0], center[1], center[2]]

# カメラ配置と画像保存
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with rep.new_layer():
    cameras = []
    render_products = []
    extrinsic_params = {}

    grid_size = 4  # グリッドの分割数（幅と奥行き）
    camera_height = 30000  # カメラの高さ（固定値）

    # グリッドごとにカメラを配置
    x_spacing = FIELD_SIZE[0] / grid_size
    y_spacing = FIELD_SIZE[1] / grid_size

    for i in range(grid_size):
        for j in range(grid_size):
            # グリッド内のカメラ位置を計算
            x = -FIELD_SIZE[0] / 2 + x_spacing * (i + 0.5)
            y = -FIELD_SIZE[1] / 2 + y_spacing * (j + 0.5)
            z = camera_height
            position = [x, y, z]
            look_at = [x, y, 0]  # 真下を見る

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
            print(f"カメラ {camera_name} を配置しました。")

    with rep.trigger.on_frame(num_frames=1, rt_subframes=64):
        pass

    # カメラのパラメータ保存
    with open(os.path.join(output_dir, "extrinsic_params.json"), "w") as f:
        json.dump(extrinsic_params, f, indent=4)

    writer = rep.WriterRegistry.get("BasicWriter")
    print(dir(writer.initialize))
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        distance_to_image_plane=True,
        distance_to_camera=True,
        camera_params=True,
    )
    writer.attach(render_products)
    rep.orchestrator.run()

print(f"画像とカメラパラメータを {output_dir} に保存しました。")
