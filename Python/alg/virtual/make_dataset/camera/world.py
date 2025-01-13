import json
import math
import os
import random

import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

# 定数設定
FIELD_SIZE = [50000, 50000, 0]
FIELD_NAME = "/Field"
MATERIAL_PATH = "/FieldMaterial"
TEXTURE_PATHS = [
    "omniverse://localhost/NVIDIA/Materials/Base/Natural/Dirt/Dirt_BaseColor.png",
    "omniverse://localhost/NVIDIA/Materials/Base/Natural/Dirt/Asphalt_BaseColor.png",
    "omniverse://localhost/NVIDIA/Materials/Base/Natural/Dirt/Grass_Winter_BaseColor.png",
    "omniverse://localhost/NVIDIA/Materials/Base/Natural/Dirt/Leaves_BaseColor.png",
    "omniverse://localhost/NVIDIA/Materials/Base/Natural/Dirt/Muich_Brown_BaseColor.png",
    "omniverse://localhost/NVIDIA/Materials/Base/Natural/Dirt/Sand_BaseColor.png",
    "omniverse://localhost/NVIDIA/Materials/Base/Natural/Dirt/Soil_Rocky_BaseColor.png",
]
SCENARIO = "omniverse://localhost/Projects/test.usd"
usd_files = [
    # "omniverse://localhost/Projects/test.usd",
    "omniverse://localhost/Projects/crane_truck.usd",
    "omniverse://localhost/Projects/power_shovel.usd",
]
OUTPUT_DIR = "/Users/ryout/Documents/main_desk/omniverse_driver/world"

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


# ライト追加
# def add_directional_light(stage, name, direction, intensity, color):
#     light_path = f"/{name}"
#     light = UsdLux.DistantLight.Define(stage, Sdf.Path(light_path))
#     light.CreateIntensityAttr(intensity)
#     light.CreateColorAttr(Gf.Vec3f(*color))

#     xform = UsdGeom.Xformable(light.GetPrim())
#     xform.ClearXformOpOrder()
#     target_direction = Gf.Vec3f(*direction).GetNormalized()
#     default_direction = Gf.Vec3f(0, 0, -1)
#     axis = Gf.Cross(default_direction, target_direction)
#     angle = math.degrees(math.acos(Gf.Dot(default_direction, target_direction)))
#     if axis.GetLength() > 0:
#         axis = axis.GetNormalized()
#     rotate_op = xform.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
#     rotate_op.Set(Gf.Vec3f(angle * axis[0], angle * axis[1], angle * axis[2]))
#     return light


def add_directional_light(stage, name, intensity_range, color_range):
    """
    現実世界の昼から夕方の光をシミュレートするライトを追加する。

    Args:
        stage: USDステージ
        name: ライトの名前
        intensity_range: 光の強度の範囲 (例: (100, 500))
        color_range: 光の色 (例: [(1.0, 1.0, 0.8), (1.0, 0.6, 0.4)])
    """
    light_path = f"/{name}"
    light = UsdLux.DistantLight.Define(stage, Sdf.Path(light_path))

    # ランダムな光の強度を設定
    intensity = random.uniform(*intensity_range)
    light.CreateIntensityAttr(intensity)

    # ランダムな光の色を設定
    color = [random.uniform(color_range[0][i], color_range[1][i]) for i in range(3)]
    light.CreateColorAttr(Gf.Vec3f(*color))

    xform = UsdGeom.Xformable(light.GetPrim())
    xform.ClearXformOpOrder()

    # ランダムな方向を設定 (昼は高角度、夕方は低角度)
    altitude_angle = random.uniform(15, 75)  # 仰角: 15°〜75°
    azimuth_angle = random.uniform(0, 360)  # 方位角: 0°〜360°

    # 方位角と仰角から方向ベクトルを計算
    altitude_radians = math.radians(altitude_angle)
    azimuth_radians = math.radians(azimuth_angle)
    x = math.cos(altitude_radians) * math.cos(azimuth_radians)
    y = math.cos(altitude_radians) * math.sin(azimuth_radians)
    z = math.sin(altitude_radians)

    target_direction = Gf.Vec3f(x, y, z).GetNormalized()
    default_direction = Gf.Vec3f(0, 0, -1)
    axis = Gf.Cross(default_direction, target_direction)
    angle = math.degrees(math.acos(Gf.Dot(default_direction, target_direction)))
    if axis.GetLength() > 0:
        axis = axis.GetNormalized()

    rotate_op = xform.AddXformOp(UsdGeom.XformOp.TypeRotateXYZ)
    rotate_op.Set(Gf.Vec3f(angle * axis[0], angle * axis[1], angle * axis[2]))

    return light


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


# マテリアル作成
def create_material(stage, material_path, texture_path):
    material_prim = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(material_path + "/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    texture = UsdShade.Shader.Define(stage, Sdf.Path(material_path + "/Albedo"))
    texture.CreateIdAttr("UsdUVTexture")
    texture_input = texture.CreateInput("file", Sdf.ValueTypeNames.Asset)
    texture_input.Set(texture_path)
    texture_output = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3)
    diffuse_input.ConnectToSource(texture_output)
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material_prim.CreateSurfaceOutput().ConnectToSource(shader_output)
    return material_prim


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
def place_objects_on_field(field_size, prim, coordinates=[]):
    """
    フィールド上にオブジェクトをランダムに配置します。
    xy座標がcoordinatesリストの半径4000に入らないように配置します。
    端から3000以上離れるように配置します。
    """

    is_overlapping = True
    while is_overlapping:
        x = random.uniform(-field_size[0] / 2, field_size[0] / 2)
        y = random.uniform(-field_size[1] / 2, field_size[1] / 2)
        is_overlapping = False
        for coord in coordinates:
            if (
                math.sqrt((x - coord[0]) ** 2 + (y - coord[1]) ** 2) < 4000
                or abs(x) > 22000
                or abs(y) > 22000
            ):
                is_overlapping = True
                break

    z = field_size[2]
    angle = random.uniform(0, 360)

    xform = UsdGeom.Xformable(prim)
    if not xform:
        print(
            f"警告: Prim {prim.GetPath()} は Xformable ではありません。スキップします。"
        )

    xform.ClearXformOpOrder()

    translate_op = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate)
    translate_op.Set(Gf.Vec3d(x, y, z))

    rotate_op = xform.AddXformOp(UsdGeom.XformOp.TypeRotateZ)
    rotate_op.Set(angle)

    return [x, y, z, angle]


def load_and_place_usd_files(stage, usd_files, field_size, first_coordinate):
    """
    Load the specified USD files into the current stage and place them randomly on the field.
    """
    placed_prims = []
    coordinates = [first_coordinate]
    for usd_file in usd_files:
        prim_path = Sdf.Path(f"/{os.path.basename(usd_file).split('.')[0]}")
        root_prim = stage.DefinePrim(prim_path)
        if not root_prim:
            print(f"Warning: Could not define Prim for USD file {usd_file}. Skipping.")
            continue

        root_prim.GetReferences().AddReference(usd_file)

        coordinate = place_objects_on_field(field_size, root_prim, coordinates)
        placed_prims.append(root_prim)
        coordinates.append(coordinate)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "object_coordinates.json"), "w") as f:
        json.dump(coordinates, f, indent=4)


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
    # add_directional_light(stage, "Sunlight", [-0.5, -0.5, -1], 500, [1.0, 1.0, 0.8])
    add_directional_light(
        stage,
        "Sunlight",
        intensity_range=(300, 1000),
        color_range=[(1.0, 1.0, 0.8), (1.0, 0.5, 0.4)],
    )
    mesh_prims = get_all_valid_mesh_prims(stage, EXCLUDE_PRIMS)
    if not mesh_prims:
        raise ValueError("除外ノード以外に有効なメッシュPrimが見つかりません")
    field = create_field(stage, FIELD_SIZE, FIELD_NAME)
    random_texture_path = random.choice(TEXTURE_PATHS)
    material = create_material(stage, MATERIAL_PATH, random_texture_path)
    UsdShade.MaterialBindingAPI(field).Bind(material)
    top_level_prim = get_top_level_prims(mesh_prims)
    first_coordinate = place_objects_on_field(FIELD_SIZE, top_level_prim)
    load_and_place_usd_files(stage, usd_files, FIELD_SIZE, first_coordinate)
    setup_cameras_and_save_images(stage, FIELD_SIZE, OUTPUT_DIR)


# if __name__ == "__main__":
#     print("start")
main()
