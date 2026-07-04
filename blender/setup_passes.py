# setup_passes.py — автонастройка рендера блокинга для AI-render пайплайна.
#
# Запуск: целиком через Blender MCP (execute_blender_code) или
#   blender -b scene.blend --python setup_passes.py
#
# Что делает:
#  1. Включает пассы: Z + Cryptomatte Material (accurate, levels 6).
#     Используем CryptoMaterial, НЕ CryptoAsset (тот группирует по parent-объекту).
#  2. Строит компоузер с нуля: beauty, нормализованный depth (ближнее = белое,
#     конвенция Flux depth) и по одной ч/б маске на каждый материал.
#     Маски сразу расширяются (Dilate) и размываются (Blur) — в ComfyUI нет
#     нод блюра, поэтому швы убираем здесь.
#  3. Рендерит кадр и пишет пак PNG в OUTPUT_DIR.
#
# ВНИМАНИЕ: существующее дерево нод компоузера сцены будет удалено.

import os

import bpy
from mathutils import Vector

# ---------------- CONFIG ----------------
OUTPUT_DIR = bpy.path.abspath("//render_pack")
# Должно совпадать с width/height в driver/materials.yaml (маски = разрешению генерации).
# FluxGeneral_fal: max 1536, кратно 16.
RES_X, RES_Y = 1344, 768
MASK_DILATE_PX = 6          # расширение маски, чтобы inpaint перекрывал швы
MASK_BLUR_PX = 4            # мягкий край маски
MAT_PREFIX = "mat_"         # маски только для материалов с этим префиксом; "" = все используемые
DEPTH_MARGIN = 0.05         # запас при нормализации depth (доля от диапазона)
# -----------------------------------------


def depth_range(scene):
    """Мин/макс расстояние вдоль оси взгляда камеры по bbox видимых мешей."""
    cam = scene.camera
    if cam is None:
        raise RuntimeError("В сцене нет активной камеры")
    cam_pos = cam.matrix_world.translation
    view_dir = (cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))).normalized()
    dists = []
    for ob in scene.objects:
        if ob.type != "MESH" or not ob.visible_get():
            continue
        for corner in ob.bound_box:
            d = (ob.matrix_world @ Vector(corner) - cam_pos).dot(view_dir)
            if d > 0.0:
                dists.append(d)
    if not dists:
        raise RuntimeError("Не найдено видимых мешей перед камерой")
    near, far = max(min(dists), 0.01), max(dists)
    margin = (far - near) * DEPTH_MARGIN
    return max(near - margin, 0.01), far + margin


def used_materials():
    mats = [m for m in bpy.data.materials
            if m.users > 0 and not m.is_grease_pencil
            and (not MAT_PREFIX or m.name.startswith(MAT_PREFIX))]
    if not mats and MAT_PREFIX:
        print(f"WARN: нет материалов с префиксом '{MAT_PREFIX}', беру все используемые. "
              f"Лучше раскидать плейсхолдеры mat_* с говорящими именами.")
        mats = [m for m in bpy.data.materials if m.users > 0 and not m.is_grease_pencil]
    return mats


def make_file_output(tree, name, base_path, color_mode="RGB", color_depth="8"):
    node = tree.nodes.new("CompositorNodeOutputFile")
    node.name = node.label = name
    node.base_path = base_path
    node.format.file_format = "PNG"
    node.format.color_mode = color_mode
    node.format.color_depth = color_depth
    node.file_slots.clear()
    return node


def add_slot(fo_node, path, save_as_render=False):
    """Слот File Output; save_as_render=False -> данные пишутся без view transform."""
    slot = fo_node.file_slots.new(path)
    slot.path = path
    if hasattr(slot, "save_as_render"):
        slot.save_as_render = save_as_render
    return fo_node.inputs[-1]


def build():
    scene = bpy.context.scene
    vl = bpy.context.view_layer

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- пассы ---
    vl.use_pass_z = True
    vl.use_pass_cryptomatte_material = True
    vl.use_pass_cryptomatte_object = False
    vl.use_pass_cryptomatte_asset = False
    if hasattr(vl, "use_pass_cryptomatte_accurate"):
        vl.use_pass_cryptomatte_accurate = True
    vl.pass_cryptomatte_depth = 6

    # --- рендер-настройки ---
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = True
    scene.use_nodes = True

    near, far = depth_range(scene)
    print(f"depth range: near={near:.3f} far={far:.3f}")

    mats = used_materials()
    if not mats:
        raise RuntimeError("В сцене нет используемых материалов — нечем строить маски")
    print("материалы для масок:", [m.name for m in mats])

    # --- компоузер ---
    tree = scene.node_tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    rl.layer = vl.name
    rl.location = (-400, 0)

    comp = tree.nodes.new("CompositorNodeComposite")
    comp.location = (300, 300)
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])

    # beauty (с view transform — как в рендере)
    fo_beauty = make_file_output(tree, "out_beauty", OUTPUT_DIR)
    fo_beauty.location = (300, 150)
    tree.links.new(rl.outputs["Image"], add_slot(fo_beauty, "beauty", save_as_render=True))

    # depth: Map Range near->1, far->0 (ближнее = белое), 16-бит BW, без transform
    map_range = tree.nodes.new("CompositorNodeMapRange")
    map_range.location = (0, -100)
    map_range.use_clamp = True
    map_range.inputs["From Min"].default_value = near
    map_range.inputs["From Max"].default_value = far
    map_range.inputs["To Min"].default_value = 1.0
    map_range.inputs["To Max"].default_value = 0.0
    tree.links.new(rl.outputs["Depth"], map_range.inputs["Value"])

    fo_depth = make_file_output(tree, "out_depth", OUTPUT_DIR, color_mode="BW", color_depth="16")
    fo_depth.location = (300, -100)
    tree.links.new(map_range.outputs["Value"], add_slot(fo_depth, "depth"))

    # маски: CryptomatteV2(material) -> Dilate -> Blur -> File Output
    fo_masks = make_file_output(tree, "out_masks", OUTPUT_DIR, color_mode="BW", color_depth="8")
    fo_masks.location = (300, -400)

    crypto_layer = f"{vl.name}.CryptoMaterial"
    for i, mat in enumerate(mats):
        y = -350 - i * 200
        cr = tree.nodes.new("CompositorNodeCryptomatteV2")
        cr.location = (-150, y)
        cr.label = f"crypto_{mat.name}"
        cr.source = "RENDER"
        cr.scene = scene
        try:
            cr.layer_name = crypto_layer
        except TypeError:
            # имя слоя enum-ом отличается между версиями — берём первый CryptoMaterial
            options = [it.identifier for it in cr.bl_rna.properties["layer_name"].enum_items
                       if "CryptoMaterial" in it.identifier]
            if not options:
                raise RuntimeError(f"Слой CryptoMaterial не найден (нода {mat.name})")
            cr.layer_name = options[0]
        cr.matte_id = mat.name

        dilate = tree.nodes.new("CompositorNodeDilateErode")
        dilate.location = (30, y)
        dilate.mode = "DISTANCE"
        dilate.distance = MASK_DILATE_PX

        blur = tree.nodes.new("CompositorNodeBlur")
        blur.location = (160, y)
        blur.filter_type = "GAUSS"
        blur.size_x = blur.size_y = MASK_BLUR_PX
        blur.use_relative = False

        tree.links.new(cr.outputs["Matte"], dilate.inputs[-1])
        tree.links.new(dilate.outputs[0], blur.inputs["Image"])
        tree.links.new(blur.outputs["Image"], add_slot(fo_masks, f"mask_{mat.name}"))

    # --- рендер ---
    print(f"рендер {RES_X}x{RES_Y} -> {OUTPUT_DIR}")
    bpy.ops.render.render(write_still=False)

    files = sorted(os.listdir(OUTPUT_DIR))
    print("готово, файлы пака:")
    for f in files:
        print("  ", f)
    return files


build()
