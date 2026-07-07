# Comfyder Pro v0.2.0 — зонный AI-рендер пайплайн в N-панели Blender.
#
# Полный цикл Blocking2Render одной кнопкой:
#   1) рендер пака (depth + Cryptomatte-маска на каждую зону, dilate/blur
#      per-material) временным компоузером — твой компоузер не трогается;
#   2) граф ComfyUI: глобальный проход (Flux + depth ControlNet) →
#      зонные проходы по маскам (fill / qwen / gemini / kontext / zturbo) →
#      финал (Gemini: кадр + depth + промпт настроения + защита фактур);
#   3) фоновый поллинг, все шаги в папку, финал — в Image Editor.
#
# Требования: ComfyUI + паки fal-нод + FAL_KEY (см. README репозитория).
# Blender 5.0+ (новый API компоузера).

bl_info = {
    "name": "Comfyder Pro",
    "author": "Stepan Vladovskiy",
    "version": (0, 2, 0),
    "blender": (5, 0, 0),
    "location": "3D View / Image Editor > Sidebar (N) > Comfyder Pro",
    "description": "Зонный AI-рендер: блокинг → ComfyUI → FAL по маскам материалов",
    "category": "Render",
}

import json
import os
import tempfile
import textwrap
import time
import uuid
import urllib.parse
import urllib.request

import bpy
from mathutils import Vector

MAT_PREFIX = "mat_"
_JOB = {"prompt_id": None, "host": None, "t0": 0.0, "scene": None,
        "out_dir": None, "expected_final": "99_final"}


# =================================================================== HTTP
def _http_json(url, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _upload(host, path, name=None):
    name = name or os.path.basename(path)
    b = uuid.uuid4().hex
    data = open(path, "rb").read()
    body = (f"--{b}\r\nContent-Disposition: form-data; "
            f"name=\"overwrite\"\r\n\r\ntrue\r\n").encode()
    body += (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; "
             f"filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n").encode()
    body += data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(
        host + "/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["name"]


# =================================================================== пак
def _depth_range(scene):
    cam = scene.camera
    if cam is None:
        raise RuntimeError("В сцене нет камеры")
    cp = cam.matrix_world.translation
    vd = (cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))).normalized()
    ds = [(ob.matrix_world @ Vector(c) - cp).dot(vd)
          for ob in scene.objects
          if ob.type in {'MESH', 'CURVE'} and ob.visible_get()
          for c in ob.bound_box]
    ds = [d for d in ds if d > 0]
    if not ds:
        raise RuntimeError("Перед камерой нет видимой геометрии")
    near, far = max(min(ds), 0.01), max(ds)
    mg = (far - near) * 0.05
    return max(near - mg, 0.01), far + mg


def _file_out(tree, outdir, item_name, color_mode, color_depth):
    """File Output под Blender 5.x: media IMAGE, формат на item'е,
    линк строго в inputs[0] (правило №11/№16 из полевых заметок)."""
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.directory = outdir
    fo.file_name = ""
    fo.format.media_type = 'IMAGE'
    fo.file_output_items.clear()
    it = fo.file_output_items.new('FLOAT', item_name)
    it.override_node_format = True
    it.format.file_format = 'PNG'
    it.format.color_mode = color_mode
    it.format.color_depth = color_depth
    it.save_as_render = False
    return fo


def _render_pack(context, zones):
    """Рендер depth + масок по зонам. zones: [(mat_name, dilate, blur)].
    Возвращает {'depth': path, 'masks': {mat: path}}."""
    scene = context.scene
    vl = context.view_layer
    outdir = os.path.join(tempfile.gettempdir(), "comfyder_pack")
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        try:
            os.remove(os.path.join(outdir, f))
        except OSError:
            pass

    near, far = _depth_range(scene)

    old_group = scene.compositing_node_group
    old_use = scene.render.use_compositing
    old = {"z": vl.use_pass_z, "cm": vl.use_pass_cryptomatte_material}
    vl.use_pass_z = True
    vl.use_pass_cryptomatte_material = True
    if hasattr(vl, "use_pass_cryptomatte_accurate"):
        vl.use_pass_cryptomatte_accurate = True
    vl.pass_cryptomatte_depth = 6
    scene.render.use_compositing = True

    tmp = bpy.data.node_groups.get("_comfyder_pack_tmp")
    if tmp:
        bpy.data.node_groups.remove(tmp)
    tree = bpy.data.node_groups.new("_comfyder_pack_tmp", "CompositorNodeTree")
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    rl.layer = vl.name
    tree.interface.new_socket("Image", in_out='OUTPUT',
                              socket_type='NodeSocketColor')
    gout = tree.nodes.new("NodeGroupOutput")
    tree.links.new(rl.outputs["Image"], gout.inputs[0])

    # depth: near -> 1 (белое), far -> 0
    mr = tree.nodes.new("ShaderNodeMapRange")
    mr.data_type = 'FLOAT'
    mr.clamp = True
    tree.links.new(rl.outputs["Depth"], mr.inputs[0])
    mr.inputs[1].default_value = near
    mr.inputs[2].default_value = far
    mr.inputs[3].default_value = 1.0
    mr.inputs[4].default_value = 0.0
    fo_d = _file_out(tree, outdir, "cp_depth", 'BW', '16')
    tree.links.new(mr.outputs[0], fo_d.inputs[0])

    crypto_layer = f"{vl.name}.CryptoMaterial"
    for mat_name, dil, blr in zones:
        cr = tree.nodes.new("CompositorNodeCryptomatteV2")
        cr.source = "RENDER"
        cr.scene = scene
        try:
            cr.layer_name = crypto_layer
        except TypeError:
            opts = [i.identifier for i in
                    cr.bl_rna.properties["layer_name"].enum_items
                    if "CryptoMaterial" in i.identifier]
            if opts:
                cr.layer_name = opts[0]
        cr.matte_id = mat_name
        di = tree.nodes.new("CompositorNodeDilateErode")
        if hasattr(di, "mode"):
            di.mode = "DISTANCE"
        if hasattr(di, "distance"):
            di.distance = dil
        bl = tree.nodes.new("CompositorNodeBlur")
        if hasattr(bl, "filter_type"):
            bl.filter_type = "GAUSS"
        if hasattr(bl, "size_x"):
            bl.size_x = bl.size_y = blr
        if hasattr(bl, "use_relative"):
            bl.use_relative = False
        tree.links.new(cr.outputs["Matte"], di.inputs[0])
        tree.links.new(di.outputs[0], bl.inputs[0])
        fo = _file_out(tree, outdir, f"cp_mask_{mat_name}", 'BW', '8')
        tree.links.new(bl.outputs[0], fo.inputs[0])

    scene.compositing_node_group = tree
    try:
        bpy.ops.render.render(write_still=False)
    finally:
        scene.compositing_node_group = old_group
        scene.render.use_compositing = old_use
        vl.use_pass_z = old["z"]
        vl.use_pass_cryptomatte_material = old["cm"]
        bpy.data.node_groups.remove(tree)

    depth = os.path.join(outdir, "cp_depth.png")
    masks = {m: os.path.join(outdir, f"cp_mask_{m}.png") for m, _, _ in zones}
    missing = [p for p in [depth] + list(masks.values())
               if not os.path.isfile(p)]
    if missing:
        raise RuntimeError("Пак не записался: " +
                           ", ".join(os.path.basename(p) for p in missing))
    return {"depth": depth, "masks": masks}


# =================================================================== граф
def _build_graph(p, zones, depth_name, mask_names, width, height):
    """zones: список dict'ов с настройками зон (из материалов, по порядку)."""
    graph = {}
    nid = [0]

    def node(class_type, _title=None, **inputs):
        nid[0] += 1
        k = str(nid[0])
        graph[k] = {"class_type": class_type, "inputs": inputs}
        if _title:
            graph[k]["_meta"] = {"title": _title}
        return k

    def save(img, name, title):
        node("SaveImage", _title=title, images=[img, 0],
             filename_prefix=f"comfyder_pro/{name}")

    def composite_back(edited, mask_node, current, zname,
                       src_w=None, src_h=None):
        """Выход движка -> (скейл/кроп) -> врезка строго по маске."""
        if src_w:  # известный размер выхода (Gemini 2K): аспект-безопасно
            sw = round(height * src_w / src_h)
            scaled = node("ImageScale", _title=f"Подгон {zname}",
                          image=[edited, 0], upscale_method="lanczos",
                          width=sw, height=height, crop="disabled")
            src = node("ImageCrop", _title=f"Кроп {zname}",
                       image=[scaled, 0], width=width, height=height,
                       x=(sw - width) // 2, y=0)
        else:
            src = node("ImageScale", _title=f"Подгон {zname}",
                       image=[edited, 0], upscale_method="lanczos",
                       width=width, height=height, crop="disabled")
        return node("ImageCompositeMasked", _title=f"Врезка — {zname}",
                    destination=[current, 0], source=[src, 0],
                    x=0, y=0, resize_source=False, mask=[mask_node, 0])

    depth_load = node("LoadImage", _title="Вход: depth", image=depth_name)
    current = node(
        "FluxGeneral_fal", _title="1) Глобальный проход (Flux + depth)",
        prompt=p.scene_prompt, image_size="custom",
        width=width, height=height,
        num_inference_steps=28, guidance_scale=3.0, real_cfg_scale=3.3,
        num_images=1, enable_safety_checker=False, use_real_cfg=False,
        sync_mode=False, seed=p.seed,
        controlnet_unions="InstantX/FLUX.1-dev-Controlnet-Union",
        controlnet_union_control_mode="depth",
        controlnet_conditioning_scale=p.conditioning,
        control_image=[depth_load, 0])
    save(current, "00_global", "Сейв 00: глобальный")

    for i, z in enumerate(zones, start=1):
        zname = z["name"]
        seed = z["seed"] or p.seed
        mask_load = node("LoadImage", _title=f"Маска {zname}",
                         image=mask_names[zname])
        m = node("ImageToMask", _title=f"MASK {zname}",
                 image=[mask_load, 0], channel="red")
        eng = z["engine"]
        if eng == "fill":
            current = node(
                "FluxPro1Fill_fal", _title=f"2.{i}) Fill — {zname}",
                prompt=z["prompt"], num_images=1, safety_tolerance="5",
                output_format="png", image=[current, 0],
                mask_image=[mask_load, 0], seed=seed,
                sync_mode=False, enhance_prompt=False)
        elif eng == "qwen":
            edited = node(
                "FalQwenImageEditInpaint", _title=f"2.{i}) Qwen — {zname}",
                image=[current, 0], mask=[m, 0], prompt=z["prompt"],
                strength=z["strength"], guidance_scale=4.0,
                num_inference_steps=30, negative_prompt=z["negative"],
                num_images=1, seed=seed)
            current = composite_back(edited, m, current, zname)
        elif eng == "zturbo":
            edited = node(
                "FalZImageTurboInpaint", _title=f"2.{i}) Z-Turbo — {zname}",
                image=[current, 0], mask=[m, 0], prompt=z["prompt"],
                strength=z["strength"], num_inference_steps=8,
                acceleration="regular", num_images=1, seed=seed)
            current = composite_back(edited, m, current, zname)
        elif eng == "gemini_zone":
            tgt = z["target"] or f"the {zname[len(MAT_PREFIX):]} zone"
            edited = node(
                "FalGeminiFlashEdit", _title=f"2.{i}) Gemini — {zname}",
                image=[current, 0], prompt=z["prompt"],
                version="3.1-flash-preview", resolution="2K",
                system_prompt=("You are editing exactly one zone of the "
                               f"image. Change ONLY {tgt}. Keep composition, "
                               "all other objects, lighting, framing and "
                               "aspect ratio exactly unchanged."),
                num_images=1, seed=seed)
            current = composite_back(edited, m, current, zname,
                                     src_w=2752, src_h=1536)
        elif eng == "kontext_zone":
            tgt = z["target"] or "the masked area"
            instr = (f"Change {tgt} to: {z['prompt']}. Keep the composition, "
                     "camera, lighting and everything else exactly unchanged.")
            edited = node(
                "FluxProKontext_fal", _title=f"2.{i}) Kontext — {zname}",
                prompt=instr, image=[current, 0], aspect_ratio="16:9",
                guidance_scale=3.5, num_images=1, safety_tolerance="5",
                output_format="png", sync_mode=False)
            current = composite_back(edited, m, current, zname)
        save(current, f"{i:02d}_{zname}", f"Сейв {i:02d}: {zname}")

    if p.final_enabled:
        protect = [z["target"] or z["prompt"].split(",")[0]
                   for z in zones if z["protect"]]
        fp = p.mood.strip()
        if protect:
            fp += " Keep exactly: " + "; ".join(protect) + "."
        fp += " No new objects, no lamps."
        current = node(
            "FalGeminiFlashEdit", _title="3) Финал — Gemini (кадр + depth)",
            image=[current, 0], image_2=[depth_load, 0],
            version="3.1-flash-preview", resolution="2K",
            system_prompt=("The first image is the artwork to refine. The "
                           "second image is its depth map (white = near "
                           "camera) — use it ONLY as geometry reference, "
                           "never draw it. Preserve the exact composition, "
                           "framing and aspect ratio of the first image."),
            prompt=fp, num_images=1, seed=p.seed)
        save(current, "99_final", "Сейв 99: финал")
    return graph


# =================================================================== статус
def _set_status(txt):
    sc = bpy.data.scenes.get(_JOB.get("scene") or "") or bpy.data.scenes[0]
    sc.comfyder_pro.status = txt
    wm = bpy.context.window_manager
    if wm:
        for w in wm.windows:
            for a in w.screen.areas:
                if a.type in ('IMAGE_EDITOR', 'VIEW_3D'):
                    a.tag_redraw()


def _poll():
    pid, host = _JOB.get("prompt_id"), _JOB.get("host")
    if not pid:
        return None
    try:
        h = _http_json(f"{host}/history/{pid}", timeout=10)
    except Exception as e:
        _set_status("Ошибка сети: " + str(e)[:60])
        _JOB["prompt_id"] = None
        return None
    entry = h.get(pid)
    st = (entry or {}).get("status", {})
    if st.get("status_str") == "error":
        _set_status("Ошибка выполнения — смотри ComfyUI")
        _JOB["prompt_id"] = None
        return None
    if not entry or not entry.get("outputs"):
        if time.time() - _JOB["t0"] > 1800:
            _set_status("Таймаут 30 мин — проверь очередь ComfyUI")
            _JOB["prompt_id"] = None
            return None
        _set_status(f"Генерация… {int(time.time() - _JOB['t0'])}с")
        return 4.0

    out_dir = _JOB["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    final_path = None
    try:
        for node_id, o in sorted(entry["outputs"].items(),
                                 key=lambda kv: int(kv[0])):
            for im in o.get("images", []):
                qs = urllib.parse.urlencode({
                    "filename": im["filename"],
                    "subfolder": im.get("subfolder", ""),
                    "type": im.get("type", "output")})
                data = urllib.request.urlopen(
                    f"{host}/view?{qs}", timeout=120).read()
                dst = os.path.join(out_dir, im["filename"])
                with open(dst, "wb") as fh:
                    fh.write(data)
                if im["filename"].startswith(_JOB["expected_final"]):
                    final_path = dst
                if final_path is None:
                    final_path = dst  # хотя бы последний шаг
        if final_path:
            img = bpy.data.images.load(final_path, check_existing=False)
            img.name = "Comfyder Pro Result"
            for w in bpy.context.window_manager.windows:
                for a in w.screen.areas:
                    if a.type == 'IMAGE_EDITOR':
                        a.spaces.active.image = img
                        a.tag_redraw()
                        break
        _set_status("Готово — шаги в " + out_dir)
    except Exception as e:
        _set_status("Ошибка загрузки результата: " + str(e)[:60])
    _JOB["prompt_id"] = None
    return None


# =================================================================== props
ENGINE_ITEMS = [
    ("fill", "Fill — перекраска",
     "FluxPro1Fill: пиксельно точен по маске. Смена цвета, тонкие структуры "
     "(ветки, провода), бутоны"),
    ("qwen", "Qwen — фактура",
     "Сохраняет подложку (ползунок strength) + негативный промпт. "
     "Дошлифовка фактуры. НЕ для тонкого — ресемплит кадр"),
    ("gemini_zone", "Gemini — широкая зона",
     "Умный edit c врезкой по маске, 2K. Стены, вода, фоны"),
    ("kontext_zone", "Kontext — структура",
     "Держит форму объекта. Склонен «лакировать» органику"),
    ("zturbo", "Z-Turbo — черновик", "Быстро и дёшево, для прикидок"),
]


class ComfyderZoneSettings(bpy.types.PropertyGroup):
    prompt: bpy.props.StringProperty(
        name="Промпт", description="Описание материала зоны (лучше EN)")
    engine: bpy.props.EnumProperty(name="Движок", items=ENGINE_ITEMS,
                                   default="qwen")
    strength: bpy.props.FloatProperty(
        name="Strength", default=0.70, min=0.0, max=1.0,
        description="Qwen/Z-Turbo: сколько перерисовывать. ~0.7 фактура, "
                    "0.85+ перекрас (цвет всё равно задаёт глобальный проход)")
    negative: bpy.props.StringProperty(
        name="Негатив", description="Qwen: чего в зоне быть не должно")
    target: bpy.props.StringProperty(
        name="Target", description="Gemini/Kontext: что именно менять "
                                   "(например: the large torus ring)")
    dilate: bpy.props.IntProperty(
        name="Dilate px", default=6, min=0, max=64,
        description="Расширение маски. Цветам/свечению 15–25")
    blur: bpy.props.IntProperty(
        name="Blur px", default=4, min=0, max=64,
        description="Мягкость края маски. Цветам 15+")
    protect: bpy.props.BoolProperty(
        name="Защитить в финале", default=True,
        description="Добавить «keep …» в финальный промпт")
    seed: bpy.props.IntProperty(
        name="Seed", default=0, min=0,
        description="0 = seed сцены. Залочь зону при итерациях")


class ComfyderZoneRef(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()


class ComfyderProProps(bpy.types.PropertyGroup):
    host: bpy.props.StringProperty(name="ComfyUI",
                                   default="http://192.168.1.2:8188")
    scene_prompt: bpy.props.StringProperty(
        name="Промпт сцены",
        description="Глобальный проход. Материалы в начале, окружение в конце",
        default="")
    mood: bpy.props.StringProperty(
        name="Настроение",
        default="cohesive hyperreal 4K render quality, unified soft studio "
                "lighting, subtle atmospheric haze, shallow depth of field.")
    conditioning: bpy.props.FloatProperty(
        name="Сила depth", default=0.75, min=0.4, max=0.95,
        description="ControlNet conditioning: как жёстко держать форму")
    seed: bpy.props.IntProperty(name="Seed", default=7, min=0)
    final_enabled: bpy.props.BoolProperty(name="Финальный проход", default=True)
    zones: bpy.props.CollectionProperty(type=ComfyderZoneRef)
    zone_index: bpy.props.IntProperty(default=0)
    status: bpy.props.StringProperty(default="")
    output_dir: bpy.props.StringProperty(
        name="Результаты", subtype='DIR_PATH', default="//comfyder_out")


# =================================================================== zone ops
def _used_mats():
    return [m.name for m in bpy.data.materials
            if m.users > 0 and not m.is_grease_pencil
            and m.name.startswith(MAT_PREFIX)]


class COMFYDERPRO_OT_zone_sync(bpy.types.Operator):
    bl_idname = "comfyder_pro.zone_sync"
    bl_label = "Синхронизировать зоны"
    bl_description = "Добавить в список все используемые материалы mat_*"

    def execute(self, context):
        p = context.scene.comfyder_pro
        known = {z.name for z in p.zones}
        added = 0
        for name in _used_mats():
            if name not in known:
                p.zones.add().name = name
                added += 1
        # убрать исчезнувшие
        for i in range(len(p.zones) - 1, -1, -1):
            if p.zones[i].name not in bpy.data.materials:
                p.zones.remove(i)
        self.report({'INFO'}, f"Добавлено зон: {added}")
        return {'FINISHED'}


class COMFYDERPRO_OT_zone_add(bpy.types.Operator):
    bl_idname = "comfyder_pro.zone_add"
    bl_label = "Новая зона"
    bl_description = "Создать материал mat_* и добавить зоной"

    name: bpy.props.StringProperty(name="Имя (латиницей)", default="new")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        import random
        slug = "".join(c if c.isalnum() else "_" for c in self.name.lower())
        mat_name = MAT_PREFIX + slug
        if mat_name in bpy.data.materials:
            self.report({'WARNING'}, f"{mat_name} уже существует")
        else:
            m = bpy.data.materials.new(mat_name)
            m.use_nodes = True
            col = (random.random(), random.random(), random.random(), 1)
            b = m.node_tree.nodes.get("Principled BSDF")
            if b:
                b.inputs["Base Color"].default_value = col
            m.diffuse_color = col
        p = context.scene.comfyder_pro
        if mat_name not in {z.name for z in p.zones}:
            p.zones.add().name = mat_name
            p.zone_index = len(p.zones) - 1
        return {'FINISHED'}


class COMFYDERPRO_OT_zone_remove(bpy.types.Operator):
    bl_idname = "comfyder_pro.zone_remove"
    bl_label = "Убрать зону из списка"

    def execute(self, context):
        p = context.scene.comfyder_pro
        if 0 <= p.zone_index < len(p.zones):
            p.zones.remove(p.zone_index)
            p.zone_index = min(p.zone_index, len(p.zones) - 1)
        return {'FINISHED'}


class COMFYDERPRO_OT_zone_move(bpy.types.Operator):
    bl_idname = "comfyder_pro.zone_move"
    bl_label = "Сдвинуть зону"
    bl_description = "Порядок в списке = порядок проходов (крупное → мелкое)"

    direction: bpy.props.EnumProperty(items=[("UP", "Up", ""),
                                             ("DOWN", "Down", "")])

    def execute(self, context):
        p = context.scene.comfyder_pro
        i = p.zone_index
        j = i - 1 if self.direction == "UP" else i + 1
        if 0 <= j < len(p.zones):
            p.zones.move(i, j)
            p.zone_index = j
        return {'FINISHED'}


class COMFYDERPRO_OT_build_prompt(bpy.types.Operator):
    bl_idname = "comfyder_pro.build_prompt"
    bl_label = "Собрать промпт сцены из зон"
    bl_description = ("Правило №1: цвета задаёт глобальный проход — "
                      "материалы в начале промпта")

    def execute(self, context):
        p = context.scene.comfyder_pro
        parts = []
        for z in p.zones:
            mat = bpy.data.materials.get(z.name)
            if mat and mat.comfyder.prompt.strip():
                parts.append(mat.comfyder.prompt.strip())
        p.scene_prompt = ", ".join(parts)
        return {'FINISHED'}


class COMFYDERPRO_OT_edit_text(bpy.types.Operator):
    bl_idname = "comfyder_pro.edit_text"
    bl_label = "Редактор"

    which: bpy.props.StringProperty()  # scene | mood | zone
    text: bpy.props.StringProperty(name="")

    def invoke(self, context, event):
        p = context.scene.comfyder_pro
        if self.which == "scene":
            self.text = p.scene_prompt
        elif self.which == "mood":
            self.text = p.mood
        else:
            mat = bpy.data.materials.get(p.zones[p.zone_index].name)
            self.text = mat.comfyder.prompt if mat else ""
        return context.window_manager.invoke_props_dialog(self, width=620)

    def draw(self, context):
        self.layout.prop(self, "text", text="")

    def execute(self, context):
        p = context.scene.comfyder_pro
        if self.which == "scene":
            p.scene_prompt = self.text
        elif self.which == "mood":
            p.mood = self.text
        else:
            mat = bpy.data.materials.get(p.zones[p.zone_index].name)
            if mat:
                mat.comfyder.prompt = self.text
        return {'FINISHED'}


# =================================================================== generate
class COMFYDERPRO_OT_generate(bpy.types.Operator):
    bl_idname = "comfyder_pro.generate"
    bl_label = "Generate"
    bl_description = "Полный прогон: пак → глобальный → зоны → финал"

    def execute(self, context):
        p = context.scene.comfyder_pro
        scene = context.scene
        if _JOB.get("prompt_id"):
            self.report({'WARNING'}, "Уже идёт генерация")
            return {'CANCELLED'}
        if not p.zones:
            self.report({'ERROR'}, "Нет зон — нажми «Синхронизировать»")
            return {'CANCELLED'}
        if not p.scene_prompt.strip():
            self.report({'ERROR'}, "Пустой промпт сцены — собери из зон")
            return {'CANCELLED'}

        zones = []
        for z in p.zones:
            mat = bpy.data.materials.get(z.name)
            if mat is None or mat.users == 0:
                self.report({'ERROR'}, f"Материал {z.name} не используется")
                return {'CANCELLED'}
            s = mat.comfyder
            if not s.prompt.strip():
                self.report({'ERROR'}, f"У зоны {z.name} пустой промпт")
                return {'CANCELLED'}
            zones.append({"name": z.name, "prompt": s.prompt.strip(),
                          "engine": s.engine, "strength": s.strength,
                          "negative": s.negative, "target": s.target,
                          "dilate": s.dilate, "blur": s.blur,
                          "protect": s.protect, "seed": s.seed})

        # разрешение: ≤1536, кратно 16 (лимит FluxGeneral)
        rw, rh = scene.render.resolution_x, scene.render.resolution_y
        w = min(rw, 1536) // 16 * 16
        h = round(w * rh / rw) // 16 * 16
        if (w, h) != (rw, rh):
            scene.render.resolution_x, scene.render.resolution_y = w, h
            self.report({'INFO'}, f"Разрешение подогнано: {w}x{h}")

        _set_status("Рендер пака…")
        try:
            pack = _render_pack(context, [(z["name"], z["dilate"], z["blur"])
                                          for z in zones])
        except Exception as e:
            self.report({'ERROR'}, f"Пак: {str(e)[:80]}")
            _set_status("")
            return {'CANCELLED'}

        host = p.host.rstrip("/")
        try:
            depth_name = _upload(host, pack["depth"])
            mask_names = {m: _upload(host, path)
                          for m, path in pack["masks"].items()}
            graph = _build_graph(p, zones, depth_name, mask_names, w, h)
            resp = _http_json(host + "/prompt",
                              {"prompt": graph,
                               "client_id": str(uuid.uuid4())}, timeout=60)
        except Exception as e:
            self.report({'ERROR'}, f"ComfyUI: {str(e)[:80]}")
            _set_status("")
            return {'CANCELLED'}
        if resp.get("node_errors"):
            self.report({'ERROR'}, str(resp["node_errors"])[:120])
            _set_status("")
            return {'CANCELLED'}

        _JOB.update(prompt_id=resp["prompt_id"], host=host, t0=time.time(),
                    scene=scene.name,
                    out_dir=bpy.path.abspath(p.output_dir))
        _set_status(f"Отправлено ({len(zones)} зон), жду…")
        if not bpy.app.timers.is_registered(_poll):
            bpy.app.timers.register(_poll, first_interval=4.0)
        return {'FINISHED'}


# =================================================================== UI
class COMFYDERPRO_UL_zones(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        mat = bpy.data.materials.get(item.name)
        row = layout.row(align=True)
        if mat:
            row.label(text=item.name, icon='MATERIAL')
            eng = mat.comfyder.engine
            short = {"fill": "Fill", "qwen": "Qwen", "gemini_zone": "Gem",
                     "kontext_zone": "Kntx", "zturbo": "ZT"}.get(eng, "?")
            row.label(text=short)
        else:
            row.label(text=item.name + " (нет)", icon='ERROR')


def _wrap_preview(col, text, width=44, lines=5):
    if len(text) > 38:
        box = col.box()
        bc = box.column(align=True)
        bc.scale_y = 0.75
        for line in textwrap.wrap(text, width=width)[:lines]:
            bc.label(text=line)


def _draw(panel, context):
    p = context.scene.comfyder_pro
    lay = panel.layout

    box = lay.box()
    box.label(text="Сцена", icon='SCENE_DATA')
    row = box.row(align=True)
    row.prop(p, "scene_prompt", text="")
    op = row.operator("comfyder_pro.edit_text", text="", icon='GREASEPENCIL')
    op.which = "scene"
    _wrap_preview(box, p.scene_prompt)
    box.operator("comfyder_pro.build_prompt", icon='FILE_REFRESH')
    box.prop(p, "conditioning", slider=True)
    box.prop(p, "seed")
    cam = context.scene.camera
    if cam and getattr(cam.data, "lens", 0) > 40:
        box.label(text=f"Объектив {cam.data.lens:.0f}мм: depth будет плоским,"
                       " лучше 24мм ближе", icon='ERROR')

    box = lay.box()
    box.label(text="Зоны (порядок = порядок проходов)", icon='MATERIAL')
    row = box.row()
    row.template_list("COMFYDERPRO_UL_zones", "", p, "zones", p, "zone_index",
                      rows=4)
    col = row.column(align=True)
    col.operator("comfyder_pro.zone_add", text="", icon='ADD')
    col.operator("comfyder_pro.zone_remove", text="", icon='REMOVE')
    col.separator()
    col.operator("comfyder_pro.zone_move", text="", icon='TRIA_UP').direction = "UP"
    col.operator("comfyder_pro.zone_move", text="", icon='TRIA_DOWN').direction = "DOWN"
    box.operator("comfyder_pro.zone_sync", icon='FILE_REFRESH')

    if 0 <= p.zone_index < len(p.zones):
        mat = bpy.data.materials.get(p.zones[p.zone_index].name)
        if mat:
            s = mat.comfyder
            zb = box.box()
            zb.label(text=mat.name)
            row = zb.row(align=True)
            row.prop(s, "prompt", text="")
            op = row.operator("comfyder_pro.edit_text", text="",
                              icon='GREASEPENCIL')
            op.which = "zone"
            _wrap_preview(zb, s.prompt)
            zb.prop(s, "engine", text="")
            if s.engine in ("qwen", "zturbo"):
                zb.prop(s, "strength", slider=True)
            if s.engine == "qwen":
                zb.prop(s, "negative", text="Негатив")
            if s.engine in ("gemini_zone", "kontext_zone"):
                zb.prop(s, "target", text="Target")
            row = zb.row(align=True)
            row.prop(s, "dilate")
            row.prop(s, "blur")
            row = zb.row(align=True)
            row.prop(s, "protect")
            row.prop(s, "seed", text="Seed")

    box = lay.box()
    box.prop(p, "final_enabled", icon='SHADERFX')
    if p.final_enabled:
        row = box.row(align=True)
        row.prop(p, "mood", text="")
        op = row.operator("comfyder_pro.edit_text", text="",
                          icon='GREASEPENCIL')
        op.which = "mood"
        _wrap_preview(box, p.mood)

    lay.separator()
    lay.operator("comfyder_pro.generate", icon='PLAY')
    if p.status:
        lay.label(text=p.status)
    lay.prop(p, "output_dir", text="")
    lay.prop(p, "host", text="")


class COMFYDERPRO_PT_view3d(bpy.types.Panel):
    bl_label = "Comfyder Pro"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Comfyder Pro"

    def draw(self, context):
        _draw(self, context)


class COMFYDERPRO_PT_image(bpy.types.Panel):
    bl_label = "Comfyder Pro"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Comfyder Pro"

    def draw(self, context):
        _draw(self, context)


# =================================================================== reg
classes = (ComfyderZoneSettings, ComfyderZoneRef, ComfyderProProps,
           COMFYDERPRO_OT_zone_sync, COMFYDERPRO_OT_zone_add,
           COMFYDERPRO_OT_zone_remove, COMFYDERPRO_OT_zone_move,
           COMFYDERPRO_OT_build_prompt, COMFYDERPRO_OT_edit_text,
           COMFYDERPRO_OT_generate, COMFYDERPRO_UL_zones,
           COMFYDERPRO_PT_view3d, COMFYDERPRO_PT_image)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Material.comfyder = bpy.props.PointerProperty(
        type=ComfyderZoneSettings)
    bpy.types.Scene.comfyder_pro = bpy.props.PointerProperty(
        type=ComfyderProProps)


def unregister():
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    del bpy.types.Scene.comfyder_pro
    del bpy.types.Material.comfyder
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
