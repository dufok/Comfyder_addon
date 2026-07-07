# Comfyder Lite v0.1 — обработка рендера промптом через ComfyUI + FAL (Gemini).
#
# Установка: Edit > Preferences > Add-ons > Install from Disk > этот файл.
# Использование: рендер (F12) > в окне Render Result нажать N > таб Comfyder >
# написать промпт настроения > Generate. Результат появится в Image Editor
# отдельным изображением "Comfyder Result".
#
# Архитектура (проверена вручную, см. vault: Comfyder Plugin §Lite):
#   кадр -> [VLM_fal: описание сцены + mood] -> FalGeminiFlashEdit -> результат
# Поллинг /history через bpy.app.timers — UI не блокируется.

bl_info = {
    "name": "Comfyder Lite",
    "author": "Stepan Vladovskiy",
    "version": (0, 1, 4),
    "blender": (5, 0, 0),
    "location": "Image Editor > Sidebar (N) > Comfyder",
    "description": "AI-обработка рендера промптом через ComfyUI + FAL",
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

# ----------------------------------------------------------------- состояние
_JOB = {"prompt_id": None, "host": None, "t0": 0.0, "scene": None}


# ----------------------------------------------------------------- HTTP
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


# ----------------------------------------------------------------- граф
def _build_graph(image_name, mood, use_vlm, resolution, seed, depth_name=None):
    keep = ("Preserve the exact composition, framing, aspect ratio and all "
            "objects of the first image.")
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": image_name},
               "_meta": {"title": "Comfyder: вход"}}}
    if depth_name:
        g["9"] = {"class_type": "LoadImage", "inputs": {"image": depth_name},
                  "_meta": {"title": "Comfyder: depth"}}
    if use_vlm:
        g["2"] = {"class_type": "VLM_fal", "_meta": {"title": "Comfyder: VLM"},
                  "inputs": {
                      "image": ["1", 0],
                      "prompt": "Describe this image.",
                      "system_prompt": (
                          "You write concise scene descriptions for image "
                          "generation models. Describe the objects, their "
                          "materials and the composition in one paragraph, "
                          "English only. Then append EXACTLY this sentence at "
                          "the end: '" + mood + "' Output ONLY the final text."),
                      "model": "google/gemini-2.5-flash",
                      "temperature": 0.4, "reasoning": False,
                      "max_tokens": 400}}
        prompt = ["2", 0]
    else:
        prompt = mood + " " + keep
    sysp = "The image is a 3D render to refine. " + keep
    inputs = {"image": ["1", 0],
              "version": "3.1-flash-preview",
              "resolution": resolution,
              "system_prompt": sysp,
              "prompt": prompt,
              "num_images": 1, "seed": seed}
    if depth_name:
        inputs["image_2"] = ["9", 0]
        inputs["system_prompt"] = (
            "The first image is a 3D render to refine. The second image is "
            "its depth map (white = near camera) — use it ONLY as geometry "
            "and spatial reference, never draw it. " + keep)
    g["3"] = {"class_type": "FalGeminiFlashEdit",
              "_meta": {"title": "Comfyder: Gemini"},
              "inputs": inputs}
    g["4"] = {"class_type": "SaveImage", "_meta": {"title": "Comfyder: сейв"},
              "inputs": {"images": ["3", 0],
                         "filename_prefix": "comfyder_lite/result"}}
    return g


# ----------------------------------------------------------------- depth
def _render_depth(context):
    """Рендер карты глубины во временный PNG (Blender 5.x компоузер).

    Временный node group: Z-pass -> Map Range (near=1, far=0) -> File Output.
    Исходный компоузер сцены сохраняется и восстанавливается.
    ВНИМАНИЕ: перерендеривает сцену — Render Result будет заменён
    (входной кадр к этому моменту уже сохранён во временный файл).
    """
    from mathutils import Vector
    scene = context.scene
    vl = context.view_layer
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
    near, far = max(near - mg, 0.01), far + mg

    old_group = scene.compositing_node_group
    old_use = scene.render.use_compositing
    old_z = vl.use_pass_z
    vl.use_pass_z = True
    scene.render.use_compositing = True

    tmp = bpy.data.node_groups.get("_comfyder_depth_tmp")
    if tmp:
        bpy.data.node_groups.remove(tmp)
    tree = bpy.data.node_groups.new("_comfyder_depth_tmp", "CompositorNodeTree")
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    rl.layer = vl.name
    tree.interface.new_socket("Image", in_out='OUTPUT',
                              socket_type='NodeSocketColor')
    gout = tree.nodes.new("NodeGroupOutput")
    tree.links.new(rl.outputs["Image"], gout.inputs[0])
    mr = tree.nodes.new("ShaderNodeMapRange")
    mr.data_type = 'FLOAT'
    mr.clamp = True
    tree.links.new(rl.outputs["Depth"], mr.inputs[0])
    mr.inputs[1].default_value = near
    mr.inputs[2].default_value = far
    mr.inputs[3].default_value = 1.0
    mr.inputs[4].default_value = 0.0
    fo = tree.nodes.new("CompositorNodeOutputFile")
    outdir = tempfile.gettempdir()
    fo.directory = outdir
    fo.file_name = ""
    fo.format.media_type = 'IMAGE'
    fo.file_output_items.clear()
    it = fo.file_output_items.new('FLOAT', 'comfyder_depth')
    it.override_node_format = True
    it.format.file_format = 'PNG'
    it.format.color_mode = 'BW'
    it.format.color_depth = '16'
    it.save_as_render = False
    tree.links.new(mr.outputs[0], fo.inputs[0])

    scene.compositing_node_group = tree
    try:
        bpy.ops.render.render(write_still=False)
    finally:
        scene.compositing_node_group = old_group
        scene.render.use_compositing = old_use
        vl.use_pass_z = old_z
        bpy.data.node_groups.remove(tree)

    path = os.path.join(outdir, "comfyder_depth.png")
    if not os.path.isfile(path):
        raise RuntimeError("Depth-файл не записался")
    return path


# ----------------------------------------------------------------- статус
def _set_status(txt):
    sc = bpy.data.scenes.get(_JOB.get("scene") or "") or bpy.data.scenes[0]
    sc.comfyder.status = txt
    wm = bpy.context.window_manager
    if wm:
        for w in wm.windows:
            for a in w.screen.areas:
                if a.type in ('IMAGE_EDITOR', 'VIEW_3D'):
                    a.tag_redraw()


# ----------------------------------------------------------------- поллинг
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
    if not entry or not entry.get("outputs"):
        st = (entry or {}).get("status", {})
        if st.get("status_str") == "error":
            _set_status("Ошибка выполнения — смотри ComfyUI")
            _JOB["prompt_id"] = None
            return None
        if time.time() - _JOB["t0"] > 420:
            _set_status("Таймаут 7 мин — проверь очередь ComfyUI")
            _JOB["prompt_id"] = None
            return None
        _set_status(f"Генерация… {int(time.time() - _JOB['t0'])}с")
        return 3.0

    # готово — качаем первую картинку
    try:
        for o in entry["outputs"].values():
            for im in o.get("images", []):
                qs = urllib.parse.urlencode({
                    "filename": im["filename"],
                    "subfolder": im.get("subfolder", ""),
                    "type": im.get("type", "output")})
                data = urllib.request.urlopen(
                    f"{host}/view?{qs}", timeout=120).read()
                out = os.path.join(tempfile.gettempdir(),
                                   f"comfyder_{pid[:8]}.png")
                with open(out, "wb") as fh:
                    fh.write(data)
                img = bpy.data.images.load(out, check_existing=False)
                img.name = "Comfyder Result"
                shown = False
                for w in bpy.context.window_manager.windows:
                    for a in w.screen.areas:
                        if a.type == 'IMAGE_EDITOR' and not shown:
                            a.spaces.active.image = img
                            a.tag_redraw()
                            shown = True
                _set_status("Готово: " + img.name)
                _JOB["prompt_id"] = None
                return None
    except Exception as e:
        _set_status("Ошибка загрузки результата: " + str(e)[:60])
        _JOB["prompt_id"] = None
        return None
    return 3.0


# ----------------------------------------------------------------- свойства
class ComfyderProps(bpy.types.PropertyGroup):
    host: bpy.props.StringProperty(
        name="ComfyUI", default="http://192.168.1.2:8188")
    mood: bpy.props.StringProperty(
        name="Промпт", description="Настроение/стиль — можно по-русски",
        default="4K high resolution render quality, cinematic soft lighting, "
                "subtle atmospheric haze")
    use_vlm: bpy.props.BoolProperty(
        name="VLM-описание сцены",
        description="Vision-модель опишет кадр и склеит описание с промптом",
        default=True)
    resolution: bpy.props.EnumProperty(
        name="Разрешение",
        items=[("1K", "1K", ""), ("2K", "2K", ""), ("4K", "4K", "")],
        default="2K")
    seed: bpy.props.IntProperty(name="Seed", default=7, min=0)
    source: bpy.props.EnumProperty(
        name="Источник",
        items=[("RENDER", "Рендер", "Последний Render Result (F12)"),
               ("VIEWPORT", "Вьюпорт", "OpenGL-снимок активного вьюпорта — "
                                       "быстрый черновик без рендера")],
        default="RENDER")
    use_depth: bpy.props.BoolProperty(
        name="Прикладывать depth",
        description="Отдать Gemini карту глубины вторым входом — "
                    "жёстче держит геометрию",
        default=False)
    auto_depth: bpy.props.BoolProperty(
        name="Авто-рендер depth",
        description="Сгенерировать свежую карту глубины самому "
                    "(Z-pass, быстрый перерендер; Render Result будет заменён)",
        default=True)
    depth_path: bpy.props.StringProperty(
        name="Depth PNG", subtype='FILE_PATH',
        default="//render_pack/depth.png")
    status: bpy.props.StringProperty(default="")


class COMFYDER_OT_edit_prompt(bpy.types.Operator):
    bl_idname = "comfyder.edit_prompt"
    bl_label = "Промпт настроения"
    bl_description = "Редактировать промпт в широком окне"

    mood: bpy.props.StringProperty(name="", default="")

    def invoke(self, context, event):
        self.mood = context.scene.comfyder.mood
        return context.window_manager.invoke_props_dialog(self, width=620)

    def draw(self, context):
        self.layout.prop(self, "mood", text="")

    def execute(self, context):
        context.scene.comfyder.mood = self.mood
        return {'FINISHED'}


# ----------------------------------------------------------------- оператор
class COMFYDER_OT_generate(bpy.types.Operator):
    bl_idname = "comfyder.generate"
    bl_label = "Generate"
    bl_description = "Отправить кадр в ComfyUI/FAL и получить AI-обработку"

    def execute(self, context):
        p = context.scene.comfyder
        if _JOB.get("prompt_id"):
            self.report({'WARNING'}, "Уже идёт генерация")
            return {'CANCELLED'}

        img = None
        if p.source == 'VIEWPORT':
            try:
                in3d = context.area is not None and context.area.type == 'VIEW_3D'
                bpy.ops.render.opengl(view_context=in3d)
                img = bpy.data.images.get("Render Result")
            except Exception as e:
                self.report({'ERROR'}, f"Вьюпорт-снимок не удался: {str(e)[:60]}")
                return {'CANCELLED'}
        else:
            sp = getattr(context, "space_data", None)
            if sp is not None and getattr(sp, "image", None) is not None:
                img = sp.image
            if img is None:
                img = bpy.data.images.get("Render Result")
        if img is None:
            self.report({'ERROR'}, "Нет изображения — сначала сделай рендер")
            return {'CANCELLED'}

        tmp = os.path.join(tempfile.gettempdir(), "comfyder_in.png")
        try:
            img.save_render(tmp, scene=context.scene)
        except Exception as e:
            self.report({'ERROR'}, f"Не смог сохранить кадр: {e}")
            return {'CANCELLED'}

        host = p.host.rstrip("/")
        depth_name = None
        if p.use_depth:
            dp = None
            if p.auto_depth:
                if p.source == 'VIEWPORT':
                    self.report({'WARNING'},
                                "Авто-depth только с источником «Рендер» — еду без")
                else:
                    try:
                        dp = _render_depth(context)
                    except Exception as e:
                        self.report({'WARNING'},
                                    f"Авто-depth не удался: {str(e)[:60]}")
            else:
                cand = bpy.path.abspath(p.depth_path)
                dp = cand if os.path.isfile(cand) else None
            if dp:
                try:
                    depth_name = _upload(host, dp, "comfyder_depth.png")
                except Exception:
                    depth_name = None
            if depth_name is None:
                self.report({'WARNING'}, "Depth не приложен — еду без него")
        try:
            name = _upload(host, tmp, "comfyder_in.png")
            graph = _build_graph(name, p.mood.strip(), p.use_vlm,
                                 p.resolution, p.seed, depth_name)
            resp = _http_json(host + "/prompt",
                              {"prompt": graph,
                               "client_id": str(uuid.uuid4())}, timeout=60)
        except Exception as e:
            self.report({'ERROR'}, f"ComfyUI недоступен: {str(e)[:80]}")
            return {'CANCELLED'}

        if resp.get("node_errors"):
            self.report({'ERROR'}, "Ошибки нод: " + str(resp["node_errors"])[:100])
            return {'CANCELLED'}

        _JOB.update(prompt_id=resp["prompt_id"], host=host,
                    t0=time.time(), scene=context.scene.name)
        _set_status("Отправлено, жду…")
        if not bpy.app.timers.is_registered(_poll):
            bpy.app.timers.register(_poll, first_interval=3.0)
        return {'FINISHED'}


# ----------------------------------------------------------------- панели
def _draw(panel, context):
    p = context.scene.comfyder
    col = panel.layout.column(align=False)
    row = col.row(align=True)
    row.prop(p, "mood", text="")
    row.operator("comfyder.edit_prompt", text="", icon='GREASEPENCIL')
    # перенос длинного промпта в читаемые строки
    if len(p.mood) > 38:
        box = col.box()
        bc = box.column(align=True)
        bc.scale_y = 0.75
        for line in textwrap.wrap(p.mood, width=44)[:6]:
            bc.label(text=line)
    col.prop(p, "use_vlm")
    row = col.row(align=True)
    row.prop(p, "source", expand=True)
    row = col.row(align=True)
    row.prop(p, "resolution", expand=True)
    col.prop(p, "seed")
    col.prop(p, "use_depth")
    if p.use_depth:
        col.prop(p, "auto_depth")
        if not p.auto_depth:
            col.prop(p, "depth_path", text="")
    col.separator()
    col.operator("comfyder.generate", icon='SHADERFX')
    if p.status:
        col.label(text=p.status)
    col.separator()
    col.prop(p, "host", text="")


class COMFYDER_PT_image(bpy.types.Panel):
    bl_label = "Comfyder Lite"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Comfyder"

    def draw(self, context):
        _draw(self, context)


class COMFYDER_PT_view3d(bpy.types.Panel):
    bl_label = "Comfyder Lite"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Comfyder"

    def draw(self, context):
        _draw(self, context)


# ----------------------------------------------------------------- регистрация
classes = (ComfyderProps, COMFYDER_OT_edit_prompt, COMFYDER_OT_generate,
           COMFYDER_PT_image, COMFYDER_PT_view3d)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.comfyder = bpy.props.PointerProperty(type=ComfyderProps)


def unregister():
    if bpy.app.timers.is_registered(_poll):
        bpy.app.timers.unregister(_poll)
    del bpy.types.Scene.comfyder
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
