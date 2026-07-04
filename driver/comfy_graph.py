"""Сборка ComfyUI workflow (API-формат) под FAL-ноды.

Цепочка:
  LoadImage(depth) -> FluxGeneral_fal(ControlNet Union, mode=depth)   # глобальный проход
  -> проход на материал (mask_1..mask_N):                             # текст или референс
  -> FluxProKontext_fal                                               # опц. сшивка света

Проход на материал:
  * без reference — FluxPro1Fill_fal (масочный inpaint по текстовому промпту);
  * с reference — edit-модель с двумя входами (текущий кадр + свотч материала),
    у неё нет входа маски, поэтому результат обрезается обратно по маске
    через ImageCompositeMasked — вне зоны картинка остаётся пиксель-в-пиксель.

Маски приходят из Blender уже с dilate+blur (в этом ComfyUI нод блюра нет),
mask_image у Fill-ноды имеет тип IMAGE — конвертация не нужна.
После каждого прохода стоит SaveImage: промежуточные шаги видны в out/.
"""

REF_ENGINES = ("nanobanana", "kontext")


def build_workflow(cfg, depth_image, mask_images, ref_images=None, prefix="b2r"):
    """cfg — dict из materials.yaml; depth_image / mask_images / ref_images — имена
    файлов, уже загруженных в input ComfyUI (mask_images: {material_name: filename},
    ref_images: {material_name: filename} для материалов с полем reference)."""
    ref_images = ref_images or {}
    scene = cfg["scene"]
    graph = {}
    next_id = [0]

    def node(class_type, _title=None, **inputs):
        next_id[0] += 1
        nid = str(next_id[0])
        graph[nid] = {"class_type": class_type, "inputs": inputs}
        if _title:
            graph[nid]["_meta"] = {"title": _title}
        return nid

    def save(img_node, name, title):
        node("SaveImage", _title=title,
             images=[img_node, 0], filename_prefix=f"{prefix}/{name}")

    # 1. глобальный проход: depth ControlNet Union
    depth_load = node("LoadImage", _title="Вход: depth из Blender", image=depth_image)
    current = node(
        "FluxGeneral_fal",
        _title="1) Глобальный проход — стиль по depth",
        prompt=scene["prompt"],
        image_size="custom",
        width=scene["width"],
        height=scene["height"],
        num_inference_steps=scene.get("steps", 28),
        guidance_scale=scene.get("guidance_scale", 3.0),
        real_cfg_scale=3.3,
        num_images=1,
        enable_safety_checker=False,
        use_real_cfg=False,
        sync_mode=False,
        seed=scene.get("seed", -1),
        controlnet_unions=scene.get("controlnet_union", "InstantX/FLUX.1-dev-Controlnet-Union"),
        controlnet_union_control_mode=scene.get("control_mode", "depth"),
        controlnet_conditioning_scale=scene.get("conditioning_scale", 0.65),
        control_image=[depth_load, 0],
    )
    save(current, "00_global", "Сейв 00: глобальный")

    # 2. проходы по материалам (порядок из конфига: крупное -> мелкое)
    for i, mat in enumerate(cfg.get("materials", []), start=1):
        mask_load = node("LoadImage", _title=f"Вход: маска {mat['name']}",
                         image=mask_images[mat["name"]])
        if mat["name"] in ref_images:
            current = _reference_pass(node, scene, mat, current, mask_load,
                                      ref_images[mat["name"]], i)
        else:
            current = node(
                "FluxPro1Fill_fal",
                _title=f"2.{i}) Inpaint текстом — {mat['name']}",
                prompt=mat["prompt"],
                num_images=1,
                safety_tolerance="5",
                output_format="png",
                image=[current, 0],
                mask_image=[mask_load, 0],
                seed=mat.get("seed", -1),
                sync_mode=False,
                # enhance_prompt переписывает промпт LLM-ом — для материалов предсказуемее без него
                enhance_prompt=bool(mat.get("enhance_prompt", False)),
            )
        save(current, f"{i:02d}_{mat['name']}", f"Сейв {i:02d}: {mat['name']}")

    # 3. опциональная сшивка освещения (у FluxGeneral_fal нет img2img — только Kontext)
    fp = cfg.get("final_pass") or {}
    if fp.get("enabled"):
        current = node(
            "FluxProKontext_fal",
            _title="3) Финал — сшивка света (Kontext)",
            prompt=fp["prompt"],
            image=[current, 0],
            guidance_scale=fp.get("guidance_scale", 3.5),
            num_images=1,
            safety_tolerance="5",
            output_format="png",
            sync_mode=False,
        )
        save(current, "99_final", "Сейв 99: финал")

    return graph


def _edit_prompt(mat):
    """Промпт для edit-модели: она не видит маску, поэтому говорим ей, куда класть."""
    if mat.get("reference_prompt"):
        return mat["reference_prompt"]
    target = mat.get("target", f"the surfaces described as: {mat['prompt']}")
    return (f"Apply the material and finish shown in the second image to {target}. "
            f"{mat['prompt']}. Keep the composition, camera, lighting and "
            f"everything else exactly unchanged.")


def _reference_pass(node, scene, mat, current, mask_load, ref_image, idx):
    """Материал по референс-свотчу: edit-модель (2 входа) + обрезка по маске.

    Edit-модели зону ищут по промпту и могут зацепить лишнее, поэтому их выход
    приводится к размеру кадра (ImageScale) и вкомпоуживается в предыдущий шаг
    строго внутри маски (ImageCompositeMasked, серые края маски дают мягкий шов).
    """
    swatch = node("LoadImage", _title=f"Вход: свотч {mat['name']}", image=ref_image)
    engine = mat.get("ref_engine", "nanobanana")
    edit_title = f"2.{idx}) Edit по референсу — {mat['name']}"
    if engine == "kontext":
        edited = node(
            "FluxProKontextMulti_fal",
            _title=edit_title,
            prompt=_edit_prompt(mat),
            image_1=[current, 0],
            image_2=[swatch, 0],
            aspect_ratio="16:9",
            guidance_scale=mat.get("guidance_scale", 3.5),
            num_images=1,
            safety_tolerance="5",
            output_format="png",
            sync_mode=False,
        )
    elif engine == "nanobanana":
        edited = node(
            "NanoBananaEdit_fal",
            _title=edit_title,
            prompt=_edit_prompt(mat),
            image_1=[current, 0],
            image_2=[swatch, 0],
            num_images=1,
            output_format="png",
        )
    else:
        raise ValueError(f"{mat['name']}: неизвестный ref_engine '{engine}', "
                         f"допустимо: {REF_ENGINES}")

    scaled = node("ImageScale", _title="Подгон к размеру кадра",
                  image=[edited, 0], upscale_method="lanczos",
                  width=scene["width"], height=scene["height"], crop="disabled")
    mask = node("ImageToMask", _title=f"Маска {mat['name']} → MASK",
                image=[mask_load, 0], channel="red")
    return node("ImageCompositeMasked", _title=f"Врезка по маске — {mat['name']}",
                destination=[current, 0], source=[scaled, 0],
                x=0, y=0, resize_source=False, mask=[mask, 0])
