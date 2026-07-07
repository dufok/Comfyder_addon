"""Сборка ComfyUI workflow v2 — расширение comfy_graph.py.

Новое против v1:
  * engine per-material: fill (FluxPro1Fill_fal, как v1) | qwen
    (FalQwenImageEditInpaint: strength + negative_prompt, маска MASK)
    | zturbo (FalZImageTurboInpaint: быстрый черновик);
  * enhance: true — промпт материала прогоняется через GonkaPromptEnhance
    прямо в графе (оплата GNK);
  * референс-ветки и final_pass (Kontext) — как в v1.

strength у qwen/zturbo: сколько перерисовывать (0..1). ~0.7 сохраняет
подложку глобального прохода (тени/перспективу), меняя фактуру.
"""

REF_ENGINES = ("nanobanana", "kontext")

GONKA_INSTRUCTION = (
    "Rewrite as a concise material description prompt for an image "
    "inpainting model. English only, vivid and concrete, keep the intent. "
    "Return only the prompt."
)


def build_workflow(cfg, depth_image, mask_images, ref_images=None, prefix="b2r"):
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

    def prompt_input(mat):
        """Строка или линк на Gonka-ноду, если enhance: true."""
        if not mat.get("enhance"):
            return mat["prompt"]
        g = node("GonkaPromptEnhance",
                 _title=f"Gonka: промпт {mat['name']}",
                 text=mat["prompt"],
                 instruction=mat.get("enhance_instruction", GONKA_INSTRUCTION),
                 model=mat.get("enhance_model", "MiniMaxAI/MiniMax-M2.7"),
                 system_prompt="", temperature=0.7,
                 max_tokens=300, seed=scene.get("seed", 7))
        return [g, 0]

    # 1. глобальный проход
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

    # 2. материальные проходы
    for i, mat in enumerate(cfg.get("materials", []), start=1):
        mask_load = node("LoadImage", _title=f"Вход: маска {mat['name']}",
                         image=mask_images[mat["name"]])
        if mat["name"] in ref_images:
            current = _reference_pass(node, scene, mat, current, mask_load,
                                      ref_images[mat["name"]], i)
        else:
            engine = mat.get("engine", "fill")
            p = prompt_input(mat)
            if engine == "gemini_zone":
                # Gemini-зона: edit без маски -> врезка по маске.
                # resolution 2K + даунскейл к кадру = суперсемплинг.
                m = node("ImageToMask", _title=f"MASK {mat['name']}",
                         image=[mask_load, 0], channel="red")
                sysp = ("You are editing exactly one zone of the image. "
                        f"Change ONLY {mat.get('target', 'the described area')}. "
                        "Keep composition, all other objects, lighting, framing "
                        "and aspect ratio exactly unchanged.")
                edited = node(
                    "FalGeminiFlashEdit",
                    _title=f"2.{i}) Gemini-зона — {mat['name']}",
                    image=[current, 0], prompt=p,
                    version=mat.get("gemini_version", "3.1-flash-preview"),
                    resolution=mat.get("resolution", "2K"),
                    system_prompt=sysp,
                    num_images=1, seed=mat.get("seed", scene.get("seed", 7)),
                )
                # Gemini 2K (16:9) = 2752x1536, аспект 1.792 != 1.778 кадра:
                # растяжка смещает тонкие структуры. Скейлим с сохранением
                # аспекта по высоте и центр-кропим до кадра.
                sw = round(scene["height"] * 2752 / 1536)
                scaled = node("ImageScale", _title=f"Подгон {mat['name']}",
                              image=[edited, 0], upscale_method="lanczos",
                              width=sw, height=scene["height"], crop="disabled")
                cropped = node("ImageCrop", _title=f"Кроп {mat['name']}",
                               image=[scaled, 0], width=scene["width"],
                               height=scene["height"],
                               x=(sw - scene["width"]) // 2, y=0)
                current = node("ImageCompositeMasked",
                               _title=f"Врезка по маске — {mat['name']}",
                               destination=[current, 0], source=[cropped, 0],
                               x=0, y=0, resize_source=False, mask=[m, 0])
            elif engine == "overlay":
                # Процедурный свет: маска -> белый градиент -> screen поверх кадра.
                # Ноль AI-вызовов; intensity = blend_factor (ползунок в плагине).
                m = node("ImageToMask", _title=f"MASK {mat['name']}",
                         image=[mask_load, 0], channel="red")
                beam = node("MaskToImage", _title=f"Луч {mat['name']}",
                            mask=[m, 0])
                current = node("ImageBlend",
                               _title=f"2.{i}) Overlay (screen) — {mat['name']}",
                               image1=[current, 0], image2=[beam, 0],
                               blend_mode=mat.get("blend_mode", "screen"),
                               blend_factor=mat.get("intensity", 0.4))
            elif engine == "kontext_zone":
                # Kontext-зона (Вариант B): держит структуру, не ресемплит кадр.
                m = node("ImageToMask", _title=f"MASK {mat['name']}",
                         image=[mask_load, 0], channel="red")
                instr = mat.get("instruction") or (
                    f"Change {mat.get('target', 'the masked area')} to: "
                    f"{mat['prompt']}. Keep the composition, camera, lighting "
                    f"and everything else exactly unchanged.")
                edited = node(
                    "FluxProKontext_fal",
                    _title=f"2.{i}) Kontext-зона — {mat['name']}",
                    prompt=instr, image=[current, 0],
                    aspect_ratio=mat.get("aspect_ratio", "16:9"),
                    guidance_scale=mat.get("guidance_scale", 3.5),
                    num_images=1, safety_tolerance="5",
                    output_format="png", sync_mode=False,
                )
                scaled = node("ImageScale", _title=f"Подгон {mat['name']}",
                              image=[edited, 0], upscale_method="lanczos",
                              width=scene["width"], height=scene["height"],
                              crop="disabled")
                current = node("ImageCompositeMasked",
                               _title=f"Врезка по маске — {mat['name']}",
                               destination=[current, 0], source=[scaled, 0],
                               x=0, y=0, resize_source=False, mask=[m, 0])
            elif engine in ("qwen", "zturbo"):
                m = node("ImageToMask", _title=f"MASK {mat['name']}",
                         image=[mask_load, 0], channel="red")
                if engine == "qwen":
                    edited = node(
                        "FalQwenImageEditInpaint",
                        _title=f"2.{i}) Qwen inpaint — {mat['name']}",
                        image=[current, 0], mask=[m, 0], prompt=p,
                        strength=mat.get("strength", 0.7),
                        guidance_scale=mat.get("guidance_scale", 4.0),
                        num_inference_steps=mat.get("steps", 30),
                        negative_prompt=mat.get("negative", ""),
                        num_images=1, seed=mat.get("seed", scene.get("seed", 7)),
                    )
                else:
                    edited = node(
                        "FalZImageTurboInpaint",
                        _title=f"2.{i}) Z-Turbo inpaint — {mat['name']}",
                        image=[current, 0], mask=[m, 0], prompt=p,
                        strength=mat.get("strength", 0.8),
                        num_inference_steps=mat.get("steps", 8),
                        acceleration="regular",
                        num_images=1, seed=mat.get("seed", scene.get("seed", 7)),
                    )
                # выход строго по маске: вне зоны кадр пиксель-в-пиксель,
                # шум перекодирования не накапливается
                scaled = node("ImageScale", _title=f"Подгон {mat['name']}",
                              image=[edited, 0], upscale_method="lanczos",
                              width=scene["width"], height=scene["height"],
                              crop="disabled")
                current = node("ImageCompositeMasked",
                               _title=f"Врезка по маске — {mat['name']}",
                               destination=[current, 0], source=[scaled, 0],
                               x=0, y=0, resize_source=False, mask=[m, 0])
            else:  # fill
                current = node(
                    "FluxPro1Fill_fal",
                    _title=f"2.{i}) Inpaint текстом — {mat['name']}",
                    prompt=p, num_images=1, safety_tolerance="5",
                    output_format="png", image=[current, 0],
                    mask_image=[mask_load, 0],
                    seed=mat.get("seed", -1), sync_mode=False,
                    enhance_prompt=bool(mat.get("enhance_prompt", False)),
                )
        save(current, f"{i:02d}_{mat['name']}", f"Сейв {i:02d}: {mat['name']}")

    # 3. финальная сшивка
    fp = cfg.get("final_pass") or {}
    if fp.get("enabled") and fp.get("engine") == "gemini":
        current = node(
            "FalGeminiFlashEdit",
            _title="3) Финал — Gemini (кадр + depth)",
            image=[current, 0], image_2=[depth_load, 0],
            version=fp.get("gemini_version", "3.1-flash-preview"),
            resolution=fp.get("resolution", "2K"),
            system_prompt=("The first image is the artwork to refine. The second "
                           "image is its depth map (white = near camera) — use it "
                           "ONLY as geometry and spatial reference, never draw it. "
                           "Preserve the exact composition, framing and aspect "
                           "ratio of the first image."),
            prompt=fp["prompt"],
            num_images=1, seed=fp.get("seed", scene.get("seed", 7)),
        )
        save(current, "99_final", "Сейв 99: финал (Gemini)")
    elif fp.get("enabled"):
        current = node(
            "FluxProKontext_fal",
            _title="3) Финал — сшивка (Kontext)",
            prompt=fp["prompt"],
            image=[current, 0],
            aspect_ratio=fp.get("aspect_ratio", "16:9"),
            guidance_scale=fp.get("guidance_scale", 3.5),
            num_images=1, safety_tolerance="5",
            output_format="png", sync_mode=False,
        )
        save(current, "99_final", "Сейв 99: финал")

    return graph


def _edit_prompt(mat):
    if mat.get("reference_prompt"):
        return mat["reference_prompt"]
    target = mat.get("target", f"the surfaces described as: {mat['prompt']}")
    return (f"Apply the material and finish shown in the second image to {target}. "
            f"{mat['prompt']}. Keep the composition, camera, lighting and "
            f"everything else exactly unchanged.")


def _reference_pass(node, scene, mat, current, mask_load, ref_image, idx):
    swatch = node("LoadImage", _title=f"Вход: свотч {mat['name']}", image=ref_image)
    engine = mat.get("ref_engine", "nanobanana")
    title = f"2.{idx}) Edit по референсу — {mat['name']}"
    if engine == "kontext":
        edited = node("FluxProKontextMulti_fal", _title=title,
                      prompt=_edit_prompt(mat), image_1=[current, 0],
                      image_2=[swatch, 0], aspect_ratio="16:9",
                      guidance_scale=mat.get("guidance_scale", 3.5),
                      num_images=1, safety_tolerance="5",
                      output_format="png", sync_mode=False)
    elif engine == "nanobanana":
        edited = node("NanoBananaEdit_fal", _title=title,
                      prompt=_edit_prompt(mat), image_1=[current, 0],
                      image_2=[swatch, 0], num_images=1, output_format="png")
    else:
        raise ValueError(f"{mat['name']}: неизвестный ref_engine '{engine}'")
    scaled = node("ImageScale", _title="Подгон к размеру кадра",
                  image=[edited, 0], upscale_method="lanczos",
                  width=scene["width"], height=scene["height"], crop="disabled")
    mask = node("ImageToMask", _title=f"Маска {mat['name']} → MASK",
                image=[mask_load, 0], channel="red")
    return node("ImageCompositeMasked", _title=f"Врезка по маске — {mat['name']}",
                destination=[current, 0], source=[scaled, 0],
                x=0, y=0, resize_source=False, mask=[mask, 0])
