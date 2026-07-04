# blocking2render — Blender blocking → ComfyUI → FAL

Пайплайн «AI-рендер»: блокинг из Blender превращается в 2D-борд.
Геометрию держит depth ControlNet, материалы задаются по зонам через
Cryptomatte-маски и последовательные inpaint-проходы. Всё тяжёлое считает
FAL API (Flux), локальный ComfyUI (CPU, 192.168.1.2:8188) — только оркестратор.

## Схема

```
Blender (setup_passes.py)          ComfyUI (граф собирает driver)
┌────────────────────────┐        ┌──────────────────────────────────────┐
│ Z-pass ─→ depth.png    │──────→ │ FluxGeneral_fal (CN Union, depth)    │
│ CryptoMaterial ─→      │        │   ↓                                  │
│   mask_<mat>.png       │──────→ │ FluxPro1Fill_fal × N (inpaint)       │
│   (dilate+blur уже тут)│        │   ↓                                  │
│ beauty.png             │        │ FluxProKontext_fal (опц. сшивка)     │
└────────────────────────┘        └──────────────────────────────────────┘
```

## Порядок работы

1. **Blender**: на блокинг раскидать плейсхолдер-материалы `mat_*`
   (латиница, без пробелов: `mat_wood`, `mat_concrete`). Выполнить
   [blender/setup_passes.py](blender/setup_passes.py) через Blender MCP или
   `blender -b scene.blend --python blender/setup_passes.py`.
   Результат — `render_pack/` рядом с .blend: `beauty*.png`, `depth*.png`,
   `mask_mat_*.png` (File Output дописывает номер кадра, драйвер это ждёт).
2. **Конфиг**: в [driver/materials.yaml](driver/materials.yaml) — общий промпт
   сцены и промпт на каждый материал. Порядок материалов = порядок inpaint,
   от крупных поверхностей к мелким (стены → пол → мебель → детали).

   Материал можно задать не только текстом, но и **фото-референсом**
   (`reference: swatches/chrome.jpg`, путь относительно yaml). Тогда вместо
   Fill-inpaint строится ветка: edit-модель с двумя входами (кадр + свотч,
   `ref_engine: nanobanana` или `kontext`) → ImageScale → ImageCompositeMasked —
   результат обрезается обратно по маске материала, вне зоны картинка
   не меняется. `target:` подсказывает edit-модели, куда класть материал
   (маску она не видит), `reference_prompt:` заменяет шаблонный промпт целиком.
3. **Driver** (Docker, на proserver):

   ```bash
   docker build -t b2r-driver driver/
   docker run --rm --network host \
     -v "$PWD/render_pack:/data/pack" -v "$PWD/out:/data/out" \
     -v "$PWD/driver/materials.yaml:/data/materials.yaml:ro" \
     b2r-driver --pack /data/pack --config /data/materials.yaml \
                --out /data/out --host http://192.168.1.2:8188
   ```

   `--dump workflow.json` — записать граф без отправки.
   FAL_KEY нужен процессу ComfyUI (ноды зовут FAL оттуда), драйверу — нет.

Промежуточные результаты каждого прохода сохраняются (`00_global`,
`01_<mat>`, ..., `99_final`) — удобно ловить, на каком шаге что-то поехало.

## Ключевые решения и грабли

- `FluxGeneral_fal` — единственная FAL-нода с ControlNet Union
  (`InstantX/FLUX.1-dev-Controlnet-Union`), режим `depth`,
  `conditioning_scale` 0.6–0.7. Один union на вызов, max 1536px, кратно 16.
- У `FluxGeneral_fal` **нет img2img-входа** → финальная сшивка света только
  через `FluxProKontext_fal` (без denoise-ручки, управляется промптом).
- `FluxPro1Fill_fal.mask_image` — тип IMAGE, не MASK → маски идут напрямую
  из LoadImage без конвертаций.
- В этом ComfyUI **нет нод блюра** (нет даже core `ImageBlur`) → dilate+blur
  масок делает компоузер Blender (Dilate/Erode +6px, Gauss 4px).
- CryptoMaterial, не CryptoAsset (тот группирует по parent-объекту).
- Depth из Z-пасса точный — никаких MiDaS/Depth-Anything поверх.
- Маски рендерятся в том же разрешении, что генерация:
  `RES_X/RES_Y` в setup_passes.py = `width/height` в materials.yaml.
- Depth и маски пишутся с `save_as_render=False` (без view transform),
  depth — 16-бит BW PNG, near=white (конвенция Flux depth).

## Структура

```
blender/setup_passes.py   # bpy: пассы + компоузер + рендер пака
comfy/workflow.json       # референсный граф (генерат run.py --dump)
driver/                   # Dockerfile, run.py, comfy_graph.py, materials.yaml
```
