---
tags: [pipeline, blender, comfyui, fal, archviz]
created: 2026-07-04
---

# Blocking2Render — AI-рендер из блокинга (Blender → ComfyUI → FAL)

Пайплайн превращает грубый блокинг из Blender в 2D-борд для клиентов
(event/exhibition, archviz). Геометрию держит depth ControlNet, материалы
задаются по зонам через Cryptomatte-маски — текстом или фото-референсом.
Ничего не запекается обратно на геометрию: результат — картинка.

Репо: `~/Projects/Blender-ComfyUI` · github.com/dufok/Blender-ComfyUI

## Как устроено

```
 Mac (Blender)                proserver (Docker)                Облако
┌──────────────────┐   PNG   ┌─────────────┐  HTTP  ┌──────────────┐  API   ┌─────────┐
│ setup_passes.py  │ ──────→ │ driver      │ ─────→ │ ComfyUI :8188│ ─────→ │ FAL     │
│ блокинг → пак    │         │ run.py      │        │ CPU, только  │        │ Flux    │
│ beauty/depth/маски│        │ граф+очередь│        │ оркестратор  │        │ GPU     │
└──────────────────┘         └─────────────┘        └──────────────┘        └─────────┘
```

**Blender** (`blender/setup_passes.py`, запуск через Blender MCP или `blender -b`):
одним запуском включает Z-pass и Cryptomatte Material (accurate, levels 6),
строит компоузер и рендерит пак PNG: `beauty`, `depth` (Map Range, ближнее =
белое, 16-бит BW, save_as_render=False) и `mask_<материал>.png` на каждый
материал `mat_*`. Маски выходят уже с dilate +6px и Gauss-блюром 4px — в
ComfyUI на сервере нет ни одной ноды блюра, поэтому швы убираются здесь.
Диапазон depth считается автоматически по bbox видимых мешей от камеры.

**Driver** (`driver/run.py`, Docker на proserver): читает `materials.yaml`,
заливает пак через `/upload/image`, собирает workflow JSON на лету под текущий
список материалов (`comfy_graph.py`), шлёт в `/prompt`, поллит `/history`,
скачивает результаты. FAL_KEY живёт в окружении ComfyUI — драйверу не нужен.

**ComfyUI** (proserver :8188, CPU) сам ничего не считает: FAL-ноды — обёртки
над API. Даёт очередь, промежуточные картинки и визуальную отладку.
Воркспейс сохранён на сервере: Workflows → `blocking2render` (все ноды
названы по функции: «1) Глобальный проход…», «2.N) Inpaint…», «Сейв NN…»).

## Цепочка генерации

1. **Глобальный проход** — `FluxGeneral_fal` + ControlNet Union
   (`InstantX/FLUX.1-dev-Controlnet-Union`, mode=depth,
   `conditioning_scale` 0.6–0.7): depth диктует форму, промпт сцены — стиль.
   Опционально есть IP-Adapter (`ip_adapter_image`) — стилевой референс на весь борд.
2. **Проходы по материалам**, строго последовательно, порядок из YAML
   (крупное → мелкое). Два вида:
   - *текстом*: `FluxPro1Fill_fal` (масочный inpaint, `enhance_prompt` выключен);
   - *по референсу*: `NanoBananaEdit_fal` или `FluxProKontextMulti_fal`
     (кадр + фото-свотч) → `ImageScale` → `ImageCompositeMasked` — выход
     edit-модели врезается обратно строго по маске материала, вне зоны
     картинка не меняется. Edit-модель маску не видит — зону подсказывает
     поле `target:`.
3. **Финал** (опция) — `FluxProKontext_fal` сшивает свет/рефлексы между зонами.

После каждого шага `SaveImage` (`00_global`, `01_mat_wall`, … `99_final`) —
видно, на каком проходе что-то поехало.

## materials.yaml

```yaml
scene:            # общий промпт, width/height (= RES в setup_passes.py),
  ...             # controlnet union, conditioning_scale, seed
materials:
  - name: mat_counter                  # = имя материала в Blender = имя маски
    prompt: polished chrome metal panels
    reference: swatches/chrome.jpg     # опц.: фото-свотч → референс-ветка
    target: the counter panels         # опц.: куда класть (для edit-модели)
    ref_engine: nanobanana             # nanobanana (default) | kontext
final_pass:
  enabled: false
  prompt: harmonize lighting ...
```

## Запуск

```bash
# 1. Blender: раскидать материалы mat_* по блокингу, выполнить setup_passes.py
# 2. proserver:
docker build -t b2r-driver driver/
docker run --rm --network host \
  -v "$PWD/render_pack:/data/pack" -v "$PWD/out:/data/out" \
  -v "$PWD/driver/materials.yaml:/data/materials.yaml:ro" \
  b2r-driver --pack /data/pack --config /data/materials.yaml \
             --out /data/out --host http://192.168.1.2:8188
```

`--dump workflow.json` — собрать граф без отправки.

## FAL-ноды: что установлено и как искать

В ComfyUI на proserver стоит пак FAL-нод (суффикс `_fal`, категории `FAL/*`).
Они не считают локально — это обёртки над fal.ai API (FAL_KEY в окружении
процесса ComfyUI). Категории: `FAL/Image` (генерация/edit/upscale, ~35 нод),
`FAL/3D` (Tripo, Hunyuan3D, Trellis — image→glb), `FAL/VideoGeneration`
(Kling, Veo, Wan, Sora…), `FAL/LLM`, `FAL/VLM`, `FAL/Training` (LoRA-тренеры).

Используемые в пайплайне:

| Нода | Роль | Ключевое |
|---|---|---|
| `FluxGeneral_fal` | глобальный проход | единственная с ControlNet Union + IP-Adapter; нет img2img; max 1536px |
| `FluxPro1Fill_fal` | inpaint текстом | `mask_image` тип IMAGE; `enhance_prompt` (LLM) лучше выключать |
| `NanoBananaEdit_fal` | материал по референсу | до 4 входных картинок, маску не видит |
| `FluxProKontextMulti_fal` | то же, вариант 2 | аспект фиксируется явно (`aspect_ratio`) |
| `FluxProKontext_fal` | финальная сшивка | edit одной картинки промптом, denoise-ручки нет |

**Как инвентаризировать заново** (после обновления пака или на другом сервере) —
весь каталог нод отдаёт один эндпоинт, UI не нужен:

```bash
# все ноды с fal/flux в имени + категория
curl -s http://192.168.1.2:8188/object_info | python3 -c "
import json,sys
d=json.load(sys.stdin)
for n,i in d.items():
    if 'fal' in n.lower() or 'flux' in n.lower():
        print(n,'|',i.get('category'))"

# точная схема входов конкретной ноды (required/optional, типы, дефолты, диапазоны)
curl -s http://192.168.1.2:8188/object_info | python3 -c "
import json,sys
i=json.load(sys.stdin)['FluxGeneral_fal']['input']
print(json.dumps(i,indent=1,ensure_ascii=False))"
```

Именно так были найдены все ограничения из раздела «Грабли»: смотри на типы
входов (IMAGE vs MASK), списки-enum'ы (какие ControlNet'ы доступны) и
min/max (пределы разрешения). Схема входа = контракт для workflow JSON.

## Грабли (проверено)

- FAL `flux-general`: один ControlNet Union на вызов, режима seg нет →
  сегментация только масками. Max 1536px, кратно 16.
- У `FluxGeneral_fal` нет img2img-входа → сшивка света только через Kontext.
- `FluxPro1Fill_fal.mask_image` — тип IMAGE, не MASK.
- CryptoMaterial, не CryptoAsset (тот группирует по верхнему parent-объекту).
- Имена материалов: латиница, без пробелов (`mat_wood`) — они же ключи конфига.
- Depth из Z-пасса точный — MiDaS/Depth-Anything поверх только портят.
- Маски = разрешению генерации (RES в setup_passes.py = width/height в yaml).
- Сайдбар Workflows в ComfyUI открывает только UI-формат; API-формат
  конвертируется фронтендом: `app.loadApiJson()` → `app.graph.serialize()`
  → POST `/userdata/workflows%2F<имя>.json`.
- NanoBanana может вернуть иную пропорцию — если текстура «плывёт» внутри
  зоны, переключить материал на `ref_engine: kontext` (аспект фиксирован).

## Статус (2026-07-04)

Код и воркспейс готовы, тестовый прогон на реальном блокинге не делался.
Дальше: тест на сцене, тюнинг `conditioning_scale` и порядка проходов.
