# Comfyder — AI render pipeline for Blender

> **Requires: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) + [fal.ai](https://fal.ai) API key.**
> ComfyUI is only the orchestrator (CPU is fine) — all GPU work runs on FAL, pay-per-call.

Turn a rough Blender blocking (or any render) into a polished AI board.
Two tools in one repo:

| | |
|---|---|
| **Comfyder Lite** — Blender add-on | One button: render → prompt → refined AI render, right inside Blender |
| **Blocking2Render** — full pipeline | Depth ControlNet + Cryptomatte masks + per-material AI passes |

![lite demo](docs/img/demo_lite.jpg)
*Comfyder Lite: raw viewport-color render → "hyperreal 4K, light beams, atmospheric smoke" → one click*

![pipeline demo](docs/img/demo_pipeline.jpg)
*Full pipeline: blocking with placeholder materials → per-zone passes (brick / water / flowers by masks) → final*

## Requirements

- **ComfyUI** reachable over HTTP (LAN or localhost). CPU-only is fine — it never samples locally.
- ComfyUI custom node packs:
  - [gokayfem/ComfyUI-fal-API](https://github.com/gokayfem/ComfyUI-fal-API) — `FluxGeneral_fal`, `FluxPro1Fill_fal`, `VLM_fal`, …
  - ComfyUI-FAL (Image Edit pack) — `FalGeminiFlashEdit`, `FalQwenImageEditInpaint`, …
- **`FAL_KEY`** in the environment of the ComfyUI process (get one at [fal.ai](https://fal.ai)). Every generation is a paid API call (cents per image).
- **Blender 5.0+** (the add-on and scripts use the 5.x compositor API).

## Comfyder Lite (add-on)

Install: `Edit > Preferences > Add-ons > Install from Disk` → `addon/comfyder_lite.py`.

Use: press **F12** (or pick *Viewport* as source) → in the render window press **N** → **Comfyder** tab → type a mood prompt (any language) → **Generate**. ~1 min later the result appears as a new image *"Comfyder Result"*.

Options: VLM scene auto-description, 1K/2K/4K output, seed, **auto-rendered depth map** as a geometry hint (recommended), ComfyUI address.

## Full pipeline (Blocking2Render)

1. **Blender**: assign placeholder materials `mat_*` per zone, run `blender/setup_passes.py` → renders `depth.png` + one Cryptomatte mask per material.
2. **Config**: describe each zone in `driver/materials.yaml` (engine per zone: `fill` / `qwen` / `gemini_zone` / `kontext_zone` / reference swatch / procedural light `overlay`).
3. **Run**: `driver/run.py` (or build the graph with `driver/comfy_graph_v2.py`) — uploads the pack, submits the chain, saves every intermediate step.

Chain: global pass (Flux + depth ControlNet) → sequential masked passes per material → final refine (Gemini, frame + depth). Fixed seed = re-run any single zone for cents.

Field notes with 20 battle-tested rules (mask dilate/blur values, engine choice per zone type, aspect-ratio traps, non-determinism and pinning) live in [docs/Blocking2Render.md](docs/Blocking2Render.md).

## Структура / Structure

```
addon/    comfyder_lite.py         ← Blender add-on (Lite)
blender/  setup_passes.py          ← depth + Cryptomatte masks renderer
driver/   comfy_graph_v2.py, run.py, materials.yaml
comfy/    workflow.json            ← reference graph (API format)
docs/     Blocking2Render.md, img/
```

## RU: кратко

Comfyder Lite — аддон для Blender: рендер → промпт настроения → полированный
AI-рендер через ComfyUI + FAL (Gemini). Полный пайплайн Blocking2Render —
зонная генерация по маскам Cryptomatte с depth ControlNet. Нужны: ComfyUI
с паками fal-API нод и ключ `FAL_KEY` в окружении ComfyUI (вся генерация —
платные вызовы fal.ai, центы за картинку). Blender 5.0+.

## License

MIT
