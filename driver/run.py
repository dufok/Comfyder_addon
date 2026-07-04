"""Driver: пак из Blender -> ComfyUI (/upload/image, /prompt, /history) -> результаты.

Пример:
  python run.py --pack ./render_pack --config materials.yaml --out ./out \
      --host http://192.168.1.2:8188

FAL_KEY живёт в окружении самого ComfyUI (FAL-ноды зовут API оттуда) —
драйверу ключ не нужен.
"""

import argparse
import glob
import json
import os
import sys
import time
import uuid

import requests
import yaml

from comfy_graph import build_workflow

POLL_INTERVAL = 3
TIMEOUT = 1800  # цепочка FAL-вызовов может идти долго


def find_one(pack_dir, pattern):
    hits = sorted(glob.glob(os.path.join(pack_dir, pattern)))
    if not hits:
        sys.exit(f"не найден файл по маске '{pattern}' в {pack_dir}")
    if len(hits) > 1:
        print(f"WARN: несколько файлов под '{pattern}', беру {hits[-1]}")
    return hits[-1]


def upload(host, path):
    with open(path, "rb") as fh:
        r = requests.post(
            f"{host}/upload/image",
            files={"image": (os.path.basename(path), fh, "image/png")},
            data={"overwrite": "true"},
            timeout=60,
        )
    r.raise_for_status()
    info = r.json()
    name = info["name"]
    if info.get("subfolder"):
        name = f"{info['subfolder']}/{name}"
    return name


def submit_and_wait(host, graph):
    client_id = str(uuid.uuid4())
    r = requests.post(f"{host}/prompt", json={"prompt": graph, "client_id": client_id}, timeout=60)
    if r.status_code != 200:
        sys.exit(f"ComfyUI отверг workflow: {r.status_code}\n{r.text}")
    resp = r.json()
    if resp.get("node_errors"):
        sys.exit("ошибки нод:\n" + json.dumps(resp["node_errors"], indent=2, ensure_ascii=False))
    pid = resp["prompt_id"]
    print(f"prompt_id={pid}, жду выполнения (poll {POLL_INTERVAL}s)...")

    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        h = requests.get(f"{host}/history/{pid}", timeout=30).json()
        if pid not in h:
            continue
        entry = h[pid]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            sys.exit("выполнение упало:\n" + json.dumps(status, indent=2, ensure_ascii=False))
        if entry.get("outputs"):
            return entry["outputs"]
    sys.exit(f"таймаут {TIMEOUT}s — проверь очередь ComfyUI")


def download_outputs(host, outputs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for node_id, out in sorted(outputs.items(), key=lambda kv: int(kv[0])):
        for img in out.get("images", []):
            r = requests.get(
                f"{host}/view",
                params={"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output")},
                timeout=120,
            )
            r.raise_for_status()
            dst = os.path.join(out_dir, img["filename"])
            with open(dst, "wb") as fh:
                fh.write(r.content)
            saved.append(dst)
            print("  ->", dst)
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="папка с beauty/depth/mask_*.png из Blender")
    ap.add_argument("--config", required=True, help="materials.yaml")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--host", default=os.environ.get("COMFY_HOST", "http://192.168.1.2:8188"))
    ap.add_argument("--dump", metavar="FILE", help="только записать workflow JSON и выйти")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    depth_path = find_one(args.pack, "depth*.png")
    mask_paths = {m["name"]: find_one(args.pack, f"mask_{m['name']}*.png")
                  for m in cfg.get("materials", [])}

    # референс-свотчи: путь относительно materials.yaml (или абсолютный)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    ref_paths = {}
    for m in cfg.get("materials", []):
        if m.get("reference"):
            path = os.path.join(cfg_dir, m["reference"])
            if not os.path.isfile(path):
                sys.exit(f"{m['name']}: референс не найден: {path}")
            ref_paths[m["name"]] = path

    if args.dump:
        graph = build_workflow(cfg, os.path.basename(depth_path),
                               {k: os.path.basename(v) for k, v in mask_paths.items()},
                               {k: os.path.basename(v) for k, v in ref_paths.items()})
        with open(args.dump, "w") as fh:
            json.dump(graph, fh, indent=2, ensure_ascii=False)
        print("workflow записан в", args.dump)
        return

    print(f"загружаю входы в {args.host} ...")
    depth_name = upload(args.host, depth_path)
    mask_names = {name: upload(args.host, path) for name, path in mask_paths.items()}
    ref_names = {name: upload(args.host, path) for name, path in ref_paths.items()}

    graph = build_workflow(cfg, depth_name, mask_names, ref_names)
    outputs = submit_and_wait(args.host, graph)
    saved = download_outputs(args.host, outputs, args.out)
    print(f"готово: {len(saved)} файлов в {args.out}")


if __name__ == "__main__":
    main()
