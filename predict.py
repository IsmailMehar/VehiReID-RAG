#!/usr/bin/env python3
# predict.py
import os, sys, json, glob, argparse, random
from pathlib import Path

# --- make 'src/' imports work when running from repo root ---
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM

# Optional: pretty names from .mat files (if scipy present)
try:
    from scipy.io import loadmat  # pip install scipy
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

from models.vit_multitask import ViTMultiTask
from datasets.compcars_dataset import CompCarsCls

VIEW_IDX_TO_NAME = ['uncertain','front','rear','side','front-side','rear-side']

# ---------------- utils ----------------
def _sanitize_state_dict(sd: dict) -> dict:
    """Strip wrapper prefixes produced by torch.compile / DDP."""
    out = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."): k = k[len("_orig_mod."):]
        if k.startswith("module."):    k = k[len("module."):]
        out[k] = v
    return out

def _classes_from_ckpt(sd: dict):
    """Infer classifier output sizes from checkpoint head tensors."""
    sizes = {}
    for head in ("model","year","view","type"):
        w = sd.get(f"head_{head}.weight")
        sizes[head] = int(w.shape[0]) if w is not None else None
    return sizes

def _resize_centercrop_transform(img_size, normalize=True):
    resize_short = int(round(img_size / 224 * 256))  # e.g., 448 -> 512
    tfms = [
        T.Resize(resize_short, interpolation=IM.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ]
    if normalize:
        tfms.append(T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)))
    return T.Compose(tfms)

def _first_data_key(d, preferred):
    for k in preferred:
        if k in d:
            return k
    for k in d.keys():
        if not k.startswith("__"):
            return k
    raise KeyError("No usable keys found in .mat dict")

def _cellstr_to_list(arr):
    import numpy as np
    a = np.array(arr, dtype=object).squeeze()
    if isinstance(a, str):
        return [a]
    out = []
    for elem in np.ravel(a):
        if isinstance(elem, str):
            out.append(elem); continue
        e = np.array(elem)
        if e.dtype.kind in ("U", "S"):
            out.append("".join(e.squeeze().tolist()))
        elif e.size == 1:
            out.append(str(e.item()))
        else:
            out.append(str(e))
    return [s.strip() for s in out]

def load_name_lookups(make_model_mat_path, car_type_mat_path):
    mm = loadmat(make_model_mat_path)
    ct = loadmat(car_type_mat_path)
    makes_key  = _first_data_key(mm, ["make_names", "makes", "make"])
    models_key = _first_data_key(mm, ["model_names", "models", "model"])
    types_key  = _first_data_key(ct, ["types", "type_names", "car_type", "car_types"])
    make_names  = _cellstr_to_list(mm[makes_key])    # 1-based
    model_names = _cellstr_to_list(mm[models_key])   # 1-based
    type_names  = _cellstr_to_list(ct[types_key])    # 1-based
    return {"make": make_names, "model": model_names, "type": type_names}

def invert_mapping(d):
    """Invert a dict[int -> int] -> list s.t. inv[idx] = original_id (or None)."""
    if not d:
        return []
    n = max(d.values()) + 1
    inv = [None] * n
    for orig, idx in d.items():
        inv[idx] = int(orig)
    return inv

def first_existing(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def topk_with_names(logits, head, idx_to_orig_list, name_lookup=None, k=5):
    probs = F.softmax(logits, dim=1)
    k = min(k, probs.shape[1])
    vals, inds = probs.topk(k, dim=1)
    vals = vals[0].tolist()
    inds = [int(i) for i in inds[0].tolist()]

    names = []
    for idx in inds:
        if head == "view":
            name = VIEW_IDX_TO_NAME[idx] if 0 <= idx < len(VIEW_IDX_TO_NAME) else f"view_{idx}"
        else:
            orig = idx_to_orig_list[idx] if idx < len(idx_to_orig_list) else None
            if head == "model" and name_lookup is not None:
                # 1-based lookup
                name = name_lookup["model"][orig-1] if (isinstance(orig, int) and 1 <= orig <= len(name_lookup["model"])) else f"model_{idx}"
            elif head == "type" and name_lookup is not None:
                name = name_lookup["type"][orig-1] if (isinstance(orig, int) and 1 <= orig <= len(name_lookup["type"])) else f"type_{idx}"
            else:
                name = f"{head}_{orig if orig is not None else idx}"
        names.append(name)
    return list(zip(names, [float(v) for v in vals]))

# --------------- model loader ---------------
def load_model_from_ckpt(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["cfg"]

    # sanitize keys
    sd_raw = ck.get("model", ck.get("state_dict", ck))
    sd = _sanitize_state_dict(sd_raw)

    # infer class counts from checkpoint heads
    ck_sizes = _classes_from_ckpt(sd)
    # fallback to vocabs.json if any head missing (rare)
    root = os.path.join(cfg["root"], "compcars") if str(cfg["root"]).endswith("data") else cfg["root"]
    vocabs_path = os.path.join(root, "indexes", "vocabs.json")
    if not all(ck_sizes[h] for h in ("model","year","view","type")) and os.path.exists(vocabs_path):
        with open(vocabs_path, "r", encoding="utf-8") as f:
            voc = json.load(f)
        def cnt(key): return len(voc[key]["to_idx"])
        ck_sizes = {
            "model": ck_sizes["model"] or cnt("model_id"),
            "year":  ck_sizes["year"]  or cnt("year"),
            "view":  ck_sizes["view"]  or cnt("view"),
            "type":  ck_sizes["type"]  or cnt("type_id"),
        }

    n_model, n_year, n_view, n_type = ck_sizes["model"], ck_sizes["year"], ck_sizes["view"], ck_sizes["type"]
    assert all(isinstance(x, int) and x > 0 for x in (n_model, n_year, n_view, n_type)), f"Bad class counts: {ck_sizes}"

    model = ViTMultiTask(
        cfg["backbone"], n_model, n_year, n_view, n_type,
        img_size=cfg["img_size"], drop_path_rate=cfg.get("drop_path_rate", 0.0)
    ).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval().to(memory_format=torch.channels_last)
    return model, cfg, root, vocabs_path

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/best.pt")
    ap.add_argument("--image", default="", help="Path to a single image")
    ap.add_argument("--dir",   default="", help="Directory of images (recursive)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--make_model_mat", type=str, default=None, help="Override make_model_name.mat")
    ap.add_argument("--car_type_mat",   type=str, default=None, help="Override car_type.mat")
    ap.add_argument("--save_csv", default="", help="Optional: write predictions CSV for --dir")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, root, vocabs_path = load_model_from_ckpt(args.ckpt, device=device)
    img_size = int(cfg["img_size"])
    tf = _resize_centercrop_transform(img_size, normalize=True)

    # dataset wrapper (for vocabs + optional random test sample)
    ds = CompCarsCls(root=root, csv_file=vocabs_path.replace("vocabs.json","index.csv"),
                     transform=tf, use_bbox=cfg.get("use_bbox", True),
                     vocabs_path=vocabs_path, year_unknown=cfg.get("year_unknown","map_unk"))
    rows_test = [i for i,r in enumerate(ds.rows) if r.get("split")=="test"]

    # invert vocabs for id→name mapping
    with open(vocabs_path, "r", encoding="utf-8") as f:
        vocabs = json.load(f)
    to_idx = {k: {int(kk): int(vv) for kk, vv in v["to_idx"].items()} for k, v in vocabs.items()}
    idx_to_model_id = invert_mapping(to_idx["model_id"])
    idx_to_type_id  = invert_mapping(to_idx["type_id"])
    idx_to_year_val = invert_mapping(to_idx["year"])
    if ds.year_unk_idx is not None and len(idx_to_year_val) == ds.n_year - 1:
        idx_to_year_val = idx_to_year_val + [None]  # append unk slot

    # optional .mat lookups
    name_lookup = None
    if _SCIPY_OK:
        mm_path = first_existing(
            args.make_model_mat,
            cfg.get("make_model_mat"),
            os.path.join(root,"misc","make_model_name.mat"),
            os.path.join(cfg["root"],"compcars","misc","make_model_name.mat") if "root" in cfg else None,
        )
        ct_path = first_existing(
            args.car_type_mat,
            cfg.get("car_type_mat"),
            os.path.join(root,"misc","car_type.mat"),
            os.path.join(cfg["root"],"compcars","misc","car_type.mat") if "root" in cfg else None,
        )
        if mm_path and ct_path:
            try:
                name_lookup = load_name_lookups(mm_path, ct_path)
                if args.debug:
                    print(f"[names] using:\n  make_model: {mm_path}\n  car_type:   {ct_path}")
            except Exception as e:
                if args.debug:
                    print(f"[names] failed to load .mat files: {e}")
                name_lookup = None
    else:
        if args.debug:
            print("[names] scipy not installed; using raw IDs for names.")

    # collect paths
    paths = []
    used_random = False
    if args.image:
        assert os.path.exists(args.image), f"--image not found: {args.image}"
        paths = [args.image]
    elif args.dir:
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            paths.extend(glob.glob(os.path.join(args.dir, "**", ext), recursive=True))
        assert paths, f"No images found under {args.dir}"
    else:
        # pick a random TEST sample
        idx = random.choice(rows_test) if rows_test else random.randrange(len(ds))
        img, _, meta = ds[idx]
        # write it to a temp tensor path-less flow
        paths = []
        used_random = True
        random_sample = (img, meta)

    # inference helpers
    @torch.no_grad()
    def predict_tensor(img_tensor):
        logits = model(img_tensor.unsqueeze(0).to(device, memory_format=torch.channels_last))
        logits["model"] = logits["model"] / args.temperature
        # YEAR pretty text
        year_idx = int(logits['year'].argmax(1).item())
        y = idx_to_year_val[year_idx] if 0 <= year_idx < len(idx_to_year_val) else None
        year_text = str(y) if (isinstance(y, int) and y >= 0) else "unknown"
        # top-k heads
        view_topk  = topk_with_names(logits['view'],  'view',  list(range(ds.n_view)), name_lookup=None, k=args.topk)
        type_topk  = topk_with_names(logits['type'],  'type',  idx_to_type_id, name_lookup=name_lookup, k=args.topk)
        model_topk = topk_with_names(logits['model'],'model', idx_to_model_id, name_lookup=name_lookup, k=args.topk)
        return year_text, view_topk, type_topk, model_topk

    def pretty_print(path_or_meta, year_text, view_topk, type_topk, model_topk):
        view_name  = view_topk[0][0]  if view_topk else 'uncertain'
        type_name  = type_topk[0][0]  if type_topk else 'unknown type'
        model_name = model_topk[0][0] if model_topk else 'model_?'
        src = path_or_meta if isinstance(path_or_meta, str) else path_or_meta.get("rel_path","<dataset sample>")
        print(f"{src}\n  -> {year_text} {model_name}; body type: {type_name}; view: {view_name}.")
        def show_topk(tag, pairs):
            disp = ", ".join([f"{n} ({p:.3f})" for n,p in pairs])
            print(f"     {tag} top-{len(pairs)}: {disp}")
        show_topk("model", model_topk)
        show_topk("type ", type_topk)
        show_topk("view ", view_topk)

    # single random dataset sample path-less flow
    if used_random:
        img, meta = random_sample
        year_text, view_topk, type_topk, model_topk = predict_tensor(img)
        pretty_print(meta, year_text, view_topk, type_topk, model_topk)
        return

    # file(s) flow
    rows_for_csv = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
            continue
        x = tf(img)
        year_text, view_topk, type_topk, model_topk = predict_tensor(x)
        pretty_print(p, year_text, view_topk, type_topk, model_topk)

        if args.save_csv:
            row = {
                "path": os.path.abspath(p),
                **{f"model_top{i+1}_name": model_topk[i][0] if i < len(model_topk) else "" for i in range(args.topk)},
                **{f"model_top{i+1}_prob": f"{model_topk[i][1]:.6f}" if i < len(model_topk) else "" for i in range(args.topk)},
            }
            rows_for_csv.append(row)

    if args.save_csv and rows_for_csv:
        import csv
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            w.writeheader(); w.writerows(rows_for_csv)
        print(f"[saved] {args.save_csv}")

if __name__ == "__main__":
    main()
