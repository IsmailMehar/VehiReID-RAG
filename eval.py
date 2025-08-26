import os, csv, json, time, argparse
from collections import defaultdict, Counter

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM

from datasets.compcars_dataset import CompCarsCls
from models.vit_multitask import ViTMultiTask
from utils.metrics import year_mae_indices 

# helpers 
def init_dist_if_needed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1 and torch.cuda.is_available()
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        local_rank = 0
    return use_ddp, local_rank

def is_main_process(use_ddp):
    return True if not use_ddp else dist.get_rank() == 0

def macro_f1_from_tp_fp_fn(tp, fp, fn) -> float:
    tp, fp, fn = tp.float(), fp.float(), fn.float()
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
    return float(f1.mean())

def center_crop_or_pad(x, out_size: int):
    B, C, H, W = x.shape
    if H == out_size and W == out_size:
        return x
    if H >= out_size and W >= out_size:
        y1 = (H - out_size) // 2
        x1 = (W - out_size) // 2
        return x[:, :, y1:y1+out_size, x1:x1+out_size]
    pad_h = max(0, out_size - H)
    pad_w = max(0, out_size - W)
    pt, pb = pad_h // 2, pad_h - (pad_h // 2)
    pl, pr = pad_w // 2, pad_w - (pad_w // 2)
    return F.pad(x, (pl, pr, pt, pb))

@torch.no_grad()
def tta_forward(model, imgs, dtype, img_size, scales=(1.0,), flips=False):
    agg = None
    for s in scales:
        if s == 1.0:
            x = imgs
        else:
            new = max(8, int(round(img_size * s)))
            x = F.interpolate(imgs, size=(new, new), mode="bilinear", align_corners=False)
            x = center_crop_or_pad(x, img_size)
        views = [x, torch.flip(x, dims=[3])] if flips else [x]
        for xv in views:
            with torch.amp.autocast('cuda', dtype=dtype):
                out = model(xv)
            if agg is None:
                agg = {k: out[k].float() for k in out}
            else:
                for k in agg:
                    agg[k] += out[k].float()
    denom = len(scales) * (2 if flips else 1)
    for k in agg:
        agg[k] /= denom
    return agg

def build_type_map(index_csv_path, vocabs_json_path):
    with open(vocabs_json_path, "r", encoding="utf-8") as f:
        voc = json.load(f)
    model_to_idx = {int(k): v for k, v in voc["model_id"]["to_idx"].items()}
    type_to_idx  = {int(k): v for k, v in voc["type_id"]["to_idx"].items()}
    from collections import defaultdict, Counter
    counts = defaultdict(Counter)
    with open(index_csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r.get("split") != "train": continue
            try:
                raw_model = int(r["model_id"]); raw_type = int(r["type_id"])
            except Exception:
                continue
            if raw_model in model_to_idx and raw_type in type_to_idx:
                counts[model_to_idx[raw_model]][type_to_idx[raw_type]] += 1
    return {m: ctr.most_common(1)[0][0] for m, ctr in counts.items() if len(ctr)}

def _sanitize_state_dict(sd: dict) -> dict:
    """Strip wrapper prefixes produced by torch.compile / DDP, etc."""
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


def type_rerank_topk(model_logits, pred_type_idx, modelidx_to_typeidx, topk=5):
    scores, idxs = torch.topk(model_logits, k=min(topk, model_logits.numel()), dim=0)
    for midx in idxs.tolist():
        if midx in modelidx_to_typeidx and modelidx_to_typeidx[midx] == pred_type_idx:
            return midx
    return int(idxs[0].item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./runs/best.pt")
    ap.add_argument("--rerank", action="store_true",
                    help="Enable type-aware re-ranking of model predictions.")
    ap.add_argument("--save_csv", default="", help="Optional: write per-rank CSVs.")
    args = ap.parse_args()

    use_ddp, local_rank = init_dist_if_needed()
    main_proc = is_main_process(use_ddp)

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["cfg"]
    cfg["tta"] = {"scales": [1.0], "flips": False}

    raw_sd = ck.get("model", ck.get("state_dict", ck))
    sd = _sanitize_state_dict(raw_sd)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # dataset
    root = os.path.join(cfg["root"], "compcars")
    csv_path = os.path.join(root, "indexes", "index.csv")
    vocabs_path = os.path.join(root, "indexes", "vocabs.json")

    resize_short = int(round(cfg["img_size"] / 224 * 256))
    tf = T.Compose([
        T.Resize(resize_short, interpolation=IM.BICUBIC),
        T.CenterCrop(cfg["img_size"]),
        T.ToTensor(),
        T.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    ds = CompCarsCls(root=root, csv_file=csv_path, transform=tf,
                     use_bbox=cfg.get("use_bbox", True),
                     vocabs_path=vocabs_path,
                     year_unknown=cfg.get("year_unknown", "map_unk"))
    ds.rows = [r for r in ds.rows if r["split"] == "test"]

    n_model, n_year, n_view, n_type = ds.n_model, ds.n_year, ds.n_view, ds.n_type

    # model
    model = ViTMultiTask(cfg["backbone"], n_model, n_year, n_view, n_type,
                     img_size=cfg["img_size"],
                     drop_path_rate=cfg.get("drop_path_rate", 0.0)).to(device)

    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        raise RuntimeError(f"Strict load failed.\nMissing: {sorted(missing)}\nUnexpected: {sorted(unexpected)}") from e

    model = model.to(memory_format=torch.channels_last).eval()


    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    val_bs = int(cfg.get("val_batch_size", cfg["batch_size"] * 2))
    sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if use_ddp else None
    loader = DataLoader(
        ds, batch_size=val_bs, shuffle=False, sampler=sampler,
        num_workers=cfg.get("num_workers", 8), pin_memory=True, persistent_workers=False
    )

    # TTA from YAML
    tta_cfg = cfg.get("tta", {})
    tta_scales = tuple(tta_cfg.get("scales", [1.0]))
    tta_flips  = bool(tta_cfg.get("flips", False))

    # year mapping for MAE
    with open(vocabs_path, "r", encoding="utf-8") as f:
        voc = json.load(f)
    year_values = voc["year"]["values"]
    idx2year = torch.tensor(year_values, device=device, dtype=torch.long)
    unk_idx = ds.year_unk_idx

    # optional type map for re-rank
    modelidx_to_typeidx = build_type_map(csv_path, vocabs_path) if args.rerank else {}

    # metrics (per-rank)
    m_tp = torch.zeros(n_model, device=device, dtype=torch.int64)
    m_fp = torch.zeros(n_model, device=device, dtype=torch.int64)
    m_fn = torch.zeros(n_model, device=device, dtype=torch.int64)

    t_tp = torch.zeros(n_type,  device=device, dtype=torch.int64)
    t_fp = torch.zeros(n_type,  device=device, dtype=torch.int64)
    t_fn = torch.zeros(n_type,  device=device, dtype=torch.int64)

    v_tp = torch.zeros(n_view,  device=device, dtype=torch.int64)
    v_fp = torch.zeros(n_view,  device=device, dtype=torch.int64)
    v_fn = torch.zeros(n_view,  device=device, dtype=torch.int64)

    y_tp = torch.zeros(n_year,  device=device, dtype=torch.int64)
    y_fp = torch.zeros(n_year,  device=device, dtype=torch.int64)
    y_fn = torch.zeros(n_year,  device=device, dtype=torch.int64)

    n_total        = torch.zeros((), device=device, dtype=torch.int64)
    m_top5_correct = torch.zeros((), device=device, dtype=torch.int64)
    year_abs_err   = torch.zeros((), device=device)
    year_known_n   = torch.zeros((), device=device, dtype=torch.int64)

    rows_for_csv = []

    def update_tp_fp_fn(pred, tgt, n_classes, tp, fp, fn):
        valid = (tgt >= 0) & (tgt < n_classes)
        if not valid.any(): return
        p = pred[valid]; t = tgt[valid]
        eq = (p == t)
        if eq.any():
            tp.index_add_(0, t[eq], torch.ones_like(t[eq], dtype=tp.dtype))
        if (~eq).any():
            wp = p[~eq]; wt = t[~eq]
            fp.index_add_(0, wp, torch.ones_like(wp, dtype=fp.dtype))
            fn.index_add_(0, wt, torch.ones_like(wt, dtype=fn.dtype))

    # eval loop 
    t0 = time.time()
    if use_ddp and isinstance(sampler, DistributedSampler):
        sampler.set_epoch(0)  

    with torch.inference_mode():
        for imgs, tgts, meta in loader:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)

            logits = tta_forward(model, imgs, amp_dtype, cfg["img_size"],
                                 scales=tta_scales, flips=tta_flips)

            m_logits = logits["model"]
            t_pred = logits["type"].argmax(dim=1)
            v_pred = logits["view"].argmax(dim=1)
            y_pred = logits["year"].argmax(dim=1)

            if args.rerank and len(modelidx_to_typeidx):
                m_pred = []
                for i in range(m_logits.size(0)):
                    best = type_rerank_topk(m_logits[i], int(t_pred[i].item()),
                                            modelidx_to_typeidx, topk=5)
                    m_pred.append(best)
                m_pred = torch.tensor(m_pred, device=device, dtype=torch.long)
            else:
                m_pred = m_logits.argmax(dim=1)

            tgt_model = tgts["model"].to(device, non_blocking=True)
            tgt_type  = tgts["type"].to(device, non_blocking=True)
            tgt_view  = tgts["view"].to(device, non_blocking=True)
            tgt_year  = tgts["year"].to(device, non_blocking=True)

            update_tp_fp_fn(m_pred, tgt_model, n_model, m_tp, m_fp, m_fn)
            update_tp_fp_fn(t_pred, tgt_type,  n_type,  t_tp, t_fp, t_fn)
            update_tp_fp_fn(v_pred, tgt_view,  n_view,  v_tp, v_fp, v_fn)
            update_tp_fp_fn(y_pred, tgt_year,  n_year,  y_tp, y_fp, y_fn)

            top5 = m_logits.topk(5, dim=1).indices
            m_top5_correct += (top5 == tgt_model.unsqueeze(1)).any(dim=1).sum()

            abs_err_sum, known_cnt = year_mae_indices(y_pred, tgt_year, idx2year, unk_idx)
            year_abs_err += abs_err_sum
            year_known_n += known_cnt

            n_total += imgs.size(0)

            if args.save_csv:
                for i in range(imgs.size(0)):
                    rows_for_csv.append({
                        "rel_path": meta["rel_path"][i] if isinstance(meta["rel_path"], list) else meta["rel_path"],
                        "model_pred": int(m_pred[i].item()),
                        "model_tgt": int(tgt_model[i].item()),
                        "type_pred": int(t_pred[i].item()),
                        "type_tgt": int(tgt_type[i].item()),
                        "view_pred": int(v_pred[i].item()),
                        "view_tgt": int(tgt_view[i].item()),
                        "year_pred": int(y_pred[i].item()),
                        "year_tgt": int(tgt_year[i].item()),
                    })

    # DDP reduce (sum) 
    if use_ddp:
        for ten in (m_tp, m_fp, m_fn, t_tp, t_fp, t_fn, v_tp, v_fp, v_fn, y_tp, y_fp, y_fn):
            dist.all_reduce(ten, op=dist.ReduceOp.SUM)
        for ten in (m_top5_correct, n_total, year_abs_err, year_known_n):
            dist.all_reduce(ten, op=dist.ReduceOp.SUM)

    # Rank-0 prints metrics 
    if main_proc:
        total = n_total.clamp_min(1).float()
        model_top1 = float(m_tp.float().sum() / total)
        type_top1  = float(t_tp.float().sum() / total)
        view_top1  = float(v_tp.float().sum() / total)
        year_top1  = float(y_tp.float().sum() / total)
        model_top5 = float(m_top5_correct.float() / total)

        model_f1 = macro_f1_from_tp_fp_fn(m_tp, m_fp, m_fn)
        type_f1  = macro_f1_from_tp_fp_fn(t_tp, t_fp, t_fn)
        view_f1  = macro_f1_from_tp_fp_fn(v_tp, v_fp, v_fn)
        year_f1  = macro_f1_from_tp_fp_fn(y_tp, y_fp, y_fn)

        year_mae = float((year_abs_err / year_known_n.clamp_min(1)).item()) if year_known_n.item() > 0 else float("nan")

        took = time.time() - t0
        print(f"[Eval done in {took:.1f}s]  (DDP={use_ddp}, scales={tta_scales}, flips={tta_flips}, rerank={args.rerank})")
        print(
            f"Model  @1={model_top1:.3f}  @5={model_top5:.3f}  F1={model_f1:.3f}\n"
            f"Type   @1={type_top1:.3f}   F1={type_f1:.3f}\n"
            f"View   @1={view_top1:.3f}   F1={view_f1:.3f}\n"
            f"Year   @1={year_top1:.3f}   F1={year_f1:.3f}   MAE={year_mae:.2f}"
        )

    if args.save_csv:
        suffix = f".rank{dist.get_rank()}.csv" if use_ddp else ".csv"
        out = args.save_csv
        if not out.endswith(".csv"):  
            os.makedirs(out, exist_ok=True)
            out = os.path.join(out, "eval_preds" + suffix)
        elif use_ddp:
            base, ext = os.path.splitext(out)
            out = base + suffix
        with open(out, "w", newline="") as f:
            if rows_for_csv:
                w = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
                w.writeheader(); w.writerows(rows_for_csv)
        if main_proc:
            print(f"Wrote per-rank CSV(s) to: {args.save_csv}")

    # cleanup
    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

#run using torchrun --nproc_per_node=2 src/eval.py --ckpt ./runs/best.pt --rerank

