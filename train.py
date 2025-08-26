import os, yaml, torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode as IM
from torchvision.transforms import RandAugment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.distributed as dist
from tqdm import tqdm

from models.vit_multitask import ViTMultiTask
from losses import multitask_loss
from datasets.compcars_dataset import CompCarsCls
from utils.metrics import topk_correct, macro_f1_end, year_mae_indices
from utils.samplers import DistributedWeightedSampler, model_balanced_weights
from utils.mats import WarmupCosine

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import time

from torch.amp import GradScaler  

def tick_all(msg: str, main_process: bool):
    torch.cuda.synchronize()
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if main_process:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


#  logging helpers (CSV + PNG) 
def init_train_sums():
    return {"n": 0.0, "total": 0.0, "model": 0.0, "type": 0.0, "view": 0.0, "year": 0.0}

def accumulate_train_sums(sums, total_loss, parts, batch_size: int):
    sums["n"]     += batch_size
    sums["total"] += float(total_loss.item()) * batch_size
    for k in ("model", "type", "view", "year"):
        if k in parts:
            sums[k] += float(parts[k].item()) * batch_size

def render_pngs_and_write_csv(logs, out_dir: str):
    if not logs: return
    import csv
    os.makedirs(out_dir, exist_ok=True)
    # CSV
    fn = list(logs[0].keys())
    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(logs)
    # PNGs
    epochs = [r["epoch"] for r in logs]
    plt.figure()
    plt.plot(epochs, [r["train_loss"] for r in logs], label="total")
    plt.plot(epochs, [r["train_loss_model"] for r in logs], label="model")
    plt.plot(epochs, [r["train_loss_type"] for r in logs],  label="type")
    plt.plot(epochs, [r["train_loss_view"] for r in logs],  label="view")
    plt.plot(epochs, [r["train_loss_year"] for r in logs],  label="year")
    plt.xlabel("epoch"); plt.ylabel("train loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png")); plt.close()

    plt.figure()
    plt.plot(epochs, [r["val_model_top1"] for r in logs],       label="model@1")
    plt.plot(epochs, [r["val_model_top5"] for r in logs],       label="model@5")
    plt.plot(epochs, [r["val_model_macro_f1"] for r in logs],   label="model macro-F1")
    plt.plot(epochs, [r["val_type_top1"] for r in logs],        label="type@1")
    plt.plot(epochs, [r["val_view_top1"] for r in logs],        label="view@1")
    plt.plot(epochs, [r["val_year_top1"] for r in logs],        label="year@1")
    plt.xlabel("epoch"); plt.ylabel("val metric"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_curves.png")); plt.close()

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

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

def make_llrd_param_groups(model, base_lr, weight_decay, layer_decay=0.8):
    groups = []
    num_blocks = len(getattr(model.backbone, 'blocks', []))

    def layer_id(n: str) -> int:
        if n.startswith('head_'):
            return num_blocks + 1
        if 'backbone.blocks.' in n:
            return int(n.split('backbone.blocks.')[1].split('.')[0]) + 1
        return 0

    by_layer = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        lid = layer_id(n)
        scale = layer_decay ** (num_blocks + 1 - lid)
        wd = 0.0 if (p.ndim < 2 or n.endswith('bias')) else weight_decay
        key = (lid, wd)
        if key not in by_layer:
            by_layer[key] = {'params': [], 'lr': base_lr * scale, 'weight_decay': wd}
        by_layer[key]['params'].append(p)

    for (_, _), g in sorted(by_layer.items(), key=lambda kv: kv[0]):
        groups.append(g)
    return groups

def is_main_process(use_ddp):
    return True if not use_ddp else dist.get_rank() == 0

def main(cfg_path='./config/default.yaml'):
    cfg = load_cfg(cfg_path)
    use_ddp, local_rank = init_dist_if_needed()
    main_process = is_main_process(use_ddp)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if bf16_ok else torch.float16

    #  transforms 
    mean = (0.485, 0.456, 0.406); std = (0.229, 0.224, 0.225)
    resize_short = int(round(cfg['img_size'] / 224 * 256))

    train_tf = T.Compose([
        T.RandomResizedCrop(cfg['img_size'], scale=tuple(cfg.get('random_resized_crop_scale', (0.8, 1.0))),
                            ratio=(0.8, 1.25), interpolation=IM.BICUBIC),
        T.RandomHorizontalFlip(p=0.0 if not cfg.get('hflip_for_view', False) else 0.5),
        RandAugment(num_ops=cfg.get('randaugment', {}).get('N', 1),
                    magnitude=cfg.get('randaugment', {}).get('M', 7)),
        T.ToTensor(),
        T.Normalize(mean, std),
        *( [T.RandomErasing(p=0.25, value='random')] if cfg.get('random_erasing', True) else [] )
    ])
    val_tf = T.Compose([
        T.Resize(resize_short, interpolation=IM.BICUBIC),
        T.CenterCrop(cfg['img_size']),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    #  datasets 
    csv_path    = os.path.join(cfg['root'], 'compcars', 'indexes', 'index.csv')
    vocabs_path = os.path.join(cfg['root'], 'compcars', 'indexes', 'vocabs.json')

    train_set = CompCarsCls(root=os.path.join(cfg['root'], 'compcars'),
                            csv_file=csv_path, transform=train_tf,
                            use_bbox=cfg.get('use_bbox', True),
                            vocabs_path=vocabs_path, year_unknown="map_unk")
    val_set   = CompCarsCls(root=os.path.join(cfg['root'], 'compcars'),
                            csv_file=csv_path, transform=val_tf,
                            use_bbox=cfg.get('use_bbox', True),
                            vocabs_path=vocabs_path, year_unknown="map_unk")

    train_set.rows = [r for r in train_set.rows if r['split'] == 'train']
    val_set.rows   = [r for r in val_set.rows   if r['split'] == 'test']

    import json
    with open(vocabs_path, "r", encoding="utf-8") as f:
        vocabs = json.load(f)
    year_values = vocabs["year"]["values"]
    year_idx_to_val = torch.tensor(year_values, device=device, dtype=torch.long)
    unk_idx = train_set.year_unk_idx
    if unk_idx is not None:
        assert unk_idx == year_idx_to_val.numel(), \
            f"unk_idx={unk_idx} must equal number of real-year classes={year_idx_to_val.numel()}"

    n_model, n_year, n_view, n_type = train_set.n_model, train_set.n_year, train_set.n_view, train_set.n_type

    #  model 
    model = ViTMultiTask(cfg["backbone"], n_model, n_year, n_view, n_type,
                         img_size=cfg["img_size"], drop_path_rate=cfg.get("drop_path_rate", 0.0)).to(device)
    
    model = model.to(memory_format=torch.channels_last)
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
    
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            static_graph=True, gradient_as_bucket_view=True
        )

    #  sampler & loaders 
    sample_weights = model_balanced_weights(train_set)
    if use_ddp:
        sampler = DistributedWeightedSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            seed=42,
        )
    else:
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_set, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        sampler=sampler,
        num_workers=cfg["num_workers"], 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=4, 
        drop_last=True
    )

    if use_ddp:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set, shuffle=False, drop_last=False
        )
    else:
        val_sampler = None

    val_bs = int(cfg.get("val_batch_size", cfg["batch_size"] * 2))  

    
    val_loader = DataLoader(
        val_set, 
        batch_size=val_bs, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg["num_workers"], 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2
    )

    #  optimiser + scheduler 
    base_lr, wd = cfg["lr"], cfg["weight_decay"]
    decay = float(cfg.get("layerwise_lr_decay", 0.8))
    pg = make_llrd_param_groups(model if not hasattr(model, "module") else model.module,
                                base_lr=base_lr, weight_decay=wd, layer_decay=decay)
    try:
        opt = torch.optim.AdamW(pg, lr=base_lr, weight_decay=wd, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(pg, lr=base_lr, weight_decay=wd)
    sched = WarmupCosine(opt, base_lrs=[g['lr'] for g in pg], epochs=cfg["epochs"],
                         warmup_epochs=int(cfg.get("warmup_epochs", 10)))

    loss_weights = cfg['loss_weights']
    best_f1 = 0.0
    os.makedirs(cfg['out_dir'], exist_ok=True)

    logs = []
    grad_accum_steps = int(cfg.get("grad_accum_steps", 1))
    scaler = GradScaler('cuda', enabled=(autocast_dtype == torch.float16))

    if main_process:
        print(f"DDP: {use_ddp} | Device: {device} | AMP: {autocast_dtype} | "
              f"Batch: {cfg['batch_size']} | Accum: {grad_accum_steps}")

    #  training loop 
    for epoch in range(cfg["epochs"]):
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)
            if isinstance(val_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

        model.train()
        opt.zero_grad(set_to_none=True)
        train_sums = init_train_sums()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", disable=not main_process)
        for step, (imgs, tgts, meta) in enumerate(pbar):
            imgs = imgs.to(device=device, non_blocking=True, memory_format=torch.channels_last)
            tgts = {k: v.to(device, non_blocking=True) for k, v in tgts.items()}

            with torch.amp.autocast('cuda', dtype=autocast_dtype):
                total_loss, parts = multitask_loss(
                    model(imgs), tgts, loss_weights, cfg["label_smoothing"], return_parts=True
                )
                loss = total_loss / grad_accum_steps

            if main_process:
                accumulate_train_sums(train_sums, total_loss, parts, imgs.size(0))

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

            if main_process and (step % 20 == 0):
                pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.3f}")

        sched.step()

        # ALL RANKS hit this barrier & only rank0 prints 
        tick_all(f"epoch {epoch+1} end of train step", main_process=main_process)

        # Validation (distributed) 
        t_val = time.time()
        model.eval()

        # per-class TP/FP/FN vectors (much smaller than confusion matrices)
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

        @torch.no_grad()
        def update_tp_fp_fn(pred, tgt, n_classes, tp, fp, fn):
            valid = (tgt >= 0) & (tgt < n_classes)
            if not valid.any(): return
            p = pred[valid]; t = tgt[valid]
            eq = (p == t)
            if eq.any():
                tp.index_add_(0, t[eq], torch.ones_like(t[eq], dtype=tp.dtype))
            if (~eq).any():
                wrong_p = p[~eq]; wrong_t = t[~eq]
                fp.index_add_(0, wrong_p, torch.ones_like(wrong_p, dtype=fp.dtype))
                fn.index_add_(0, wrong_t, torch.ones_like(wrong_t, dtype=fn.dtype))

        def macro_f1_from_tp_fp_fn(tp, fp, fn) -> float:
            tp, fp, fn = tp.float(), fp.float(), fn.float()
            denom = 2*tp + fp + fn
            f1 = torch.where(denom > 0, 2*tp/denom, torch.zeros_like(tp))
            return float(f1.mean())

        with torch.inference_mode():
            for imgs, tgts, _ in val_loader:
                imgs = imgs.to(device=device, non_blocking=True, memory_format=torch.channels_last)
                tgts = {k: v.to(device, non_blocking=True) for k, v in tgts.items()}

                with torch.amp.autocast('cuda', dtype=autocast_dtype):
                    logits = model(imgs)

                m_pred = logits["model"].argmax(dim=1)
                t_pred = logits["type"].argmax(dim=1)
                v_pred = logits["view"].argmax(dim=1)
                y_pred = logits["year"].argmax(dim=1)

                update_tp_fp_fn(m_pred, tgts["model"], n_model, m_tp, m_fp, m_fn)
                update_tp_fp_fn(t_pred, tgts["type"],  n_type,  t_tp, t_fp, t_fn)
                update_tp_fp_fn(v_pred, tgts["view"],  n_view,  v_tp, v_fp, v_fn)
                update_tp_fp_fn(y_pred, tgts["year"],  n_year,  y_tp, y_fp, y_fn)

                top5 = logits["model"].topk(5, dim=1).indices
                m_top5_correct += (top5 == tgts["model"].unsqueeze(1)).any(dim=1).sum()

                abs_err_sum, known_cnt = year_mae_indices(y_pred, tgts["year"], year_idx_to_val, unk_idx)
                year_abs_err += abs_err_sum
                year_known_n += known_cnt

                n_total += imgs.size(0)

        if use_ddp:
            for ten in (m_tp, m_fp, m_fn, t_tp, t_fp, t_fn, v_tp, v_fp, v_fn, y_tp, y_fp, y_fn):
                dist.all_reduce(ten, op=dist.ReduceOp.SUM)
            for ten in (m_top5_correct, n_total, year_abs_err, year_known_n):
                dist.all_reduce(ten, op=dist.ReduceOp.SUM)

        # Rank-0 metrics
        if main_process:
            total = n_total.clamp_min(1).float()
            m_top1 = float(m_tp.float().sum() / total)
            t_top1 = float(t_tp.float().sum() / total)
            v_top1 = float(v_tp.float().sum() / total)
            y_top1 = float(y_tp.float().sum() / total)
            m_top5 = float(m_top5_correct.float() / total)

            m_f1 = macro_f1_from_tp_fp_fn(m_tp, m_fp, m_fn)
            t_f1 = macro_f1_from_tp_fp_fn(t_tp, t_fp, t_fn)
            v_f1 = macro_f1_from_tp_fp_fn(v_tp, v_fp, v_fn)
            y_f1 = macro_f1_from_tp_fp_fn(y_tp, y_fp, y_fn)

            year_mae = float((year_abs_err / year_known_n.clamp_min(1)).item()) if year_known_n.item() > 0 else float("nan")

            print(
                f"Val | model@1={m_top1:.3f} model@5={m_top5:.3f} model_F1={m_f1:.3f}  |  "
                f"type@1={t_top1:.3f} type_F1={t_f1:.3f}  |  "
                f"view@1={v_top1:.3f} view_F1={v_f1:.3f}  |  "
                f"year@1={y_top1:.3f} year_F1={y_f1:.3f} year_MAE={year_mae:.2f}"
            )

            tl_total = train_sums["total"] / max(1.0, train_sums["n"])
            tl_model = train_sums["model"] / max(1.0, train_sums["n"])
            tl_type  = train_sums["type"]  / max(1.0, train_sums["n"])
            tl_view  = train_sums["view"]  / max(1.0, train_sums["n"])
            tl_year  = train_sums["year"]  / max(1.0, train_sums["n"])
            row = {
                "epoch": epoch + 1,
                "lr": opt.param_groups[0]["lr"],
                "train_loss": tl_total,
                "train_loss_model": tl_model,
                "train_loss_type": tl_type,
                "train_loss_view": tl_view,
                "train_loss_year": tl_year,
                "val_model_top1": m_top1,
                "val_model_top5": m_top5,
                "val_model_macro_f1": m_f1,
                "val_type_top1": t_top1,
                "val_view_top1": v_top1,
                "val_year_top1": y_top1,
                "val_year_mae": year_mae,
            }
            logs.append(row)
            render_pngs_and_write_csv(logs, cfg["out_dir"])

        if use_ddp:
            dist.barrier()
        tick_all(f"after validation (took {time.time() - t_val:.1f}s)", main_process=main_process)

        # Checkpoint 
        t_ckpt = time.time()
        if main_process:
            if m_f1 > best_f1:
                best_f1 = m_f1
                to_save = model.module if hasattr(model, "module") else model
                torch.save({"model": to_save.state_dict(), "cfg": cfg},
                        os.path.join(cfg["out_dir"], "best.pt"))
                print("✓ Saved best checkpoint")

        if use_ddp:
            dist.barrier()
        tick_all(f"after checkpoint save (took {time.time() - t_ckpt:.1f}s)", main_process=main_process)


    # Clean up DDP
    if use_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
