import random, math, torch
import torch.nn as nn
from collections import defaultdict

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def stratified_split_by_model(ds, val_ratio=0.1):
    """Return train_idx, val_idx stratified by model id."""
    by_cls = defaultdict(list)
    for i in range(len(ds)):
        _, tgt, _ = ds[i]
        y = int(tgt["model"])
        if y != -100:
            by_cls[y].append(i)
    tr_idx, va_idx = [], []
    for idxs in by_cls.values():
        random.shuffle(idxs)
        k = max(1, int(len(idxs) * val_ratio))
        va_idx.extend(idxs[:k]); tr_idx.extend(idxs[k:])
    random.shuffle(tr_idx); random.shuffle(va_idx)
    return tr_idx, va_idx

class WarmupCosine:
    """Warmup (linear) then cosine decay over epochs (call .step() once per epoch)."""
    def __init__(self, optimizer, base_lrs, epochs, warmup_epochs=5):
        self.opt = optimizer
        self.base_lrs = base_lrs
        self.epochs = epochs
        self.warm = warmup_epochs
        self.t = 0
    def step(self):
        self.t += 1
        for pg, base in zip(self.opt.param_groups, self.base_lrs):
            if self.t <= self.warm:
                lr = base * self.t / max(1, self.warm)
            else:
                tt = (self.t - self.warm) / max(1, (self.epochs - self.warm))
                lr = 0.5 * base * (1 + math.cos(math.pi * tt))
            pg["lr"] = lr

@torch.no_grad()
def evaluate_cls(model, loader, device, n_model):
    """Top-1 / Top-5 / Mean-per-class accuracy for classification."""
    model.eval()
    correct1 = correct5 = total = 0
    per_cls_correct = torch.zeros(n_model, dtype=torch.long)
    per_cls_total   = torch.zeros(n_model, dtype=torch.long)

    for imgs, targets, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        y = targets["model"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            logits = model(imgs)["logits"]["model"]

        pred1 = logits.argmax(1)
        _, pred5 = logits.topk(5, dim=1)

        mask = (y != -100)
        y_m = y[mask]; p1 = pred1[mask]; p5 = pred5[mask]
        total    += y_m.numel()
        correct1 += (p1 == y_m).sum().item()
        correct5 += (p5 == y_m.unsqueeze(1)).any(dim=1).sum().item()

        for cls in y_m.unique():
            m = (y_m == cls)
            per_cls_correct[cls] += (p1[m] == cls).sum()
            per_cls_total[cls]   += m.sum()

    top1 = correct1 / max(1, total)
    top5 = correct5 / max(1, total)
    mpca = (per_cls_correct.float() / per_cls_total.clamp_min(1)).mean().item()
    return top1, top5, mpca

def make_losses():
    """Factory for losses used in training."""
    ce = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=-100)
    return ce
