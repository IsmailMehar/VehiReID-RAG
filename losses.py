import torch
import torch.nn.functional as F

def multitask_loss(logits, targets, weights, label_smoothing=0.0, return_parts: bool = False):
    """
    Returns a scalar CUDA tensor. No .item() here (avoid sync).
    """
    device = next(iter(logits.values())).device
    total = torch.zeros((), device=device)

    parts = {}
    for k, t in targets.items():
        if k not in logits:
            continue
        loss_k = F.cross_entropy(logits[k], t, label_smoothing=label_smoothing)
        parts[k] = loss_k
        total = total + weights.get(k, 1.0) * loss_k
    return (total, parts) if return_parts else total