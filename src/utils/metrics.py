import torch
from sklearn.metrics import f1_score

@torch.no_grad()
def topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int = 1):
    """
    Returns (num_correct, num_samples) as CUDA tensors.
    No .item() here; reduce once at the end if using DDP.
    """
    _, pred = logits.topk(k, dim=1)        # [B, k]
    correct_vec = pred.eq(targets.view(-1, 1)).any(dim=1)  # [B]
    num_correct = correct_vec.sum()
    num_samples = torch.tensor(logits.size(0), device=logits.device)
    return num_correct, num_samples

@torch.no_grad()
def macro_f1_end(all_logits: torch.Tensor, all_targets: torch.Tensor):
    """
    Compute macro-F1 once at the end (CPU). Use only on the full val set
    (e.g., rank-0 with non-distributed val sampler).
    """
    pred = all_logits.argmax(dim=1).cpu().numpy()
    tgt  = all_targets.cpu().numpy()
    return f1_score(tgt, pred, average='macro')

@torch.no_grad()
def year_mae_indices(pred_idx: torch.Tensor,
                     tgt_idx: torch.Tensor,
                     idx_to_year: torch.Tensor,
                     unk_idx: int | None):
    """
    Compute sum |pred_year - true_year| and count over samples with KNOWN year
    *and* where the prediction is also a KNOWN class (not UNK).
    Returns (sum_abs_error, count) as CUDA tensors.
    """
    device = tgt_idx.device
    if unk_idx is None:
        mask = torch.ones_like(tgt_idx, dtype=torch.bool, device=device)
    else:
        mask = (tgt_idx != unk_idx) & (pred_idx != unk_idx)

    if not mask.any():
        return torch.zeros((), device=device), torch.zeros((), device=device)

    err = (idx_to_year[pred_idx[mask]] - idx_to_year[tgt_idx[mask]]).abs().sum()
    cnt = mask.sum()
    return err, cnt
