# src/utils/samplers.py
import math
from collections import Counter
from typing import Optional, Sequence

import torch
from torch.utils.data import Sampler


__all__ = ["DistributedWeightedSampler", "model_balanced_weights"]


class DistributedWeightedSampler(Sampler[int]):
    """
    Deterministic, epoch-dependent weighted sampler sharded across DDP ranks.

    - Draws `total_size = num_replicas * num_samples_per_rank` indices once,
      then assigns them to ranks by striding (no overlap).
    - Guarantees each rank yields exactly `num_samples_per_rank` samples.
    - Assumes replacement sampling (matches WeightedRandomSampler's common use).

    Args:
        weights (Tensor | Sequence[float]): per-sample weights (len == dataset size).
        num_samples (int | None): If provided, interpreted as the TOTAL number of samples
            across all ranks per epoch. Each rank will then yield floor(num_samples / num_replicas).
            If None, each rank yields ceil(dataset_size / num_replicas).
        replacement (bool): must be True (non-replacement across ranks is not supported here).
        num_replicas (int | None): DDP world size; inferred if None or if DDP uninitialized.
        rank (int | None): DDP rank; inferred if None or if DDP uninitialized.
        seed (int): base seed; epoch is added for per-epoch determinism.
    """
    def __init__(
        self,
        weights: Sequence[float] | torch.Tensor,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(None)
        if not replacement:
            raise NotImplementedError("DistributedWeightedSampler supports replacement=True only.")

        w = torch.as_tensor(weights, dtype=torch.float32, device="cpu")
        if w.numel() == 0:
            raise ValueError("weights must be non-empty")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        if float(w.sum()) == 0.0:
            raise ValueError("sum(weights) must be > 0")

        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0

        self.weights = w.contiguous()
        self.replacement = True
        self.seed = int(seed)
        self.epoch = 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.dataset_size = int(self.weights.numel())

        if num_samples is None:
            self.num_samples_per_rank = int(math.ceil(self.dataset_size / self.num_replicas))
        else:
            if num_samples <= 0:
                raise ValueError("num_samples must be positive")
            self.num_samples_per_rank = int(num_samples // self.num_replicas)
            if self.num_samples_per_rank == 0:
                raise ValueError("num_samples too small for given num_replicas")

        self.total_size = self.num_samples_per_rank * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples_per_rank

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        global_indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g)
        
        rank_indices = global_indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices.tolist())


def model_balanced_weights(
    dataset,
    key: str = "model_id",
    alpha: float = 1.0,
    device: Optional[torch.device | str] = None,
) -> torch.Tensor:
    """
    Per-sample weights inverse to class frequency of `key`.

    weight = 1 / (count ** alpha), with alpha in (0, 1] to smooth extremes.
    Expects `dataset.rows` to be an iterable of dicts/objects with `key`.

    Args:
        dataset: object with `.rows`
        key: field name used to define classes (default "model_id")
        alpha: smoothing exponent in (0, 1]; 1.0 = pure inverse frequency
        device: optional device to place the returned tensor on

    Returns:
        torch.DoubleTensor of shape [len(dataset.rows)]
    """
    try:
        rows = dataset.rows
    except AttributeError as e:
        raise AttributeError("dataset must expose `.rows`") from e

    ids = []
    for r in rows:
        ids.append(int(r[key] if isinstance(r, dict) else getattr(r, key)))

    counts = Counter(ids)
    if any(c <= 0 for c in counts.values()):
        raise ValueError("All class counts must be positive")

    alpha = float(alpha)
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")

    weights = [1.0 / (counts[i] ** alpha) for i in ids]
    t = torch.tensor(weights, dtype=torch.double)
    if device is not None:
        t = t.to(device)
    return t
