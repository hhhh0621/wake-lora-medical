from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import masked_mean, token_ce_loss
from .utils import perplexity_from_nll


def model_input_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate_nll(
    model: Any,
    dataloader: DataLoader,
    max_batches: int | None = None,
    desc: str = "eval",
) -> dict[str, float]:
    model.eval()
    device = model_input_device(model)
    total_loss = 0.0
    total_tokens = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc=desc, leave=False)):
        if max_batches is not None and step >= max_batches:
            break
        batch = move_batch(batch, device)
        outputs = model(**batch)
        ce_tok, mask = token_ce_loss(outputs.logits, batch["labels"])
        token_count = float(mask.float().sum().cpu())
        loss = float(masked_mean(ce_tok, mask).cpu())
        total_loss += loss * token_count
        total_tokens += token_count

    nll = total_loss / max(total_tokens, 1.0)
    return {
        "nll": nll,
        "perplexity": perplexity_from_nll(nll),
        "tokens": total_tokens,
    }
