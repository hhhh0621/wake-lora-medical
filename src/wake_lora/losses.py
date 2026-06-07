from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F


def shifted_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    return shift_logits, shift_labels, mask


def token_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits, shift_labels, mask = shifted_logits_and_labels(logits, labels)
    vocab = shift_logits.size(-1)
    flat_loss = F.cross_entropy(
        shift_logits.float().view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    return flat_loss.view_as(shift_labels), mask


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.masked_fill(~mask, 0.0)
    denom = mask.float().sum().clamp_min(1.0)
    return values.sum() / denom


def disable_adapter_context(model: Any):
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


class TokenFeatureMemoryBank:
    def __init__(self, memory_size: int = 4) -> None:
        self.memory_size = memory_size
        self.features: dict[int, torch.Tensor] = {}
        self.ptr: dict[int, int] = {}

    @torch.no_grad()
    def centroids(self, token_ids: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        rows = []
        for idx, token_id in enumerate(token_ids.detach().cpu().tolist()):
            stored = self.features.get(int(token_id))
            if stored is None or stored.numel() == 0:
                rows.append(fallback[idx].detach())
            else:
                rows.append(stored.to(fallback.device, dtype=fallback.dtype).mean(dim=0))
        return torch.stack(rows, dim=0)

    @torch.no_grad()
    def update(self, token_ids: torch.Tensor, features: torch.Tensor) -> None:
        for token_id, feature in zip(token_ids.detach().cpu().tolist(), features.detach()):
            key = int(token_id)
            value = feature.float().detach()
            if key not in self.features:
                self.features[key] = value.new_zeros((self.memory_size, value.numel()))
                self.ptr[key] = 0
            slot = self.ptr[key]
            self.features[key][slot].copy_(value.to(self.features[key].device))
            self.ptr[key] = (slot + 1) % self.memory_size


def segment_distance(
    point: torch.Tensor,
    anchor_a: torch.Tensor,
    anchor_b: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    vec_ab = anchor_b - anchor_a
    vec_ap = point - anchor_a
    len_sq_ab = (vec_ab * vec_ab).sum(dim=1, keepdim=True).clamp_min(1e-8)
    t_raw = (vec_ap * vec_ab).sum(dim=1, keepdim=True) / len_sq_ab
    t_clamped = t_raw.clamp(0.0, 1.0)
    closest = anchor_a + t_clamped * vec_ab
    dist = ((point - closest) ** 2).mean(dim=1)
    metrics = {
        "segment_t_mean": float(t_clamped.mean().detach().cpu()),
        "segment_out_ratio": float(((t_raw < 0) | (t_raw > 1)).float().mean().detach().cpu()),
        "segment_ab_dist": float(len_sq_ab.sqrt().mean().detach().cpu()),
    }
    return dist.mean(), metrics


class WakeLoraLoss:
    def __init__(
        self,
        lambda_kl: float = 0.1,
        lambda_ce_reuse: float = 0.0,
        lambda_segment: float = 0.0,
        segment_memory_size: int = 4,
        eps: float = 1e-4,
        alpha_min: float = 0.0,
        alpha_max: float = 2.0,
        ce_reuse_max: float = 2.0,
        temperature: float = 1.0,
    ) -> None:
        self.lambda_kl = lambda_kl
        self.lambda_ce_reuse = lambda_ce_reuse
        self.lambda_segment = lambda_segment
        self.eps = eps
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ce_reuse_max = ce_reuse_max
        self.temperature = temperature
        self.segment_bank = TokenFeatureMemoryBank(memory_size=segment_memory_size)

    def __call__(self, model: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        outputs_lora = model(**batch, output_hidden_states=self.lambda_segment > 0)
        logits_lora = outputs_lora.logits
        ce_lora_tok, mask = token_ce_loss(logits_lora, batch["labels"])

        with torch.no_grad():
            with disable_adapter_context(model):
                outputs_base = model(**batch)
            logits_base = outputs_base.logits.detach()
            ce_base_tok, _ = token_ce_loss(logits_base, batch["labels"])

        ce_reuse = self.lambda_ce_reuse / (ce_base_tok.detach() + self.eps)
        ce_reuse = ce_reuse.clamp(min=0.0, max=self.ce_reuse_max)
        ce_weight = 1.0 + ce_reuse
        weighted_ce_num = (ce_weight * ce_lora_tok).masked_fill(~mask, 0.0).sum()
        weighted_ce_den = ce_weight.masked_fill(~mask, 0.0).sum().clamp_min(1.0)
        ce_lora = weighted_ce_num / weighted_ce_den

        lora_shift, _, _ = shifted_logits_and_labels(logits_lora, batch["labels"])
        base_shift, _, _ = shifted_logits_and_labels(logits_base, batch["labels"])

        temp = self.temperature
        log_p_lora = F.log_softmax(lora_shift.float() / temp, dim=-1)
        p_base = F.softmax(base_shift.float() / temp, dim=-1)
        kl_tok = F.kl_div(log_p_lora, p_base, reduction="none").sum(dim=-1) * (temp * temp)

        alpha = self.lambda_kl / (ce_base_tok.detach() + self.eps)
        alpha = alpha.clamp(min=self.alpha_min, max=self.alpha_max)
        wake_kl = masked_mean(alpha * kl_tok, mask)
        segment_loss = logits_lora.new_zeros(())
        segment_metrics = {
            "segment_loss": 0.0,
            "segment_t_mean": 0.0,
            "segment_out_ratio": 0.0,
            "segment_ab_dist": 0.0,
            "segment_memory_tokens": float(len(self.segment_bank.features)),
        }
        if self.lambda_segment > 0:
            hidden = outputs_lora.hidden_states[-1][:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            valid_hidden = hidden[mask].float()
            valid_labels = shift_labels[mask]
            if valid_hidden.numel() > 0:
                output_weight = model.get_output_embeddings().weight
                anchor_a = output_weight[valid_labels].float()
                anchor_b = self.segment_bank.centroids(valid_labels, fallback=anchor_a)
                segment_loss, segment_metrics = segment_distance(valid_hidden, anchor_a, anchor_b)
                self.segment_bank.update(valid_labels, valid_hidden)
                segment_metrics["segment_loss"] = float(segment_loss.detach().cpu())
                segment_metrics["segment_memory_tokens"] = float(len(self.segment_bank.features))

        total = ce_lora + wake_kl + self.lambda_segment * segment_loss

        metrics = {
            "loss": float(total.detach().cpu()),
            "ce_lora": float(ce_lora.detach().cpu()),
            "wake_kl": float(wake_kl.detach().cpu()),
            "ce_base": float(masked_mean(ce_base_tok, mask).detach().cpu()),
            "alpha_mean": float(masked_mean(alpha, mask).detach().cpu()),
            "ce_reuse_weight_mean": float(masked_mean(ce_weight, mask).detach().cpu()),
            "token_count": float(mask.float().sum().detach().cpu()),
        }
        metrics.update(segment_metrics)
        return total, metrics
