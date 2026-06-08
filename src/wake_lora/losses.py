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
        self.counts: dict[int, int] = {}

    @torch.no_grad()
    def centroids(
        self,
        token_ids: torch.Tensor,
        fallback: torch.Tensor,
        min_count: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        reliable = []
        for idx, token_id in enumerate(token_ids.detach().cpu().tolist()):
            key = int(token_id)
            stored = self.features.get(key)
            count = self.counts.get(key, 0)
            if stored is None or stored.numel() == 0 or count < min_count:
                rows.append(fallback[idx].detach())
                reliable.append(min_count <= 0)
            else:
                valid_count = min(count, self.memory_size)
                rows.append(stored[:valid_count].to(fallback.device, dtype=fallback.dtype).mean(dim=0))
                reliable.append(True)
        return torch.stack(rows, dim=0), torch.tensor(reliable, dtype=torch.bool, device=fallback.device)

    @torch.no_grad()
    def update(self, token_ids: torch.Tensor, features: torch.Tensor) -> None:
        for token_id, feature in zip(token_ids.detach().cpu().tolist(), features.detach()):
            key = int(token_id)
            value = feature.float().detach()
            if key not in self.features:
                self.features[key] = value.new_zeros((self.memory_size, value.numel()))
                self.ptr[key] = 0
                self.counts[key] = 0
            slot = self.ptr[key]
            self.features[key][slot].copy_(value.to(self.features[key].device))
            self.ptr[key] = (slot + 1) % self.memory_size
            self.counts[key] += 1


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


def simplex_projection_loss(
    point: torch.Tensor,
    anchor_ids: torch.Tensor,
    output_weight: torch.Tensor,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    anchors = output_weight[anchor_ids].float().detach()
    point_norm = F.normalize(point.float(), dim=-1)
    anchor_norm = F.normalize(anchors, dim=-1)
    logits = torch.einsum("nh,nkh->nk", point_norm, anchor_norm) / max(temperature, 1e-4)
    weights = F.softmax(logits, dim=-1)
    closest = torch.einsum("nk,nkh->nh", weights, anchor_norm)
    dist = ((point_norm - closest) ** 2).sum(dim=-1)
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
    metrics = {
        "simplex_loss": float(dist.mean().detach().cpu()),
        "simplex_anchor_count": float(anchor_ids.size(1)),
        "simplex_weight_entropy": float(entropy.mean().detach().cpu()),
        "simplex_max_weight": float(weights.max(dim=-1).values.mean().detach().cpu()),
    }
    return dist.mean(), metrics


def simplex_ce_loss(
    lora_logits: torch.Tensor,
    base_logits: torch.Tensor,
    target_ids: torch.Tensor,
    top_k: int,
    label_mix: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    top_k = max(1, min(top_k, base_logits.size(-1)))
    label_mix = min(max(label_mix, 0.0), 1.0)
    top_base = base_logits.float().topk(k=top_k, dim=-1)
    anchor_ids = torch.cat([target_ids.unsqueeze(1), top_base.indices], dim=1)
    base_weights = F.softmax(top_base.values / max(temperature, 1e-4), dim=-1)
    q = lora_logits.new_zeros(anchor_ids.shape, dtype=torch.float32)
    q[:, 0] = 1.0 - label_mix
    q[:, 1:] = label_mix * base_weights
    log_probs = F.log_softmax(lora_logits.float(), dim=-1)
    anchor_log_probs = log_probs.gather(dim=-1, index=anchor_ids)
    token_loss = -(q * anchor_log_probs).sum(dim=-1)
    metrics = {
        "simplex_ce_loss": float(token_loss.mean().detach().cpu()),
        "simplex_ce_target_mass": float(q[:, 0].mean().detach().cpu()),
        "simplex_ce_base_entropy": float(
            (-(base_weights * base_weights.clamp_min(1e-8).log()).sum(dim=-1)).mean().detach().cpu()
        ),
    }
    return token_loss.mean(), metrics


class WakeLoraLoss:
    def __init__(
        self,
        lambda_kl: float = 0.1,
        lambda_ce_reuse: float = 0.0,
        lambda_self_reuse: float = 0.0,
        lambda_consistency: float = 0.0,
        lambda_segment: float = 0.0,
        lambda_simplex: float = 0.0,
        lambda_simplex_ce: float = 0.0,
        hard_ce_threshold: float = 0.0,
        hard_ce_min_weight: float = 0.1,
        segment_memory_size: int = 4,
        segment_min_count: int = 0,
        simplex_top_k: int = 8,
        simplex_label_mix: float = 0.2,
        eps: float = 1e-4,
        alpha_min: float = 0.0,
        alpha_max: float = 2.0,
        ce_reuse_max: float = 2.0,
        self_reuse_max: float = 4.0,
        temperature: float = 1.0,
        consistency_temperature: float = 1.0,
        simplex_temperature: float = 0.2,
        collect_segment_features: bool = False,
    ) -> None:
        self.lambda_kl = lambda_kl
        self.lambda_ce_reuse = lambda_ce_reuse
        self.lambda_self_reuse = lambda_self_reuse
        self.lambda_consistency = lambda_consistency
        self.lambda_segment = lambda_segment
        self.lambda_simplex = lambda_simplex
        self.lambda_simplex_ce = lambda_simplex_ce
        self.hard_ce_threshold = hard_ce_threshold
        self.hard_ce_min_weight = hard_ce_min_weight
        self.segment_min_count = segment_min_count
        self.simplex_top_k = simplex_top_k
        self.simplex_label_mix = simplex_label_mix
        self.eps = eps
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ce_reuse_max = ce_reuse_max
        self.self_reuse_max = self_reuse_max
        self.temperature = temperature
        self.consistency_temperature = consistency_temperature
        self.simplex_temperature = simplex_temperature
        self.collect_segment_features = collect_segment_features
        self.segment_bank = TokenFeatureMemoryBank(memory_size=segment_memory_size)

    def __call__(self, model: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        needs_segment_features = self.lambda_segment > 0 or self.lambda_simplex > 0 or self.collect_segment_features
        needs_base_forward = (
            self.lambda_kl > 0
            or self.lambda_ce_reuse > 0
            or self.lambda_simplex > 0
            or self.lambda_simplex_ce > 0
        )
        needs_consistency = self.lambda_consistency > 0 and model.training
        outputs_lora = model(**batch, output_hidden_states=needs_segment_features)
        logits_lora = outputs_lora.logits
        ce_lora_tok, mask = token_ce_loss(logits_lora, batch["labels"])
        if needs_consistency:
            outputs_lora_2 = model(**batch)
            logits_lora_2 = outputs_lora_2.logits
            ce_lora_tok_2, _ = token_ce_loss(logits_lora_2, batch["labels"])
            ce_objective_tok = 0.5 * (ce_lora_tok + ce_lora_tok_2)
        else:
            logits_lora_2 = None
            ce_objective_tok = ce_lora_tok

        if needs_base_forward:
            with torch.no_grad():
                with disable_adapter_context(model):
                    outputs_base = model(**batch)
                logits_base = outputs_base.logits.detach()
                ce_base_tok, _ = token_ce_loss(logits_base, batch["labels"])
        else:
            logits_base = None
            ce_base_tok = torch.zeros_like(ce_lora_tok)

        if self.lambda_ce_reuse > 0:
            ce_reuse = self.lambda_ce_reuse / (ce_base_tok.detach() + self.eps)
            ce_reuse = ce_reuse.clamp(min=0.0, max=self.ce_reuse_max)
            ce_weight = 1.0 + ce_reuse
        else:
            ce_weight = torch.ones_like(ce_lora_tok)
        if self.lambda_self_reuse > 0:
            self_reuse = self.lambda_self_reuse / (ce_objective_tok.detach() + self.eps)
            self_reuse = self_reuse.clamp(min=0.0, max=self.self_reuse_max)
            ce_weight = ce_weight + self_reuse
        hard_ce_weight = torch.ones_like(ce_lora_tok)
        if self.hard_ce_threshold > 0:
            hard_ce_weight = self.hard_ce_threshold / (ce_objective_tok.detach() + self.eps)
            hard_ce_weight = hard_ce_weight.clamp(min=self.hard_ce_min_weight, max=1.0)
            ce_weight = ce_weight * hard_ce_weight
        weighted_ce_num = (ce_weight * ce_objective_tok).masked_fill(~mask, 0.0).sum()
        weighted_ce_den = ce_weight.masked_fill(~mask, 0.0).sum().clamp_min(1.0)
        ce_lora = weighted_ce_num / weighted_ce_den

        consistency_loss = logits_lora.new_zeros(())
        if logits_lora_2 is not None:
            lora_shift_1, _, _ = shifted_logits_and_labels(logits_lora, batch["labels"])
            lora_shift_2, _, _ = shifted_logits_and_labels(logits_lora_2, batch["labels"])
            temp_cons = self.consistency_temperature
            log_p_1 = F.log_softmax(lora_shift_1.float() / temp_cons, dim=-1)
            log_p_2 = F.log_softmax(lora_shift_2.float() / temp_cons, dim=-1)
            p_1 = log_p_1.exp()
            p_2 = log_p_2.exp()
            kl_12 = F.kl_div(log_p_1, p_2.detach(), reduction="none").sum(dim=-1)
            kl_21 = F.kl_div(log_p_2, p_1.detach(), reduction="none").sum(dim=-1)
            consistency_loss = masked_mean(0.5 * (kl_12 + kl_21) * (temp_cons * temp_cons), mask)

        if self.lambda_kl > 0 and logits_base is not None:
            lora_shift, _, _ = shifted_logits_and_labels(logits_lora, batch["labels"])
            base_shift, _, _ = shifted_logits_and_labels(logits_base, batch["labels"])

            temp = self.temperature
            log_p_lora = F.log_softmax(lora_shift.float() / temp, dim=-1)
            p_base = F.softmax(base_shift.float() / temp, dim=-1)
            kl_tok = F.kl_div(log_p_lora, p_base, reduction="none").sum(dim=-1) * (temp * temp)
            alpha = self.lambda_kl / (ce_base_tok.detach() + self.eps)
            alpha = alpha.clamp(min=self.alpha_min, max=self.alpha_max)
            wake_kl = masked_mean(alpha * kl_tok, mask)
        else:
            alpha = torch.zeros_like(ce_lora_tok)
            wake_kl = logits_lora.new_zeros(())
        segment_loss = logits_lora.new_zeros(())
        segment_metrics = {
            "segment_loss": 0.0,
            "segment_t_mean": 0.0,
            "segment_out_ratio": 0.0,
            "segment_ab_dist": 0.0,
            "segment_memory_tokens": float(len(self.segment_bank.features)),
            "segment_reliable_ratio": 0.0,
        }
        if needs_segment_features:
            hidden = outputs_lora.hidden_states[-1][:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            valid_hidden = hidden[mask].float()
            valid_labels = shift_labels[mask]
            if valid_hidden.numel() > 0:
                output_weight = model.get_output_embeddings().weight
                anchor_a = output_weight[valid_labels].float()
                anchor_b, reliable = self.segment_bank.centroids(
                    valid_labels,
                    fallback=anchor_a,
                    min_count=self.segment_min_count,
                )
                if reliable.any():
                    segment_loss, segment_metrics = segment_distance(
                        valid_hidden[reliable],
                        anchor_a[reliable],
                        anchor_b[reliable],
                    )
                self.segment_bank.update(valid_labels, valid_hidden)
                segment_metrics["segment_loss"] = float(segment_loss.detach().cpu())
                segment_metrics["segment_memory_tokens"] = float(len(self.segment_bank.features))
                segment_metrics["segment_reliable_ratio"] = float(reliable.float().mean().detach().cpu())

        simplex_loss = logits_lora.new_zeros(())
        simplex_metrics = {
            "simplex_loss": 0.0,
            "simplex_anchor_count": 0.0,
            "simplex_weight_entropy": 0.0,
            "simplex_max_weight": 0.0,
        }
        if self.lambda_simplex > 0 and logits_base is not None and needs_segment_features:
            hidden = outputs_lora.hidden_states[-1][:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            valid_hidden = hidden[mask].float()
            valid_labels = shift_labels[mask]
            if valid_hidden.numel() > 0:
                base_shift, _, _ = shifted_logits_and_labels(logits_base, batch["labels"])
                valid_base_logits = base_shift[mask].float()
                top_k = max(1, min(self.simplex_top_k, valid_base_logits.size(-1)))
                top_ids = valid_base_logits.topk(k=top_k, dim=-1).indices
                anchor_ids = torch.cat([valid_labels.unsqueeze(1), top_ids], dim=1)
                output_weight = model.get_output_embeddings().weight
                simplex_loss, simplex_metrics = simplex_projection_loss(
                    valid_hidden,
                    anchor_ids,
                    output_weight,
                    temperature=self.simplex_temperature,
                )

        simplex_ce = logits_lora.new_zeros(())
        simplex_ce_metrics = {
            "simplex_ce_loss": 0.0,
            "simplex_ce_target_mass": 0.0,
            "simplex_ce_base_entropy": 0.0,
        }
        if self.lambda_simplex_ce > 0 and logits_base is not None:
            lora_shift, shift_labels, _ = shifted_logits_and_labels(logits_lora, batch["labels"])
            base_shift, _, _ = shifted_logits_and_labels(logits_base, batch["labels"])
            valid_lora_logits = lora_shift[mask].float()
            valid_base_logits = base_shift[mask].float()
            valid_labels = shift_labels[mask]
            if valid_lora_logits.numel() > 0:
                simplex_ce, simplex_ce_metrics = simplex_ce_loss(
                    valid_lora_logits,
                    valid_base_logits,
                    valid_labels,
                    top_k=self.simplex_top_k,
                    label_mix=self.simplex_label_mix,
                    temperature=self.simplex_temperature,
                )

        total = (
            ce_lora
            + wake_kl
            + self.lambda_segment * segment_loss
            + self.lambda_consistency * consistency_loss
            + self.lambda_simplex * simplex_loss
            + self.lambda_simplex_ce * simplex_ce
        )

        metrics = {
            "loss": float(total.detach().cpu()),
            "ce_lora": float(ce_lora.detach().cpu()),
            "wake_kl": float(wake_kl.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "ce_base": float(masked_mean(ce_base_tok, mask).detach().cpu()),
            "alpha_mean": float(masked_mean(alpha, mask).detach().cpu()),
            "ce_reuse_weight_mean": float(masked_mean(ce_weight, mask).detach().cpu()),
            "hard_ce_weight_mean": float(masked_mean(hard_ce_weight, mask).detach().cpu()),
            "ce_weight_ess_ratio": float(
                (
                    ce_weight.masked_fill(~mask, 0.0).sum().pow(2)
                    / (
                        ce_weight.masked_fill(~mask, 0.0).pow(2).sum()
                        * mask.float().sum().clamp_min(1.0)
                    ).clamp_min(1e-8)
                )
                .detach()
                .cpu()
            ),
            "token_count": float(mask.float().sum().detach().cpu()),
        }
        metrics.update(segment_metrics)
        metrics.update(simplex_metrics)
        metrics.update(simplex_ce_metrics)
        return total, metrics
