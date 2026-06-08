from __future__ import annotations

import gc
import math
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from .data import DataCollatorForCausalSFT, TokenizedSFTDataset, build_train_eval_split
from .evaluate import evaluate_nll, model_input_device, move_batch
from .losses import WakeLoraLoss
from .modeling import prepare_model_and_tokenizer
from .utils import (
    append_jsonl,
    count_trainable_parameters,
    ensure_dir,
    save_json,
    set_seed,
    setup_logger,
)


def add_common_training_args(parser: ArgumentParser) -> None:
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="FreedomIntelligence/medical-o1-reasoning-SFT")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--question_column", type=str, default=None)
    parser.add_argument("--answer_column", type=str, default=None)
    parser.add_argument("--reasoning_column", type=str, default=None)
    parser.add_argument("--no_reasoning", action="store_true")
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=512)
    parser.add_argument("--max_eval_samples", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="outputs/debug")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_use_rslora", action="store_true")
    parser.add_argument("--lora_use_dora", action="store_true")
    parser.add_argument("--lora_init", type=str, default="default")
    parser.add_argument("--lambda_kl", type=float, default=0.1)
    parser.add_argument("--lambda_ce_reuse", type=float, default=0.0)
    parser.add_argument("--lambda_self_reuse", type=float, default=0.0)
    parser.add_argument("--lambda_consistency", type=float, default=0.0)
    parser.add_argument("--lambda_segment", type=float, default=0.0)
    parser.add_argument("--lambda_simplex", type=float, default=0.0)
    parser.add_argument("--lambda_simplex_ce", type=float, default=0.0)
    parser.add_argument("--hard_ce_threshold", type=float, default=0.0)
    parser.add_argument("--hard_ce_min_weight", type=float, default=0.1)
    parser.add_argument("--segment_memory_size", type=int, default=4)
    parser.add_argument("--segment_min_count", type=int, default=0)
    parser.add_argument("--simplex_top_k", type=int, default=8)
    parser.add_argument("--simplex_label_mix", type=float, default=0.2)
    parser.add_argument("--alpha_min", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=2.0)
    parser.add_argument("--ce_reuse_max", type=float, default=2.0)
    parser.add_argument("--self_reuse_max", type=float, default=4.0)
    parser.add_argument("--wake_temperature", type=float, default=1.0)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    parser.add_argument("--simplex_temperature", type=float, default=0.2)
    parser.add_argument(
        "--wake_start_ratio",
        type=float,
        default=0.0,
        help="Fraction of optimizer updates to run before enabling Wake KL/segment regularizers.",
    )
    parser.add_argument(
        "--wake_ramp_ratio",
        type=float,
        default=0.0,
        help="Fraction of optimizer updates used to linearly ramp Wake regularizers after wake_start_ratio.",
    )
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_max_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)


def build_dataloaders(args: Namespace, tokenizer: Any) -> tuple[DataLoader, DataLoader]:
    train_raw, eval_raw = build_train_eval_split(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    train_ds = TokenizedSFTDataset(
        train_raw,
        tokenizer=tokenizer,
        max_length=args.max_length,
        question_column=args.question_column,
        answer_column=args.answer_column,
        reasoning_column=args.reasoning_column,
        include_reasoning=not args.no_reasoning,
    )
    eval_ds = TokenizedSFTDataset(
        eval_raw,
        tokenizer=tokenizer,
        max_length=args.max_length,
        question_column=args.question_column,
        answer_column=args.answer_column,
        reasoning_column=args.reasoning_column,
        include_reasoning=not args.no_reasoning,
    )
    collator = DataCollatorForCausalSFT(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, eval_loader


def run_base_eval(args: Namespace) -> dict[str, float]:
    out_dir = ensure_dir(Path(args.output_dir))
    logger = setup_logger(out_dir)
    set_seed(args.seed)
    logger.info("Loading base model for direct evaluation.")
    model, tokenizer, stats = prepare_model_and_tokenizer(
        model_path=args.model_path,
        use_lora=False,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    _, eval_loader = build_dataloaders(args, tokenizer)
    metrics = evaluate_nll(model, eval_loader, max_batches=args.eval_max_batches, desc="base eval")
    metrics.update({"method": "base", "parameter_stats": stats})
    save_json(metrics, out_dir / "base_eval.json")
    logger.info("Base eval: nll=%.4f ppl=%.2f", metrics["nll"], metrics["perplexity"])
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _standard_lora_loss(model: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(**batch)
    loss = outputs.loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "ce_lora": float(loss.detach().cpu()),
        "wake_kl": 0.0,
        "ce_base": 0.0,
        "alpha_mean": 0.0,
    }


def normalize_lora_init(value: str) -> str | bool:
    if value == "default":
        return True
    if value.lower() in {"false", "none"}:
        return False
    return value


@torch.no_grad()
def trainable_parameter_stats(model: Any) -> dict[str, float]:
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        value = param.detach().float()
        sq_sum += float((value * value).sum().cpu())
        abs_sum += float(value.abs().sum().cpu())
        count += value.numel()
    denom = max(count, 1)
    return {
        "trainable_param_count": float(count),
        "trainable_param_norm": math.sqrt(sq_sum),
        "trainable_param_rms": math.sqrt(sq_sum / denom),
        "trainable_param_abs_mean": abs_sum / denom,
    }


def wake_regularizer_multiplier(
    global_step: int,
    total_updates: int,
    start_ratio: float,
    ramp_ratio: float,
) -> float:
    if start_ratio <= 0.0 and ramp_ratio <= 0.0:
        return 1.0
    progress = global_step / max(total_updates, 1)
    if progress < start_ratio:
        return 0.0
    if ramp_ratio <= 0.0:
        return 1.0
    return min(1.0, max(0.0, (progress - start_ratio) / ramp_ratio))


def train_lora_method(args: Namespace, method: str) -> dict[str, Any]:
    if method not in {"standard_lora", "wake_lora"}:
        raise ValueError(f"Unsupported method: {method}")

    out_dir = ensure_dir(Path(args.output_dir) / method)
    logger = setup_logger(out_dir)
    set_seed(args.seed)
    logger.info("Loading model for %s.", method)

    model, tokenizer, stats = prepare_model_and_tokenizer(
        model_path=args.model_path,
        use_lora=True,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_use_rslora=args.lora_use_rslora,
        lora_use_dora=args.lora_use_dora,
        lora_init=normalize_lora_init(args.lora_init),
    )
    logger.info("Parameter stats: %s", count_trainable_parameters(model))
    train_loader, eval_loader = build_dataloaders(args, tokenizer)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    wake_loss = WakeLoraLoss(
        lambda_kl=args.lambda_kl,
        lambda_ce_reuse=args.lambda_ce_reuse,
        lambda_self_reuse=args.lambda_self_reuse,
        lambda_consistency=args.lambda_consistency,
        lambda_segment=args.lambda_segment,
        lambda_simplex=args.lambda_simplex,
        lambda_simplex_ce=args.lambda_simplex_ce,
        hard_ce_threshold=args.hard_ce_threshold,
        hard_ce_min_weight=args.hard_ce_min_weight,
        segment_memory_size=args.segment_memory_size,
        segment_min_count=args.segment_min_count,
        simplex_top_k=args.simplex_top_k,
        simplex_label_mix=args.simplex_label_mix,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        ce_reuse_max=args.ce_reuse_max,
        self_reuse_max=args.self_reuse_max,
        temperature=args.wake_temperature,
        consistency_temperature=args.consistency_temperature,
        simplex_temperature=args.simplex_temperature,
        collect_segment_features=args.lambda_segment > 0 and args.wake_start_ratio > 0,
    )
    base_lambda_kl = args.lambda_kl
    base_lambda_self_reuse = args.lambda_self_reuse
    base_lambda_consistency = args.lambda_consistency
    base_lambda_segment = args.lambda_segment
    base_lambda_simplex = args.lambda_simplex
    base_lambda_simplex_ce = args.lambda_simplex_ce
    device = model_input_device(model)
    best_nll = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running: dict[str, float] = {}
        progress = tqdm(train_loader, desc=f"{method} epoch {epoch + 1}/{args.epochs}")
        for micro_step, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            if method == "wake_lora":
                wake_multiplier = wake_regularizer_multiplier(
                    global_step=global_step,
                    total_updates=total_updates,
                    start_ratio=args.wake_start_ratio,
                    ramp_ratio=args.wake_ramp_ratio,
                )
                wake_loss.lambda_kl = base_lambda_kl * wake_multiplier
                wake_loss.lambda_self_reuse = base_lambda_self_reuse * wake_multiplier
                wake_loss.lambda_consistency = base_lambda_consistency * wake_multiplier
                wake_loss.lambda_segment = base_lambda_segment * wake_multiplier
                wake_loss.lambda_simplex = base_lambda_simplex * wake_multiplier
                wake_loss.lambda_simplex_ce = base_lambda_simplex_ce * wake_multiplier
                loss, metrics = wake_loss(model, batch)
                metrics["wake_multiplier"] = wake_multiplier
            else:
                loss, metrics = _standard_lora_loss(model, batch)

            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + float(value)

            should_step = micro_step % args.gradient_accumulation_steps == 0
            should_step = should_step or micro_step == len(train_loader)
            if should_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_metrics = {
                    key: value / max(micro_step, 1)
                    for key, value in running.items()
                }
                avg_metrics.update(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm.detach().cpu()),
                    }
                )
                avg_metrics.update(trainable_parameter_stats(model))
                append_jsonl(avg_metrics, out_dir / "train_metrics.jsonl")
                progress.set_postfix(loss=f"{avg_metrics.get('loss', 0.0):.4f}")

        eval_metrics = evaluate_nll(
            model,
            eval_loader,
            max_batches=args.eval_max_batches,
            desc=f"{method} eval",
        )
        eval_metrics.update({"epoch": epoch + 1, "global_step": global_step, "method": method})
        append_jsonl(eval_metrics, out_dir / "eval_metrics.jsonl")
        logger.info(
            "%s epoch %d eval: nll=%.4f ppl=%.2f",
            method,
            epoch + 1,
            eval_metrics["nll"],
            eval_metrics["perplexity"],
        )
        if eval_metrics["nll"] < best_nll:
            best_nll = eval_metrics["nll"]
            model.save_pretrained(out_dir / "adapter_best")

    final_metrics = evaluate_nll(
        model,
        eval_loader,
        max_batches=args.eval_max_batches,
        desc=f"{method} final eval",
    )
    final_metrics.update(
        {
            "method": method,
            "global_step": global_step,
            "parameter_stats": stats,
            "adapter_dir": str(out_dir / "adapter_final"),
        }
    )
    model.save_pretrained(out_dir / "adapter_final")
    save_json(final_metrics, out_dir / "eval_final.json")
    logger.info("%s final: nll=%.4f ppl=%.2f", method, final_metrics["nll"], final_metrics["perplexity"])

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_metrics
