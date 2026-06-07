from __future__ import annotations

import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from tqdm.auto import tqdm

from .evaluate import model_input_device
from .modeling import load_causal_lm, load_tokenizer
from .utils import ensure_dir, parse_args_with_config, save_json, set_seed


ANSWER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)


def clean_question_text(record: dict[str, Any]) -> str:
    question = str(record.get("question", "")).strip()
    if "Answer Choices:" in question:
        question = question.split("Answer Choices:", 1)[0].strip()
    return question


def render_options(record: dict[str, Any]) -> str:
    options = record.get("options") or {}
    if isinstance(options, dict) and options:
        return "\n".join(f"({key}) {value}" for key, value in sorted(options.items()))
    return ""


def format_mcqa_prompt(
    record: dict[str, Any],
    tokenizer: Any | None = None,
    prompt_style: str = "sft",
    scoring_method: str = "letter_likelihood",
) -> str:
    question = clean_question_text(record)
    rendered = render_options(record)
    if scoring_method in {"letter_likelihood", "generate"}:
        directive = "Choose the single best answer. Return only the answer letter."
    else:
        directive = "Choose the single best answer. Return the selected option."

    if prompt_style == "chat":
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("prompt_style='chat' requires a tokenizer with apply_chat_template.")
        messages = [
            {
                "role": "system",
                "content": "You are a careful medical assistant.",
            },
            {
                "role": "user",
                "content": (
                    f"{directive}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Options:\n{rendered}"
                ),
            },
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if prompt_style != "sft":
        raise ValueError(f"Unsupported prompt style: {prompt_style}")

    return (
        "You are a careful medical assistant.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{rendered}\n\n"
        f"{directive}\nAnswer:"
    )


def parse_answer(text: str) -> str:
    match = ANSWER_RE.search(text.strip())
    return match.group(1).upper() if match else ""


def option_labels(record: dict[str, Any]) -> list[str]:
    options = record.get("options") or {}
    if isinstance(options, dict) and options:
        return sorted(str(key).strip().upper() for key in options if str(key).strip())
    return list("ABCDEFGHIJ")


def normalize_gold_label(record: dict[str, Any]) -> str:
    label = str(record.get("label", "")).strip().upper()
    if label:
        return label
    answer_idx = record.get("answer_idx")
    if isinstance(answer_idx, int) and 0 <= answer_idx < 26:
        return chr(ord("A") + answer_idx)
    return ""


def option_text(record: dict[str, Any], label: str) -> str:
    options = record.get("options") or {}
    if isinstance(options, dict):
        return str(options.get(label, "")).strip()
    return ""


@torch.no_grad()
def score_option_letters(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidates: dict[str, str],
    device: torch.device,
    length_normalize: bool = False,
) -> tuple[str, dict[str, float]]:
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
    if not prompt_ids:
        raise ValueError("Prompt tokenization produced no tokens.")

    sequences = []
    answer_spans = []
    for label, candidate_text in candidates.items():
        candidate_ids = tokenizer(
            candidate_text,
            add_special_tokens=False,
        )["input_ids"]
        if not candidate_ids:
            continue
        full_ids = prompt_ids + candidate_ids
        sequences.append((label, full_ids))
        answer_spans.append((len(prompt_ids), len(full_ids)))

    if not sequences:
        return "", {}

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    max_len = max(len(ids) for _, ids in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row_idx, (_, ids) in enumerate(sequences):
        row = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[row_idx, : row.numel()] = row
        attention_mask[row_idx, : row.numel()] = 1

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    scores: dict[str, float] = {}
    for row_idx, (label, ids) in enumerate(sequences):
        start, end = answer_spans[row_idx]
        pred_positions = torch.arange(start - 1, end - 1, device=device)
        target_tokens = input_ids[row_idx, start:end]
        log_probs = F.log_softmax(outputs.logits[row_idx, pred_positions].float(), dim=-1)
        token_scores = log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)
        score = token_scores.mean() if length_normalize else token_scores.sum()
        scores[label] = float(score.detach().cpu())

    pred = max(scores.items(), key=lambda item: item[1])[0]
    return pred, scores


def build_candidates(
    record: dict[str, Any],
    labels: list[str],
    scoring_method: str,
    candidate_prefix: str,
) -> dict[str, str]:
    if scoring_method == "letter_likelihood":
        return {label: f"{candidate_prefix}{label}" for label in labels}
    if scoring_method == "option_likelihood":
        return {
            label: f"{candidate_prefix}({label}) {option_text(record, label)}".rstrip()
            for label in labels
        }
    raise ValueError(f"Unsupported scoring method for candidate building: {scoring_method}")


@torch.no_grad()
def generate_answer(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_answer(pred_text), pred_text


@torch.no_grad()
def evaluate_mcqa(args) -> dict[str, Any]:
    set_seed(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    model = load_causal_lm(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    device = model_input_device(model)

    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    if args.max_samples and args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    correct = 0
    total = 0
    examples = []
    for record in tqdm(dataset, desc="mcqa eval"):
        prompt = format_mcqa_prompt(
            record,
            tokenizer=tokenizer,
            prompt_style=args.prompt_style,
            scoring_method=args.scoring_method,
        )
        scores = None
        pred_text = ""
        if args.scoring_method in {"letter_likelihood", "option_likelihood"}:
            labels = option_labels(record)
            candidates = build_candidates(
                record=record,
                labels=labels,
                scoring_method=args.scoring_method,
                candidate_prefix=args.candidate_prefix,
            )
            pred, scores = score_option_letters(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                candidates=candidates,
                device=device,
                length_normalize=args.length_normalize,
            )
        elif args.scoring_method == "generate":
            pred, pred_text = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            raise ValueError(f"Unsupported scoring method: {args.scoring_method}")

        gold = normalize_gold_label(record)
        total += 1
        correct += int(pred == gold)
        if len(examples) < args.save_examples:
            example = {
                "id": record.get("id"),
                "gold": gold,
                "pred": pred,
                "raw_prediction": pred_text,
            }
            if scores is not None:
                example["option_scores"] = scores
            examples.append(example)

    metrics = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "adapter_path": args.adapter_path,
        "prompt_style": args.prompt_style,
        "scoring_method": args.scoring_method,
        "candidate_prefix": args.candidate_prefix,
        "length_normalize": args.length_normalize,
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "examples": examples,
    }
    out_dir = ensure_dir(args.output_dir)
    save_json(metrics, Path(out_dir) / "mcqa_eval.json")
    return metrics


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate base or adapter on medical MCQA.")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/mcqa_eval")
    parser.add_argument("--dataset_name", type=str, default="TsinghuaC3I/MedXpertQA")
    parser.add_argument("--dataset_config", type=str, default="Text")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--save_examples", type=int, default=20)
    parser.add_argument(
        "--scoring_method",
        choices=["letter_likelihood", "option_likelihood", "generate"],
        default="letter_likelihood",
    )
    parser.add_argument("--prompt_style", choices=["sft", "chat"], default="sft")
    parser.add_argument("--candidate_prefix", type=str, default=" ")
    parser.add_argument("--length_normalize", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = parse_args_with_config(build_parser())
    if not args.model_path:
        raise ValueError("Please provide --model_path or set model_path in a JSON config.")
    metrics = evaluate_mcqa(args)
    print(f"accuracy={metrics['accuracy']:.4f} correct={metrics['correct']} total={metrics['total']}")


if __name__ == "__main__":
    main()
