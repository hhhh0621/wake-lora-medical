from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import Dataset as TorchDataset


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    target: str


SYSTEM_PROMPT = (
    "You are a careful medical assistant. Answer the medical question "
    "accurately and avoid unsupported claims."
)


def _first_text(record: dict[str, Any], names: list[str]) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def format_medical_record(
    record: dict[str, Any],
    question_column: str | None = None,
    answer_column: str | None = None,
    reasoning_column: str | None = None,
    include_reasoning: bool = True,
) -> SFTExample:
    question_names = [question_column] if question_column else []
    question_names += ["Question", "question", "instruction", "input", "query", "prompt"]
    answer_names = [answer_column] if answer_column else []
    answer_names += ["Response", "response", "answer", "output", "chosen"]
    reasoning_names = [reasoning_column] if reasoning_column else []
    reasoning_names += ["Complex_CoT", "cot", "reasoning", "rationale", "chain_of_thought"]

    question = _first_text(record, question_names)
    answer = _first_text(record, answer_names)
    reasoning = _first_text(record, reasoning_names)

    if not question:
        question = str(record)
    if include_reasoning and reasoning:
        target = f"<reasoning>\n{reasoning}\n</reasoning>\n\n{answer}".strip()
    else:
        target = answer.strip()

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )
    return SFTExample(prompt=prompt, target=target)


def load_raw_sft_dataset(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
) -> Dataset:
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    return load_dataset(dataset_name, split=split)


def build_train_eval_split(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    validation_fraction: float,
    seed: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> tuple[Dataset, Dataset]:
    raw = load_raw_sft_dataset(dataset_name, dataset_config, dataset_split)
    raw = raw.shuffle(seed=seed)
    split = raw.train_test_split(test_size=validation_fraction, seed=seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    if max_train_samples is not None and max_train_samples > 0:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
    if max_eval_samples is not None and max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(max_eval_samples, len(eval_ds))))
    return train_ds, eval_ds


class TokenizedSFTDataset(TorchDataset):
    def __init__(
        self,
        raw_dataset: Dataset,
        tokenizer: Any,
        max_length: int,
        question_column: str | None = None,
        answer_column: str | None = None,
        reasoning_column: str | None = None,
        include_reasoning: bool = True,
    ) -> None:
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.question_column = question_column
        self.answer_column = answer_column
        self.reasoning_column = reasoning_column
        self.include_reasoning = include_reasoning

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.raw_dataset[int(idx)]
        example = format_medical_record(
            record,
            question_column=self.question_column,
            answer_column=self.answer_column,
            reasoning_column=self.reasoning_column,
            include_reasoning=self.include_reasoning,
        )
        prompt_ids = self.tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
        target_ids = self.tokenizer(example.target, add_special_tokens=False)["input_ids"]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            target_ids = target_ids + [eos]

        if len(target_ids) >= self.max_length:
            target_ids = target_ids[: self.max_length]
            prompt_ids = []
        else:
            prompt_budget = self.max_length - len(target_ids)
            prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DataCollatorForCausalSFT:
    def __init__(self, pad_token_id: int, label_pad_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(x["input_ids"].numel() for x in features)
        batch: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for item in features:
            pad = max_len - item["input_ids"].numel()
            batch["input_ids"].append(
                torch.nn.functional.pad(item["input_ids"], (0, pad), value=self.pad_token_id)
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0)
            )
            batch["labels"].append(
                torch.nn.functional.pad(item["labels"], (0, pad), value=self.label_pad_id)
            )
        return {key: torch.stack(value, dim=0) for key, value in batch.items()}
