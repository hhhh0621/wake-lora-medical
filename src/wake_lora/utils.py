from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logger(output_dir: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger("wake_lora")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if output_dir is not None:
        ensure_dir(output_dir)
        file_handler = logging.FileHandler(Path(output_dir) / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def count_trainable_parameters(model: torch.nn.Module) -> dict[str, Any]:
    trainable = 0
    total = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def resolve_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def perplexity_from_nll(nll: float) -> float:
    if not math.isfinite(nll):
        return float("inf")
    return float(math.exp(min(nll, 50.0)))


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    add_config_arg(parser)
    pre_args, remaining = parser.parse_known_args()
    if pre_args.config:
        data = json.loads(Path(pre_args.config).read_text(encoding="utf-8"))
        parser.set_defaults(**data)
    args = parser.parse_args()
    if args.config:
        args.config = os.path.abspath(args.config)
    return args
