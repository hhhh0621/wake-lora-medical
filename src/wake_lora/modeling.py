from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import count_trainable_parameters, resolve_dtype


QWEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_tokenizer(model_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_causal_lm(
    model_path: str,
    dtype: str = "bf16",
    device_map: str = "auto",
    gradient_checkpointing: bool = True,
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(dtype),
        device_map=device_map,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
    return model


def attach_lora(
    model: Any,
    r: int,
    alpha: int,
    dropout: float,
    use_rslora: bool = False,
    use_dora: bool = False,
    init_lora_weights: str | bool = True,
    target_modules: list[str] | None = None,
) -> Any:
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules or QWEN_LORA_TARGETS,
        use_rslora=use_rslora,
        use_dora=use_dora,
        init_lora_weights=init_lora_weights,
    )
    model = get_peft_model(model, config)
    return model


def prepare_model_and_tokenizer(
    model_path: str,
    use_lora: bool,
    dtype: str,
    device_map: str,
    gradient_checkpointing: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_use_rslora: bool = False,
    lora_use_dora: bool = False,
    lora_init: str | bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    tokenizer = load_tokenizer(model_path)
    model = load_causal_lm(
        model_path=model_path,
        dtype=dtype,
        device_map=device_map,
        gradient_checkpointing=gradient_checkpointing,
    )
    if use_lora:
        model = attach_lora(
            model,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
            use_rslora=lora_use_rslora,
            use_dora=lora_use_dora,
            init_lora_weights=lora_init,
        )
    else:
        for param in model.parameters():
            param.requires_grad_(False)
    stats = count_trainable_parameters(model)
    return model, tokenizer, stats
