from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from .train import add_common_training_args, run_base_eval, train_lora_method
from .utils import ensure_dir, parse_args_with_config, save_json


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run Base vs LoRA vs Wake-LoRA on medical SFT.")
    add_common_training_args(parser)
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip direct base model evaluation.",
    )
    parser.add_argument(
        "--skip_standard",
        action="store_true",
        help="Skip standard LoRA training.",
    )
    parser.add_argument(
        "--skip_wake",
        action="store_true",
        help="Skip Wake-LoRA training.",
    )
    return parser


def main() -> None:
    args = parse_args_with_config(build_parser())
    if not args.model_path:
        raise ValueError("Please provide --model_path or set model_path in the JSON config.")
    out_dir = ensure_dir(args.output_dir)
    summary = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_length": args.max_length,
        "lambda_kl": args.lambda_kl,
        "lambda_ce_reuse": args.lambda_ce_reuse,
        "lambda_segment": args.lambda_segment,
        "segment_memory_size": args.segment_memory_size,
        "segment_min_count": args.segment_min_count,
        "wake_start_ratio": args.wake_start_ratio,
        "wake_ramp_ratio": args.wake_ramp_ratio,
        "results": {},
    }

    if not args.skip_base:
        summary["results"]["base"] = run_base_eval(args)
    if not args.skip_standard:
        summary["results"]["standard_lora"] = train_lora_method(args, "standard_lora")
    if not args.skip_wake:
        summary["results"]["wake_lora"] = train_lora_method(args, "wake_lora")

    rankings = []
    for method, metrics in summary["results"].items():
        if "nll" in metrics:
            rankings.append((metrics["nll"], method))
    rankings.sort()
    summary["ranking_by_nll"] = [{"method": method, "nll": nll} for nll, method in rankings]
    save_json(summary, Path(out_dir) / "summary.json")


if __name__ == "__main__":
    main()
