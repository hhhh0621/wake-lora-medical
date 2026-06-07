from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


METHODS = {
    "standard": {
        "skip": ["--skip_wake"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
    },
    "wake_kl": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.1,
        "lambda_segment": 0.0,
    },
    "wake_segment": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.005,
    },
    "wake_kl_segment": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.1,
        "lambda_segment": 0.005,
    },
    "wake_adaptive": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "adaptive",
        "lambda_segment": 0.005,
    },
    "wake_scheduled": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "scheduled",
        "lambda_segment": "scheduled",
    },
    "wake_budget": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "budget",
        "lambda_segment": "budget",
    },
    "wake_delayed": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "budget",
        "lambda_segment": "budget",
        "wake_start_ratio": 0.25,
        "wake_ramp_ratio": 0.125,
    },
    "wake_reliable": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "scheduled",
        "lambda_segment": "scheduled",
        "segment_min_count": 2,
    },
}


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_method_list(text: str) -> list[str]:
    methods = [x.strip() for x in text.split(",") if x.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Choices: {sorted(METHODS)}")
    return methods


def adaptive_kl(sample_count: int, base_lambda: float, ref_samples: int, power: float) -> float:
    if sample_count <= ref_samples:
        return base_lambda
    return base_lambda * (ref_samples / sample_count) ** power


def scheduled_kl(sample_count: int, base_lambda: float, cutoff_samples: int) -> float:
    if sample_count <= cutoff_samples:
        return base_lambda
    return 0.0


def scheduled_segment(sample_count: int, base_lambda: float, ref_samples: int, power: float) -> float:
    if sample_count <= ref_samples:
        return base_lambda
    return base_lambda * (ref_samples / sample_count) ** power


def output_dir_name(
    prefix: str,
    samples: int,
    seed: int,
    method: str,
    lambda_kl: float,
    lambda_segment: float,
    wake_start_ratio: float = 0.0,
    wake_ramp_ratio: float = 0.0,
) -> str:
    kl_tag = f"kl{lambda_kl:.5g}".replace(".", "p")
    seg_tag = f"seg{lambda_segment:.5g}".replace(".", "p")
    name = f"{prefix}_n{samples}_s{seed}_{method}_{kl_tag}_{seg_tag}"
    if wake_start_ratio > 0 or wake_ramp_ratio > 0:
        start_tag = f"ws{wake_start_ratio:.5g}".replace(".", "p")
        ramp_tag = f"wr{wake_ramp_ratio:.5g}".replace(".", "p")
        name = f"{name}_{start_tag}_{ramp_tag}"
    return name


def resolve_epochs(args: argparse.Namespace, samples: int) -> int:
    if args.target_updates is None or args.target_updates <= 0:
        return args.epochs
    updates_per_epoch = max(1, (samples + args.batch_size - 1) // args.batch_size)
    updates_per_epoch = max(1, (updates_per_epoch + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
    return max(1, (args.target_updates + updates_per_epoch - 1) // updates_per_epoch)


def summary_exists(output_dir: Path) -> bool:
    path = output_dir / "summary.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("results"))


def build_command(args: argparse.Namespace, samples: int, seed: int, method: str) -> tuple[list[str], Path]:
    spec = METHODS[method]
    lambda_kl = spec["lambda_kl"]
    if lambda_kl == "adaptive":
        lambda_kl = adaptive_kl(
            sample_count=samples,
            base_lambda=args.adaptive_base_kl,
            ref_samples=args.adaptive_ref_samples,
            power=args.adaptive_power,
        )
    elif lambda_kl == "scheduled":
        lambda_kl = scheduled_kl(
            sample_count=samples,
            base_lambda=args.scheduled_base_kl,
            cutoff_samples=args.scheduled_kl_cutoff_samples,
        )
    elif lambda_kl == "budget":
        lambda_kl = scheduled_kl(
            sample_count=samples,
            base_lambda=args.budget_base_kl,
            cutoff_samples=args.budget_kl_cutoff_samples,
        )
    lambda_segment = spec["lambda_segment"]
    if lambda_segment == "scheduled":
        lambda_segment = scheduled_segment(
            sample_count=samples,
            base_lambda=args.scheduled_base_segment,
            ref_samples=args.scheduled_segment_ref_samples,
            power=args.scheduled_segment_power,
        )
    elif lambda_segment == "budget":
        lambda_segment = scheduled_segment(
            sample_count=samples,
            base_lambda=args.budget_base_segment,
            ref_samples=args.budget_segment_ref_samples,
            power=args.budget_segment_power,
        )
    lambda_segment = float(lambda_segment)
    segment_min_count = int(spec.get("segment_min_count", args.segment_min_count))
    wake_start_ratio = float(spec.get("wake_start_ratio", args.wake_start_ratio))
    wake_ramp_ratio = float(spec.get("wake_ramp_ratio", args.wake_ramp_ratio))
    epochs = resolve_epochs(args, samples)
    out_dir = ROOT / "outputs" / output_dir_name(
        args.output_prefix,
        samples,
        seed,
        method,
        float(lambda_kl),
        lambda_segment,
        wake_start_ratio,
        wake_ramp_ratio,
    )
    cmd = [
        args.python_bin,
        "-m",
        "wake_lora.run_three_way",
        "--config",
        args.config,
        "--output_dir",
        str(out_dir.relative_to(ROOT)),
        "--max_train_samples",
        str(samples),
        "--max_eval_samples",
        str(args.max_eval_samples),
        "--max_length",
        str(args.max_length),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--eval_max_batches",
        str(args.eval_max_batches),
        "--num_workers",
        str(args.num_workers),
        "--lambda_kl",
        str(float(lambda_kl)),
        "--lambda_ce_reuse",
        "0.0",
        "--lambda_segment",
        str(lambda_segment),
        "--segment_memory_size",
        str(args.segment_memory_size),
        "--segment_min_count",
        str(segment_min_count),
        "--wake_start_ratio",
        str(wake_start_ratio),
        "--wake_ramp_ratio",
        str(wake_ramp_ratio),
        "--seed",
        str(seed),
        *spec["skip"],
    ]
    if args.learning_rate is not None:
        cmd.extend(["--learning_rate", str(args.learning_rate)])
    return cmd, out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a low-data Wake-LoRA experiment matrix.")
    parser.add_argument("--config", default="configs/qwen_medical_o1_small.json")
    parser.add_argument("--python_bin", default="/opt/conda/bin/python")
    parser.add_argument("--output_prefix", default="matrix_medical_o1")
    parser.add_argument("--samples", default="8,16,32,64,128")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--methods", default="standard,wake_segment,wake_kl_segment,wake_adaptive")
    parser.add_argument("--max_eval_samples", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--target_updates",
        type=int,
        default=None,
        help="If set, choose epochs per sample count to reach at least this many optimizer updates.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--eval_max_batches", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--segment_memory_size", type=int, default=4)
    parser.add_argument("--segment_min_count", type=int, default=0)
    parser.add_argument("--adaptive_base_kl", type=float, default=0.1)
    parser.add_argument("--adaptive_ref_samples", type=int, default=32)
    parser.add_argument("--adaptive_power", type=float, default=2.0)
    parser.add_argument("--scheduled_base_kl", type=float, default=0.1)
    parser.add_argument("--scheduled_kl_cutoff_samples", type=int, default=32)
    parser.add_argument("--scheduled_base_segment", type=float, default=0.005)
    parser.add_argument("--scheduled_segment_ref_samples", type=int, default=64)
    parser.add_argument("--scheduled_segment_power", type=float, default=2.0)
    parser.add_argument("--budget_base_kl", type=float, default=0.1)
    parser.add_argument("--budget_kl_cutoff_samples", type=int, default=16)
    parser.add_argument("--budget_base_segment", type=float, default=0.005)
    parser.add_argument("--budget_segment_ref_samples", type=int, default=16)
    parser.add_argument("--budget_segment_power", type=float, default=2.0)
    parser.add_argument("--wake_start_ratio", type=float, default=0.0)
    parser.add_argument("--wake_ramp_ratio", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    samples = parse_int_list(args.samples)
    seeds = parse_int_list(args.seeds)
    methods = parse_method_list(args.methods)

    for sample_count in samples:
        for seed in seeds:
            for method in methods:
                cmd, out_dir = build_command(args, sample_count, seed, method)
                if summary_exists(out_dir) and not args.force:
                    print(f"[skip] {out_dir.relative_to(ROOT)}")
                    continue
                print("[run]", " ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
