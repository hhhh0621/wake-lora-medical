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
    "wake_gentle": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "gentle",
        "lambda_segment": "gentle",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_gentle_self_reuse": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "gentle",
        "lambda_segment": "gentle",
        "lambda_self_reuse": "self_reuse",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_gentle_consistency": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "gentle",
        "lambda_segment": "gentle",
        "lambda_consistency": "consistency",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_gentle_reuse_consistency": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "gentle",
        "lambda_segment": "gentle",
        "lambda_self_reuse": "self_reuse",
        "lambda_consistency": "consistency",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_utilization": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "utilization",
        "lambda_segment": "utilization",
        "lambda_self_reuse": "utilization",
        "lambda_consistency": "utilization",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_simplex": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
        "lambda_simplex": "simplex",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_simplex_ce": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
        "lambda_simplex_ce": "simplex_ce",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_utilization_simplex": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "utilization",
        "lambda_segment": "utilization",
        "lambda_self_reuse": "utilization",
        "lambda_consistency": "utilization",
        "lambda_simplex": "utilization",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_utilization_simplex_ce": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": "utilization",
        "lambda_segment": "utilization",
        "lambda_self_reuse": "utilization",
        "lambda_consistency": "utilization",
        "lambda_simplex_ce": "utilization",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_self_reuse": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
        "lambda_self_reuse": "self_reuse",
    },
    "wake_self_reuse_delayed": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
        "lambda_self_reuse": "self_reuse",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_self_reuse_segment": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": "gentle",
        "lambda_self_reuse": "self_reuse",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_consistency": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
        "lambda_consistency": "consistency",
    },
    "wake_consistency_delayed": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
        "lambda_consistency": "consistency",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_consistency_segment": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": "gentle",
        "lambda_consistency": "consistency",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_reuse_consistency": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
        "lambda_self_reuse": "self_reuse",
        "lambda_consistency": "consistency",
    },
    "wake_reuse_consistency_delayed": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
        "lambda_self_reuse": "self_reuse",
        "lambda_consistency": "consistency",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_reuse_consistency_segment": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": "gentle",
        "lambda_self_reuse": "self_reuse",
        "lambda_consistency": "consistency",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
    },
    "wake_hard_cap": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": 0.0,
        "hard_ce_threshold": "hard_cap",
    },
    "wake_hard_cap_segment": {
        "skip": ["--skip_base", "--skip_standard"],
        "lambda_kl": 0.0,
        "lambda_segment": "gentle",
        "hard_ce_threshold": "hard_cap",
        "wake_start_ratio": "gentle",
        "wake_ramp_ratio": "gentle",
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


def gentle_kl(sample_count: int, base_lambda: float, max_samples: int) -> float:
    if sample_count <= max_samples:
        return base_lambda
    return 0.0


def gentle_segment(
    sample_count: int,
    extreme_lambda: float,
    mid_lambda: float,
    large_ref_lambda: float,
    mid_max_samples: int,
    large_ref_samples: int,
    power: float,
) -> float:
    if sample_count <= 8:
        return extreme_lambda
    if sample_count <= mid_max_samples:
        return mid_lambda
    if sample_count <= large_ref_samples:
        return large_ref_lambda
    return large_ref_lambda * (large_ref_samples / sample_count) ** power


def gentle_wake_ratio(sample_count: int, ratio: float, max_samples: int) -> float:
    if sample_count <= max_samples:
        return ratio
    return 0.0


def gated_value(sample_count: int, value: float, max_samples: int) -> float:
    if sample_count <= max_samples:
        return value
    return 0.0


def output_dir_name(
    prefix: str,
    samples: int,
    seed: int,
    method: str,
    lambda_kl: float,
    lambda_segment: float,
    lambda_self_reuse: float = 0.0,
    lambda_consistency: float = 0.0,
    lambda_simplex: float = 0.0,
    lambda_simplex_ce: float = 0.0,
    hard_ce_threshold: float = 0.0,
    wake_start_ratio: float = 0.0,
    wake_ramp_ratio: float = 0.0,
) -> str:
    kl_tag = f"kl{lambda_kl:.5g}".replace(".", "p")
    seg_tag = f"seg{lambda_segment:.5g}".replace(".", "p")
    name = f"{prefix}_n{samples}_s{seed}_{method}_{kl_tag}_{seg_tag}"
    if lambda_self_reuse > 0:
        self_tag = f"sr{lambda_self_reuse:.5g}".replace(".", "p")
        name = f"{name}_{self_tag}"
    if lambda_consistency > 0:
        cons_tag = f"cons{lambda_consistency:.5g}".replace(".", "p")
        name = f"{name}_{cons_tag}"
    if lambda_simplex > 0:
        simplex_tag = f"simp{lambda_simplex:.5g}".replace(".", "p")
        name = f"{name}_{simplex_tag}"
    if lambda_simplex_ce > 0:
        simplex_ce_tag = f"sce{lambda_simplex_ce:.5g}".replace(".", "p")
        name = f"{name}_{simplex_ce_tag}"
    if hard_ce_threshold > 0:
        hard_tag = f"hct{hard_ce_threshold:.5g}".replace(".", "p")
        name = f"{name}_{hard_tag}"
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
    elif lambda_kl == "gentle":
        lambda_kl = gentle_kl(
            sample_count=samples,
            base_lambda=args.gentle_base_kl,
            max_samples=args.gentle_kl_max_samples,
        )
    elif lambda_kl == "utilization":
        lambda_kl = gated_value(
            sample_count=samples,
            value=args.gentle_base_kl,
            max_samples=args.utilization_kl_max_samples,
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
    elif lambda_segment == "gentle":
        lambda_segment = gentle_segment(
            sample_count=samples,
            extreme_lambda=args.gentle_extreme_segment,
            mid_lambda=args.gentle_mid_segment,
            large_ref_lambda=args.gentle_large_ref_segment,
            mid_max_samples=args.gentle_mid_max_samples,
            large_ref_samples=args.gentle_large_ref_samples,
            power=args.gentle_segment_power,
        )
    elif lambda_segment == "utilization":
        lambda_segment = gated_value(
            sample_count=samples,
            value=args.gentle_extreme_segment,
            max_samples=args.utilization_segment_max_samples,
        )
    lambda_segment = float(lambda_segment)
    lambda_self_reuse = spec.get("lambda_self_reuse", 0.0)
    if lambda_self_reuse == "self_reuse":
        lambda_self_reuse = args.self_reuse_lambda
    elif lambda_self_reuse == "utilization":
        lambda_self_reuse = gated_value(
            sample_count=samples,
            value=args.self_reuse_lambda,
            max_samples=args.utilization_self_reuse_max_samples,
        )
    lambda_self_reuse = float(lambda_self_reuse)
    lambda_consistency = spec.get("lambda_consistency", 0.0)
    if lambda_consistency == "consistency":
        lambda_consistency = args.consistency_lambda
    elif lambda_consistency == "utilization":
        lambda_consistency = gated_value(
            sample_count=samples,
            value=args.consistency_lambda,
            max_samples=args.utilization_consistency_max_samples,
        )
    lambda_consistency = float(lambda_consistency)
    lambda_simplex = spec.get("lambda_simplex", 0.0)
    if lambda_simplex == "simplex":
        lambda_simplex = args.simplex_lambda
    elif lambda_simplex == "utilization":
        lambda_simplex = gated_value(
            sample_count=samples,
            value=args.simplex_lambda,
            max_samples=args.utilization_simplex_max_samples,
        )
    lambda_simplex = float(lambda_simplex)
    lambda_simplex_ce = spec.get("lambda_simplex_ce", 0.0)
    if lambda_simplex_ce == "simplex_ce":
        lambda_simplex_ce = args.simplex_ce_lambda
    elif lambda_simplex_ce == "utilization":
        lambda_simplex_ce = gated_value(
            sample_count=samples,
            value=args.simplex_ce_lambda,
            max_samples=args.utilization_simplex_ce_max_samples,
        )
    lambda_simplex_ce = float(lambda_simplex_ce)
    hard_ce_threshold = spec.get("hard_ce_threshold", 0.0)
    if hard_ce_threshold == "hard_cap":
        hard_ce_threshold = args.hard_ce_threshold
    hard_ce_threshold = float(hard_ce_threshold)
    segment_min_count = int(spec.get("segment_min_count", args.segment_min_count))
    wake_start_ratio = spec.get("wake_start_ratio", args.wake_start_ratio)
    if wake_start_ratio == "gentle":
        wake_start_ratio = gentle_wake_ratio(
            sample_count=samples,
            ratio=args.gentle_wake_start_ratio,
            max_samples=args.gentle_delay_max_samples,
        )
    wake_ramp_ratio = spec.get("wake_ramp_ratio", args.wake_ramp_ratio)
    if wake_ramp_ratio == "gentle":
        wake_ramp_ratio = gentle_wake_ratio(
            sample_count=samples,
            ratio=args.gentle_wake_ramp_ratio,
            max_samples=args.gentle_delay_max_samples,
        )
    wake_start_ratio = float(wake_start_ratio)
    wake_ramp_ratio = float(wake_ramp_ratio)
    epochs = resolve_epochs(args, samples)
    out_dir = ROOT / "outputs" / output_dir_name(
        args.output_prefix,
        samples,
        seed,
        method,
        float(lambda_kl),
        lambda_segment,
        lambda_self_reuse,
        lambda_consistency,
        lambda_simplex,
        lambda_simplex_ce,
        hard_ce_threshold,
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
        "--lambda_self_reuse",
        str(lambda_self_reuse),
        "--lambda_consistency",
        str(lambda_consistency),
        "--lambda_segment",
        str(lambda_segment),
        "--lambda_simplex",
        str(lambda_simplex),
        "--lambda_simplex_ce",
        str(lambda_simplex_ce),
        "--hard_ce_threshold",
        str(hard_ce_threshold),
        "--hard_ce_min_weight",
        str(args.hard_ce_min_weight),
        "--segment_memory_size",
        str(args.segment_memory_size),
        "--segment_min_count",
        str(segment_min_count),
        "--simplex_top_k",
        str(args.simplex_top_k),
        "--simplex_label_mix",
        str(args.simplex_label_mix),
        "--self_reuse_max",
        str(args.self_reuse_max),
        "--consistency_temperature",
        str(args.consistency_temperature),
        "--simplex_temperature",
        str(args.simplex_temperature),
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
    parser.add_argument("--self_reuse_lambda", type=float, default=0.025)
    parser.add_argument("--self_reuse_max", type=float, default=4.0)
    parser.add_argument("--consistency_lambda", type=float, default=0.5)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    parser.add_argument("--simplex_lambda", type=float, default=0.01)
    parser.add_argument("--simplex_ce_lambda", type=float, default=0.05)
    parser.add_argument("--simplex_top_k", type=int, default=8)
    parser.add_argument("--simplex_label_mix", type=float, default=0.2)
    parser.add_argument("--simplex_temperature", type=float, default=0.2)
    parser.add_argument("--hard_ce_threshold", type=float, default=1.5)
    parser.add_argument("--hard_ce_min_weight", type=float, default=0.1)
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
    parser.add_argument("--gentle_base_kl", type=float, default=0.1)
    parser.add_argument("--gentle_kl_max_samples", type=int, default=8)
    parser.add_argument("--gentle_extreme_segment", type=float, default=0.005)
    parser.add_argument("--gentle_mid_segment", type=float, default=0.000625)
    parser.add_argument("--gentle_mid_max_samples", type=int, default=16)
    parser.add_argument("--gentle_large_ref_segment", type=float, default=0.0)
    parser.add_argument("--gentle_large_ref_samples", type=int, default=32)
    parser.add_argument("--gentle_segment_power", type=float, default=2.0)
    parser.add_argument("--gentle_delay_max_samples", type=int, default=8)
    parser.add_argument("--gentle_wake_start_ratio", type=float, default=0.25)
    parser.add_argument("--gentle_wake_ramp_ratio", type=float, default=0.125)
    parser.add_argument("--utilization_kl_max_samples", type=int, default=8)
    parser.add_argument("--utilization_segment_max_samples", type=int, default=8)
    parser.add_argument("--utilization_self_reuse_max_samples", type=int, default=8)
    parser.add_argument("--utilization_consistency_max_samples", type=int, default=16)
    parser.add_argument("--utilization_simplex_max_samples", type=int, default=8)
    parser.add_argument("--utilization_simplex_ce_max_samples", type=int, default=8)
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
