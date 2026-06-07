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


def output_dir_name(prefix: str, samples: int, seed: int, method: str, lambda_kl: float, lambda_segment: float) -> str:
    kl_tag = f"kl{lambda_kl:.5g}".replace(".", "p")
    seg_tag = f"seg{lambda_segment:.5g}".replace(".", "p")
    return f"{prefix}_n{samples}_s{seed}_{method}_{kl_tag}_{seg_tag}"


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
    lambda_segment = float(spec["lambda_segment"])
    out_dir = ROOT / "outputs" / output_dir_name(args.output_prefix, samples, seed, method, float(lambda_kl), lambda_segment)
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
        str(args.epochs),
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
        "--seed",
        str(seed),
        *spec["skip"],
    ]
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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_max_batches", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--segment_memory_size", type=int, default=4)
    parser.add_argument("--adaptive_base_kl", type=float, default=0.1)
    parser.add_argument("--adaptive_ref_samples", type=int, default=32)
    parser.add_argument("--adaptive_power", type=float, default=2.0)
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
