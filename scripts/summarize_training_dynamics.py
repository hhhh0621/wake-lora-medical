from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def parse_matrix_name(path: Path) -> tuple[int, int, str] | None:
    name = path.parent.name
    parts = name.split("_")
    sample = None
    seed = None
    method_parts = []
    for idx, part in enumerate(parts):
        if part.startswith("n") and part[1:].isdigit():
            sample = int(part[1:])
        elif part.startswith("s") and part[1:].isdigit():
            seed = int(part[1:])
            method_parts = parts[idx + 1 :]
            break
    if sample is None or seed is None or not method_parts:
        return None
    method_parts = [x for x in method_parts if not re.fullmatch(r"(kl|seg|sr|cons|simp|sce|hct|ws|wr)[0-9p]+", x)]
    return sample, seed, "_".join(method_parts)


def mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return st.mean(values)


def value(row: dict[str, Any], key: str) -> float:
    raw = row.get(key)
    if raw is None:
        return math.nan
    try:
        return float(raw)
    except Exception:
        return math.nan


def best_eval(eval_rows: list[dict[str, Any]], final_nll: float) -> tuple[int | None, float, float]:
    eval_rows = [row for row in eval_rows if "nll" in row]
    if not eval_rows:
        return None, final_nll, 0.0
    best = min(eval_rows, key=lambda row: float(row["nll"]))
    best_epoch = best.get("epoch")
    best_nll = float(best["nll"])
    return best_epoch, best_nll, final_nll - best_nll


def collect_run(path: Path) -> list[dict[str, Any]]:
    parsed = parse_matrix_name(path)
    if parsed is None:
        return []
    sample, seed, method = parsed
    data = read_json(path)
    if not data:
        return []

    rows = []
    result_keys = []
    if "standard_lora" in data.get("results", {}):
        result_keys.append(("standard", "standard_lora"))
    if "wake_lora" in data.get("results", {}):
        result_keys.append((method, "wake_lora"))

    for display_method, result_key in result_keys:
        result = data["results"].get(result_key)
        if not result or "nll" not in result:
            continue
        run_dir = path.parent / result_key
        train_rows = read_jsonl(run_dir / "train_metrics.jsonl")
        eval_rows = read_jsonl(run_dir / "eval_metrics.jsonl")
        final_nll = float(result["nll"])
        best_epoch, best_nll, gap = best_eval(eval_rows, final_nll)
        tail = train_rows[-5:]
        last = train_rows[-1] if train_rows else {}
        rows.append(
            {
                "sample_count": sample,
                "seed": seed,
                "method": display_method,
                "final_nll": final_nll,
                "best_epoch": best_epoch,
                "best_nll": best_nll,
                "final_best_gap": gap,
                "train_steps": len(train_rows),
                "final_train_loss": value(last, "loss"),
                "final_ce_lora": value(last, "ce_lora"),
                "final_regularizer_total": value(last, "regularizer_total"),
                "tail_regularizer_total": mean([value(row, "regularizer_total") for row in tail]),
                "final_grad_norm": value(last, "grad_norm"),
                "tail_grad_norm": mean([value(row, "grad_norm") for row in tail]),
                "max_grad_norm": max([value(row, "grad_norm") for row in train_rows], default=math.nan),
                "final_param_norm": value(last, "trainable_param_norm"),
                "final_param_rms": value(last, "trainable_param_rms"),
                "tail_wake_multiplier": mean([value(row, "wake_multiplier") for row in tail]),
                "final_ce_weight_ess_ratio": value(last, "ce_weight_ess_ratio"),
                "path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def fmt(value_: Any, digits: int = 6) -> str:
    if value_ is None:
        return ""
    if isinstance(value_, float):
        if math.isnan(value_):
            return ""
        return f"{value_:.{digits}f}"
    return str(value_)


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "training_dynamics.csv"
    md_path = output_dir / "training_dynamics.md"
    fieldnames = [
        "sample_count",
        "seed",
        "method",
        "final_nll",
        "best_epoch",
        "best_nll",
        "final_best_gap",
        "train_steps",
        "final_train_loss",
        "final_ce_lora",
        "final_regularizer_total",
        "tail_regularizer_total",
        "final_grad_norm",
        "tail_grad_norm",
        "max_grad_norm",
        "final_param_norm",
        "final_param_rms",
        "tail_wake_multiplier",
        "final_ce_weight_ess_ratio",
        "path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "# Training Dynamics Summary",
        "",
        "Lower NLL and final-best gap are better. Tail metrics average the last five optimizer steps.",
        "",
        "| Samples | Seed | Method | Final NLL | Best NLL | Gap | Steps | Tail Grad | Tail Reg | Param RMS | ESS |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["sample_count"], item["seed"], item["method"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sample_count"]),
                    str(row["seed"]),
                    str(row["method"]),
                    fmt(row["final_nll"]),
                    fmt(row["best_nll"]),
                    fmt(row["final_best_gap"]),
                    str(row["train_steps"]),
                    fmt(row["tail_grad_norm"]),
                    fmt(row["tail_regularizer_total"]),
                    fmt(row["final_param_rms"]),
                    fmt(row["final_ce_weight_ess_ratio"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Wake-LoRA training dynamics for a matrix prefix.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output_dir", default="reports/training_dynamics")
    args = parser.parse_args()

    rows = []
    for path in sorted((ROOT / "outputs").glob(f"{args.prefix}_*/summary.json")):
        rows.extend(collect_run(path))
    write_outputs(rows, ROOT / args.output_dir)
    print(ROOT / args.output_dir / "training_dynamics.md")
    print(ROOT / args.output_dir / "training_dynamics.csv")


if __name__ == "__main__":
    main()
