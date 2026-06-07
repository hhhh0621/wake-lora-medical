from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics as st
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


LEGACY_RUNS = [
    ("pilot32", 32, 42, "standard", "outputs/pilot_medical_o1_en_32_e1_lam01/summary.json", "standard_lora"),
    ("pilot32", 32, 42, "wake_kl", "outputs/pilot_medical_o1_en_32_e1_lam01/summary.json", "wake_lora"),
    ("pilot32", 32, 42, "wake_segment", "outputs/pilot_medical_o1_en_32_e1_segment005_kl0/summary.json", "wake_lora"),
    ("pilot32", 32, 42, "wake_kl_segment", "outputs/pilot_medical_o1_en_32_e1_segment005_kl01/summary.json", "wake_lora"),
    ("pilot32", 32, 43, "standard", "outputs/pilot_medical_o1_en_32_e1_lam01_seed43/summary.json", "standard_lora"),
    ("pilot32", 32, 43, "wake_kl", "outputs/pilot_medical_o1_en_32_e1_lam01_seed43/summary.json", "wake_lora"),
    ("pilot32", 32, 43, "wake_kl_segment", "outputs/pilot_medical_o1_en_32_e1_segment005_kl01_seed43/summary.json", "wake_lora"),
    ("pilot32", 32, 44, "standard", "outputs/pilot_medical_o1_en_32_e1_lam01_seed44/summary.json", "standard_lora"),
    ("pilot32", 32, 44, "wake_kl", "outputs/pilot_medical_o1_en_32_e1_lam01_seed44/summary.json", "wake_lora"),
    ("pilot32", 32, 44, "wake_kl_segment", "outputs/pilot_medical_o1_en_32_e1_segment005_kl01_seed44/summary.json", "wake_lora"),
    ("pilot64", 64, 42, "standard", "outputs/pilot_medical_o1_en_64_e1_lam01/summary.json", "standard_lora"),
    ("pilot64", 64, 42, "wake_kl", "outputs/pilot_medical_o1_en_64_e1_lam01/summary.json", "wake_lora"),
    ("pilot64", 64, 42, "wake_segment", "outputs/pilot_medical_o1_en_64_e1_segment005_kl0/summary.json", "wake_lora"),
    ("pilot64", 64, 42, "wake_kl_segment", "outputs/pilot_medical_o1_en_64_e1_segment005_kl01/summary.json", "wake_lora"),
]


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
    method_parts = [x for x in method_parts if not re.fullmatch(r"(kl|seg)[0-9p]+", x)]
    method = "_".join(method_parts)
    return sample, seed, method


def collect_rows(prefix: str) -> list[dict]:
    rows = []
    seen = set()
    for group, sample, seed, method, rel_path, result_key in LEGACY_RUNS:
        path = ROOT / rel_path
        data = read_json(path)
        if not data:
            continue
        result = data.get("results", {}).get(result_key)
        if not result:
            continue
        key = (group, sample, seed, method)
        seen.add(key)
        rows.append(
            {
                "group": group,
                "sample_count": sample,
                "seed": seed,
                "method": method,
                "nll": result["nll"],
                "perplexity": result["perplexity"],
                "tokens": result["tokens"],
                "path": rel_path,
            }
        )

    for path in sorted((ROOT / "outputs").glob(f"{prefix}_*/summary.json")):
        parsed = parse_matrix_name(path)
        if parsed is None:
            continue
        sample, seed, method = parsed
        key = ("matrix", sample, seed, method)
        if key in seen:
            continue
        data = read_json(path)
        if not data:
            continue
        result_key = "standard_lora" if method == "standard" else "wake_lora"
        result = data.get("results", {}).get(result_key)
        if not result:
            continue
        rows.append(
            {
                "group": "matrix",
                "sample_count": sample,
                "seed": seed,
                "method": method,
                "nll": result["nll"],
                "perplexity": result["perplexity"],
                "tokens": result["tokens"],
                "path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return values[0], 0.0
    return st.mean(values), st.pstdev(values)


def markdown_summary(rows: list[dict]) -> str:
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["sample_count"], row["method"]), []).append(float(row["nll"]))

    lines = [
        "# Wake-LoRA Low-Data Matrix Summary",
        "",
        "Lower NLL is better.",
        "",
        "| Samples | Method | Runs | Mean NLL | Std |",
        "|---:|---|---:|---:|---:|",
    ]
    for (sample, method), values in sorted(grouped.items()):
        mean, std = mean_std(values)
        lines.append(f"| {sample} | {method} | {len(values)} | {mean:.6f} | {std:.6f} |")

    lines.extend(["", "## Per-Seed Rows", "", "| Samples | Seed | Method | NLL | PPL | Path |", "|---:|---:|---|---:|---:|---|"])
    for row in sorted(rows, key=lambda x: (x["sample_count"], x["seed"], x["method"], x["path"])):
        lines.append(
            f"| {row['sample_count']} | {row['seed']} | {row['method']} | "
            f"{row['nll']:.6f} | {row['perplexity']:.4f} | `{row['path']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize low-data Wake-LoRA matrix outputs.")
    parser.add_argument("--prefix", default="matrix_medical_o1")
    parser.add_argument("--output_dir", default="reports")
    args = parser.parse_args()

    rows = collect_rows(args.prefix)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "low_data_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "sample_count", "seed", "method", "nll", "perplexity", "tokens", "path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "low_data_matrix.md"
    md_path.write_text(markdown_summary(rows), encoding="utf-8")
    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
