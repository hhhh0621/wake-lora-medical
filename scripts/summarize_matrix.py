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


def read_best_eval(path: Path, result_key: str) -> tuple[int | None, float | None]:
    metrics_path = path.parent / result_key / "eval_metrics.jsonl"
    if not metrics_path.exists():
        return None, None
    rows = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    rows = [row for row in rows if "nll" in row]
    if not rows:
        return None, None
    best = min(rows, key=lambda row: float(row["nll"]))
    return best.get("epoch"), float(best["nll"])


def best_gap(result: dict, path: Path, result_key: str) -> tuple[int | None, float, float]:
    best_epoch, best_nll = read_best_eval(path, result_key)
    if best_nll is None:
        best_nll = float(result["nll"])
    return best_epoch, best_nll, float(result["nll"]) - best_nll


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
    method = "_".join(method_parts)
    return sample, seed, method


def collect_rows(prefix: str, include_legacy: bool = False) -> list[dict]:
    rows = []
    seen = set()
    if include_legacy:
        for group, sample, seed, method, rel_path, result_key in LEGACY_RUNS:
            path = ROOT / rel_path
            data = read_json(path)
            if not data:
                continue
            result = data.get("results", {}).get(result_key)
            if not result:
                continue
            base = data.get("results", {}).get("base")
            base_key = (group, sample, seed, "base")
            if base and base_key not in seen:
                seen.add(base_key)
                rows.append(
                    {
                        "group": group,
                        "sample_count": sample,
                        "seed": seed,
                        "method": "base",
                        "nll": base["nll"],
                        "perplexity": base["perplexity"],
                        "tokens": base["tokens"],
                        "best_epoch": None,
                        "best_nll": base["nll"],
                        "final_minus_best": 0.0,
                        "path": rel_path,
                    }
                )
            key = (group, sample, seed, method)
            seen.add(key)
            best_epoch, best_nll, gap = best_gap(result, path, result_key)
            rows.append(
                {
                    "group": group,
                    "sample_count": sample,
                    "seed": seed,
                    "method": method,
                    "nll": result["nll"],
                    "perplexity": result["perplexity"],
                    "tokens": result["tokens"],
                    "best_epoch": best_epoch,
                    "best_nll": best_nll,
                    "final_minus_best": gap,
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
        base = data.get("results", {}).get("base")
        base_key = ("matrix", sample, seed, "base")
        if base and base_key not in seen:
            seen.add(base_key)
            rows.append(
                {
                    "group": "matrix",
                    "sample_count": sample,
                    "seed": seed,
                    "method": "base",
                        "nll": base["nll"],
                        "perplexity": base["perplexity"],
                        "tokens": base["tokens"],
                        "best_epoch": None,
                        "best_nll": base["nll"],
                        "final_minus_best": 0.0,
                        "path": str(path.relative_to(ROOT)),
                    }
                )
        result_key = "standard_lora" if method.startswith("standard") else "wake_lora"
        result = data.get("results", {}).get(result_key)
        if not result:
            continue
        best_epoch, best_nll, gap = best_gap(result, path, result_key)
        rows.append(
            {
                "group": "matrix",
                "sample_count": sample,
                "seed": seed,
                "method": method,
                "nll": result["nll"],
                "perplexity": result["perplexity"],
                "tokens": result["tokens"],
                "best_epoch": best_epoch,
                "best_nll": best_nll,
                "final_minus_best": gap,
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


def fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def markdown_summary(rows: list[dict]) -> str:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["sample_count"], row["method"]), []).append(row)

    lines = [
        "# Wake-LoRA Low-Data Matrix Summary",
        "",
        "Lower NLL is better.",
        "",
        "| Samples | Method | Runs | Mean Final NLL | Std | Mean Best NLL | Mean Final-Best Gap |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (sample, method), group_rows in sorted(grouped.items()):
        final_values = [float(row["nll"]) for row in group_rows]
        best_values = [float(row.get("best_nll", row["nll"])) for row in group_rows]
        gap_values = [float(row.get("final_minus_best", 0.0)) for row in group_rows]
        mean, std = mean_std(final_values)
        best_mean, _ = mean_std(best_values)
        gap_mean, _ = mean_std(gap_values)
        lines.append(
            f"| {sample} | {method} | {len(group_rows)} | {mean:.6f} | {std:.6f} | "
            f"{best_mean:.6f} | {gap_mean:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Rows",
            "",
            "| Samples | Seed | Method | Final NLL | Best Epoch | Best NLL | Gap | PPL | Path |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda x: (x["sample_count"], x["seed"], x["method"], x["path"])):
        best_epoch = "" if row.get("best_epoch") is None else str(row["best_epoch"])
        lines.append(
            f"| {row['sample_count']} | {row['seed']} | {row['method']} | "
            f"{row['nll']:.6f} | {best_epoch} | {fmt_float(row.get('best_nll'))} | "
            f"{fmt_float(row.get('final_minus_best'))} | {row['perplexity']:.4f} | `{row['path']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize low-data Wake-LoRA matrix outputs.")
    parser.add_argument("--prefix", default="matrix_medical_o1")
    parser.add_argument("--include_legacy", action="store_true")
    parser.add_argument("--output_dir", default="reports")
    args = parser.parse_args()

    rows = collect_rows(args.prefix, include_legacy=args.include_legacy)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "low_data_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "sample_count",
                "seed",
                "method",
                "nll",
                "perplexity",
                "tokens",
                "best_epoch",
                "best_nll",
                "final_minus_best",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "low_data_matrix.md"
    md_path.write_text(markdown_summary(rows), encoding="utf-8")
    print(md_path)
    print(csv_path)


if __name__ == "__main__":
    main()
