from __future__ import annotations

import html
import importlib.metadata as md
import json
import statistics as st
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "wake_lora_experiment_summary.html"


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(path: str, method: str):
    data = read_json(ROOT / path)
    if not data:
        return None
    return data["results"].get(method)


def fmt(x, digits=4):
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def env_rows():
    names = ["numpy", "torch", "torchvision", "transformers", "peft", "datasets", "accelerate", "tqdm"]
    rows = []
    for name in names:
        try:
            ver = md.version(name)
        except Exception:
            ver = "missing"
        rows.append((name, ver))
    return rows


def row(cells, tag="td"):
    return "<tr>" + "".join(f"<{tag}>{html.escape(str(c))}</{tag}>" for c in cells) + "</tr>"


def build_main_table():
    runs = [
        ("8 / 1 epoch / seed 42 / KL 0.1", "outputs/smoke_medical_o1_8_v2/summary.json"),
        ("16 / 1 epoch / seed 42 / KL 0.1", "outputs/pilot_medical_o1_en_16_e1_lam01/summary.json"),
        ("32 / 1 epoch / seed 42 / KL 0.1", "outputs/pilot_medical_o1_en_32_e1_lam01/summary.json"),
        ("64 / 1 epoch / seed 42 / KL 0.1", "outputs/pilot_medical_o1_en_64_e1_lam01/summary.json"),
        ("64 / 3 epochs / seed 42 / KL 0.005", "outputs/pilot_medical_o1_en_64_e3_lam005/summary.json"),
    ]
    out = [row(["Setting", "Base NLL", "Standard LoRA NLL", "Wake-LoRA NLL", "Wake - Standard"], "th")]
    for label, path in runs:
        data = read_json(ROOT / path)
        if not data:
            continue
        res = data["results"]
        base = res.get("base", {}).get("nll")
        std = res.get("standard_lora", {}).get("nll")
        wake = res.get("wake_lora", {}).get("nll")
        diff = None if std is None or wake is None else wake - std
        out.append(row([label, fmt(base), fmt(std), fmt(wake), fmt(diff)]))
    return "\n".join(out)


def build_seed_table():
    runs = [
        ("42", "outputs/pilot_medical_o1_en_32_e1_lam01/summary.json"),
        ("43", "outputs/pilot_medical_o1_en_32_e1_lam01_seed43/summary.json"),
        ("44", "outputs/pilot_medical_o1_en_32_e1_lam01_seed44/summary.json"),
    ]
    stds, wakes, diffs = [], [], []
    out = [row(["Seed", "Base NLL", "Standard LoRA NLL", "Wake-LoRA NLL", "Wake - Standard"], "th")]
    for seed, path in runs:
        data = read_json(ROOT / path)
        if not data:
            continue
        res = data["results"]
        base = res["base"]["nll"]
        std = res["standard_lora"]["nll"]
        wake = res["wake_lora"]["nll"]
        diff = wake - std
        stds.append(std)
        wakes.append(wake)
        diffs.append(diff)
        out.append(row([seed, fmt(base), fmt(std), fmt(wake), fmt(diff)]))
    if diffs:
        out.append(
            row(
                [
                    "Mean +/- population std",
                    "-",
                    f"{st.mean(stds):.4f} +/- {st.pstdev(stds):.4f}",
                    f"{st.mean(wakes):.4f} +/- {st.pstdev(wakes):.4f}",
                    f"{st.mean(diffs):.4f} +/- {st.pstdev(diffs):.4f}",
                ]
            )
        )
    return "\n".join(out)


def build_segment_seed_table():
    runs = [
        (
            "42",
            "outputs/pilot_medical_o1_en_32_e1_lam01/summary.json",
            "outputs/pilot_medical_o1_en_32_e1_segment005_kl01/summary.json",
        ),
        (
            "43",
            "outputs/pilot_medical_o1_en_32_e1_lam01_seed43/summary.json",
            "outputs/pilot_medical_o1_en_32_e1_segment005_kl01_seed43/summary.json",
        ),
        (
            "44",
            "outputs/pilot_medical_o1_en_32_e1_lam01_seed44/summary.json",
            "outputs/pilot_medical_o1_en_32_e1_segment005_kl01_seed44/summary.json",
        ),
    ]
    stds, wakes, segs = [], [], []
    out = [
        row(
            [
                "Seed",
                "Standard LoRA",
                "Wake-KL",
                "Wake-KL+Segment",
                "Segment - Standard",
                "Segment - Wake-KL",
            ],
            "th",
        )
    ]
    for seed, base_path, seg_path in runs:
        base_data = read_json(ROOT / base_path)
        seg_data = read_json(ROOT / seg_path)
        if not base_data or not seg_data:
            continue
        std = base_data["results"]["standard_lora"]["nll"]
        wake = base_data["results"]["wake_lora"]["nll"]
        seg = seg_data["results"]["wake_lora"]["nll"]
        stds.append(std)
        wakes.append(wake)
        segs.append(seg)
        out.append(row([seed, fmt(std), fmt(wake), fmt(seg), fmt(seg - std), fmt(seg - wake)]))
    if segs:
        out.append(
            row(
                [
                    "Mean",
                    fmt(st.mean(stds)),
                    fmt(st.mean(wakes)),
                    fmt(st.mean(segs)),
                    fmt(st.mean(segs) - st.mean(stds)),
                    fmt(st.mean(segs) - st.mean(wakes)),
                ]
            )
        )
    return "\n".join(out)


def build_64_segment_table():
    base_data = read_json(ROOT / "outputs/pilot_medical_o1_en_64_e1_lam01/summary.json")
    seg_only = read_json(ROOT / "outputs/pilot_medical_o1_en_64_e1_segment005_kl0/summary.json")
    kl_seg = read_json(ROOT / "outputs/pilot_medical_o1_en_64_e1_segment005_kl01/summary.json")
    out = [row(["Method", "lambda_kl", "lambda_segment", "NLL", "Delta vs Standard"], "th")]
    if not base_data:
        return "\n".join(out)
    std = base_data["results"]["standard_lora"]["nll"]
    rows = [
        ("Standard LoRA", "0.0", "0.0", std),
        ("Wake-KL", "0.1", "0.0", base_data["results"]["wake_lora"]["nll"]),
    ]
    if seg_only:
        rows.append(("Wake-Segment", "0.0", "0.005", seg_only["results"]["wake_lora"]["nll"]))
    if kl_seg:
        rows.append(("Wake-KL+Segment", "0.1", "0.005", kl_seg["results"]["wake_lora"]["nll"]))
    for label, lam_kl, lam_seg, nll in rows:
        out.append(row([label, lam_kl, lam_seg, fmt(nll), fmt(nll - std)]))
    return "\n".join(out)


def build_sweep_table():
    out = [row(["Variant", "NLL", "Perplexity"], "th")]
    for p in sorted((ROOT / "outputs").glob("sweep_medical_o1_en_64_e1_lam_*/summary.json")):
        data = read_json(p)
        m = data["results"]["wake_lora"]
        out.append(row([p.parent.name, fmt(m["nll"]), fmt(m["perplexity"])]))
    for p in sorted((ROOT / "outputs").glob("sweep_medical_o1_en_64_e1_ce_*_kl0/summary.json")):
        data = read_json(p)
        m = data["results"]["wake_lora"]
        out.append(row([p.parent.name, fmt(m["nll"]), fmt(m["perplexity"])]))
    return "\n".join(out)


def build_mcqa_table(paths):
    out = [row(["Method", "Protocol", "Accuracy", "Correct / Total"], "th")]
    for label, path in paths:
        data = read_json(path)
        if data:
            out.append(
                row(
                    [
                        label,
                        f"{data.get('prompt_style', 'sft')} / {data.get('scoring_method', 'generate')}",
                        fmt(data["accuracy"]),
                        f"{data['correct']} / {data['total']}",
                    ]
                )
            )
    return "\n".join(out)


def build_mcqa_likelihood_table():
    return build_mcqa_table(
        [
            ("Base Qwen", ROOT / "outputs/medxpertqa_text_50_chat_option/base/mcqa_eval.json"),
            (
                "Standard LoRA 32 seed42",
                ROOT / "outputs/medxpertqa_text_50_chat_option/standard_lora_32_seed42/mcqa_eval.json",
            ),
            (
                "Wake-LoRA 32 seed42",
                ROOT / "outputs/medxpertqa_text_50_chat_option/wake_lora_32_seed42/mcqa_eval.json",
            ),
        ]
    )


def build_mcqa_legacy_table():
    return build_mcqa_table(
        [
        ("Base Qwen", ROOT / "outputs/medxpertqa_text_50/base/mcqa_eval.json"),
        ("Standard LoRA 32 seed42", ROOT / "outputs/medxpertqa_text_50/standard_lora_32_seed42/mcqa_eval.json"),
        ("Wake-LoRA 32 seed42", ROOT / "outputs/medxpertqa_text_50/wake_lora_32_seed42/mcqa_eval.json"),
        ]
    )


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    env = "\n".join(row(x) for x in env_rows())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Wake-LoRA Medical Experiment Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2937; }}
    h1 {{ margin-bottom: 0; }}
    h2 {{ margin-top: 28px; border-bottom: 1px solid #d1d5db; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 18px; font-size: 14px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .note {{ background: #f9fafb; border-left: 4px solid #2563eb; padding: 12px 14px; margin: 14px 0; }}
    .warn {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; margin: 14px 0; }}
    code {{ background: #eef2ff; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Wake-LoRA Medical Experiment Summary</h1>
  <p>Generated at {html.escape(now)} on the LabServer project.</p>

  <div class="note">
    Objective: test whether an SGFR-inspired Wake-Zone regularizer improves low-data LoRA fine-tuning for medical SFT.
    The current positive signal is on the 32-sample setting, where Wake-LoRA beats Standard LoRA across three seeds.
  </div>

  <h2>Environment</h2>
  <table>{env}</table>
  <p>Model path: <code>/home/jovyan/imagenet-1k/qwen_model</code>. Dataset: <code>FreedomIntelligence/medical-o1-reasoning-SFT</code>, config <code>en</code>, accessed through <code>https://hf-mirror.com</code>.</p>

  <h2>Main Pilot Runs</h2>
  <table>{build_main_table()}</table>

  <h2>32-Sample Three-Seed Check</h2>
  <table>{build_seed_table()}</table>
  <p>Lower NLL is better. Wake-LoRA uses <code>lambda_kl=0.1</code>, <code>lambda_ce_reuse=0.0</code>, 1 epoch, 32 train samples, 64 eval samples, max length 512.</p>

  <h2>Token Segment Memory Extension</h2>
  <table>{build_segment_seed_table()}</table>
  <p>This extension follows the image SGFR geometry more closely: target-token hidden states are constrained near the segment between the target token output weight and a per-token memory centroid. The 32-sample run uses <code>lambda_kl=0.1</code> and <code>lambda_segment=0.005</code>.</p>

  <h2>64-Sample KL Schedule Probe</h2>
  <table>{build_64_segment_table()}</table>
  <p>At 64 samples, the segment term helps but the KL term should be reduced or disabled. This supports a sample-count-aware or progress-aware schedule.</p>

  <h2>Hyperparameter Sweep Notes</h2>
  <table>{build_sweep_table()}</table>
  <div class="warn">
    The direct CE-reuse variant did not help in the 64-sample pilot. KL strength must be controlled carefully:
    stronger KL improved neither the 64-sample 1-epoch run nor the 3-epoch run.
  </div>

  <h2>External MCQA: Constrained Option Scoring</h2>
  <table>{build_mcqa_likelihood_table()}</table>
  <p>This evaluates each candidate answer letter by conditional log-likelihood under the prompt and selects the highest-scoring option. This protocol is preferred over free-form answer generation for multiple-choice evaluation.</p>

  <h2>External MCQA: Legacy Free Generation</h2>
  <table>{build_mcqa_legacy_table()}</table>
  <p>The legacy 50-question MedXpertQA check used greedy free-form generation followed by regex parsing. It is retained only for provenance and should not be used as the main external MCQA claim.</p>

  <h2>Interpretation</h2>
  <ul>
    <li>Base Qwen has strong language modeling ability but does not specialize to this medical reasoning SFT target.</li>
    <li>Standard LoRA improves held-out SFT NLL substantially.</li>
    <li>Wake-LoRA shows a stable advantage in the 32-sample regime across seeds 42, 43, and 44.</li>
    <li>The new token segment memory term improves the 32-sample mean NLL and recovers the 64-sample setting when KL is disabled.</li>
    <li>Wake-LoRA is not universally better yet: in the 64-sample setting, the KL anchor can slow adaptation unless it is scheduled down.</li>
  </ul>

  <h2>Next Improvements</h2>
  <ol>
    <li>Run a full matrix over sample counts 8/16/32/64/128 and seeds 42/43/44.</li>
    <li>Introduce a scheduled KL coefficient that decays as sample count or training progress increases.</li>
    <li>Broaden the token segment memory test to 8/16/32/64/128 samples and three seeds.</li>
    <li>Improve MCQA evaluation with constrained option scoring rather than free-form generation parsing.</li>
    <li>Add a second medical dataset to test transfer rather than only same-distribution validation NLL.</li>
  </ol>
</body>
</html>
"""
    OUT.write_text(html_text, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
