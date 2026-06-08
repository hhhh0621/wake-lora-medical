# Wake-LoRA Medical SFT

This folder is a small, reproducible experiment for moving SGFR-style sample
reuse into LoRA fine-tuning.

The first target task is medical instruction tuning under low-data supervision.
The comparison has three arms:

1. Base Qwen: evaluate the downloaded Qwen model without fine-tuning.
2. Standard LoRA: train LoRA with ordinary next-token cross entropy.
3. Wake-LoRA: train LoRA with cross entropy plus SGFR-inspired memory terms.

## Method Sketch

The image SGFR code uses:

- parameter anchor: current classifier weight for the label
- memory anchor: historical class feature mean
- sample feature: current feature to be rectified into the wake zone

For language model LoRA, the first implementation moves this idea to token
distributions:

```text
loss = CE(P_lora, y) + alpha * KL(P_base || P_lora)
alpha = lambda_kl / (CE(P_base, y) + eps)
```

If the base model already explains a token well, `CE(P_base, y)` is small and
the KL anchor is strong. If the base model is wrong, the anchor relaxes and the
label can pull the LoRA adapter harder.

The current research branch also includes a direct sample-utilization term:

```text
w = 1 + clip(lambda_ce_reuse / (CE(P_base, y) + eps), 0, w_max)
CE_reuse = sum(w * CE(P_lora, y)) / sum(w)
```

This anti-dropping term increases the contribution of base-easy tokens without
changing the average loss scale.

The newer branch adds a closer translation of the original segment geometry:

```text
A_i = lm_head weight of target token y_i
B_i = memory-bank centroid of historical hidden states for token y_i
P_i = current hidden state that predicts y_i
loss_segment = mean_i distance(P_i, segment(A_i, B_i))
```

Use `--lambda_segment 0.005` to enable this term. Early pilots suggest
`lambda_kl=0.1, lambda_segment=0.005` is useful at 32 samples, while 64 samples
prefer reducing or disabling KL and keeping the segment term.

## Default Dataset

The default dataset is `FreedomIntelligence/medical-o1-reasoning-SFT`, a recent
medical reasoning SFT dataset on Hugging Face. The code also accepts another
dataset name and column mapping through command-line flags.

Useful links:

- Dataset: https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT
- Optional external benchmark: https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA

## Server Command

Use the conda Python on the lab server, not `/usr/bin/python3`.

```bash
cd /home/jovyan/imagenet-1k/wake_lora_medical
/opt/conda/bin/python -m pip install -r requirements.txt
PYTHONPATH=src /opt/conda/bin/python -m wake_lora.run_three_way \
  --model_path /home/jovyan/imagenet-1k/qwen_model \
  --output_dir outputs/medical_o1_qwen_small \
  --dataset_name FreedomIntelligence/medical-o1-reasoning-SFT \
  --dataset_config en \
  --max_train_samples 512 \
  --max_eval_samples 128 \
  --max_length 1024 \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 8
```

Example SGFR-style Wake-LoRA run:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src /opt/conda/bin/python -m wake_lora.run_three_way \
  --config configs/qwen_medical_o1_small.json \
  --output_dir outputs/medical_o1_qwen_32_segment \
  --max_train_samples 32 \
  --max_eval_samples 64 \
  --max_length 512 \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lambda_kl 0.1 \
  --lambda_segment 0.005
```

For the Chinese split, change `--dataset_config en` to the matching split if the
dataset exposes it on the server.

## Outputs

Each run writes JSON summaries under `outputs/...`:

- `base_eval.json`
- `standard_lora/eval_final.json`
- `wake_lora/eval_final.json`
- `summary.json`

The ranking metric for the first pass is validation negative log-likelihood and
perplexity on a held-out split. After the training pipeline is stable, add an
external QA accuracy evaluation to avoid overclaiming from same-dataset splits.

## Optional External MCQA Evaluation

After the adapters are trained, evaluate on the text split of MedXpertQA.
For multiple choice, prefer constrained option scoring over free-form answer
generation:

```bash
PYTHONPATH=src /opt/conda/bin/python -m wake_lora.mcqa_eval \
  --model_path /home/jovyan/imagenet-1k/qwen_model \
  --adapter_path outputs/medical_o1_qwen_small/wake_lora/adapter_final \
  --output_dir outputs/medical_o1_qwen_small/wake_lora/medxpertqa_text \
  --dataset_name TsinghuaC3I/MedXpertQA \
  --dataset_config Text \
  --dataset_split test \
  --max_samples 200 \
  --prompt_style chat \
  --scoring_method option_likelihood \
  --length_normalize
```

The current MedXpertQA 50-question sanity check is very low in absolute
accuracy, so it should be treated as a hard external diagnostic, not as the main
evidence for method quality.

## Low-Data Matrix

The next-stage experiment is driven by a reproducible matrix runner. It skips
finished runs by default.

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src /opt/conda/bin/python scripts/run_low_data_matrix.py \
  --samples 8,16,32,64,128 \
  --seeds 42,43,44 \
  --methods standard,wake_segment,wake_kl_segment,wake_reliable
```

`wake_adaptive` uses the current sample-aware KL rule:

```text
lambda_kl(n) = 0.1,                         if n <= 32
lambda_kl(n) = 0.1 * (32 / n)^2,            if n > 32
lambda_segment = 0.005
```

`wake_scheduled` is the current preferred rule after the first 8/16/32/64/128
probe:

```text
lambda_kl(n) = 0.1,                         if n <= 32
lambda_kl(n) = 0,                           if n > 32
lambda_segment(n) = 0.005,                  if n <= 64
lambda_segment(n) = 0.005 * (64 / n)^2,     if n > 64
```

`wake_budget` is the more conservative fixed-update rule introduced after the
32-sample budget-controlled runs showed over-regularization from the stronger
schedule:

```text
lambda_kl(n) = 0.1,                         if n <= 16
lambda_kl(n) = 0,                           if n > 16
lambda_segment(n) = 0.005,                  if n <= 16
lambda_segment(n) = 0.005 * (16 / n)^2,     if n > 16
```

`wake_delayed` uses the same sample-count budget, but delays the Wake KL/segment
regularizers until the LoRA adapter has completed the first quarter of its
optimizer updates:

```text
wake_multiplier(t) = 0,                     if progress(t) < 0.25
wake_multiplier(t) = linear ramp to 1,      over the next 0.125 progress
lambda_kl(t) = wake_multiplier(t) * lambda_kl(n)
lambda_segment(t) = wake_multiplier(t) * lambda_segment(n)
```

`wake_gentle` is the current sample-aware follow-up after the low-learning-rate
matrix showed that strong Wake regularization helps at 8 samples but is already
too much at 16 samples:

```text
lambda_kl(n) = 0.1,                         if n <= 8
lambda_kl(n) = 0,                           if n > 8
lambda_segment(n) = 0.005,                  if n <= 8
lambda_segment(n) = 0.000625,               if 8 < n <= 16
lambda_segment(n) = 0,                      if n > 16
wake_start_ratio = 0.25 and wake_ramp_ratio = 0.125, if n <= 8
wake_start_ratio = 0 and wake_ramp_ratio = 0,         if n > 8
```

`wake_utilization` keeps the V3 delayed KL+segment anchor at 8 samples, adds
normalized current-model token reuse and two-view dropout consistency, keeps
only consistency at 16 samples, and disables Wake-utilization terms above 16
samples:

```text
if n <= 8:
    lambda_kl = 0.1
    lambda_segment = 0.005
    lambda_self_reuse = 0.025
    lambda_consistency = 0.5
    wake_start_ratio = 0.25
    wake_ramp_ratio = 0.125
elif n <= 16:
    lambda_consistency = 0.5
else:
    all Wake-utilization terms = 0
```

Clean three-seed low-LR fixed-update results:

| Train samples | Standard LoRA | Wake-gentle | Wake-utilization |
|---:|---:|---:|---:|
| 8 | 1.760924 | 1.738797 | 1.723933 |
| 16 | 1.656166 | 1.656163 | 1.656016 |
| 32 | 1.626196 | 1.625987 | 1.625920 |

Stronger LR sanity checks show that standard LoRA is very competitive at
`learning_rate=5e-5` for 32 updates. The current strongest Wake-utilization
claim is therefore narrower: it substantially reduces late-training drift when
the 8-sample set is reused for longer budgets. See
`docs/method_note.md` for the full positive and negative ablations.

The current best candidate is `wake_utilization_strong` with PiSSA
initialization:

```text
--lora_init pissa_niter_4
--learning_rate 5e-5
--target_updates 32
if n <= 8:
    lambda_kl = 0.3
    lambda_segment = 0.015
    lambda_self_reuse = 0.025
    lambda_consistency = 0.5
    wake_start_ratio = 0.25
    wake_ramp_ratio = 0.125
elif n <= 16:
    lambda_consistency = 0.5
else:
    all Wake-utilization terms = 0
```

Independent three-seed results against PiSSA LoRA:

| Train samples | Standard PiSSA | Wake strong PiSSA | Final-best gap, standard | Final-best gap, Wake |
|---:|---:|---:|---:|---:|
| 8 | 1.767560 | 1.697446 | 0.064367 | 0.000277 |
| 16 | 1.677108 | 1.660998 | 0.004337 | 0.000130 |

The result is promising because Wake makes the final checkpoint almost match
the best checkpoint, while standard PiSSA still drifts late. The claim should
remain conservative until it is validated on a second medical dataset and
compared against tuned ordinary LoRA plus PiSSA/DoRA baselines. EMA adapter
averaging was tested as a fair no-validation trick, but did not beat the raw
Wake strong checkpoint under the current settings.

External validation has started on
`huzaifa525/Medical_Intelligence_Dataset_76k_2026_Edition`, using the committed
`configs/qwen_medical_intelligence_2026.json` config. With 5 seeds, 64 eval
samples, `lr=5e-5`, and 32 updates:

| Dataset | Train samples | Standard PiSSA | Wake strong PiSSA | Paired delta |
|---|---:|---:|---:|---:|
| Medical Intelligence 2026 | 8 | 1.796237 | 1.735205 | -0.061032 |
| Medical Intelligence 2026 | 16 | 1.750075 | 1.718800 | -0.031275 |

The same external run also checked default-initialized LoRA at `5e-5`,
`7.5e-5`, and `1e-4`; the best default LoRA settings were still worse than
Wake strong PiSSA on the final checkpoint. This is the first cross-dataset
evidence strong enough to shape the paper narrative, though the final paper
still needs a broader tuned-baseline table.

Run the candidate matrix with:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src /opt/conda/bin/python scripts/run_low_data_matrix.py \
  --output_prefix v5_utilization_final_lr1e4 \
  --samples 8,16,32 \
  --seeds 42,43,44 \
  --methods standard,wake_utilization \
  --target_updates 32 \
  --learning_rate 1e-4
```

Summarize training dynamics for any matrix prefix:

```bash
PYTHONPATH=src /opt/conda/bin/python scripts/summarize_training_dynamics.py \
  --prefix pissa_lr5e5_n8_u32 \
  --output_dir reports/pissa_lr5e5_n8_u32
```

`wake_reliable` uses the same schedule but only applies the segment memory loss
when a target token already has at least two historical hidden states in the
memory bank. This guards the extreme 8/16-sample regime, where the memory anchor
can otherwise be too noisy.

After a run, summarize results without generating server-side HTML:

```bash
PYTHONPATH=src /opt/conda/bin/python scripts/summarize_matrix.py
```

The matrix summary reports final NLL as the primary metric and also includes
best-epoch NLL plus the final-best gap as an overfitting diagnostic.

For fair low-data comparisons, use a fixed optimizer-update budget instead of a
fixed epoch count. For example, this gives each sample-count setting about 32
optimizer updates:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src /opt/conda/bin/python scripts/run_low_data_matrix.py \
  --output_prefix matrix_budget_o1 \
  --samples 8,16,32,64,128 \
  --seeds 42,43,44 \
  --methods standard,wake_budget,wake_scheduled \
  --target_updates 32
```

Then summarize that budget-controlled matrix:

```bash
PYTHONPATH=src /opt/conda/bin/python scripts/summarize_matrix.py --prefix matrix_budget_o1
```

For extreme 8-sample runs, the fixed-update curves can overfit late in training.
Use a separate output prefix when sweeping learning rates, for example:

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src /opt/conda/bin/python scripts/run_low_data_matrix.py \
  --output_prefix matrix_lr1e4_o1 \
  --samples 8 \
  --seeds 42,43,44 \
  --methods standard,wake_budget,wake_delayed \
  --target_updates 32 \
  --learning_rate 1e-4
```
