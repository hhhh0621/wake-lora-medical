# Wake-LoRA Method Note

This note records the first LoRA versions of the SGFR idea.

## Problem

In low-data supervised learning, ordinary cross entropy gives most gradient
mass to currently wrong tokens or samples. Easy but structurally informative
tokens quickly stop contributing. In image classification, SGFR uses a wake zone
between a plastic parameter anchor and a stable memory anchor to keep the full
class distribution involved in learning.

## V1: Distribution-Space Translation

For causal language model fine-tuning, the natural object is the next-token
distribution. We use:

- Memory anchor: `P_base`, the frozen base model distribution with LoRA disabled.
- Parameter point: `P_lora`, the current LoRA-adapted distribution.
- Label force: next-token CE on the LoRA distribution.

The training objective is:

```text
L = CE(P_lora, y) + alpha * KL(P_base || P_lora)
alpha = lambda_kl / (CE(P_base, y) + eps)
```

This is conservative by design. It does not claim to be the final simplex
projection analogue. It is a stable first test that checks whether adaptive
sample reuse improves LoRA behavior on small medical SFT data.

## V2: Token Segment Memory Translation

The newer image implementation is closer to a representation-space geometry:

```text
A = classifier weight of the target class
B = memory-bank centroid of historical features for that class
P = current sample feature
loss_segment = distance(P, segment(A, B))
```

For causal language modeling, the direct analogue is built at supervised target
token positions:

```text
A_i = output embedding / lm_head weight of target token y_i
B_i = memory-bank centroid of historical hidden states for token y_i
P_i = current hidden state that predicts y_i
loss_segment = mean_i distance(P_i, segment(A_i, B_i))
```

The training objective can then be:

```text
L = CE(P_lora, y)
  + alpha * KL(P_base || P_lora)
  + lambda_segment * loss_segment
```

The current implementation stores a small per-token hidden-state memory bank.
This is still a first approximation, but it is materially closer to SGFR than a
pure KL anchor because it reintroduces the line-segment wake-zone geometry.

The reliable-memory variant adds a simple gate: a target token only contributes
to the segment loss after its memory bank has at least `segment_min_count`
historical hidden states. This avoids using a noisy or empty memory anchor in
the extreme few-sample regime.

## Direct Anti-Dropping Variant

The KL-only form behaves mostly as a memory regularizer. To target the data
dropping pathology more directly, the current code also supports a normalized
token-level CE reweighting term:

```text
w_i = 1 + clip(lambda_ce_reuse / (CE_base_i + eps), 0, w_max)
CE_reuse = sum_i w_i CE_lora_i / sum_i w_i
```

Tokens that the frozen base already predicts well receive more gradient mass.
The denominator normalizes the loss scale, so improvements cannot be explained
only by a larger effective learning rate.

## What To Measure First

1. Held-out token NLL and perplexity.
2. Small generated answer inspection.
3. External multiple-choice medical QA accuracy after the training loop is
   stable.

## Expected Failure Modes

- `alpha` can become too large on trivial tokens, so it is clipped.
- If the base model is poor on the medical domain, KL can slow adaptation.
- If sample count increases, the KL anchor may need to decay or turn off while
  the segment term remains active.
- If the dataset target contains very long chain-of-thought traces, use a
  smaller `max_length` first and increase only after the smoke test is stable.

## Pilot Findings

- 32 train samples, 64 eval samples, 1 epoch, three seeds:
  - Standard LoRA mean NLL: 1.7116.
  - Wake-KL mean NLL: 1.7016.
  - Wake-KL+segment (`lambda_kl=0.1`, `lambda_segment=0.005`) mean NLL: 1.6986.
- 64 train samples, 64 eval samples, 1 epoch, seed 42:
  - Standard LoRA NLL: 1.5944.
  - Wake-KL NLL: 1.5997.
  - Wake-segment only (`lambda_segment=0.005`) NLL: 1.5926.

Interpretation: the segment memory term improves the SGFR-to-LoRA translation.
The KL term helps in the stricter 32-sample setting, but at 64 samples it should
be reduced or scheduled down.

## Fixed-Update Budget Findings

Using a fixed budget of about 32 optimizer updates changes the low-data story
and is a fairer protocol than one epoch per sample count:

- 8 train samples, 64 eval samples, three seeds:
  - Standard LoRA mean NLL: 2.6083.
  - Wake scheduled mean NLL: 2.0148.
- 16 train samples, 64 eval samples, three seeds:
  - Standard LoRA mean NLL: 1.8573.
  - Wake scheduled mean NLL: 1.7904.
- 32 train samples, 64 eval samples, three seeds:
  - Standard LoRA mean NLL: 1.6055.
  - Strong Wake scheduled mean NLL: 1.6124.
  - Segment-only Wake mean NLL: 1.6100.

Interpretation: the current Wake losses clearly help in the extreme 8/16-sample
regime, where standard LoRA overfits under equal update budget. At 32 samples,
the same regularization is already too strong. The `wake_budget` schedule keeps
the strong loss for 8/16 samples, disables KL above 16 samples, and decays the
segment term with sample count.

## Low-LR Fixed-Update Matrix

With 64 eval samples, three seeds, 32 optimizer updates, and
`learning_rate=1e-4`, the low-data matrix is substantially stronger than the
earlier `2e-4` runs:

| Train samples | Base | Standard LoRA | Wake-budget | Wake-delayed |
|---:|---:|---:|---:|---:|
| 8 | 1.9395 | 1.7609 | 1.7499 | 1.7389 |
| 16 | 1.9395 | 1.6562 | 1.6584 | 1.6579 |
| 32 | 1.9395 | 1.6262 | 1.6260 | 1.6263 |

`wake_delayed` lets the LoRA adapter adapt under ordinary CE for the first 25%
of optimizer updates, then linearly ramps Wake KL/segment regularization over
the next 12.5% of updates. This preserves most of standard LoRA's early
plasticity while using Wake geometry as a late-stage anti-drift constraint.

Interpretation:

- At 8 samples, Wake-delayed is the current best final model and reduces the
  final-best gap from 0.0703 for standard LoRA to 0.0474.
- At 16 samples, low learning rate nearly removes late overfitting; standard
  LoRA is best by about 0.0017 NLL, so strong KL/segment regularization is no
  longer helpful.
- At 32 samples, methods are effectively tied; the tiny segment-only budget is
  marginally best, but the difference is too small to overstate.

The next method direction should therefore be sample-aware rather than uniformly
strong: keep delayed Wake for the extreme 8-sample regime, reduce or disable KL
at 16 samples, and keep only a very light segment memory term when the sample
count is larger.

## V3: Gentle Sample-Aware Wake Schedule

The next controlled variant is `wake_gentle`. It treats Wake regularization as
an extreme-low-data stabilizer rather than a uniformly strong add-on:

```text
lambda_kl(n) = 0.1,                         if n <= 8
lambda_kl(n) = 0,                           if n > 8
lambda_segment(n) = 0.005,                  if n <= 8
lambda_segment(n) = 0.000625,               if 8 < n <= 16
lambda_segment(n) = 0,                      if n > 16
wake_start_ratio = 0.25 and ramp_ratio = 0.125, if n <= 8
wake_start_ratio = 0 and ramp_ratio = 0,         if n > 8
```

This directly tests the current interpretation of the low-LR matrix:

- 8 samples: keep the delayed KL+segment Wake geometry because it improves both
  final NLL and late-training drift.
- 16 samples: remove KL because standard LoRA is already strong at low learning
  rate; keep only a very weak segment memory term.
- 32+ samples: disable Wake regularization, so the method does not convert
  noise-level segment effects into a false claim.

The goal is not to force Wake to beat standard LoRA everywhere. A publishable
claim should be narrower and cleaner: Wake-style sample utilization improves
extreme low-data stability, and the strength of the memory geometry must decay
as the supervised sample count grows.

### 16-Sample Gentle Ablation

The 16-sample low-LR matrix showed that strong KL/segment regularization is not
needed. A focused three-seed ablation confirms that reducing the segment term is
the right move:

| Variant | Mean final NLL | Mean best NLL | Mean final-best gap |
|---|---:|---:|---:|
| Standard LoRA | 1.656166 | 1.655676 | 0.000489 |
| `wake_gentle`, segment 0.0025, delayed | 1.656489 | 1.655835 | 0.000654 |
| Segment 0.00125, delayed | 1.656596 | 1.655868 | 0.000728 |
| Segment 0.00125, no delay | 1.656548 | 1.655891 | 0.000657 |
| Segment 0.000625, no delay | 1.656199 | 1.655720 | 0.000479 |

The last variant is nearly indistinguishable from standard LoRA and avoids the
small penalty from stronger Wake terms. It is therefore the new default for
`wake_gentle` at 16 samples.

### 32-Sample Conservative Decision

The 32-sample low-LR comparisons are effectively tied. The strongest observed
difference is around `1e-4` NLL, and rerunning the same tiny segment setting can
move in either direction. For the next default schedule, the safer scientific
choice is to turn Wake off above 16 samples and report the positive result as an
extreme-low-data effect rather than overfit the method to noise.

### Clean V3 Low-LR Run

With the conservative `wake_gentle` default, the clean three-seed fixed-update
run gives:

| Train samples | Standard LoRA | Wake-gentle V3 | Delta |
|---:|---:|---:|---:|
| 8 | 1.760924 | 1.738797 | -0.022127 |
| 16 | 1.656166 | 1.656163 | -0.000003 |
| 32 | 1.626196 | 1.625987 | -0.000209 |

Only the 8-sample improvement should be described as substantive. The 16- and
32-sample rows show that the sample-aware schedule avoids the penalty from
over-regularizing once standard LoRA already has enough data. The 32-sample row
has Wake regularization disabled, so the tiny numerical difference should be
treated as run noise rather than a method win.

The loss implementation now skips the frozen-base forward pass whenever both
KL and CE-reuse are disabled. This keeps segment-only and CE-only ablations
faster without changing the active objective.
