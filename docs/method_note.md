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

## V4: Exploratory Sample-Utilization Losses

The V3 result is scientifically clean but too small to be the final method.
The next exploration keeps the same central claim, but attacks the sample
dropping behavior more directly:

```text
w_i = 1 + clip(lambda_self_reuse / (CE_lora_i + eps), 0, w_max)
CE_self_reuse = sum_i w_i CE_lora_i / sum_i w_i
```

`self_reuse` gives extra normalized gradient mass to tokens the current LoRA
model already handles well. These tokens are exactly the ones that ordinary CE
quickly lets go silent. A delayed schedule is now applied to self-reuse and
consistency losses as well as KL/segment losses, so the adapter first learns
under ordinary CE and only then turns on the anti-dropping terms.

The second branch is a two-view consistency loss:

```text
L_cons = 0.5 * KL(P_lora_dropout_1 || P_lora_dropout_2)
       + 0.5 * KL(P_lora_dropout_2 || P_lora_dropout_1)
```

This treats one scarce training sample as two stochastic LoRA views and asks
the adapter to preserve its prediction geometry across views. It is closer to
"using each sample more than once" than a pure base-model KL anchor, while still
being compatible with the Wake segment memory.

Early single-seed probes:

- `self_reuse` with `lambda=0.05` improves the 8-sample seed-42 final NLL, but
  too much weight (`0.1` or `0.2`) increases late drift.
- `consistency` with `lambda=0.5` or `1.0` is stable and competitive at 8/16
  samples, but it needs multi-seed validation because it doubles the LoRA
  forward pass.
- `hard_cap`, which downweights high-CE tokens directly, is a useful negative
  result: it harms 8-sample training badly and contradicts the goal of better
  sample utilization.

The current V4 screening methods are:

- `wake_self_reuse_delayed`
- `wake_consistency_delayed`
- `wake_reuse_consistency_delayed`
- `wake_reuse_consistency_segment`
- `wake_gentle_self_reuse`
- `wake_gentle_consistency`
- `wake_gentle_reuse_consistency`

These variants are intentionally more aggressive than V3. The goal is to learn
whether a clearly distinct LoRA objective can beat standard LoRA by more than
the small regularization gains seen so far.

The `wake_gentle_*` variants keep the V3 KL+segment stabilizer and add V4
sample-reuse terms. They test whether V4 failed because the new terms are weak,
or because removing the V3 anti-drift anchor made final-model overfitting worse.

## V5: Sample-Aware Utilization Schedule

The best V4 signal is not one uniform loss. It is a sample-aware mixture:

```text
if n <= 8:
    KL = 0.1
    segment = 0.005
    self_reuse = 0.025
    consistency = 0.5
    delay/ramp = 0.25 / 0.125
elif n <= 16:
    KL = 0
    segment = 0
    self_reuse = 0
    consistency = 0.5
else:
    all Wake-utilization terms = 0
```

This is implemented as `wake_utilization`. The rationale is empirical and
mechanistic:

- At 8 samples, the adapter needs the V3 anti-drift anchor plus explicit
  token/view reuse. This gives a clearer improvement over standard LoRA than
  V3 alone.
- At 16 samples, self-reuse starts to overweight easy tokens and hurts NLL.
  Consistency alone remains stable.
- At 32 samples and above, ordinary LoRA is already strong under the current
  low learning-rate fixed-update protocol, so forcing Wake terms risks turning
  noise into a false method claim.

### V5 Final Low-LR Matrix

Using 64 eval samples, three seeds, 32 optimizer updates, and
`learning_rate=1e-4`:

| Train samples | Standard LoRA | Wake-gentle V3 | Wake-utilization V5 |
|---:|---:|---:|---:|
| 8 | 1.760924 | 1.738797 | 1.723933 |
| 16 | 1.656166 | 1.656163 | 1.656016 |
| 32 | 1.626196 | 1.625987 | 1.625920 |

The strongest result is still the extreme 8-sample regime, where V5 improves
standard LoRA by `0.036991` NLL and V3 by `0.014864` NLL. V5 also cuts the
8-sample final-best gap from `0.070286` for standard LoRA to `0.033167`, which
supports the intended claim: the method improves final-model sample utilization
and late-training stability rather than only finding a better early checkpoint.

The 16- and 32-sample rows should be reported conservatively. They show that the
sample-aware schedule does not hurt when ordinary LoRA already has enough
signal; they are not the main claim.

## V6: High-Dimensional Simplex Probes

The one-dimensional segment term may be too narrow for language modeling. A
higher-dimensional translation was added in two forms:

```text
hidden simplex:
    anchors = {target token embedding} union {base top-k token embeddings}
    loss = distance(hidden, soft_projection_to_convex_hull(anchors))

probability simplex CE:
    q = (1 - mix) * one_hot(target) + mix * P_base(top-k)
    loss = CE(P_lora restricted to local top-k simplex, q)
```

Implementation methods:

- `wake_simplex`
- `wake_utilization_simplex`
- `wake_simplex_ce`
- `wake_utilization_simplex_ce`

### V6 Probe Findings

The hidden-simplex projection is stable but weak. On 8 samples, three seeds,
`learning_rate=1e-4`, 32 updates, `top_k=16`, `temperature=0.5`, and
`lambda_simplex=0.01`:

| Method | Mean final NLL | Mean best NLL | Mean final-best gap |
|---|---:|---:|---:|
| Wake-utilization V5 | 1.723933 | 1.690766 | 0.033167 |
| Wake-utilization + hidden simplex | 1.723592 | 1.691019 | 0.032573 |

This is a real but tiny change (`0.000341` NLL). It should be recorded as a
weak positive diagnostic, not as a main method contribution.

The probability-simplex CE variant is negative under the tested settings. With
seed 42, `lambda_simplex_ce` values of `0.025`, `0.05`, and `0.1` all make
Wake-utilization worse. Reducing `label_mix` to `0.05` or `0.1` still does not
beat V5. The likely reason is that base top-k probability mass introduces
neighbor tokens that are plausible to the base model but not necessarily useful
for the supervised medical target.

## Stronger Baseline Sanity Check

The earlier V5 matrix used `learning_rate=1e-4`. A stricter LR sanity check
shows that standard LoRA becomes much stronger at lower LR:

| 8-sample setting | Standard LoRA | Wake-utilization |
|---|---:|---:|
| 32 updates, `lr=1e-4` | 1.760924 | 1.723933 |
| 32 updates, `lr=7.5e-5` | 1.708081 | 1.706623 |
| 32 updates, `lr=5e-5` | 1.705007 | 1.712501 |

This weakens the original V5 claim. Against a strongly tuned standard LoRA,
the current Wake-utilization schedule is no longer clearly better at 32
updates. This should guide the next round: the method must beat a tuned
standard baseline, not only the first low-LR baseline.

Wake-utilization is still clearly useful in the long-budget overfitting regime:

| 8-sample setting | Standard LoRA | Wake-utilization |
|---|---:|---:|
| 48 updates, `lr=5e-5`, strong Wake | 1.716109 | 1.711938 |
| 64 updates, `lr=5e-5`, default Wake | 1.849259 | 1.745226 |
| 64 updates, `lr=5e-5`, strong Wake | 1.849259 | 1.713941 |

The 64-update result supports a narrower but more robust claim: Wake-style
sample utilization strongly reduces late-training drift when the tiny training
set is reused many times. It is not yet enough for the final paper claim,
because tuned 32-update standard LoRA remains competitive.

Next directions:

- Search for a schedule that keeps the 32-update standard LoRA plasticity while
  inheriting the 48/64-update Wake stability.
- Try optimizer-state or adapter-weight averaging as a fair no-validation
  alternative to best-checkpoint selection.
- Validate on a second small medical dataset before treating any single-dataset
  improvement as publishable.
