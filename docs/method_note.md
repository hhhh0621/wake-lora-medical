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
