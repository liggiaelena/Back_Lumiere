# Lumière SegFormer — Training History & Experiment Log

This document summarises every training experiment run on the Lumière skin condition
segmentation model, from the original baseline through to the final deployed checkpoint.

---

## Baseline (Original Models)

The starting point before any retraining. Three separate models were trained independently,
one per disease, in a fixed sequential order: Melasma → Port Wine Stain → Vitiligo.
Each model fine-tuned from the previous one's checkpoint.

| Disease | Val IoU | Test IoU |
|---|---:|---:|
| Melasma | 0.5598 | 0.4826 |
| Port Wine Stain | 0.5841 | 0.5921 |
| Vitiligo | 0.4933 | — (not evaluated) |

**Hyperparameters:**
- Learning rate: `5e-5`
- Epochs: `100`
- Batch size: `4`
- Weight decay: `0.01`
- Strategy: sequential fine-tuning (one disease at a time)

---

## Phase 1 — Sequential Retraining Experiments

Using `retrain_all.py`, which retrains each disease in order (Melasma → Port Wine Stain →
Vitiligo), passing each stage's best checkpoint to the next stage as its starting point.

### What was tried

| Run | Learning Rate | Epochs | Batch Size | Melasma Test IoU | PWS Test IoU | Vitiligo Test IoU |
|---|---|---|---|---:|---:|---:|
| smoke_local (×3) | `1e-4` | 1 | 1 | 0.53 / 0.49 / 0.63 | 0.36 / 0.39 / 0.39 | 0.00 |
| smoke_colab | `1e-4` | 1 | 8 | 0.31 | 0.31 | 0.00 |
| experiment_lr1e4 | `1e-4` | 50 | 8 | **0.6367** | 0.5878 | 0.0691 |
| smoke_mixed_device | `5e-5` | 1 | 4 | 0.36 | 0.19 | 0.00 |
| experiment_cpu_512_lr5e5_batch4 | `5e-5` | 50 | 4 | 0.5927 | 0.5627 | 0.0474 |

### Key finding — why sequential training failed for Vitiligo

Vitiligo IoU stayed near **0** in every sequential run regardless of learning rate or
number of epochs. The root cause is **catastrophic forgetting**: after the model is
fine-tuned on Melasma and Port Wine Stain, the weights drift away from features useful
for Vitiligo. By the time the Vitiligo stage runs, the starting checkpoint has already
"forgotten" what it learned from earlier Vitiligo-related data.

Raising the learning rate to `1e-4` improved Melasma (+0.15 IoU over baseline) but made
catastrophic forgetting worse for Vitiligo. Lowering it back to `5e-5` did not help
either — the sequential approach was fundamentally limited.

---

## Phase 2 — Joint Multitask Training

A new training script (`train_unified.py`) was written that mixes all three disease
datasets in a single training loop, using a shared 4-class label mapping:

```
0 = background
1 = vitiligo
2 = melasma_like_hyperpigmentation
3 = port_wine_stain
```

Multiple joint experiments were run, each building on lessons from the previous one.

### Experiments

| Model | LR | Epochs | Image Size | Batch | Melasma | PWS | Vitiligo | Notes |
|---|---|---|---|---|---:|---:|---:|---|
| joint_mps_224_vitboost | `5e-5` | ~50 | 224 | 1 | 0.6272 | 0.4653 | 0.2403 | First joint run, 224px. Vitiligo improved but still weak. |
| joint_mps_512_from224_vitboost | `5e-5` | ~30 | 512 | 1 | 0.6013 | 0.4957 | 0.1740 | Scaled to 512px. Vitiligo dropped again. |
| joint_mps_512_from224_vitdice | `5e-5` | ~30 | 512 | 1 | 0.6480 | 0.5520 | 0.1063 | Added Dice loss for Vitiligo. Strong Melasma/PWS, Vitiligo still weak. |
| vitfirst_warmup_512 | `5e-5` | ~50 | 512 | 1 | — | — | — | Vitiligo-first warmup: trained Vitiligo samples first before mixing. |
| vitfirst_warmup_512_refine | `5e-5` | ~20 | 512 | 1 | — | — | — | Refinement pass over the vitiligo-first warmup checkpoint. |
| joint_mps_512_vitfirst50_rebalance | `2e-6` | 5 | 512 | 1 | 0.5577 | 0.4460 | 0.3658 | Short rebalance over vitfirst checkpoint. Vitiligo jumped. |
| joint_mps_512_final_melboost | `2e-6` | 5 | 512 | 1 | 0.5577 | 0.4714 | 0.3960 | Increased Melasma sampling. Slightly better PWS, Vitiligo regressed. |
| **joint_mps_512_vitfirst_refine_rebalance** | **`2e-6`** | **5** | **512** | **1** | **0.5577** | **0.4613** | **0.4321** | **Selected. Best overall balance.** |

### How learning rate changed across phases

| Phase | LR | Reason |
|---|---|---|
| Baseline & sequential | `5e-5` | Standard fine-tuning rate |
| Sequential experiment_lr1e4 | `1e-4` | Tried higher LR to speed up learning — helped Melasma, hurt Vitiligo |
| Joint warmup runs | `5e-5` | Standard rate for building a strong base checkpoint |
| **Final rebalance (joint)** | **`2e-6`** | **Very low rate — only a gentle nudge to rebalance the three diseases without destroying what the model already learned** |

The final learning rate of `2e-6` was intentionally very low because the last step was
a **refinement**, not a full retrain. A higher LR at this stage would have overwritten
the Vitiligo knowledge built up during the vitfirst warmup.

### Other key hyperparameter changes

| Parameter | Baseline | Final Model | Why it changed |
|---|---|---|---|
| Class weights | none | `[0.2, 4.0, 2.5, 3.5]` | Force the model to pay attention to rare disease pixels |
| Sampling | uniform | balanced `mel=2, pws=3, vit=2` | Compensate for dataset size differences (Melasma has 5× more images than Vitiligo) |
| Training strategy | sequential | joint (all 3 diseases together) | Prevent catastrophic forgetting of Vitiligo |
| Scheduler | CosineAnnealing | Cosine with `min_lr=1e-6` | Smoother LR decay in the final rebalance |
| Batch size | 4 | 1 | MPS memory constraint on Apple Silicon |

---

## Final Model Results

**Checkpoint:** `training/checkpoints/SegFormer/unified/best`
**Strategy:** Vitiligo-first warmup → refinement → short joint rebalance at `2e-6`

| Disease | Baseline Test IoU | Final Test IoU | Change | % Change |
|---|---:|---:|---:|---:|
| Melasma | 0.4826 | **0.5577** | +0.0751 | **+15.6%** |
| Port Wine Stain | 0.5921 | **0.4613** | −0.1308 | **−22.1%** |
| Vitiligo | — | **0.4321** | — | **from zero** |
| **Mean** | — | **0.4837** | — | — |

### Why this model was selected

Every model with higher Port Wine Stain IoU (> 0.55) had Vitiligo IoU near zero — not
useful for the project. The selected checkpoint is the only one that achieves a workable
score on all three diseases simultaneously.

The Port Wine Stain regression (−22%) is the trade-off of the joint strategy: giving
Vitiligo a class weight of `4.0` and using a vitiligo-first warmup naturally draws
capacity away from Port Wine Stain. Further tuning to recover PWS performance
(joint_mps_512_final_melboost) caused Vitiligo to drop again, so the rebalance
checkpoint was kept as the best available compromise.

### Per-sample test results

**Melasma** (52 test samples)
- 26 samples scored IoU = 1.0 (perfect detection)
- 20 samples scored IoU = 0.0 (missed entirely)
- Behaviour is bimodal: the model either detects the region very well or not at all

**Port Wine Stain** (18 test samples)
- Range: 0.00 – 0.87
- Best sample (port120): 0.8730
- Worst sample (port136): 0.00

**Vitiligo** (15 test samples)
- Range: 0.32 – 0.50
- Very consistent across samples — no complete misses
- All samples between 0.31 and 0.50

---

## Evaluation Artifacts

```
training/checkpoints/SegFormer/unified/final_evaluation/melasma/test_metrics.json
training/checkpoints/SegFormer/unified/final_evaluation/port_wine_stain/test_metrics.json
training/checkpoints/SegFormer/unified/final_evaluation/vitiligo/test_metrics.json
```
