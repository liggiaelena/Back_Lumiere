# Hyperparameter Experiment Report

Generated from `training/experiments.json`.

## Melasma

| Run | Learning Rate | Epochs | Batch Size | Weight Decay | Val IoU | Test IoU | Delta Test IoU |
|-----|---------------|--------|------------|--------------|---------|----------|----------------|
| baseline | 5e-05 | 100 | 4 | 0.0100 | 0.5598 | 0.4826 | 0.0000 |
| smoke_local | 1e-04 | 1 | 1 | 0.0100 | 0.6337 | 0.5292 | +0.0466 |
| smoke_local | 1e-04 | 1 | 1 | 0.0100 | 0.4588 | 0.4868 | +0.0041 |
| smoke_local | 1e-04 | 1 | 1 | 0.0100 | 0.5731 | 0.6294 | +0.1467 |
| smoke_colab | 1e-04 | 1 | 8 | 0.0100 | 0.3144 | 0.3148 | -0.1679 |
| experiment_lr1e4 | 1e-04 | 50 | 8 | 0.0100 | 0.5498 | 0.6367 | +0.1540 |
| smoke_mixed_device | 5e-05 | 1 | 4 | 0.0100 | 0.3456 | 0.3567 | -0.1259 |
| experiment_cpu_512_lr5e5_batch4 | 5e-05 | 50 | 4 | 0.0100 | 0.5868 | 0.5927 | +0.1100 |

**Best run:** experiment_lr1e4 (+0.1540 test IoU vs baseline)

## Port Wine Stain

| Run | Learning Rate | Epochs | Batch Size | Weight Decay | Val IoU | Test IoU | Delta Test IoU |
|-----|---------------|--------|------------|--------------|---------|----------|----------------|
| baseline | 5e-05 | 100 | 4 | 0.0100 | 0.5841 | 0.5921 | 0.0000 |
| smoke_local | 1e-04 | 1 | 1 | 0.0100 | 0.3765 | 0.3600 | -0.2321 |
| smoke_local | 1e-04 | 1 | 1 | 0.0100 | 0.4049 | 0.3855 | -0.2066 |
| smoke_local | 1e-04 | 1 | 1 | 0.0100 | 0.3672 | 0.3917 | -0.2004 |
| smoke_colab | 1e-04 | 1 | 8 | 0.0100 | 0.4421 | 0.3101 | -0.2820 |
| experiment_lr1e4 | 1e-04 | 50 | 8 | 0.0100 | 0.5065 | 0.5878 | -0.0044 |
| smoke_mixed_device | 5e-05 | 1 | 4 | 0.0100 | 0.2011 | 0.1937 | -0.3984 |
| experiment_cpu_512_lr5e5_batch4 | 5e-05 | 50 | 4 | 0.0100 | 0.5394 | 0.5627 | -0.0294 |

**Best run:** baseline (0.0000 test IoU vs baseline)

## Vitiligo

| Run | Learning Rate | Epochs | Batch Size | Weight Decay | Val IoU | Test IoU | Delta Test IoU |
|-----|---------------|--------|------------|--------------|---------|----------|----------------|
| baseline | 5e-05 | 100 | 4 | 0.0100 | 0.4933 | - | - |
| smoke_local | 1e-04 | 1 | 1 | 0.0100 | 0e+00 | 0e+00 | - |
| smoke_local | 1e-04 | 1 | 1 | 0.0100 | 0e+00 | 0e+00 | - |
| smoke_local | 1e-04 | 1 | 1 | 0.0100 | 0e+00 | 0e+00 | - |
| smoke_colab | 1e-04 | 1 | 8 | 0.0100 | 0e+00 | 0e+00 | - |
| experiment_lr1e4 | 1e-04 | 50 | 8 | 0.0100 | 0.0904 | 0.0691 | - |
| smoke_mixed_device | 5e-05 | 1 | 4 | 0.0100 | 0e+00 | 0e+00 | - |
| experiment_cpu_512_lr5e5_batch4 | 5e-05 | 50 | 4 | 0.0100 | 0.0625 | 0.0474 | - |

**Best run:** experiment_lr1e4

## Final Unified Model Selection

The tables above contain historical stage-checkpoint metrics. They do not represent a single final checkpoint that performs all three segmentation tasks at once. The final deployable unified checkpoint is selected from the joint/rebalance experiments below.

**Final deployable checkpoint:**

```text
training/checkpoints/SegFormer/unified/best
```

Source experiment run:

```text
training/checkpoints/SegFormer/unified/joint_mps_512_vitfirst_refine_rebalance/best
```

This checkpoint was produced by:

1. Training a vitiligo-first warmup checkpoint at 512px.
2. Refining the vitiligo warmup checkpoint.
3. Running a short joint rebalance over melasma, port wine stain, and vitiligo.

### Final Test Metrics

| Model | Image Size | Melasma Test IoU | Port Wine Stain Test IoU | Vitiligo Test IoU | Mean Test IoU |
|---|---:|---:|---:|---:|---:|
| joint_mps_512_vitfirst_refine_rebalance/best | 512 | 0.5577 | 0.4613 | 0.4321 | 0.4837 |

### Final Model Comparison

| Model | Melasma Test IoU | Port Wine Stain Test IoU | Vitiligo Test IoU | Notes |
|---|---:|---:|---:|---|
| joint_mps_512_vitfirst_refine_rebalance/best | 0.5577 | 0.4613 | 0.4321 | Selected final model; best overall balance. |
| joint_mps_512_final_melboost/best | 0.5577 | 0.4714 | 0.3960 | Slightly better PWS, weaker vitiligo. Not selected. |
| joint_mps_512_vitfirst50_rebalance/best | 0.5577 | 0.4460 | 0.3658 | Earlier vitiligo-first rebalance. |
| joint_mps_512_from224_vitdice/best | 0.6480 | 0.5520 | 0.1063 | Strong melasma/PWS, weak vitiligo. Not selected. |
| joint_mps_512_from224_vitboost/best | 0.6013 | 0.4957 | 0.1740 | More balanced than direct 512, but weaker vitiligo. |
| joint_mps_224_vitboost/best | 0.6272 | 0.4653 | 0.2403 | Best 224px balanced model, not selected for 512px final. |

### Selection Rationale

`joint_mps_512_vitfirst_refine_rebalance/best` is selected because it gives the strongest vitiligo performance among unified 512px checkpoints while keeping port wine stain above 0.45 and melasma stable. Later melasma-focused tuning improved PWS slightly but reduced vitiligo, so it was not selected.

### Final Evaluation Artifacts

Canonical final evaluation artifacts:

```text
training/checkpoints/SegFormer/unified/final_evaluation/melasma/test_metrics.json
training/checkpoints/SegFormer/unified/final_evaluation/port_wine_stain/test_metrics.json
training/checkpoints/SegFormer/unified/final_evaluation/vitiligo/test_metrics.json
```

Source experiment directory before promotion:

```text
training/checkpoints/SegFormer/unified/joint_mps_512_vitfirst_refine_rebalance/evaluation/
```
