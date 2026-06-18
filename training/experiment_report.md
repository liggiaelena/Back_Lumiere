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

**Best run:** experiment_lr1e4

