# Unified SegFormer Retraining

This guide explains how to use `training/retrain_all.py` for Task218.

The script retrains the three skin-condition segmentation models in the required order:

1. Melasma
2. Port Wine Stain
3. Vitiligo

Each stage uses the previous stage's `best` checkpoint as its starting checkpoint.

## Repository Inputs

The default dataset paths are:

| Disease | Dataset path |
|---|---|
| Melasma | `data-collection/melasma` |
| Port Wine Stain | `data-collection/port_wine_stain/processed` |
| Vitiligo | `data-collection/vitiligo` |

The default initial checkpoint is:

```text
training/checkpoints/SegFormer/segformer_b2_melasma_colab_export
```

The default output checkpoints are:

```text
training/checkpoints/SegFormer/unified/tmp/melasma/best
training/checkpoints/SegFormer/unified/tmp/port_wine_stain/best
training/checkpoints/SegFormer/unified/tmp/vitiligo/best
training/checkpoints/SegFormer/unified/best
```

By default, these generated outputs are written under:

```text
training/checkpoints/SegFormer/unified/tmp/
```

`unified/best` is the final deployable checkpoint copied from `unified/tmp/vitiligo/best` after the full sequential retraining run completes.

## Local Usage

Run from the repository root:

```bash
python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 50 \
  --batch-size 4 \
  --weight-decay 0.01 \
  --run-name experiment_lr5e5_batch4
```

Use `--device cuda` on a CUDA machine:

```bash
python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 50 \
  --batch-size 4 \
  --weight-decay 0.01 \
  --run-name experiment_lr5e5_batch4 \
  --device cuda
```

On Apple Silicon Mac, use `--device auto` for CPU smoke tests. CUDA is only for NVIDIA GPU environments such as Colab. MPS can be requested with `--device mps`; the training losses use an explicit reshape workaround for PyTorch/MPS backward stride issues, but you should still run a one-epoch smoke test first.

```bash
python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 1 \
  --batch-size 1 \
  --weight-decay 0.01 \
  --run-name smoke_local \
  --image-size 224 \
  --num-workers 0 \
  --device auto
```

For local Apple Silicon training, use CPU with a smaller image size first:

```bash
python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 20 \
  --batch-size 1 \
  --weight-decay 0.01 \
  --run-name experiment_local_cpu_lr5e5 \
  --image-size 224 \
  --num-workers 0 \
  --device auto
```

For full 512px training, Colab CUDA is still the recommended environment.

## Recommended Unified Joint Training

`retrain_all.py` is a sequential fine-tuning workflow. It is useful for reproducing the ordered Task218 run, but its final exported checkpoint is copied from the last vitiligo stage. The per-disease rows in `experiment_report.md` are stage-checkpoint metrics, not proof that the final `unified/best` model keeps all three disease classes at the same time.

For a deployable four-class model, use the joint trainer instead:

```bash
python training/SegFormer/unified/train_unified.py \
  --checkpoint training/checkpoints/SegFormer/segformer_b2_4class_port_wine_stain_finetune/best \
  --output-dir training/checkpoints/SegFormer/unified/joint \
  --image-size 512 \
  --batch-size 2 \
  --epochs 50 \
  --learning-rate 5e-5 \
  --weight-decay 0.01 \
  --device cuda
```

The joint trainer mixes melasma, port wine stain, and vitiligo samples in one training loop with the shared label mapping:

```text
0 = background
1 = vitiligo
2 = melasma_like_hyperpigmentation
3 = port_wine_stain
```

It saves `best` by a combined validation score over all three validation IoUs, with a small weight on the weakest class IoU so a checkpoint that collapses one disease class is less likely to be selected.

For local Apple Silicon training, start with a smaller MPS smoke run:

```bash
python training/SegFormer/unified/train_unified.py \
  --output-dir training/checkpoints/SegFormer/unified/joint_mps_smoke \
  --image-size 224 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --epochs 2 \
  --learning-rate 5e-5 \
  --num-workers 0 \
  --device mps
```

If that is stable, run a longer local MPS experiment:

```bash
python training/SegFormer/unified/train_unified.py \
  --output-dir training/checkpoints/SegFormer/unified/joint_mps_512_lr5e5 \
  --image-size 512 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --epochs 50 \
  --learning-rate 5e-5 \
  --weight-decay 0.01 \
  --num-workers 0 \
  --device mps
```

If MPS runs out of memory, reduce `--image-size` to `384` or `320`. If an operation fails on MPS in your local PyTorch build, rerun the same command with `--device cpu` to verify the training path.

## Per-Stage Hyperparameters

`retrain_all.py` accepts global hyperparameters and optional per-stage overrides. If an override is omitted, that disease uses the global value.

Useful vitiligo-safe example:

```bash
python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 50 \
  --batch-size 4 \
  --weight-decay 0.01 \
  --run-name experiment_stage_safe_lr5e5_batch4 \
  --vitiligo-learning-rate 5e-5 \
  --vitiligo-epochs 100 \
  --vitiligo-batch-size 4 \
  --image-size 512 \
  --device cuda
```

Available override groups:

```text
--melasma-learning-rate
--melasma-epochs
--melasma-batch-size
--melasma-weight-decay
--melasma-device

--port-wine-stain-learning-rate
--port-wine-stain-epochs
--port-wine-stain-batch-size
--port-wine-stain-weight-decay
--port-wine-stain-device

--vitiligo-learning-rate
--vitiligo-epochs
--vitiligo-batch-size
--vitiligo-weight-decay
--vitiligo-device
```

If one stage fails on MPS, you can mix devices. For example, run melasma on CPU and later stages on MPS:

```bash
python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 20 \
  --batch-size 4 \
  --weight-decay 0.01 \
  --run-name experiment_mixed_device \
  --image-size 224 \
  --num-workers 0 \
  --device mps \
  --melasma-device auto
```

For a quick path check, run a one-epoch smoke test:

```bash
python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 1 \
  --batch-size 1 \
  --weight-decay 0.01 \
  --run-name smoke_test \
  --image-size 224 \
  --num-workers 0 \
  --device auto
```

## Resume Interrupted Training

Each training stage writes a resumable checkpoint after every epoch:

```text
training/checkpoints/SegFormer/unified/tmp/melasma/training_state.pt
training/checkpoints/SegFormer/unified/tmp/port_wine_stain/training_state.pt
training/checkpoints/SegFormer/unified/tmp/vitiligo/training_state.pt
```

`training_state.pt` contains the model weights, optimizer state, scheduler state when used, best validation IoU, and training history. If a run is interrupted, rerun the same command with `--resume-training`:

```bash
python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 50 \
  --batch-size 4 \
  --weight-decay 0.01 \
  --run-name experiment_lr5e5_batch4 \
  --image-size 512 \
  --num-workers 0 \
  --device auto \
  --resume-training
```

Use the same output directory, image size, batch size, learning rate, and model architecture when resuming. Increasing `--epochs` is OK if you want to continue training longer; decreasing it below the saved epoch will skip further training for stages that already reached that epoch.

## Outputs

Each successful run appends one entry to:

```text
training/experiments.json
```

The comparison report is regenerated at:

```text
training/experiment_report.md
```

Each disease also writes:

```text
training/checkpoints/SegFormer/unified/tmp/<disease>/training_history.json
training/checkpoints/SegFormer/unified/tmp/<disease>/training_state.pt
training/checkpoints/SegFormer/unified/tmp/<disease>/evaluation/test_metrics.json
training/checkpoints/SegFormer/unified/tmp/<disease>/evaluation/overlays/
```

The final unified model is exported to:

```text
training/checkpoints/SegFormer/unified/best
```

## Colab Package

From the repository root, create a training package:

```bash
zip -r lumiere_retrain_pack.zip \
  training/retrain_all.py \
  training/SegFormer/melasma \
  training/SegFormer/port_wine_stain \
  training/SegFormer/vitiligo \
  training/experiments.json \
  training/experiment_report.md \
  training/checkpoints/SegFormer/segformer_b2_melasma_colab_export \
  data-collection/melasma \
  data-collection/port_wine_stain/processed \
  data-collection/vitiligo \
  requirements.txt \
  -x "*/__pycache__/*" "*.DS_Store"
```

Upload `lumiere_retrain_pack.zip` to Colab, then run the cells in:

```text
training/colab_retrain_all.ipynb
```

## Colab Commands

After uploading and unzipping the package in Colab:

```bash
%cd /content/Back_Lumiere
!pip install -q torch torchvision transformers safetensors pillow numpy tqdm
```

Run a smoke test:

```bash
!python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 1 \
  --batch-size 2 \
  --weight-decay 0.01 \
  --run-name smoke_colab \
  --device cuda \
  --num-workers 2
```

Run the full experiment:

```bash
!python training/retrain_all.py \
  --learning-rate 5e-5 \
  --epochs 50 \
  --batch-size 4 \
  --weight-decay 0.01 \
  --run-name experiment_lr5e5_batch4 \
  --device cuda \
  --num-workers 2
```

Download results:

```bash
!zip -r lumiere_retrain_results.zip \
  training/checkpoints/SegFormer/unified \
  training/experiments.json \
  training/experiment_report.md

from google.colab import files
files.download("lumiere_retrain_results.zip")
```

## After Colab Training

After the Colab run finishes, download `lumiere_retrain_results.zip` and copy these artifacts back into the repository:

```text
training/checkpoints/SegFormer/unified/best
training/experiments.json
training/experiment_report.md
```

These are the files/directories that should replace the local versions:

| Colab result | Local repository target | Commit? |
|---|---|---|
| `training/checkpoints/SegFormer/unified/best/` | `training/checkpoints/SegFormer/unified/best/` | Yes |
| `training/experiments.json` | `training/experiments.json` | Yes |
| `training/experiment_report.md` | `training/experiment_report.md` | Yes |

Do not commit intermediate training outputs:

```text
training/checkpoints/SegFormer/unified/tmp/
```

That directory is ignored by `.gitignore` and is only used for melasma, port wine stain, and vitiligo intermediate checkpoints during retraining.

Before committing the final model, verify that the model weight file is tracked by Git LFS:

```bash
git check-attr filter -- training/checkpoints/SegFormer/unified/best/model.safetensors
```

Expected output:

```text
training/checkpoints/SegFormer/unified/best/model.safetensors: filter: lfs
```

Then stage the final model and experiment artifacts:

```bash
git add training/checkpoints/SegFormer/unified/best \
        training/experiments.json \
        training/experiment_report.md
```

## Notes

- Always run the one-epoch smoke test first in Colab.
- If Colab runs out of GPU memory, reduce `--batch-size` to `2`.
- Do not rename the dataset folders inside the zip unless you also pass custom dataset paths to `retrain_all.py`.
- The script appends experiments instead of overwriting existing runs.
