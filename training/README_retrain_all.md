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
  --learning-rate 1e-4 \
  --epochs 150 \
  --batch-size 4 \
  --weight-decay 0.01 \
  --run-name experiment_lr1e4
```

Use `--device cuda` on a CUDA machine:

```bash
python training/retrain_all.py \
  --learning-rate 1e-4 \
  --epochs 150 \
  --batch-size 4 \
  --weight-decay 0.01 \
  --run-name experiment_lr1e4 \
  --device cuda
```

On Apple Silicon Mac, use `--device auto` for CPU smoke tests. CUDA is only for NVIDIA GPU environments such as Colab. MPS can be requested explicitly with `--device mps`, but SegFormer training may fail on some PyTorch/MPS builds during backward pass.

```bash
python training/retrain_all.py \
  --learning-rate 1e-4 \
  --epochs 1 \
  --batch-size 1 \
  --weight-decay 0.01 \
  --run-name smoke_local \
  --image-size 224 \
  --num-workers 0 \
  --device auto
```

For a quick path check, run a one-epoch smoke test:

```bash
python training/retrain_all.py \
  --learning-rate 1e-4 \
  --epochs 1 \
  --batch-size 1 \
  --weight-decay 0.01 \
  --run-name smoke_test \
  --image-size 224 \
  --num-workers 0 \
  --device auto
```

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
  --learning-rate 1e-4 \
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
  --learning-rate 1e-4 \
  --epochs 150 \
  --batch-size 4 \
  --weight-decay 0.01 \
  --run-name experiment_lr1e4 \
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
