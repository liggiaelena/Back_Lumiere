# Lumiere SegFormer-B2 Port Wine Stain Finetune

## Task
Finetune SegFormer from the existing Lumiere SegFormer-B2 checkpoint for pixel-level segmentation of vascular red/purple port wine stain regions.

## Model
- Architecture: SegFormer-B2 semantic segmentation
- Starting checkpoint: `training/checkpoints/SegFormer/segformer_b2_melasma_colab_export`
- Finetuned checkpoint: `best/`
- Final epoch checkpoint: `last/`

## Classes
- 0 = background
- 1 = vitiligo
- 2 = melasma_like_hyperpigmentation
- 3 = port_wine_stain

## Dataset
- Source: CVAT `Segmentation mask 1.1` export
- Original image source: https://www.kaggle.com/datasets/roshni2404/rare-skin-disease-dataset
- Processed dataset: `training/datasets/port_wine_stain/processed`
- Total paired samples: 136
- Split:
  - Train: 100
  - Validation: 18
  - Test: 18
- Processed mask values:
  - 0 = background
  - 3 = port_wine_stain

## Training
- Environment: Google Colab GPU
- Image size: 512
- Batch size: 4
- Epochs: 50
- Learning rate: 5e-5
- Optimizer: AdamW
- Best validation checkpoint selection: highest validation port wine stain IoU

## Colab Reproduction Steps
From the project root, create the upload package:

```bash
zip -r port_wine_stain_training_pack.zip \
  training/SegFormer/train_port_wine_stain.py \
  training/datasets/port_wine_stain/processed \
  training/checkpoints/SegFormer/segformer_b2_melasma_colab_export \
  requirements.txt
```

In Google Colab:

1. Create a new notebook.
2. Select `Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU`.
3. Upload `port_wine_stain_training_pack.zip`.
4. Unzip and enter the project directory:

```bash
!rm -rf /content/Back_Lumiere
!unzip -q port_wine_stain_training_pack.zip -d /content/Back_Lumiere
%cd /content/Back_Lumiere
```

Install training dependencies:

```bash
!pip install -q torch torchvision transformers safetensors pillow numpy tqdm
```

Verify CUDA:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Run a one-epoch smoke test:

```bash
!python training/SegFormer/train_port_wine_stain.py \
  --dataset-dir training/datasets/port_wine_stain/processed \
  --checkpoint training/checkpoints/SegFormer/segformer_b2_melasma_colab_export \
  --output-dir training/checkpoints/SegFormer/segformer_b2_4class_port_wine_stain_finetune_smoke_test \
  --image-size 512 \
  --batch-size 4 \
  --epochs 1 \
  --learning-rate 5e-5 \
  --device cuda
```

Run full training:

```bash
!python training/SegFormer/train_port_wine_stain.py \
  --dataset-dir training/datasets/port_wine_stain/processed \
  --checkpoint training/checkpoints/SegFormer/segformer_b2_melasma_colab_export \
  --output-dir training/checkpoints/SegFormer/segformer_b2_4class_port_wine_stain_finetune \
  --image-size 512 \
  --batch-size 4 \
  --epochs 50 \
  --learning-rate 5e-5 \
  --device cuda
```

After training, package the result:

```bash
!zip -r segformer_b2_4class_port_wine_stain_finetune.zip \
  training/checkpoints/SegFormer/segformer_b2_4class_port_wine_stain_finetune
```

Download from Colab:

```python
from google.colab import files

files.download("segformer_b2_4class_port_wine_stain_finetune.zip")
```

The Colab notebook used for this workflow is available at `training/SegFormer/colab_port_wine_stain_training.ipynb`.

## Results
- Best validation port wine stain IoU: 0.5841 at epoch 21
- Final epoch validation port wine stain IoU: 0.5646 at epoch 50
- Test port wine stain IoU: 0.5921 on 18 test samples

## Evaluation Artifacts
- Test metrics: `evaluation/test_metrics.json`
- Prediction overlays: `evaluation/overlays/`
- Overlay legend:
  - Green = ground truth
  - Magenta = prediction
  - Yellow = prediction and ground truth overlap

## Important Limitation
This is an academic prototype model only. It is not intended for medical diagnosis or clinical decision-making.

Because this finetuning run only used port wine stain annotations, it mainly improves class 3 behavior. For a production-ready four-class model, vitiligo, melasma, and port wine stain datasets should be trained or rehearsed together to reduce forgetting and improve all class outputs.
