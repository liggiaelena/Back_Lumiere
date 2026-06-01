# Lumiere SegFormer-B2 Vitiligo Finetune

## Task
Task 7 — Finetune SegFormer for vitiligo detection

## Model
- Architecture: SegFormer-B2 semantic segmentation
- Starting checkpoint: `training/checkpoints/SegFormer/segformer_b2_4class_port_wine_stain_finetune/best`
- Finetuned checkpoint: `vitiligo_finetune/best/`
- Final epoch checkpoint: `vitiligo_finetune/last/`

## Classes
- 0 = background
- 1 = vitiligo
- 2 = melasma_like_hyperpigmentation
- 3 = port_wine_stain

## Dataset
- Source: Label Studio brush annotation export
- Total paired samples: 100
- Split:
  - Train: 70
  - Validation: 15
  - Test: 15
- Processed mask values:
  - 0 = background
  - 1 = vitiligo

## Training
- Environment: Google Colab GPU (T4)
- Image size: 512
- Batch size: 4
- Epochs: 50
- Learning rate: 5e-5 (CosineAnnealingLR, eta_min=1e-6)
- Optimizer: AdamW
- Loss: CrossEntropyLoss with class weights [0.3, 3.0, 1.0, 1.0]
- Augmentation: RandomHorizontalFlip + ColorJitter (train split only)
- Best validation checkpoint selection: highest validation vitiligo IoU

## Results
- Best validation vitiligo IoU: 0.4933 at epoch 22
- Final epoch validation vitiligo IoU: see `vitiligo_finetune/training_history.json`

## Important Limitation
This is an academic prototype model only. It is not intended for medical diagnosis or clinical decision-making.

Because this finetuning run only used vitiligo annotations, classes 2 (melasma) and 3 (port_wine_stain) behaviour is inherited from the starting checkpoint and was not retrained. For a production-ready four-class model, all three conditions should be trained together to reduce catastrophic forgetting.
