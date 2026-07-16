# Lumiere SegFormer-B2 Melasma Detection Model

## Task
Task 166 - Finetune SegFormer for melasma detection

## Training Environment
Google Colab GPU

## Model
SegFormer-B2 fine-tuned for binary semantic segmentation.

## Classes
- 0 = background
- 1 = melasma_like_hyperpigmentation

## Current Results
- Best validation positive-only IoU: 0.5598
- Best validation threshold: 0.3
- Test positive-only IoU: 0.4826
- Best test threshold: 0.3
- Shown sample average IoU: 0.4630

## Important Limitation
This is an academic prototype model only. It is not intended for medical diagnosis or clinical decision-making.
