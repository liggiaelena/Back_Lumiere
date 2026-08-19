# Lumiere SegFormer Melasma Detection Model

## Task
Task 166 - Finetune SegFormer for melasma detection

## Training Environment
Google Colab GPU

## Model
SegFormer-B0 fine-tuned for binary semantic segmentation.

## Classes
- 0 = background
- 1 = melasma_like_hyperpigmentation

## Current Results
- Best validation IoU: 0.6399
- Best validation threshold: 0.8
- Test IoU: 0.5598
- Best test threshold: 0.6

## Important Limitation
This is an academic prototype model only. It is not intended for medical diagnosis or clinical decision-making.
