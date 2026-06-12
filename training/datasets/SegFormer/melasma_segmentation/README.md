# Melasma Segmentation Dataset

## Purpose

This folder contains the prepared melasma segmentation dataset used for SegFormer prototype training in the Lumiere project.

## Azure Task

Task 216 - Upload melasma dataset to the shared repository

## Source Dataset

Roboflow Universe: melasma Instance Segmentation Dataset by ayezka

Dataset link:
https://universe.roboflow.com/ayezka/melasma-wf8lz

## Dataset Type

Instance segmentation dataset converted into binary semantic segmentation masks.

## Prepared Dataset Size

- Total images: 512
- Positive melasma/hyperpigmentation masks: 255
- Empty/background masks: 257
- Train images: 358
- Validation images: 102
- Test images: 52

## Classes Used

- 0 = background / non-target skin
- 1 = melasma_like_hyperpigmentation

## Included Files

- melasma_segmentation_prepared_for_colab.zip
- melasma_dataset_preparation_summary.json
- README.md

## Notes

The dataset was prepared for academic prototype training only. It is not a clinical-grade dataset and should not be used for medical diagnosis.

Known limitations:
- Some images are close-up patches instead of full-face images.
- Some images are rotated or watermarked.
- Some annotation masks are broad or rough.
- Dataset quality affects model accuracy.
