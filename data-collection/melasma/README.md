# Melasma Dataset

## Purpose

This folder contains the prepared melasma segmentation dataset for the Lumiere project.

## Source Dataset

Roboflow Universe: melasma Instance Segmentation Dataset by ayezka

Dataset link:
https://universe.roboflow.com/ayezka/melasma-wf8lz

## Folder Structure

data-collection/melasma/images/ contains all image files.

data-collection/melasma/masks/ contains matching binary mask files.

data-collection/melasma/splits/ contains:
- train.txt
- val.txt
- test.txt

Each split file lists the image filenames used for that split.

## Prepared Dataset Summary

- Total images: 512
- Positive melasma/hyperpigmentation masks: 255
- Empty/background masks: 257
- Train images: 358
- Validation images: 102
- Test images: 52

## Mask Meaning

- 0 = background / non-target skin
- 1 = melasma-like hyperpigmentation region

## Limitation

This dataset is for academic prototype use only. It is not intended for clinical diagnosis or medical decision-making.
