# Vitiligo Dataset

## Purpose

This folder contains the prepared vitiligo segmentation dataset for the Lumiere project.

## Source Dataset

Images collected and annotated manually using LabelMe/CVAT from open dermatology sources (ISIC Archive, DermNet, HAM10000).

## Folder Structure

data-collection/vitiligo/images/ contains all image files.

data-collection/vitiligo/masks/ contains matching binary mask files.

data-collection/vitiligo/splits/ contains:
- train.txt
- val.txt
- test.txt

Each split file lists the image stems (filename without extension) used for that split.

## Prepared Dataset Summary

- Total images: 100
- Train images: 70
- Validation images: 15
- Test images: 15

## Mask Meaning

- 0 = background / healthy skin
- 1 = vitiligo (depigmented region)

## Limitation

This dataset is for academic prototype use only. It is not intended for clinical diagnosis or medical decision-making.
