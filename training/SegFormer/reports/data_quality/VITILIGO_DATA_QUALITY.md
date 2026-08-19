# Vitiligo mask integrity finding

The current vitiligo masks must not be used for model selection or deployment.

## Evidence

- Dataset: `../Model_Segformer/vitiligo` (identical copies also exist under
  `data-collection/vitiligo`).
- All 100 PNG masks contain values 0 and 1, but their median directional edge
  ratio is 65.53.
- Median horizontal transition rate: 0.0074.
- Median vertical transition rate: 0.4688.
- Example `0.png`: horizontal transition rate 0.0073 versus vertical 0.5037.
- The copy in Git commit `2294ef7` has the same blob hash as the current file,
  so Git does not contain an earlier healthy version.

This scanline pattern is incompatible with coherent lesion polygons and is
consistent with a corrupt bit-stream, row ordering, or mask export operation.
Simple width/height, C/F-order, and transpose inversions do not restore a
spatially coherent mask.

## Consequence

Models trained on these masks cannot be honestly evaluated against the stated
pixel IoU/precision/recall deployment gates. The best experimental model was
not promoted because its validation precision remained near 0.50.

## Required remediation

Regenerate all masks from the original LabelMe/CVAT polygon annotations (or
manually re-annotate the source images) as 8-bit PNG files with values 0 and 1.
Run `validate_mask_integrity()` in `train_independent.py` before retraining.
After clean masks exist, retrain and require both validation and held-out test:

- IoU >= 0.40
- precision >= 0.60
- recall >= 0.50

Only then move the checkpoint to
`training/checkpoints/SegFormer/models/vitiligo`.
