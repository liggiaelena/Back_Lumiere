# Vitiligo classifier gate

This promoted ResNet-18 checkpoint is an image-level safety gate for the
existing vitiligo SegFormer output. It suppresses a positive segmentation when
the classifier probability is below the threshold in `deployment.json`; it
does not replace or alter the promoted segmentation checkpoint.

Held-out evaluation at the promoted threshold:

- precision: 98.82%
- recall: 98.24%
- healthy false-positive rate: 1.33%

The complete evaluation output is retained in `metrics.json`.