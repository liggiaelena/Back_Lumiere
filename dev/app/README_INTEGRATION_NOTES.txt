Files in this package:

1. app/segmentation.py
   - Loads SegFormer checkpoints.
   - Produces condition_mask and condition_map.
   - Uses your checkpoint labels:
       melasma_like_hyperpigmentation -> melasma
       port_wine_stain -> wine_stain
   - Uses melasma threshold 0.3 for segformer_b2_melasma_colab_export.

2. app/pipeline.py
   - Runs SegFormer first.
   - Sends condition_mask to skin_tone_analyzer.
   - Sends condition_map to Claude through vision.analyze_region().
   - Adds segformer_condition_map to final report.

3. app/skin_tone_analyzer.py
   - Excludes non-zero SegFormer condition_mask pixels from healthy skin tone calculation.

4. app/vision.py
   - Accepts condition_map=None.
   - Adds SegFormer condition context into Claude prompt.
   - Keeps your existing JSON output format.
