Files in this package:

1. app/segmentation.py
   - Loads SegFormer checkpoints.
   - Produces condition_mask and condition_map.
   - Uses your checkpoint labels:
       melasma_like_hyperpigmentation -> melasma
       port_wine_stain -> wine_stain
   - Loads promoted independent models from training/checkpoints/SegFormer/models/.
   - The promoted melasma model uses 512px input and threshold 0.50.
   - Preserves melasma evidence from threshold 0.20 for multimodal confirmation when
     the hard deployment mask does not trigger.

2. app/pipeline.py
   - Runs SegFormer first.
   - Sends condition_mask to skin_tone_analyzer.
   - Sends condition_map to Claude through vision.analyze_region().
   - Adds segformer_condition_map to final report.
   - Confirms a soft melasma candidate only when Claude independently reports spots
     in at least two matching facial regions; confirmed results are marked
     source=segformer_spot_fusion and suspected=true.

3. app/skin_tone_analyzer.py
   - Excludes non-zero SegFormer condition_mask pixels from healthy skin tone calculation.

4. app/vision.py
   - Accepts condition_map=None.
   - Adds SegFormer condition context into Claude prompt.
   - Keeps your existing JSON output format.
