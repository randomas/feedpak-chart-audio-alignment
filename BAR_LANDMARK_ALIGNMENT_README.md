# Bar-landmark regularized alignment

This is a new independent alignment mode. It does not replace or modify the
behaviour of `continuous-anchor-dtw`, checkpoint modes, Phase 3D candidates, or
the guarded baseline.

## Install

Copy these files into the current project, preserving `tests/`:

- `bar_landmark_alignment.py`
- `tests/test_bar_landmark_alignment.py`
- `apply_bar_landmark_alignment_patch.py`

Then run:

```powershell
python .\apply_bar_landmark_alignment_patch.py
python -m pytest -q
```

## Run DSMN and Supremacy first

```powershell
.\run_all_test_materials.ps1 `
  -MaterialsRoot "C:\Users\Matteo\Documents\Guitar\Audio alignment" `
  -OutputRoot "C:\Users\Matteo\Documents\Guitar\Audio alignment\bar_landmark_outputs" `
  -AlignmentMode bar-landmark-regularized `
  -CheckpointMeasures 4 `
  -CheckpointSearchRadius 1.0 `
  -TimelineMode both `
  -SkipUnitTests `
  -Only "my_test_song_2_gp", "my_test_song_6_gp"
```

Do not enable Phase 3D or unsafe continuous comparison in the first run. The aim
is to compare one new method directly against the packages already preserved.

## Method

1. Retains the existing guarded opening-and-scale warp as the coarse map.
2. Selects the first meaningful score event near the start of every playback bar.
3. Searches only near the predicted audio time on the corresponding available stems.
4. Builds a raw signed residual trace, making accumulating and dissipating drift visible.
5. Uses a robust weighted median across neighboring bars.
6. Interpolates bars without reliable observations.
7. Limits abrupt residual movement and clamps each interval to 0.90-1.10 stretch.
8. Applies the resulting monotonic bar map to all arrangements through the existing warp path.

The alignment report contains `decision.bar_landmark_alignment`, including every
bar's predicted time, observed time, raw residual, fitted residual, confidence,
sources, and whether it was observed or interpolated.
