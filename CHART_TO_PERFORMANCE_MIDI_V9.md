# Chart-to-Performance MIDI Streak Fit v9

This build corrects the direction and materialization of incipit-role timing.
Prepared MuScriptor MIDI remains on the fixed performance/audio clock. The
chart clock is translated and stretched onto it.

## Changes

- Phrase matching exports matched chart-onset to MIDI-onset landmarks.
- Short-phrase local stretch remains diagnostic and no longer rejects an
  otherwise recognizable sequential phrase.
- A robust affine fit is estimated over all landmarks in the confirmed streak:
  `midi_time = offset + chart_stretch * chart_time`.
- Median/MAD outlier rejection protects the fit from bad onset assignments.
- Production authority requires at least 8 inlier landmarks, at least 2 seconds
  of span, chart stretch in 0.9-1.1, median residual <= 120 ms, and maximum
  residual <= 350 ms.
- The first inlier landmark becomes an explicit
  `incipit_acquisition_onset` control, so the actual opening is materialized
  rather than leaving baseline timing active until the first phrase centre.
- Later controls are sampled from the robust fitted chart transform rather than
  from unstable phrase-centre fits.
- The MIDI remains unchanged. Only chart times are transformed.
- Authority remains interval-limited; before and after the confirmed streak,
  timing returns to the selected baseline.
- Light diagnostics now include timeline direction, acquisition control,
  chart stretch, offset, landmark/inlier counts, outlier count, and residuals.

## Validation

Full test suite: 223 passed, 1 skipped.
