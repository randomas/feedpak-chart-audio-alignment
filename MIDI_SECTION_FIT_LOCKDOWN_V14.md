# MIDI section-fit lock-down v14

## Regional fusion recovery

Fusion preserves every longest valid monotonic prefix, records the failed layer,
skips the incompatible layer, and resumes later. Independent regions must be
non-overlapping and globally forward. No support is interpolated across splits.

## Evidence policy

- multi-role agreement increases confidence;
- a coherent single-role streak spanning at least two windows can be strong;
- isolated single-window evidence is provisional;
- missing roles are neutral;
- contradictory role clusters are never averaged.

## Drum families

Chart names and prepared General-MIDI drum notes are normalized to kick, snare,
hi-hat, cymbal, tom, or other percussion before matching. Original MIDI notes
and canonical pieces remain present in the provider diagnostics.

## GP5 regression

The missing `compute_anchors()` helper is restored. It emits the historical
`{time, fret, width}` shape with ordered times and a default four-fret span.

## Listening mode

`-ForceMidiSectionFit` / `--force-midi-section-fit` requires MIDI section fit.
It is the final routing layer, applies only eligible regions with safe entry and
exit handoffs, leaves unsupported gaps on the selected baseline, and applies no
subsequent audio refinement. Output timing is labelled `forced-midi-section-fit`.

Regions with fewer than three controls, fewer than two strong controls, unsafe
handoffs, or non-monotonic handoff controls remain diagnostic-only.

## Validation

Full suite: 262 passed, 1 skipped.
