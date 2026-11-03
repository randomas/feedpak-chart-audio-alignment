# MIDI section materialization v15

## Interface

Use `-MidiSectionFitMaterialization` in the PowerShell corpus runner, or
`--midi-section-fit-materialization` in the Python CLIs, with one of:

- `conservative`
- `permissive`
- `diagnostic-unsafe`

All modes require MIDI section fit. The old force switch remains as a deprecated
alias for permissive mode.

## Core correction

Raw endpoint difference from the selected baseline is never a rejection rule.
A long reliable section may accumulate a large displacement from a very small
clock difference. Regions are judged by symbolic and geometric evidence; gaps
are judged by the stretch needed to connect region displacements.

## Evidence tiers

- Tier A: at least three strong controls, at least 75% strong, no catch-up event.
- Tier B: at least two strong controls, at least 40% strong, no more than one
  catch-up event.
- Tier C: provisional evidence that remains structurally monotonic.

Conservative enables A, permissive A+B, and diagnostic-unsafe A+B+C.
Structural reversals and non-positive local stretches are catastrophic and are
never forced.

## Transition policies

- conservative: 0.90-1.10
- permissive: 0.85-1.15
- diagnostic-unsafe: 0.70-1.30

Before the first region and after the last region, the nearest trusted timing
displacement is carried over the baseline shape. Between accepted regions, the
offset is interpolated across the unsupported interval only if the resulting
transition warp fits the selected policy. Unsupported intervals are explicitly
not marked as MIDI-validated.

## Rejection severity

- tier_1_close: evidence or required transition narrowly misses the selected mode.
- tier_2_high_risk: provisional evidence or a materially wider transition.
- catastrophic: temporal reversal, overlapping regions, non-positive map, or
  global monotonicity failure.

## Validation

268 tests passed; 1 skipped.
