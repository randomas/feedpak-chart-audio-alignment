# MIDI Section Structural Repeat Diagnostics v16.6A.2

## Scope

v16.6A.2 is diagnostic-only. It does not change candidate generation, scoring, role paths, fusion, solo-role authority, materialization, or final timing.

## Playback structure inventory

Every expanded playback measure records playback and written identities, source occurrence, playback segment, nominal span, meter/navigation metadata when available, active roles, and empty status. This exposes repeat expansion separately from written chart order.

## Repeat-run discovery

For each role, the diagnostic builds coarse per-measure symbolic signatures from normalized onset rhythm, event count, and pitch/drum-family content. Consecutive sufficiently similar measures form a repeat run. Only observable measures with at least two event groups participate.

## MIDI correspondence

Each chart run is searched independently in the corresponding prepared performance MIDI. Candidate starts are real MIDI onset times. Nearby duplicates are suppressed, and the longest approximately consecutive occurrence chain is retained.

Correspondences are classified as:

- `matching_occurrence_count`;
- `possible_extra_chart_or_missing_midi_occurrence`;
- `possible_missing_chart_or_extra_midi_occurrence`;
- `repeat_count_mismatch`;
- `inconclusive`.

A count mismatch is reported, never repaired by stretching or forced timing.

## Diagnostic proposals

High-confidence equal-count runs may emit a `repeat_run_ordinal_constraint` proposal. Count discrepancies may emit a `structural_mismatch_review` proposal. Both remain diagnostic-only and cannot alter timing in this release.

## Generality and safety

- No song names, timestamps, or corpus-specific measure rules are used.
- Missing or contaminated MIDI is neutral.
- Roles are analysed independently.
- Equal repetition count is not sufficient without symbolic similarity.
- Repeat constraints remain proposals until later authority policy is validated.

## Light diagnostics

The light MIDI-section report retains the compact playback structure inventory, chart repeat runs, run correspondences, and diagnostic proposals. Detailed role signatures remain in the full report.

## Validation

Integrated suite: **349 passed, 1 skipped**.
