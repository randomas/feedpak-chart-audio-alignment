# MIDI Section Final Phase-Diagnostic Cleanup v16.6E.3

## Scope

v16.6E.3 completes the diagnostic phase branch. It remains diagnostic-only and does not change candidates, scores, paths, controls, regions, materialization, evidence authority, method selection, or final timing.

## Positive error qualifier

A rooted episode may be called `persistent_migration` only when at least one positive error qualifier is present:

- abrupt phase step;
- persistent metrical plateau;
- repeated-structure ambiguity;
- unsupported/recovery-gap context;
- significant single-role displacement.

Multi-role support alone increases confidence that movement occurred, but is not evidence that the movement is wrong. Root departure alone no longer creates persistent migration.

## Separate confidence dimensions

Every episode now reports:

- `movement_confidence`: confidence that the measured timing movement is coherent and real;
- `phase_error_confidence`: confidence that the movement represents a phase error;
- `positive_error_qualifier`: whether a concrete error qualifier exists.

## Materialization-aware gaps

The final post-materialization phase pass receives the materialization application report. Gap context is compiled from:

- production phase gaps;
- materialization transitions;
- rejected regions;
- spaces between applied regions.

Episodes report the nearest gap kind and span. Unrooted beat-scale movement near a final gap may become `probable_occurrence_switch`.

## Safety

- `diagnostic_only: true`
- `selection_behavior_changed: false`
- `timing_changes_applied: false`

## Validation

Integrated suite: **382 passed, 1 skipped**.
