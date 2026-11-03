# MIDI Section Phase-Migration Calibration v16.6E.2

## Scope

v16.6E.2 recalibrates diagnostic phase interpretation. It remains diagnostic-only and preserves all candidates, paths, controls, regions, materialization, and final timing.

## Smooth tempo evolution

Directional movement may be classified as `supported_tempo_evolution` when it is smooth, multi-step, supported by multiple roles, free of repeated-structure ambiguity, not close to a persistent metrical plateau, and lacks an abrupt phase step. Exact bounded phrase support can provide the same protection for a smooth single-role episode.

## Migration qualifiers

Persistent migration now requires contextual evidence such as:

- abrupt beat-scale movement;
- a persistent plateau near an integral beat displacement;
- ambiguous repeated structure;
- large persistent single-role movement;
- non-smooth departure from a validated root.

The report exposes the qualifying reasons rather than only the final label.

## Occurrence-switch diagnostic

An unrooted large phase movement near an unsupported or recovered gap may be classified as `probable_occurrence_switch`. This is intended to expose repeated-pattern relatching without forcing a structural correction.

## Added episode fields

- median step size;
- smooth-evolution flag;
- abrupt-step flag;
- role count;
- root departure in beats;
- metrical-plateau flag and error;
- exact and ambiguous phrase context;
- overlapping phrase-run IDs;
- gap proximity;
- classification reasons.

## Absolute root state

The report separately records `absolute_root_state` as `rooted_and_consistent`, `audio_root_required`, or `unrooted`. Internal migration analysis does not claim to validate absolute incipit timing.

## Safety

- `diagnostic_only: true`
- `selection_behavior_changed: false`
- `timing_changes_applied: false`

## Validation

Integrated suite: **375 passed, 1 skipped**.
