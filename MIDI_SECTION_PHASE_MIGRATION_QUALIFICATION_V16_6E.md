# MIDI Section Phase-Migration Qualification v16.6E

## Scope

v16.6E adds a diagnostic-only phase-stability dimension. It does not alter candidate generation, candidate scores, path selection, fusion, rooted phrase support, materialization, evidence tiers, method selection, or final timing.

## Episode construction

For each fused region, control-to-control changes in performance-minus-chart offset are grouped into directional episodes. Neutral changes may remain within an episode; a material reversal closes the episode and begins another.

Each episode reports:

- chart start, end, and duration;
- entry and exit offsets;
- total movement in seconds and local beats;
- movement rate;
- maximum individual step;
- supporting roles;
- strong-step count and total step count;
- validated-root compatibility;
- persistence after the episode;
- whether later controls return toward the entry phase;
- severity and suggested listening point.

## Phase classifications

Episodes are classified as:

- `stable`
- `supported_tempo_evolution`
- `temporary_correction`
- `correction_and_catch_up`
- `provisional_phase_change`
- `persistent_migration`
- `structurally_unrooted`

This separates evidence support from phase stability. A section may have strong evidence support while its selected path is migrating.

## Section integration

The report includes `section_phase_states`, pairing every v16.6C.1 section with:

- `evidence_support`
- `phase_stability`
- `production_status`

The existing evidence-quality label is not overwritten.

## Production integration

Phase qualification is generated after fusion and regenerated after materialization so the light report reflects the final applied section map and its current section summary.

## Safety

Every report states:

- `diagnostic_only: true`
- `selection_behavior_changed: false`
- `timing_changes_applied: false`

The module does not modify its input objects and does not mark controls provisional yet. Functional authority changes are deferred until corpus diagnostics and listening confirm the classifications.

## Validation

Integrated suite: **370 passed, 1 skipped**.
