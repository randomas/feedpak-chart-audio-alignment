# MIDI Section Production Phase Diagnostics v16.6A.1

## Scope

v16.6A.1 is diagnostic-only. It preserves v16.6A candidate diagnostics and adds analysis at the fused-control and materialization layers that actually determine production timing. It does not change search, scores, ordering, role paths, fusion, materialization, or final beat times.

## Fused-control diagnostics

Every consecutive fused-control edge now records:

- chart interval and production offsets;
- offset change in seconds, estimated beats, and estimated measures;
- local playback-measure and beat duration;
- roles before, after, and common to the edge;
- approximate beat-sized and measure-sized phase flags;
- persistence classification.

Persistence classifications are:

- `temporary_correction`;
- `persistent_offset_state`;
- `gradual_cumulative_drift`;
- `mixed_or_noisy`;
- `unresolved_end_of_region`.

These are review diagnostics, not authority rules.

## Region diagnostics

Every fused region now reports:

- entry and exit offsets;
- cumulative offset trend in seconds, beats, and measures;
- maximum departure from entry phase;
- whether the region returns toward its entry phase;
- all fused-control edge diagnostics.

## Opening and terminal diagnostics

The report records:

- first fused-control time;
- whether an acquisition root was validated;
- first applied region time;
- unrooted leading span;
- chart terminal time;
- final fused-control time and offset;
- unsupported terminal-tail duration;
- final applied region and carried terminal offset.

## Materialization and gap diagnostics

Applied fragments and baseline-shaped gaps now report:

- fragment entry/exit offsets and trend;
- source-region and trimming provenance;
- gap entry/exit offsets;
- gap phase change;
- explicit `midi_validated: false` state.

## Ranked review points

The report emits ranked review points for:

- unrooted openings;
- persistent production-control phase changes;
- beat- or measure-sized movements;
- gradual region drift;
- baseline-shaped gap phase changes.

## Light diagnostics

`collect_run_diagnostics.py` retains `production_phase_diagnostics` in the light MIDI-section report. Existing candidate caches remain compatible.

## Validation

Integrated suite: **345 passed, 1 skipped**.
