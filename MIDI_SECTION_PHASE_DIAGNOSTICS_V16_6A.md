# MIDI Section Phase Diagnostics v16.6A

## Scope

v16.6A is diagnostic-only. It does not change candidate search, candidate ordering, selected role paths, fusion, materialization, or final timing. It measures whether a locally strong symbolic match is uniquely located in performance MIDI and whether selected paths exhibit beat-sized, measure-sized, or cumulative offset changes.

## Candidate occurrence diagnostics

Every retained distinct temporal occurrence now reports:

- occurrence rank;
- candidate score;
- strongest competing occurrence score;
- absolute score margin;
- score ratio;
- temporal separation from the strongest competitor;
- number of near-equivalent occurrences;
- total distinct occurrence count;
- phase-distinctiveness classification.

The initial classifications are:

- `unique_only_occurrence`;
- `dominant`;
- `ambiguous_best`;
- `alternative_occurrence`.

These labels are observations only. v16.6A does not use them for authority or selection.

## Selected-path diagnostics

For each selected role streak, the report now includes:

- entry offset;
- exit offset;
- cumulative offset trend;
- offset jump at every candidate transition;
- estimated local beat and measure duration;
- beat-sized phase-jump flag;
- measure-sized phase-jump flag.

Beat and measure flags are deliberately diagnostic approximations derived from the chart-window duration. They are intended to locate listening-review points, not to reject timing in v16.6A.

## Cache compatibility

The persistent Phase-1 candidate-cache schema remains unchanged. Existing exact candidate caches are reused, and diagnostics are computed from the retained distinct candidate occurrences after cache loading. A full cold candidate-search rerun is therefore not required merely to enable v16.6A diagnostics.

## Light diagnostics

The light MIDI-section report now retains:

- `candidate_phase_diagnostics` summary;
- `selected_phase_diagnostics` with per-role streak and transition details.

## Test result

Integrated suite: **343 passed, 1 skipped**.
