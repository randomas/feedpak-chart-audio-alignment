# MIDI Section Evidence Quality Calibration v16.6C.1

## Scope

v16.6C.1 calibrates the diagnostic-only section evidence summary. Alignment search, scoring, role paths, fusion, materialization, evidence tiers, and final timing remain unchanged.

## Selected production path versus rejected alternatives

Role conflicts are no longer automatically treated as section-wide production conflicts.

- With a strong applied path, alternatives are reported as `rejected_alternative_clusters` and apply only a small confidence deduction.
- With an applied but less strongly supported path, they are reported as `dominant_selected_cluster_with_alternatives`.
- Only an unapplied or unresolved selected phase becomes `selected_path_unresolved_conflict` and receives the hard `conflicting` label.

## Production-path support

Applied materialized regions now receive primary positive weight. The summary considers fused-region coverage, strong-control ratio, selected production use, rooted acquisition, phrase boundaries, and phase diagnostics.

## Local phase review

A long drift episode contributes one review onset rather than contaminating every overlapping subsection. Region-wide phase evolution remains contextual risk, scaled by strong-control ratio and root state. Review timestamps are deduplicated.

## Semantic intervals

Sections now distinguish:

- `pre_roll`
- `validated_evidence`
- `provisional_evidence`
- `baseline_shaped_gap`
- `unsupported_playable_content`
- `terminal_tail`

Pre-roll and terminal-tail durations are excluded from the musical quality distribution.

## Primary concern

Concerns affecting applied playable timing are ranked before uncovered pre-roll, post-roll, or inactive intervals. This prevents a silent or intentional lead-in from masking the main production risk.

## Safety

The summary remains descriptive:

- `diagnostic_only: true`
- `selection_behavior_changed: false`
- `timing_changes_applied: false`

No candidate, control, region, phrase, or timing map is changed.

## Validation

Integrated suite: **360 passed, 1 skipped**.
