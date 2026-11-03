# MIDI Section Evidence Quality Summary v16.6C

## Scope

v16.6C adds a diagnostic-only section-level interpretation layer. It does not change candidate search, scoring, path selection, fusion, materialization, evidence tiers, or final timing.

## Section construction

Section boundaries are derived from existing diagnostic structure: fused-region boundaries, production-phase review boundaries, phrase-run boundaries, applied-region boundaries, transition boundaries, and chart endpoints. Nearby boundaries are merged to avoid unusably small diagnostic fragments. Adjacent short intervals with the same quality and production use are merged.

## Quality labels

Sections are described as:

- `strong`
- `good`
- `average`
- `weak`
- `unsupported`
- `conflicting`
- `structurally_uncertain`

The labels describe evidence quality and are not acceptance or pruning decisions.

## Causal summaries

Each section exposes:

- start, end, and optional name;
- quality and confidence;
- concise summary;
- up to three positive factors;
- up to three risk factors;
- production use and provenance;
- primary warning;
- suggested listening-review time.

Inputs include acquisition-root state, fused-region coverage, strong controls, role conflicts, phrase support or ambiguity, production-phase review points, materialized-region coverage, and baseline/unsupported intervals.

## Song overview

The report includes overall quality, confidence, duration-weighted quality distribution, primary concern, and ranked suggested review points.

## Production integration

A preliminary summary is generated with the section-fit report. When materialization runs, the summary is regenerated with applied-region provenance so `production_use` reflects the final MIDI-section application. The light diagnostic retains the complete compact section summary.

## Safety

Every report explicitly states:

- `diagnostic_only: true`
- `selection_behavior_changed: false`
- `timing_changes_applied: false`

No evidence is removed or downgraded in the alignment engine.

## Validation

Integrated suite: **356 passed, 1 skipped**.
