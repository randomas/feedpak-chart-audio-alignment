# MIDI Section Materialization Recovery v16.5

## Purpose

v16.5 preserves policy-safe MIDI timing evidence on either side of an internal edge that the selected production materialization mode rejects. Fusion already recovered later monotonic paths; this release applies equivalent recovery at the stricter production-policy boundary.

## Behaviour

For each fused region, materialization now:

1. assesses every internal edge under the selected materialization mode;
2. retains normal, short-delta, and length-adjusted correction/catch-up exceptions;
3. splits at remaining rejected edges;
4. recalculates evidence tier and geometry for every fragment;
5. preserves eligible safe prefixes;
6. independently reconsiders later fragments;
7. may trim leading controls from a later fragment until a safe baseline-shaped re-entry has sufficient runway;
8. never labels the rejected edge or recovery gap as MIDI-validated;
9. validates the final global warp for strict monotonicity.

## Diagnostics

Materialization reports now include:

- `materialization_split_edges`;
- `materialization_recovery_applied`;
- `source_region_id` and fragment identity;
- `trimmed_leading_control_count` where restart recovery required trimming;
- `available_recovered` status when recovered fragments reach production;
- rejected short fragments and rejected unsafe transitions.

## Compatibility

- Search grid, candidate scores, role paths, and fusion output are unchanged.
- Existing materialization modes and geometry limits are unchanged.
- Existing accepted correction/catch-up exceptions remain accepted.
- Unsupported gaps continue to use baseline shape and remain explicitly non-MIDI-validated.
- Catastrophic non-positive geometry is never forced.

## Tests

The release adds focused coverage for:

- safe prefix plus recovered suffix;
- safe prefix with an unrecoverable suffix;
- preservation of an accepted episode exception;
- existing catastrophic, tier, transition, and unsupported-gap behaviour.

Integrated suite result: **340 passed, 1 skipped**.
