# MIDI section hybrid geometry v15.1

## Acceptance rule

After evidence and hard monotonicity checks, an internal region edge or
unsupported-gap transition is accepted when either:

1. its stretch is inside the selected materialization mode's range; or
2. it is short enough and its absolute timing correction is inside that mode's
   short-region allowance.

## Mode limits

- conservative: stretch 0.90-1.10, or <=120 ms over <=6 s
- permissive: stretch 0.85-1.15, or <=250 ms over <=10 s
- diagnostic-unsafe: stretch 0.70-1.30, or <=500 ms over <=16 s

A long section with a small drift therefore passes on stretch even if its total
endpoint displacement is large. A short malformed section may pass on bounded
absolute correction even if its local stretch is outside the normal range.

## Diagnostics

Every assessed edge reports:

- `acceptance_basis`: `normal_stretch_range`,
  `short_region_delta_exception`, `geometry_out_of_policy`, or
  `catastrophic_non_positive_geometry`;
- stretch;
- absolute correction;
- edge duration;
- the mode-specific delta and duration limits.

Catch-up pairs involving a short-delta exception are labelled
`possible_local_chart_timing_defect`. Other accepted opposite corrections are
labelled `baseline_return_catch_up`; invalid pairs are
`probable_artificial_catch_up`.

## Fusion

Fusion retains short bounded high-stretch edges up to the diagnostic-unsafe
ceiling so the selected materialization mode can make the final decision. Hard
reversal and large unbounded corrections remain excluded.

## Validation

273 tests passed; 1 skipped.
