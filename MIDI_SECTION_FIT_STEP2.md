# MIDI Section Fit, Step 2

Step 2 adds independent globally monotonic overlapping paths per role. It is
still diagnostic-only and does not alter FeedPak timing.

## Hard invariants

- chart window starts and ends strictly increase;
- MIDI window starts and ends strictly increase;
- every local chart transform has positive stretch;
- candidate maps in a shared measure agree within a hard tolerance;
- switching ownership inside an overlap cannot reverse mapped MIDI time;
- no selector repairs or sorts a non-monotonic result after selection;
- unobservable or candidate-empty windows split authority into separate
  supported streaks;
- no evidence is interpolated across those gaps.

## Selector

A bounded beam (default width 12) evaluates the retained Step-1 candidates.
Local score, overlap agreement, stretch continuity, and translation continuity
contribute to ranking. Invalid temporal transitions are excluded from the
search graph rather than penalized softly.

If consecutive observable windows have no compatible transition, the block is
split and each side becomes an independent evidence streak.

## Still deferred

- multi-role fusion;
- sparse timing-map construction;
- FeedPak materialization;
- production authority.
