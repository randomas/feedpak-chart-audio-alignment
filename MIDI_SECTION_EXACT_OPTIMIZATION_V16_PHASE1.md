# MIDI section exact optimization v16 phase 1

This patch is based on v15.1 hybrid geometry. It preserves the exhaustive
stretch/translation grid, candidate limits, scoring constants, tie ordering,
path selection, fusion, and materialization rules.

## Exact optimizations

1. Prepared performance MIDI is grouped and sorted once per role.
2. A parallel time array and `bisect_left`/`bisect_right` replace full-list
   scans while preserving inclusive interval boundaries.
3. Each chart window owns an alignment cache keyed by exact MIDI slice indices.
   Both search hypotheses share the cache.
4. Ordered-DP symbolic similarity is computed once per chart/MIDI group pair
   and reused during forward scoring and backtrace.
5. Timers and counters are written to full and light MIDI-section diagnostics.

## Diagnostics

Global and per-role counters include:

- transform trials
- MIDI slice lookups
- unique MIDI slices
- DP evaluations
- alignment-cache hits
- cache-hit rate
- prepared MIDI group count
- MIDI grouping seconds
- candidate generation seconds
- path selection seconds
- fusion seconds
- total section-fit seconds
- materialization seconds

## Deliberately deferred

This release does not add multiprocessing, Numba, GPU execution, merged
hypothesis grids, coarse-to-fine search, approximate prefilters, wider grid
steps, fewer candidates, or early termination.

## Parity protection

Tests compare the optimized DP with the original reference DP, indexed slices
with inclusive full scans, raw and pre-grouped search results, candidate object
serialization, cache accounting, and stage diagnostics.

Container validation: 278 passed, 1 skipped.
