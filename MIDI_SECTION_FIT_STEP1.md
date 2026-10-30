# MIDI Section Fit, Step 1

This patch implements diagnostic-only measure-aligned section generation and
candidate search. It does not select a global path or alter FeedPak timing.

## Included

- expanded-playback-measure windows;
- four-measure target, one-measure overlap;
- duration-aware adjustment between three and six measures;
- eight-measure context fingerprints;
- independent per-role windows;
- asymmetric ordered group matching;
- fixed performance-MIDI clock;
- chart translation and stretch search;
- distinct repeated-occurrence retention;
- observability-neutral missing evidence;
- compact JSON-ready candidate reports.

## Deferred to Step 2

- globally monotonic per-role path selection;
- overlap-map crossing rejection;
- supported-streak separation across gaps;
- multi-role fusion;
- FeedPak timing-map materialization.

## Integration contract

`build_section_candidate_report()` accepts:

1. expanded playback measures with start/end times;
2. chart events grouped by role;
3. prepared MIDI events grouped by role;
4. an optional baseline time function.

The result is diagnostic-only and explicitly reports that no timing changes
were applied.
