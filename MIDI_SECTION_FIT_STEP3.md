# MIDI Section Fit, Step 3

Step 3 fuses compatible per-role paths into sparse diagnostic timing-map
proposals. It still does not modify FeedPak timing.

## Fusion policy

- role predictions are clustered by mapped MIDI time;
- incompatible clusters are retained as explicit conflicts and are never
  averaged together;
- matching roles contribute corroboration and provenance;
- strong single-role evidence may remain provisional;
- a beam selects a globally monotonic control sequence;
- every fused edge must remain within normal 0.90-1.10 production limits;
- large evidence gaps split independent regions;
- no controls are interpolated across those gaps;
- correction-and-catch-up patterns are reported;
- sparse controls retain role, streak, and source-window provenance.

## Monotonicity certificate

The report certifies strict increase of both chart and MIDI control times and
states explicitly that unsupported gaps were not interpolated.

## Deferred

- integration with the main runner and real corpus reports;
- forced FeedPak materialization;
- normal production authority;
- optional wide diagnostic timing-map proposal.
