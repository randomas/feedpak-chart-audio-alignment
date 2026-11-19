# v16.7K3.1.2 inner-fusion progress

## Scope

K3.1.2 moves heartbeat visibility inside the fused monotonic-control dynamic-programming search. K3.1.1 could report only between complete selector calls, so one expensive selector invocation could still leave PowerShell silent for more than a minute.

## Inner heartbeat

The selector accepts an optional observational callback. After every DP layer it reports:

- DP layer index and count;
- previous beam state count;
- current candidate count;
- next-state count before beam truncation;
- cumulative transition checks;
- cumulative transition rejection counts.

The caller time-throttles display to approximately 15 seconds and prints global layer position, fusion block position, DP layer position, states, candidates, transition checks, unsafe-geometry rejections, non-advancing rejections, elapsed time, and ETA. Output uses `flush=True`.

## Equivalence

The callback is invoked only after the existing transition and state calculations. It does not mutate candidates, beams, sorting, truncation, scores, rejections, controls, or regions. Tests compare the direct selector result with and without a callback and require exact equality.
