# MIDI section length-adjusted episode geometry v16.2

This release combines catch-up episode grouping and length-normalized episode density with a bounded length-adjusted delta exception. Ordinary edges retain fixed mode stretch limits. Only an out-of-policy edge inside an allowed catch-up episode may use the growing delta allowance, and it must remain inside a fixed hard stretch ceiling.

For permissive mode the allowance is `min(500 ms, 250 ms + 5 ms * max(0, supported measures - 16))` with hard stretch 0.70-1.30. The 50-measure DSMN region therefore receives a 420 ms allowance and accepts its 384.6 ms, 1.25-stretch local correction.

Search grids, exact caching, candidate scoring, role paths, fusion ordering, and catastrophic geometry checks are unchanged.
