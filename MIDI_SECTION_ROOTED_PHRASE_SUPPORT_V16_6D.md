# MIDI Section Rooted Phrase-Ordinal Support v16.6D

## Scope

v16.6D adds a guarded phrase-authority pass while preserving the existing section engine. Phrase evidence does not create timing controls, replace section candidates, or delete evidence. It may only adjust the scores of existing section candidates before the normal monotonic path and fusion stages are rerun.

## Eligibility

A phrase run may influence selection only when all conditions hold:

- the opening has a validated three-clock or solo-role acquisition root;
- the run is the selected representative of its overlapping phrase cluster;
- chart and MIDI occurrence counts match exactly;
- boundary-context score is at least 0.45;
- joint confidence is at least 0.76;
- chart and MIDI ordinal counts agree;
- the first phrase-origin offset agrees with the validated root within 0.75 seconds.

Missing phrase evidence remains neutral. Count mismatches, excess MIDI instances, weak boundaries, unrooted songs, and non-selected shifted phrase origins receive no authority.

## Candidate support

For an eligible run, existing candidates of the same role are evaluated at phrase-occurrence starts.

- A candidate whose predicted MIDI phrase times remain within 0.35 seconds receives a bounded positive score adjustment, up to +0.22.
- An ordinally incompatible candidate receives a bounded negative adjustment, no lower than -0.10.

No candidate is removed. No phrase-derived control is inserted. Candidate generation and cache contents remain unchanged.

## Second selection pass

When at least one candidate is adjusted, v16.6D reruns the existing:

1. per-role monotonic path selector;
2. fused role-path selector;
3. solo-role authority annotation;
4. three-clock incipit reconciliation;
5. production phase diagnostics;
6. ordinary materialization pipeline.

Thus any timing change can occur only through the already established section-path and fusion logic.

## Diagnostics

The light and full MIDI-section reports include `rooted_phrase_ordinal_support`, covering:

- root validation and root offset;
- eligible and rejected run IDs with reasons;
- compatible and incompatible candidate counts;
- affected roles;
- policy thresholds;
- initial and final fusion summaries;
- explicit evidence-preservation guarantees.

## Regression safeguards

- unrooted songs are unchanged;
- ambiguous or excess-occurrence runs are unchanged;
- non-selected shifted hypotheses are unchanged;
- mismatch evidence is never forced;
- original candidate reports are deep-copied before score support;
- candidate caches are not invalidated.

## Validation

Integrated suite: **365 passed, 1 skipped**.
