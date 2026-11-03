# MIDI Section Phrase Repeat Diagnostics v16.6A.3

## Scope

v16.6A.3 is diagnostic-only. Candidate search, scoring, role paths, fusion, materialization, and final timing are unchanged.

## Phrase hierarchy

The diagnostic analyses two- and four-measure phrases. Single-measure repetition is no longer eligible for an anchor proposal. Phrase signatures combine normalized onset rhythm, event count, pitch or canonical drum-family content, duration, and role.

## Joint run matching

Consecutive similar chart phrases form a repeat run. Performance-MIDI hypotheses are assembled jointly by phrase-duration spacing before duplicate suppression, preventing a high-scoring interior onset from destroying a valid ordinal chain.

## Boundary context

The phrase immediately before and after a run is compared on chart and MIDI sides. Boundary context contributes to joint confidence and is required for a diagnostic ordinal-anchor proposal.

## Occurrence accounting

Each correspondence reports separately:

- chart occurrence count;
- confirmed MIDI occurrences;
- probable additional occurrences;
- unresolved occurrences;
- boundary-context score;
- joint confidence.

Count discrepancies are reported and never repaired by forced stretching.

## Cross-role corroboration

Roles remain independent. Overlapping runs are grouped only when phrase length and structural classification agree. This produces a corroboration report without averaging incompatible locations.

## Diagnostic proposals

- `phrase_run_ordinal_constraint` requires an equal confirmed count, sufficient joint confidence, and boundary support.
- `phrase_structure_review` exposes incomplete or mismatched phrase runs.

All proposals remain diagnostic-only and cannot alter timing.

## Validation

Integrated suite: **349 passed, 1 skipped**.
