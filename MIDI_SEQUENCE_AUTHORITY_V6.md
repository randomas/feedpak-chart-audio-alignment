# MIDI Sequence Authority v6

This build changes prepared-MIDI alignment from a single expected-origin search to a dual-origin, sequence-led investigation.

## Implemented

- Expected prepared-timeline origin and offset-method incipit are searched as independent soft hypotheses.
- The offset incipit is not installed as an immutable whole-song boundary.
- Candidate discovery is permissive; authority is assigned later.
- MIDI timing controls require a bidirectionally confirmed sequence of at least six onset groups.
- Single notes and silence boundaries remain diagnostic and cannot become MIDI controls.
- Pitched roles use a deliberately modest phrase-distinctiveness floor; drums require a higher floor.
- Candidates receive authority tiers A-D.
- Paths are selected per role before cross-role fusion, preventing a dense role from dominating immediately.
- Late chart roles remain visible as observations but cannot authoritatively vote for the incipit.
- Incipit diagnostics compare expected-origin and offset-incipit residuals and rank sequence hypotheses.

## Validation

Full test suite: 215 passed, 1 skipped.
