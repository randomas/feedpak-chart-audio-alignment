# Incipit Role Tracking v7

This build adds a final, separate refinement to prepared-MIDI sequence authority.

## Behavior

1. Classifies the chart opening as solo, dominant, ensemble, or uncertain.
2. For solo/dominant openings, nominates the lead role from chart activity only.
3. Acquires the first recognizable phrase of that role around both expected origin and offset-method incipit.
4. Tracks subsequent phrases sequentially around the active projected transform.
5. Treats insufficient MIDI as an unobserved phrase, coasts without creating controls, and permits one cautious reacquisition.
6. Reports explicit lock acquisition, unobserved phrases, loss reason, reacquisition, and release action.
7. Promotes a streak only after at least three accepted phrases.
8. Applies a confirmed streak only inside its supported interval. Outside the streak, timing returns to the baseline rather than extrapolating the MIDI correction.
9. Ensemble or uncertain openings retain the existing global sequence-chain behavior.

## Important diagnostics

`transcription_investigator.json` now contains `incipit_role_tracking` with:

- `opening_role`
- `state`
- `events`
- `streaks`
- `representative_controls`
- `release_reason`

## Validation

Full test suite: 218 passed, 1 skipped.
