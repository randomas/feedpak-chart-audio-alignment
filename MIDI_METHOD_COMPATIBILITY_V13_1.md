# MIDI method compatibility v13.1

## Supported guarded methods

Both `--transcription-investigator` and diagnostic `--midi-section-fit` are
available with:

- dtw-checkpoint
- checkpoint-linear
- checkpoint-dtw-diagnostic
- checkpoint-dtw-selective
- continuous-anchor-linear
- continuous-anchor-dtw
- continuous-anchor-structural
- bar-landmark-regularized

They remain unavailable for nominal, offset, linear, and unguarded dtw.

## Dual search centres

MIDI section candidate generation now searches around two explicit hypotheses:

1. `requested_method_baseline`, supplied by the selected guarded alignment
   method;
2. `prepared_chart_clock`, the fixed prepared chart/MIDI clock fallback.

Every retained section candidate records `search_hypothesis`. Candidate
occurrences from both hypotheses are pooled, ranked, and deduplicated before the
hard-monotonic section path selector runs. A poor requested-method baseline
therefore cannot hide an otherwise useful prepared-clock MIDI occurrence.

## Scope

This compatibility patch remains diagnostic-only. It does not make section-fit
controls authoritative and does not add MIDI-only materialization.

Full suite: 255 passed, 1 skipped.
