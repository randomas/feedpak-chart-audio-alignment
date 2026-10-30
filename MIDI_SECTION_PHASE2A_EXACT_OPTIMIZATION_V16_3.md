# MIDI section Phase 2A exact optimization v16.3

Phase 2A preserves the v16.2 search grid, scoring, candidate limits, path
selection, fusion, episode density, length-adjusted geometry, and final timing.

## Changes

- Persistent content-addressed candidate-report cache in each song folder at
  `.feedpak_cache/midi_section_candidates.json`.
- SHA-256 fingerprint covers playback measures, chart events, prepared MIDI,
  candidate configuration, schema version, and sampled values from every
  search baseline at every measure boundary.
- Atomic cache writes and explicit hit/miss reasons.
- `--no-midi-section-cache`, `--rebuild-midi-section-cache`, and
  `--midi-section-cache PATH` controls.
- Exact rolling-row DP score storage and compact byte backtrace storage.
- Similarity, recurrence option order, tie behavior, normalized score, and
  matched-pair output remain reference-compatible.
- Invalid transforms are rejected before landmark dictionaries and candidate
  dataclasses are allocated.
- New rejection-allocation and cache diagnostics.

## Cache safety

The cache is reused only on an exact schema and fingerprint match. Any chart,
MIDI, playback clock, baseline sample, padding consequence, or search-config
change causes a miss and complete rebuild. Policy-only materialization changes
can reuse the exact candidate inventory.

## Validation

- Exact DP/reference parity remains covered.
- Cache hit, input invalidation, rebuild, disable, and candidate-report equality
  are tested.
- Full suite: 322 passed, 1 skipped.
