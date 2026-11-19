# v16.7K3.1 exact-equivalent MIDI candidate caching and progress

## Purpose

K3.1 reduces repeated MIDI-section candidate work and makes long searches observable without changing search grids, scores, validity gates, candidate ordering, paths, fusion, authority, or materialization.

## Legacy monolithic compatibility

The existing `.feedpak_cache/midi_section_candidates.json` remains the first lookup. A valid exact monolithic cache is accepted unchanged. Its role reports are copied losslessly into role-level caches; the legacy file is retained.

## Role-level caches

Role caches are stored under `.feedpak_cache/midi_section_candidates.roles/` as `bass.json`, `drums.json`, `guitar.json`, `piano.json`, and `synth.json`. Each fingerprint contains only that role's chart events, prepared MIDI events, playback measures, candidate configuration, and baseline samples. Adding or changing Synth does not invalidate unchanged roles.

Each completed role is committed atomically. If a later role is interrupted, completed role work survives.

## Exact equivalence

K3.1 does not prune candidates or alter measure windows, radii, translation/stretch grids, DP scoring, pitch rewards, event minimums, residual gates, occurrence filtering, candidate limits, ordering, path selection, fusion, or materialization.

## Progress

On a miss, the console prints role/window counts, transform trials, DP evaluations, memory-cache hit rate, elapsed time, and ETA at most every 15 seconds and at completion. Exact role-cache hits are explicit.

## Piano/Synth collision

When metadata defines no Piano arrangement and no explicit Piano alignment stem, a generic Keys stem is no longer implicitly routed to legacy Piano. An explicitly configured Synth stream may use that source with unverified nominal identity.
