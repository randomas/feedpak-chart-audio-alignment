# v16.7R1 audio-evidence reuse audit

v16.7R restarts from the v16.6E.3 functional baseline. Experimental v16.7A/A.1 threshold detectors remain research artifacts and are not production dependencies.

## Governing policy

- Performance MIDI remains the primary dense timing source.
- Audio evidence is stricter, sparse, and complementary.
- Missing or ambiguous audio evidence is neutral and cannot disauthorize reliable MIDI.
- Exact manual anchors outrank automatic evidence.
- Stem labels are nominal model targets, not verified instrument identity.
- `insufficient_strict_audio_evidence` is a valid result.

## Capability owners

| Concept | Existing owner | Decision |
|---|---|---|
| Physical leading silence | `feedpak_common.detect_leading_silence_seconds` | Keep |
| Score-informed opening | opening-anchor functions in `process_gp_alignment.py` | Wrap |
| Incipit clock comparison | `incipit_three_clock.py` | Keep and wrap |
| Exact/manual controls | `manual_anchors.py` | Keep |
| Source activity span and coverage | coverage functions in `process_gp_alignment.py` | Wrap |
| Full-mix terminal activity | `detect_audio_content_end` | Wrap |
| Local transition and fingerprint features | `continuous_anchors.py` | Public read-only adapter |
| Source-neutral novelty and dependence | `structural_search.py` | Wrap |
| Competing structural occurrences | `structural_search.analyse` | Query adapter later |
| Role-specific evidence | `instrument_evidence.py`, `beatmap_evidence.py` | Wrap |
| Regional fusion and proposals | `regional_source_fusion.py`, `alignment_proposals.py` | Keep |
| Dense performance timing | transcription and MIDI-section modules | Primary authority |
| Global A/A.1 event thresholds | archived experiment | Experimental only |

## Anti-overfitting rules

No named-song branches, corpus timestamps, or thresholds selected solely to recover an expected event. Behavioral changes require a general problem statement, synthetic tests, reuse of existing subsystems, holdout evaluation, and preservation of reliable MIDI by default.

## R2 boundary

R2 adds only a common JSON-serializable schema and read-only adapters. It does not import detector modules, decode audio, execute detectors, alter reports, change scores, select paths, create anchors, or modify timing.
