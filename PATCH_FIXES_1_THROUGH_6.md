# Reporting and QoL fixes 1 through 6

## Implemented

1. Optional unsafe variant failures no longer fail a successful production build. Added `--strict-variant-builds`.
2. Variant control discovery supports current, explicit, selected-path, control-list, and historical report layouts. Expected missing controls now use structured diagnostics rather than `ValueError`.
3. Checkpoint DTW reports explicit segment lifecycle counters from generation through application.
4. Production timing reporting separates base, checkpoint-linear, continuous-linear, local-DTW, sanitization, selected method, and fallback reason.
5. Continuous alignment validation includes passed/failed predicates and a validation snapshot with control, monotonicity, stretch, and manual-anchor compatibility fields.
6. Diagnostics default to a compact score-schema summary, hash-deduplicate repeated source artifacts, keep only the newest project config, and copy the full schema only with `--include-full-score-schema`.

## Verification

Full test suite: 193 passed.

## Diagnostics hotfix (2026-10-23)

- Fixed the corpus runner/collector CLI mismatch (`--diagnostics-dir` and `--log` are now accepted).
- The run log is collected for both successful and failed builds.
- All readable JSON, log, text, CSV, YAML, and YML run artefacts are collected from the output tree.
- Every `.feedpak` is expanded under `feedpak_ascii` for direct inspection.
- `active_score.json` / full active-score schema files remain omitted.
- For dated YAML snapshots, only the newest is retained; ordinary non-dated YAML files are retained.

## Diagnostics layout hotfix 2 (2026-10-23)

- Feedpak extraction now writes only UTF-8 text members with approved ASCII/report suffixes.
- Audio stems and all other binary archive members are omitted and recorded in the diagnostics manifest.
- The corpus runner now creates its main `logs` directory inside the run-level `Diagnostics` directory.

## DTW-selective reporting correction (2026-10-23)

- Separates feature-unavailable candidates from feature-evaluated candidates rejected by quality/improvement gates.
- Reports segment-level `no_source_passed_gates` separately from true `insufficient_features`.
- Adds source-candidate totals, unavailable-reason counts, per-source state counts, accepted/rejected candidate totals, and truthful boundary-assignment counts.
- Console reporting now prints feature-evaluated, unavailable, no-meaningful-improvement, and accepted source candidate counts.
- Layered production reporting now recognizes coarse scale/opening changes and checkpoint-linear changes independently of local DTW.
- Selective fallback reason now uses the production status, e.g. `checkpoint_linear_fallback_no_accepted_segments`, rather than `diagnostic_only`.
- Diagnostics ASCII-only extraction and `Diagnostics/logs` layout fixes are retained.

## Switchable drift-adjusted structural second pass (2026-10-23)

- Added `--structural-second-pass`, valid only with `continuous-anchor-structural`.
- Pass 1 builds the ordinary conservative monotonic preview.
- A bounded provisional residual prior interpolates between selected Pass 1 anchors and holds the nearest residual outside the controlled span.
- Pass 2 reruns candidate discovery around the adjusted centres.
- Baseline-centred and drift-centred hypotheses are merged and the complete monotonic path is reselected; the provisional path is not automatically accepted.
- Reports recovered events, matches found by both centres, baseline-only matches, drift-only matches, centre distances, the provisional controls, and the final selected-anchor count.
- The switch is wired through `build_feedpak.py`, `process_gp_alignment.py`, `run_all_test_materials.ps1`, and `run_all_alignment_methods.ps1`.
- Existing continuous structural behavior is unchanged when the switch is absent.

## Full alphaTab schema diagnostics omission fix (2026-10-23)

- `*.alphatab.json`, `*schema4*.json`, and active-score JSON files are now omitted from diagnostics by default even though the general source collector includes `*.json`.
- The compact `active_score_schema_summary.json` remains available by default.
- `--include-full-score-schema` explicitly restores full schema collection.

## Pluggable transcription investigator (2026-10-23)

- Added one method-independent switch: `--transcription-investigator` / `-TranscriptionInvestigator`.
- Added optional Basic Pitch adapter for piano/guitar/bass stems.
- Added optional DrumScript adapter for the isolated drums stem.
- Added SHA-256/config/provider-version caches.
- Added normalized pitched-note and drum-family event envelopes.
- Added sparse dense-anchor-matrix correspondence diagnostics using pitch/rhythm or drum/rhythm evidence.
- Added a dedicated `.transcription_investigator.json` sidecar.
- Missing or failed optional providers are nonfatal and explicitly reported.
- Initial investigator mode is diagnostic-only and cannot alter production timing.

## Prepared MuScriptor evidence ingestion

- Replaced direct Basic Pitch and DrumScript inference with externally prepared MuScriptor MIDI ingestion.
- Added canonical optional filenames for piano, guitar, bass, and drums.
- Added non-destructive `source_time + padding_added = audio_time` projection.
- Added tempo-map-aware MIDI parsing through `mido`.
- Added always-on forward and backward asymmetric partial-sequence matching.
- Extra transcription notes are low-cost skips; missing chart notes are tolerated; order is preserved.
- Added bidirectional confirmation, runner-up uniqueness, local stretch, timing prior, monotonic path preview, and explicit unsupported gaps.
- Restricted the investigator to guarded/checkpoint/continuous/bar-landmark methods.
- The all-method runner skips nominal, offset, linear, and unguarded DTW automatically.
- Kept all transcription evidence diagnostic-only.

## Phrase-chain matcher v2

- Melody instruments are grouped into simultaneous onset sets before comparison.
- Matching now operates on variable event-count phrases, not individual notes or fixed overlapping note windows.
- The asymmetric dynamic program returns concrete matched score/MIDI onset-group pairs.
- Timing offset, stretch, residuals, and phrase placement are computed only from matched pairs using a robust median affine fit.
- Adjacent hypotheses are clustered into distinct audio locations before uniqueness is measured.
- Phrase fingerprints retain directed root intervals and normalized relative spacing for solos, riffs, bass lines, and ostinati.
- Independent forward and backward global dynamic chains are reconciled at a meet-in-the-middle candidate.
- MuScriptor drum MIDI notes are translated with the project canonical `gm_drum_to_piece` map.
- Both score and transcription drums use the same reduced Rock Band 3 Pro-style kit vocabulary.
- Simultaneous drum hits are grouped into hit sets and matched as ordered groove/fill patterns.
- Kick and snare receive greater drum-pattern weight; extra detected pieces remain low-cost.
- Added per-role diagnostics for score groups, MIDI groups, viable phrase windows, candidates, and strong candidates.

## MuScriptor global placement and structural Pass 2 integration

1. MIDI phrase discovery now uses a method-independent nominal identity or accepted opening-anchor/global-scale prior, never the selected alignment method's warp as its primary centre.
2. Chart phrases are searched in a narrow nominal corridor first, then a wider nominal corridor, followed by a recovery corridor around an improved provisional path.
3. Accepted opening and compatible manual anchors are passed as immutable global path boundaries.
4. Forward and backward dynamic paths are independently propagated and reconciled at a meet-in-the-middle candidate.
5. Global edges are rejected unless their implied chart/audio stretch is within 0.9–1.1.
6. Low-local-uniqueness repeated motifs can be promoted only by a valid globally constrained chain.
7. Reliable transcription phrases may centre structural Pass 2 searches, but remain diagnostic and are never direct production controls.
8. A second MIDI recovery pass runs after structural Pass 2 around the improved search prior.
9. Reports now separate `local_phrase_confidence` from `global_placement_confidence` and record nominal-prior error.
10. Production safety is unchanged: all transcription candidates keep `production_eligible=false`, `timing_changes_applied=false`, and `candidate_count_applied=0`.

## Trusted performance-MIDI authority integration

- Consolidates overlapping selected phrase observations into robust trusted MIDI segments.
- Fits each segment independently and requires 0.9–1.1 stretch, median residual <=120 ms, and maximum residual <=350 ms.
- Materializes sparse representative MIDI controls instead of using every overlapping phrase window.
- Promotes a sufficiently covered, bounded MIDI path to the primary automatic timing clock.
- Evaluates structural audio candidates against that MIDI clock as strong corroboration, compatible support, uncertainty, or conflict.
- Incompatible audio evidence is retained diagnostically but cannot pull the trusted MIDI path.
- Structural Pass 2 searches around the trusted MIDI clock and its final warp validation is relative to that clock, avoiding enormous corrections against a displaced drum baseline.
- Existing audio methods remain active for corroboration and for regions where MIDI authority is insufficient.

## Force MIDI authority and production trace
- Adds `--force-midi-authority` and PowerShell `-ForceMidiAuthority`.
- Requires transcription investigation plus structural second pass.
- Bypasses soft MIDI authority gates but still requires at least two finite strictly monotonic MIDI controls.
- A blocked force request terminates with an explicit reason rather than silently falling back.
- Adds `midi_timing_decision` to the alignment report and a final `MIDI TIMING TRACE` log line.
- Final timing output explicitly identifies forced/trusted MIDI baseline with or without compatible audio refinement.
