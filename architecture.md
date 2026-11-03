# Architecture

## 1. Goal

The system converts a Guitar Pro score and synchronized stems into a Feedpak while maintaining an explicit and auditable mapping from score time to recorded time.

Core principles:

1. Score events propose locations; audio confirms them locally.
2. Source audio, padded audio, nominal score, and warped output clocks remain distinct.
3. Every timing refinement is layered over a known baseline.
4. Output mappings must remain monotonic.
5. Exact manual anchors are immutable constraints.
6. Missing or rejected evidence is timing-neutral.
7. GP5 and alphaTab adapters converge on common internal products.
8. Song-specific tests must not become hard-coded production behavior.

## 2. Pipeline

```text
song folder
  -> score inventory and strict role allocation
  -> synchronized stem discovery
  -> four-beat count-in top-up
  -> optional vocal analysis
  -> score parsing and repeat expansion
  -> opening anchor and guarded baseline
  -> checkpoint and continuous-anchor evidence
  -> optional linear or local-DTW refinement
  -> event endpoint warping
  -> timeline validation
  -> Feedpak files and ZIP packaging
```

`build_feedpak.py` owns orchestration and packaging. `process_gp_alignment.py` owns score parsing, timing decisions, event warping, and alignment diagnostics.

## 3. Time domains

- **Source-audio time**: original recording.
- **Padded-audio time**: source time plus physically added count-in silence.
- **Nominal score time**: expanded score ticks converted through authored tempo events.
- **Chart offset**: `audio_content_start - score_content_time`.
- **Warped time**: final mapping into packaged audio.

Attacks and releases are warped independently. A nonlinear map therefore changes both note start and duration correctly.

## 4. Score adapters

### GP5

PyGuitarPro supplies tracks, notes, effects, meter, tempo changes, repeats, and lyrics. `project_config.py` inventories every track and blocks ambiguous or invalid explicit assignments before audio processing.

### alphaTab

`alphatab_score.py` consumes schema-v4 JSON, expands playback occurrences, prevents ties crossing discontinuous repeat jumps, and emits the same products as the GP5 path.

### Common products

```text
fretted_tracks
notation_tracks
drum_hits_gp
guitar/bass/piano alignment events
guitar/bass/piano pitch events
song_timeline_gp
key_events_gp
gp_vocals
```

## 5. Baseline alignment

The earliest configured scored instrument owns opening precedence. A later drum entrance cannot replace an earlier guitar or piano entrance only because it is easier to detect.

Global scale is estimated separately from start anchoring and accepted only inside guarded limits. Legacy DTW can still generate source-specific candidates, quality metrics, and disagreement reports.

## 6. Checkpoint evidence

- **Onset checkpoints** compare local predicted attacks with detected attacks.
- **Silence checkpoints** compare score gaps with inactive audio and post-gap activity.
- **Chroma checkpoints** compare local pitch-class templates with isolated stems.

Checkpoint-linear consolidates only sufficiently strong and consistent evidence. Checkpoint local DTW preserves interval endpoints and falls back exactly to the prior map on rejection.

## 7. Continuous anchors

### Release-aware structure

Fretted events are represented as sounding intervals. Overlapping intervals are merged. A gap between attacks is not silence while a sustain, tie, chord tone, or let-ring interval remains active.

### Structural candidates

Structural candidates include instrument entrances and restarts after at least one local measure of true score silence.

### Dense candidates

Dense candidates provide references inside active passages:

1. Collect unique onsets per source.
2. Partition them by expanded playback measure.
3. Consider every configured measure stride.
4. Require a minimum event count.
5. Select the first attack in the measure.
6. Enforce minimum spacing and a per-source cap.
7. Score the candidate.

The score is:

```text
0.35 event density
+ 0.25 downbeat proximity
+ 0.20 preceding-gap support
+ 0.20 rhythmic-pattern uniqueness
```

Rhythmic uniqueness uses event count and quantized within-measure offsets. Common repeated patterns receive lower confidence. Dense candidates near structural candidates are suppressed.

Configuration lives in `ContinuousAnchorConfig`:

```text
dense_candidates_enabled
dense_measure_stride
dense_minimum_events_per_measure
dense_minimum_spacing_seconds
dense_minimum_score_confidence
dense_maximum_candidates_per_source
```

### Local audio evidence

Each score candidate is mapped through the baseline to one predicted audio time. Only a bounded local window is examined, preferably on the corresponding isolated stem. The detector combines quiet-before, RMS rise, onset support, and sustained activity.

Dense candidate path confidence is adjusted by score confidence:

```text
adjusted = raw_audio_confidence * (0.75 + 0.25 * score_confidence)
```

Raw confidence remains in diagnostics.

### Clustering and monotonic path

Nearby score events become one score cluster. Nearby audio alternatives become candidate groups. Only one observation per source contributes to a group.

Dynamic programming chooses a global path using local confidence, corresponding-stem support, multi-source agreement, residual size, segment-scale deviation, scale change, and skip cost.

Hard constraints include increasing score and audio time, no improper audio-event reuse, scale bounds, and exact manual-anchor interval boundaries.

### Continuous linear warp

Selected automatic anchors and exact manual anchors form control points. Manual controls win at the same score position. The baseline is retained before the first control. Between controls the map is linear. After the final control the baseline slope is preserved while the last residual is held constant.

A non-monotonic or over-stretched proposal returns the exact baseline.

### Trusted-interval DTW

`continuous-anchor-dtw` can refine between accepted continuous/manual controls. Segment endpoints remain fixed. Rejected segments do not alter timing.

## 8. Manual anchors

`manual_anchors.py` resolves playback bars, repeated source bars, occurrences, beats, and beat fractions. `search` anchors are diagnostic. `exact` anchors are hard controls. Time basis can be original source audio or padded audio.

## 9. Timeline validation

Warped downbeats are checked for duplicates, backward motion, short measures, excessive stretch, and implausible effective BPM. Bounded local damage may be interpolated between valid neighbors. Unresolved severe timing blocks production packaging unless explicitly overridden.

Accepted downbeats create dense per-measure tempos and optional explicit beat markers.

## 10. Content warping

- Fretted attacks, releases, and bend curves follow the final map.
- Piano preserves staff/voice notation and RB3-compatible playable pitch encoding.
- Drum hits map to a deterministic closed reduced kit.
- Vocals preserve lyrics, discrete pitch, and detailed contour as separate layers.
- Key and meter events follow the same score-to-audio map.

## 11. Packaging

The builder writes arrangement JSON, notation, drum tab, song timeline, keys, lyrics, vocal pitch, contour, stems, cover, and `manifest.yaml`. Relative manifest paths are validated. A `.feedpak` is a ZIP archive.

## 12. Diagnostics

The alignment report preserves raw evidence, rejected hypotheses, selected anchors, confidence margins, segment scales, validation gates, timing changes, and fallback status. This allows failures to be classified as candidate-generation, local-audio, ambiguity, global-path, constraint, or stretch problems.

## 13. Tests

Regression tests cover timing-origin separation, repeat-aware ties, parser roles, manual anchors, checkpoint and trusted-segment DTW, structural and dense continuous anchors, monotonic path selection, piano timing, fretted techniques, vocal safety, stem routing, and global-scale guards.

## 14. Planned extensions

- Interchangeable automatic drum transcription into coarse families: kick, snare, hi-hat, cymbal, and tom.
- Symbolic drum alignment bounded by trusted structural anchors.
- A basic dense beatmap after anchor reliability is demonstrated.
- A visual alphaTab/waveform editor with anchor placement and method display.
- Optional learned or Bayesian candidate scoring behind the same interfaces.

## 15. Safety invariants

Any new method must preserve:

1. Monotonic output.
2. Immutable exact manual anchors.
3. Explicit time domains.
4. Timing neutrality when evidence is missing.
5. Exact fallback to the prior map after rejection.
6. Endpoint warping for sustained events.
7. Machine-readable reporting of every applied timing change.
8. General logic rather than song-specific production rules.

## Joint dense-anchor fingerprinting

Pitched dense candidates are now evaluated with synchronized rhythm and chroma fingerprints. Score rhythm uses onset occupancy; score chroma uses authored MIDI pitch classes across sounding intervals. Audio rhythm uses the onset-strength envelope, while audio chroma uses normalized harmonic CQT chroma. Candidate search is broad, but a Gaussian-mixture temporal prior decays with displacement from the trusted baseline. Independent rhythm and chroma thresholds, joint-identity threshold, and best-versus-runner-up ambiguity gates must all pass. Dense candidates are diagnostic-only by default.

Content-addressed NPZ caches under `_alignment_cache/features` recycle decoded stem features across safe and unsafe comparison passes. Cache identity includes audio content, sample rate, hop length, and feature version.

---

## Current alignment architecture: v15 golden through v16 Phase 1

### Stability boundary

- `v15.0-dsmn-perfect` is the listening-validated golden tag.
- v15.1 adds hybrid geometry: a short edge may pass on bounded absolute correction even when its stretch exceeds the normal mode range; long regions pass on reasonable stretch regardless of accumulated endpoint displacement.
- v16 Phase 1 accelerates the same exhaustive candidate search without changing grid density, candidate limits, scoring, tie order, role-path selection, fusion, or materialization policy.

### Asset boundary

Source separation and audio-to-MIDI transcription are external preparation stages. FeedPak consumes stems and optional performance-derived MIDI, records provenance, and evaluates whether evidence supports the chart. The future wizard should validate available assets and explain missing capabilities, but should not install or execute UVR, Moises, `audio-separator`, MuScriptor, or equivalent tools.

Chart-exported MIDI is never independent timing evidence. It may be used for parser or playback diagnostics, but it must not be assigned performance-clock authority.

### Producer-neutral logical roles

External filenames are normalized to logical roles:

- `piano`: `piano.mid`, `keys.mid`, `keyboard.mid`, and legacy `.muscriptor.mid` aliases;
- `guitar`: `guitar.mid`, `guitars.mid`, and legacy aliases;
- `bass`: `bass.mid` and legacy aliases;
- `drums`: `drums.mid`, `drum.mid`, `percussion.mid`, and legacy aliases.

File labels are claims, not proof of instrument identity. Downstream diagnostics must remain tolerant of stem contamination, missing notes, extra transcription events, and inaccurate charts.

### MIDI-section pipeline

1. Expand the chart into playback order while preserving written and playback measure identities.
2. Build overlapping duration-aware measure windows independently for piano, guitar, bass, and drums.
3. Normalize chart and prepared MIDI events into ordered onset groups; drums use shared canonical families.
4. Search the fixed performance-MIDI clock over the full stretch/translation grid.
5. Select independent globally monotonic per-role streaks; unsupported windows split authority.
6. Cluster compatible role predictions without averaging contradictory clusters.
7. Recover multiple independent monotonic regions after unsafe edges or reversals.
8. Materialize eligible regions by `conservative`, `permissive`, or `diagnostic-unsafe` policy.
9. Keep unsupported spans explicitly non-MIDI-validated while preserving baseline shape and safe offset continuity.

### Exact Phase-1 acceleration

Performance MIDI is grouped once per role. Sorted onset times support inclusive binary-search slicing. Each chart window caches exact symbolic DP results by MIDI slice indices, and both baseline hypotheses share that cache. Pitch-group similarity matrices are calculated once per slice and reused during DP and backtrace.

Diagnostics report transform trials, slice lookups, unique slices, DP evaluations, cache hits, cache-hit rate, per-role elapsed time, candidate generation time, path-selection time, fusion time, and materialization time.

Multiprocessing, Numba, GPU batching, duplicate-grid merging, coarse-to-fine search, approximate prefilters, and early termination remain deferred until exact parity and real-run profiling justify them.

### External analysis setup that must remain documented

The project still supports the established vocal workflow based on WhisperX and CREPE/torchcrepe. Keep the installation, FFmpeg, CUDA/device, model-cache, Hugging Face token, vocal-layout, and troubleshooting instructions in `README.md` whenever architecture documentation is revised.

MuScriptor remains optional external preparation. Its setup is not a FeedPak runtime dependency, and alternative audio-derived MIDI providers are valid if their output is placed on the original unpadded source-audio clock.


## Phase 2b cache boundary

Phase 2b.1 instruments the full build, while Phase 2b.2 caches deterministic preparation artifacts. Cache fingerprints use canonical sorted JSON and SHA-256; file identities include content hashes, sizes, modification times, and basename diagnostics. Writes are atomic. Cache load errors, schema mismatches, fingerprint mismatches, and missing artifacts degrade to ordinary recomputation rather than build failure.

The prepared-stem cache owns only `_feedpak_build/stems` artifacts and never changes original audio. The parsed-chart cache owns normalized AlphaTab products and is invalidated by score, project configuration, adapter, or schema changes. The MIDI-section cache owns exact candidate/fusion evidence. Global baseline construction, materialization, final warping, report generation, and packaging remain active on every build.

Cache behavior is observable through `cache_file_missing`, `exact_cache_hit`, `schema_mismatch`, `fingerprint_mismatch`, `cached_artifact_missing`, and `cache_load_error` states. The final pipeline report separates inspection, stem preparation, vocals, GP/audio alignment, manifest, packaging, report writing, and total runtime.


### Expected-role observability and cache ownership

Inactive chart roles are neutral rather than absent corroborators. A single locally expected role can supply complete support when its control confidence is strong. Candidate-cache ownership remains inside the MIDI-section module; downstream path selection, fusion, solo-role annotation, materialization, and packaging always rerun. Invalid baselines require a validated acquisition root or materialization is blocked.
