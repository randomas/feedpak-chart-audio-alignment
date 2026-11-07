# v16.7R1 Audio-evidence reuse audit

## Status and scope

v16.7R1 is a documentation-only architecture checkpoint based on the tagged v16.6E.3 code line. It does not include the experimental v16.7A/A.1 detector implementation.

The purpose is to identify existing audio-evidence owners, stop duplicate feature extraction and threshold semantics, and define the safe adapter boundaries for a later common evidence layer.

No candidate score, path, fusion result, materialization decision, method selection, manual anchor, or final timing is changed by R1.

## Governing policy

- MIDI remains the primary dense timing source.
- Audio evidence is stricter and complementary.
- Audio should fill MIDI blind spots with a small number of strong structural observations or anchors.
- Missing or ambiguous audio evidence is neutral and must not disauthorize reliable MIDI evidence.
- Exact manual anchors remain immutable and outrank automatic evidence.
- Stem labels are nominal model targets, not verified instrument identity.
- `insufficient_strict_audio_evidence` is a valid outcome.
- No song name, corpus timestamp, or song-specific threshold may enter production logic.

## Audit decisions

Allowed decisions in R1:

```text
KEEP
WRAP
EXTRACT_WITHOUT_BEHAVIOR_CHANGE
MERGE_LATER
EXPERIMENTAL_ONLY
```

R1 makes no deletion or behavior-changing extraction.

# Capability matrix

## 1. Physical leading silence and prepared-stem clock

### Current owner

```text
feedpak_common.detect_leading_silence_seconds
build_feedpak prepared-stem/count-in stage
```

### Purpose

Measures physical leading space in the source/prepared audio and supports count-in padding. This is a file/preparation property, not a musical-incidence detector.

### Clocks

```text
source audio clock
prepared padded audio clock
padding_added
```

### Decision

```text
KEEP
```

Do not merge leading-silence measurement with musical activity or incipit qualification.

---

## 2. Score-informed opening observation

### Current owner

```text
process_gp_alignment._detect_lenient_opening_time
process_gp_alignment.build_opening_anchor_candidates
process_gp_alignment.choose_opening_anchor
```

### Existing evidence

The opening detector already provides bounded, score-informed local audio search using:

- the expected opening time;
- RMS activity;
- onset support;
- sustained-activity checks;
- source-specific candidates;
- distance from the expected opening;
- guarded candidate selection.

### Current consumers

```text
initial/global alignment construction
opening-anchor diagnostics
guarded global span alignment
```

### Risks of a second implementation

- two incompatible definitions of first musical activity;
- later loud passages dominating an independent whole-song threshold;
- duplicate audio decoding;
- different onset times in opening, coverage, and structural reports.

### Decision

```text
WRAP
```

R2 should adapt the existing candidate report into a common evidence schema without changing candidate generation or selection.

---

## 3. Opening clock comparison and root status

### Current owner

```text
incipit_three_clock.py
```

### Existing evidence

Compares prepared-chart, requested-method, and observed-MIDI incipit clocks. Reports validated, regional, or unrooted status while preserving clock provenance.

### Decision

```text
KEEP + WRAP
```

Strict audio opening observations should later be represented as an additional observation supplied to the root-qualification layer. They must not create a parallel root model.

---

## 4. Manual opening and boundary controls

### Current owner

```text
manual_anchors.py
process_gp_alignment.apply_manual_anchor_layer
```

### Existing authority

Exact manual anchors are immutable production controls. Search anchors are diagnostic and soft.

### Decision

```text
KEEP
```

Any later sparse structural-audio anchor must yield to exact manual anchors and report conflicts rather than replacing them.

---

## 5. Musical sustained-activity span and role coverage

### Current owner

```text
process_gp_alignment._activity_span_from_features
process_gp_alignment.diagnose_track_audio_coverage
```

### Existing evidence

- first sustained source activity;
- last sustained source activity;
- onset and RMS support;
- mapped score coverage;
- early/late source coverage disagreement;
- neutral handling when evidence is missing.

### Decision

```text
WRAP
```

Use coverage evidence as a source observation. Do not treat missing coverage as proof that MIDI is wrong or that the nominal instrument is absent.

---

## 6. Full-mix terminal activity

### Current owner

```text
process_gp_alignment.detect_audio_content_end
process_gp_alignment.estimate_last_measure_end
```

### Existing evidence

`detect_audio_content_end` already estimates the last sustained full-mix activity, rejects isolated codec clicks and low-level tail noise, and reports the threshold and file duration.

### Decision

```text
WRAP
```

A future terminal-boundary report should adapt this output. Do not add another whole-file terminal detector.

---

## 7. Continuous-anchor feature extraction and local audio transitions

### Current owner

```text
continuous_anchors._features
continuous_anchors.inspect_chart_event
continuous_anchors.inspect_dense_fingerprint
continuous_anchors.diagnose_continuous_anchors
```

### Existing features

The continuous engine already computes and caches source features keyed by audio content, sample rate, hop length, and feature version. It supports:

- RMS/activity transition candidates;
- onset evidence;
- score-informed local search;
- rhythm and harmonic fingerprint comparison;
- temporal priors;
- competing hypotheses;
- monotonic path preview.

### Decision

```text
WRAP PRIVATE PROVIDER
```

R2 should add a stable public adapter around the existing provider rather than moving or rewriting feature calculations. Cache keys and current continuous-anchor outputs must remain unchanged.

---

## 8. Structural source-neutral features and source quality

### Current owner

```text
structural_search._common_source_feature
structural_search.source_quality
structural_search._common_similarity_details
```

### Existing features

`_common_source_feature` uses `soundfile`, explicit mono conversion, controlled resampling, RMS, spectral flux, and low-frequency energy to create source-neutral novelty features. `source_quality` evaluates usability and dependence between stems so correlated separated outputs are not treated as fully independent evidence.

### Decision

```text
WRAP
```

This is the preferred existing source-quality and source-dependence basis for structural occurrence comparisons. Do not create a second independence model in the structural report.

---

## 9. Repeated structural occurrence search

### Current owner

```text
structural_search._evaluate_window
structural_search._sequence_window_score
structural_search._cluster_hypotheses
structural_search._rank_competing_clusters
structural_search._validate_structural_cluster
structural_search.analyse
```

### Existing evidence

- region/window-based search;
- shift and lag tolerance;
- multi-role/source combination;
- source quality and dependence;
- competing structural clusters;
- validation against alternatives;
- preservation of ambiguity.

### Decision

```text
KEEP + ADD QUERY ADAPTER LATER
```

Occurrence-switch questions should be routed to this machinery. Do not infer repeated-occurrence identity from a new global silence/re-entry detector.

---

## 10. Instrument-specific audio evidence

### Current owner

```text
instrument_evidence.py
beatmap_evidence.py
```

### Existing evidence

These modules already adapt role-specific audio observations into diagnostic evidence while preserving source role, support, limitations, and timing context.

### Decision

```text
KEEP + WRAP
```

R2 should map existing outputs into the common evidence schema. No new role-specific detector is justified in R1.

---

## 11. Regional source fusion and proposals

### Current owner

```text
regional_source_fusion.py
alignment_proposals.py
```

### Existing evidence

Combines regional observations, retains provenance and disagreement, and emits proposals without requiring every source to agree globally.

### Decision

```text
KEEP
```

A later structural-audio aggregator should present these results rather than inventing a parallel confidence/fusion formula.

---

## 12. Performance-MIDI evidence and authority

### Current owner

```text
transcription_investigator.py
midi_section_fit.py
rooted_phrase_support.py
section_evidence_quality.py
phase_migration_qualification.py
```

### Existing evidence

- independent recording-derived MIDI candidates;
- incipit-role tracking;
- phrase correspondence;
- monotonic role and global paths;
- section evidence support;
- phase stability;
- occurrence ambiguity;
- probable occurrence switches;
- gaps and rejected regions.

### Decision

```text
KEEP AS PRIMARY DENSE AUTHORITY
```

Audio queries should normally originate where these diagnostics identify a blind spot. Missing audio evidence must leave MIDI unchanged.

# Duplicate concepts identified by v16.7A/A.1

The experimental A/A.1 branch independently implemented:

```text
whole-song low/high activity thresholds
deep-silence thresholds
source re-entry after reduction
duplicate re-entry suppression
a second source-onset definition
a second terminal-activity definition
a second cross-source confidence formula
```

These are retained only in archived experimental packages and notes. They are not carried into the clean R line as production algorithms.

Useful A/A.1 design outputs to preserve later:

```text
explicit source/padded/chart/mapped clocks
detector provenance
limitations and risk factors
confidence separated from authority
compact sidecar/light-report presentation
diagnostic-only safety fields
```

# Owner map

| Concept | Owner | R1 decision |
|---|---|---|
| Physical leading silence | `feedpak_common.detect_leading_silence_seconds` | KEEP |
| Score-informed opening observation | opening-anchor functions in `process_gp_alignment.py` | WRAP |
| Incipit/root clock comparison | `incipit_three_clock.py` | KEEP + WRAP |
| Exact/manual boundary control | `manual_anchors.py` | KEEP |
| Source activity span/coverage | coverage functions in `process_gp_alignment.py` | WRAP |
| Full-mix terminal activity | `detect_audio_content_end` | WRAP |
| Local audio transitions and fingerprints | `continuous_anchors.py` | WRAP PRIVATE PROVIDER |
| Source-neutral novelty and dependence | `structural_search.py` | WRAP |
| Competing structural occurrences | `structural_search.analyse` | KEEP + QUERY ADAPTER |
| Role-specific evidence | `instrument_evidence.py`, `beatmap_evidence.py` | KEEP + WRAP |
| Regional evidence/proposals | `regional_source_fusion.py`, `alignment_proposals.py` | KEEP |
| Dense performance timing | transcription and MIDI-section modules | PRIMARY AUTHORITY |
| Experimental global event inventory | archived v16.7A/A.1 | EXPERIMENTAL_ONLY |

# Proposed R2 common evidence schema

R2 may add `audio_evidence_schema.py` with immutable, JSON-serializable adapters. The schema must preserve the origin report rather than reinterpret it.

```json
{
  "evidence_id": "audio-evidence-42",
  "evidence_type": "score_informed_opening_activity",
  "detector": {
    "module": "process_gp_alignment",
    "method": "_detect_lenient_opening_time",
    "version": 1
  },
  "source": "guitar",
  "nominal_role": "guitar",
  "observed_signal_type": "sustained_activity",
  "clocks": {
    "source_audio_time": 0.31,
    "padded_audio_time": 2.71,
    "chart_time": 2.40,
    "mapped_audio_time": 3.01
  },
  "residual_seconds": -0.30,
  "confidence": 0.72,
  "positive_factors": ["persistent_activity", "near_expected_opening"],
  "limitations": ["single_source", "nominal_role_not_verified"],
  "authority": "diagnostic",
  "provenance": {
    "origin_report": "opening_anchor_candidates",
    "origin_id": "guitar-opening-1"
  }
}
```

Schema rules:

- missing values remain missing, never zero-filled;
- clocks remain explicitly named;
- detector and version are mandatory;
- confidence does not imply authority;
- nominal stem role is separate from observed signal type;
- contradictory observations remain separate records;
- raw evidence is not merged merely because timestamps are close;
- adapters must not modify origin reports.

# Proposed generic diagnostic queries

After R2/R3, MIDI and structural diagnostics may request strict audio evidence through generic queries:

```text
opening_root_uncertain
competing_occurrence_boundary
persistent_phase_transition
materialization_gap_boundary
rejected_region_boundary
terminal_mismatch
```

A query supplies a chart interval and provenance. It does not provide a song name, hard-coded timestamp, or expected answer.

The response may be:

```text
strict_audio_anchor_candidate
supporting_audio_observation
ambiguous_audio_observation
contradictory_but_unqualified
insufficient_strict_audio_evidence
```

Only the first class may later be considered for sparse authority, and no authority is applied before v16.7C.

# Anti-overfitting rules

Every behavioral change after R1 must:

1. state a general signal or musical problem;
2. reuse an existing subsystem where possible;
3. include synthetic tests independent of named songs;
4. contain no named-song branch;
5. contain no corpus-derived timestamp;
6. preserve or improve holdout behavior;
7. avoid a second definition of an owned concept;
8. keep complexity stable or lower;
9. preserve reliable MIDI by default;
10. accept insufficient audio evidence as correct.

The six-song corpus and Caught in a Web are regression evidence, not training specifications.

# Preservation tests required before R2/R3

- Adapter output retains detector identity and original values.
- Origin reports are unchanged after adaptation.
- Source/padded/chart/mapped clocks remain distinct.
- Padding is applied exactly once.
- Continuous feature-cache keys remain unchanged.
- Continuous candidates, scores, and selected paths remain unchanged.
- Structural-search scores and cluster ranking remain unchanged.
- MIDI-section candidates and caches remain unchanged.
- Manual-anchor behavior remains unchanged.
- Final timing remains byte-equivalent or tolerance-equivalent as appropriate.
- Missing audio evidence cannot lower MIDI authority.

# R2 implementation boundary

Likely new files:

```text
audio_evidence_schema.py
tests/test_audio_evidence_schema.py
tests/test_audio_evidence_reuse_adapters.py
tests/test_audio_evidence_preservation.py
```

Likely adapters:

```text
opening candidate adapter
coverage/activity-span adapter
terminal-content adapter
continuous-anchor observation adapter
structural-search hypothesis adapter
instrument-evidence adapter
```

Default implementation approach:

```text
public wrappers over existing providers
```

Do not move private feature implementations in R2. Extraction into a neutral feature module is permitted only after equivalence tests prove unchanged arrays, frame times, cache keys, candidate scores, and selected outputs.

# R1 acceptance result

v16.7R1 is complete when this audit is reviewed and accepted. It intentionally adds no detector, threshold, query execution, anchor, or timing behavior.

Recommended next stage:

```text
v16.7R2 common evidence schema and read-only adapters
```
