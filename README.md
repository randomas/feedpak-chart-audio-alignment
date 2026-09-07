# Feedpak Audio Alignment Builder

An automated Python pipeline for converting separated audio stems and a Guitar Pro chart into a Feedpak archive for feedBack.

The pipeline extracts playable arrangements, analyzes vocals, aligns Guitar Pro timing to the recorded performance, creates song-level timing data, writes Feedpak side files, and packages the result as a `.feedpak` ZIP archive.

## Current capabilities

- Guitar Pro parsing for guitar, bass, drums, and piano/keys
- Guarded multi-reference audio alignment using available drum, bass, and piano/keys stems
- Probability-based candidate scoring, source selection, and agreement-gated fusion
- Instrument-specific frequency treatment for onset detection
- Automatic four-beat lead-in padding and matching chart offset
- Dense post-DTW per-measure tempo reconstruction
- Optional explicit bar and beat markers for feedBack compatibility testing
- Structural validation and repair of corrupted DTW-derived measure boundaries
- Severe-alignment rejection unless explicitly overridden
- Machine-readable alignment diagnostics
- WhisperX transcription, forced alignment, and speaker diarization
- Syllable-level lyric timing
- Merged, separated, or combined vocal layouts
- One representative CREPE pitch per syllable
- Independent fine-grained vocal pitch contour
- Drum-tab extraction and reduced-kit mapping
- Hand-position anchor generation
- Key-signature extraction
- Standard-notation output for piano/keys

---

## 1. Project scripts

### `build_feedpak.py`

The top-level orchestrator. It:

1. Finds the Guitar Pro file and audio stems.
2. Detects or adds the required four-beat lead-in.
3. Runs vocal analysis when requested.
4. Runs Guitar Pro parsing and guarded multi-reference alignment.
5. Writes arrangement and side-file JSON.
6. Builds `manifest.yaml`.
7. Packages the final `.feedpak`.
8. Writes an alignment report beside the archive.

### `process_gp_alignment.py`

Parses the Guitar Pro file and creates the chart-related intermediate data. It:

- extracts fretted arrangements;
- extracts drum hits;
- extracts piano/keys notation;
- reads key signatures and meter changes;
- builds independent drum, bass, and piano timing candidates;
- compares candidate quality and agreement;
- selects or safely combines timing candidates;
- validates and repairs warped measure boundaries;
- creates the dense tempo matrix;
- optionally creates an explicit beat grid;
- applies the accepted timing map to chart content;
- and produces the alignment report.

### `process_vocals.py`

Runs:

- WhisperX transcription;
- forced word alignment;
- speaker diarization;
- syllable division;
- and CREPE pitch tracking.

Lyrics, representative pitch notes, and the fine pitch contour are maintained as separate timing layers.

### `feedpak_common.py`

Contains shared helpers for:

- logging;
- Guitar Pro tick-to-time conversion;
- tuning conversion;
- drum mapping;
- DTW path cleanup;
- JSON output;
- Feedpak-relative paths;
- leading-silence detection;
- and stem padding.

---

## 2. Environment and setup

### 2.1 System requirements

#### Windows

Windows 10 or newer is recommended.

Windows Developer Mode is also recommended:

```text
Settings > Privacy & security > For developers > Developer Mode
```

This reduces symbolic-link problems when Hugging Face downloads models.

#### Python

Python 3.9 or newer is recommended.

Check the installed version:

```powershell
python --version
```

#### FFmpeg

FFmpeg is required for audio decoding, padding, conversion, and packaging.

Install it with Windows Package Manager:

```powershell
winget install Gyan.FFmpeg
```

Close and reopen PowerShell, then verify:

```powershell
ffmpeg -version
```

### 2.2 CUDA hardware guidance

The chart parsing and DTW stages are primarily CPU-driven. CUDA is most important for:

- WhisperX transcription;
- forced alignment;
- speaker diarization;
- and CREPE pitch tracking.

Practical GPU guidance:

- **4 GB VRAM:** not recommended for the complete vocal pipeline.
- **6 GB VRAM:** may work only with reduced Whisper settings and batch size.
- **8 GB VRAM:** practical minimum for CUDA vocal processing.
- **12 GB VRAM:** recommended.
- **16 GB or more:** comfortable for long songs and larger batches.

An NVIDIA RTX 3060 with 12 GB VRAM is suitable for the default CUDA workflow.

Monitor GPU memory during processing:

```powershell
nvidia-smi -l 1
```

If CUDA runs out of memory, reduce the Whisper batch size before changing alignment settings.

### 2.3 CPU fallback

If the NVIDIA GPU or eGPU is unavailable, the safest fallback is to skip vocal analysis while continuing chart and timing development:

```powershell
python build_feedpak.py my_song output_folder `
  --device cpu `
  --skip-vocals `
  --keep-work-dir
```

This still performs:

- count-in preparation;
- Guitar Pro parsing;
- drum, bass, and piano candidate alignment;
- timing validation and repair;
- tempo and beat generation;
- drum-tab extraction;
- notation output;
- and Feedpak packaging.

CPU-only WhisperX and CREPE are substantially slower. The current default vocal configuration is optimized for CUDA, so CPU vocal processing may also require a smaller Whisper model, INT8 computation, and a reduced batch size in a future configuration update.

### 2.4 Create and activate a virtual environment

Create the environment:

```powershell
python -m venv env
```

Activate it:

```powershell
.\env\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Then activate the environment again.

### 2.5 Install PyTorch with CUDA

Install the CUDA-enabled PyTorch build before WhisperX.

Example for CUDA 12.4:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Use the PyTorch build compatible with the installed NVIDIA driver.

### 2.6 Install project dependencies

Install the machine-learning and audio dependencies:

```powershell
pip install whisperx torchcrepe librosa scipy soundfile pyyaml pyphen numpy
```

Install Guitar Pro support:

```powershell
pip install pyguitarpro
```

Install utilities:

```powershell
pip install jsonschema
```

Optional upgrade command:

```powershell
pip install --upgrade whisperx torchcrepe librosa scipy soundfile pyyaml pyphen pyguitarpro jsonschema numpy
```

### 2.7 Verify CUDA

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output resembles:

```text
CUDA: True
GPU: NVIDIA GeForce RTX 3060
```

---

## 3. Hugging Face access

Speaker diarization uses gated Pyannote models through WhisperX.

Before running vocal analysis:

1. Sign in to Hugging Face.
2. Accept the access conditions for the required Pyannote diarization and segmentation models.
3. Create a read token.
4. Export it before running the builder.

PowerShell:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

Command Prompt:

```cmd
set HF_TOKEN=hf_your_token_here
```

If symbolic-link creation fails, enable Windows Developer Mode or set:

```powershell
$env:HF_HUB_DISABLE_SYMLINKS = "1"
```

> Never paste a live Hugging Face token into logs, screenshots, bug reports, or public messages. Revoke and replace any token that has been exposed.

---

## 4. Preparing audio stems

The alignment pipeline works best with clean, synchronized stems derived from the same source mix.

Recommended workflow:

1. Use a high-quality source mix.
2. Separate vocals first with a RoFormer vocal/instrumental model in UVR.
3. Run the vocal-reduced instrumental through a Demucs multi-stem model.
4. Keep useful outputs such as drums, bass, and other.
5. Optionally extract piano or keys from the instrumental or `other` stem with a suitable model.
6. Keep every stem untrimmed and sample-aligned.

Recommended intermediate formats:

- WAV;
- or lossless FLAC.

Avoid MP3 intermediates because lossy encoding can smear transients used for alignment.

Do not add different silence amounts to different stems. The builder handles the synchronized lead-in automatically.

---

## 5. Input folder layout

Each song should have its own directory:

```text
my_song/
├── song.gp5
├── drums.ogg
├── bass.ogg
├── piano.ogg
├── vocals.ogg
├── guitar.ogg
├── other.ogg
├── full.ogg
├── cover.jpg
└── metadata.json
```

The actual audio extensions may be `.ogg`, `.wav`, or `.flac`.

### 5.1 Recognized alignment stems

The builder currently recognizes these stem IDs automatically:

```text
drums
bass
piano
keys
keyboard
```

The base filename becomes the stem ID:

```text
drums.ogg -> drums
bass.flac -> bass
keys.wav  -> keys
```

Names such as these are not automatically recognized as alignment references:

```text
electric_bass.ogg
grand_piano.wav
synth.flac
bass_stem.wav
```

Rename them to a supported ID before building.

### 5.2 Supported audio formats

- OGG
- WAV
- FLAC

### 5.3 Optional `metadata.json`

```json
{
  "title": "My Song",
  "artist": "My Band",
  "album": "Album Name",
  "year": 2026,
  "genres": ["rock"]
}
```

If metadata is omitted, fallback values such as `Unknown Title` and `Unknown Artist` are used.

---

## 6. Basic usage

### 6.1 Standard CUDA build

```powershell
python build_feedpak.py my_song output_folder `
  --device cuda `
  --hf-token $env:HF_TOKEN
```

### 6.2 Recommended archival vocal build

```powershell
python build_feedpak.py my_song output_folder `
  --device cuda `
  --hf-token $env:HF_TOKEN `
  --vocal-layout both `
  --keep-work-dir
```

### 6.3 Skip vocal analysis

```powershell
python build_feedpak.py my_song output_folder `
  --skip-vocals
```

### 6.4 Skip Guitar Pro parsing and alignment

```powershell
python build_feedpak.py my_song output_folder `
  --skip-gp
```

### 6.5 Keep temporary files

```powershell
python build_feedpak.py my_song output_folder `
  --keep-work-dir
```

This preserves:

```text
_feedpak_build/
intermediate_vocals.json
intermediate_arrangements.json
```

Use this during initial testing and alignment debugging.

---

## 7. Automatic four-beat lead-in

The lead-in mechanism is automatic.

Before chart or vocal processing, the builder:

1. Selects a reference stem, preferring the full mix, then drums, then the first available stem.
2. Measures existing leading silence.
3. Reads the opening BPM from the Guitar Pro file.
4. Calculates a four-beat lead-in:

```text
count_in_seconds = 4 × 60 / initial_BPM
```

5. Adds only the missing amount:

```text
padding = max(0, count_in_seconds - existing_silence)
```

6. Applies identical padding to every stem.
7. Applies the resulting lead-in offset to chart events before alignment.

The same offset is applied to:

- guitar and bass notes;
- hand-position anchors;
- drum hits;
- bass alignment events;
- piano alignment events;
- key changes;
- measure boundaries;
- time signatures;
- and notation events.

This keeps all packaged stems, chart events, and alignment references on the same clock.

Original source files are never modified.

---

## 8. Validate the chart before alignment

Audio alignment assumes that the chart and recording represent the same arrangement.

Before treating a repeated alignment failure as a code defect, verify:

- the same measure count;
- the same opening pickup;
- the same time signatures;
- the same repeat expansion;
- the same solo and bridge lengths;
- the same inserted or removed bars;
- the same ending structure;
- and a reasonably corresponding drum, bass, or piano transcription.

A chart can begin and end correctly while being one or more bars wrong in the middle. This often indicates:

- a repeated riff matched to the wrong occurrence;
- an extra or missing bar;
- an alternate transcription revision;
- a solo of different length;
- or a repeat-expansion difference.

If the same measure range repeatedly triggers repair or disagreement warnings, inspect that section of the chart and recording before changing DTW parameters.

---

## 9. Multi-reference audio alignment

The alignment stage can use up to three independent references.

### 9.1 Drums

- Chart reference: Guitar Pro drum-hit onsets
- Audio reference: demuxed drum stem
- Frequency treatment: broadband onset analysis

Drums are normally the strongest reference because their attacks are transient-rich.

### 9.2 Bass

- Chart reference: Guitar Pro bass-note onsets
- Audio reference: demuxed bass stem
- Frequency treatment: approximately 30 to 1200 Hz

The bass candidate is intentionally concentrated on low and low-mid frequencies. Attack harmonics within this range can help identify pick, finger, and slap onsets.

Bass alignment is less reliable during:

- long sustained notes;
- slides;
- legato passages;
- weak or artifact-heavy demixing;
- and sections where the transcription differs from the recording.

### 9.3 Piano or keys

- Chart reference: piano/keyboard notation-beat onsets
- Audio reference: demuxed piano, keys, or keyboard stem
- Frequency treatment: approximately 45 to 6000 Hz

The current GP track-name detection expects `PIANO` or `KEY` in the track name.

### 9.4 Symmetric source processing

For each candidate, the real stem and synthetic reference are analyzed using the same source-specific Mel-frequency range.

Source-specific synthetic click frequencies are used to avoid comparing mismatched spectral representations.

### 9.5 Candidate construction

Each valid reference independently produces:

1. a list of nominal chart onsets;
2. a synthetic transient reference;
3. source-specific onset-strength envelopes;
4. a constrained DTW path;
5. a nominal-time to real-time mapping;
6. residual, coverage, and path metrics;
7. and an alignment quality score.

Missing stems or missing matching Guitar Pro tracks are skipped automatically.

---

## 10. Alignment decision and candidate agreement

Each candidate receives a heuristic score based on:

- median onset residual;
- 95th-percentile onset residual;
- path coverage;
- DTW constraint-band pressure;
- and number of usable chart events.

The scores are normalized with softmax and reported as relative probabilities.

These are not statistically calibrated probabilities. A probability of `0.70` means that the candidate received 70 percent of the relative quality weight for that build, not that it has a proven 70 percent chance of being correct.

### 10.1 Direct selection

If the strongest candidate exceeds the selection threshold, it is selected directly.

### 10.2 Candidate-agreement validation

Candidates are compared at common measure-boundary sample times before fusion.

The report records pairwise:

- median disagreement;
- 95th-percentile disagreement;
- and maximum disagreement.

Fusion is allowed only when candidates remain within the configured agreement limits.

If candidates disagree too strongly, the strongest candidate is selected and the report records:

```text
selected_due_to_disagreement
```

This prevents incompatible timing maps from being averaged together.

### 10.3 Probability-weighted fusion

If no candidate dominates and all candidates agree closely enough, the warp mappings may be combined:

```text
final_warp(t) = Σ probability_i × candidate_warp_i(t)
```

A weighted combination of agreed monotonic mappings remains monotonic.

### 10.4 Interpretation

Excellent residuals do not prove global correctness. Repeated material can produce two locally convincing but globally different alignments.

Always inspect the report when the decision is:

```text
selected_due_to_disagreement
```

or when alignment quality is `WARNING` or `SEVERE`.

---

## 11. Guarded timeline generation

Warped measure boundaries are validated before either `tempos[]` or `beats[]` is generated.

Checks include:

- duplicate timestamps;
- non-monotonic boundaries;
- implausibly short measures;
- excessive stretch or compression;
- and implausible effective BPM values.

### 11.1 Local repair

Bounded local failures may be repaired through interpolation between nearby valid boundaries.

Example console output:

```text
ALIGNMENT QUALITY: WARNING
  REPAIRED measures 48..52 by bounded interpolation
```

The repair is also recorded in the alignment report.

Repair prevents output such as:

- duplicate downbeat timestamps;
- zero-duration measures;
- backward-moving boundaries;
- and multi-thousand-BPM tempo entries.

### 11.2 Severe alignment rejection

If structural corruption remains after repair, the build refuses to package the timing map by default.

For diagnostic use only, this can be overridden:

```powershell
python build_feedpak.py my_song output_folder `
  --allow-severe-alignment
```

Do not use the override for production content unless the resulting timing has been manually inspected.

---

## 12. Timeline output modes

The builder can create dense tempo data, explicit beat data, or both.

Use:

```text
--timeline-mode tempos
--timeline-mode beats
--timeline-mode both
```

### 12.1 Dense tempos only

```powershell
python build_feedpak.py my_song output_tempos `
  --skip-vocals `
  --timeline-mode tempos `
  --keep-work-dir
```

This writes the dense per-measure tempo curve and time signatures without an explicit `beats[]` array.

### 12.2 Explicit beats only

```powershell
python build_feedpak.py my_song output_beats `
  --skip-vocals `
  --timeline-mode beats `
  --keep-work-dir
```

This is primarily a controlled compatibility test.

### 12.3 Both representations

```powershell
python build_feedpak.py my_song output_both `
  --skip-vocals `
  --timeline-mode both `
  --keep-work-dir
```

This writes both the dense tempo curve and explicit beat markers.

### 12.4 Why the modes exist

Some known working Feedpak examples contain dense `tempos[]` and `time_signatures[]` without explicit `beats[]`. Other feedBack versions or workflows may benefit from explicit beat markers.

Do not assume that `beats[]` is universally required. Use controlled A/B packages built from the same accepted alignment result to determine the behavior of the installed feedBack version.

---

## 13. Dynamic timing matrix

The accepted time mapping is applied to:

- playable guitar and bass notes;
- drum events;
- anchors;
- key changes;
- notation;
- and measure boundaries.

### 13.1 Dense tempo matrix

For each pair of accepted warped measure boundaries, the builder calculates an effective per-measure tempo:

```text
effective_BPM = beats_per_measure × 60 / warped_measure_duration
```

This is a dense curve designed to follow local performance timing rather than only sparse authored Guitar Pro tempo events.

### 13.2 Explicit highway beat grid

When enabled, each accepted warped measure interval is subdivided into beat markers.

Example 4/4 output:

```json
[
  {"time": 10.0000, "measure": 12},
  {"time": 10.5500, "measure": -1},
  {"time": 11.1000, "measure": -1},
  {"time": 11.6500, "measure": -1},
  {"time": 12.2000, "measure": 13}
]
```

Rules:

- numbered entries identify downbeats;
- `measure: -1` identifies internal beats;
- downbeats remain based on accepted post-DTW measure boundaries;
- and internal markers are interpolated within the same accepted measure interval.

This avoids creating a second independent timing map.

### 13.3 Compound meters

The current implementation divides measures by the time-signature numerator:

- 4/4 produces four subdivisions;
- 3/4 produces three;
- 6/8 produces six;
- 12/8 produces twelve.

It does not currently add compound-meter accent metadata.

### 13.4 Final measure

The final measure has no following downbeat from which to calculate its real duration. The current fallback uses the preceding valid measure duration for final-bar subdivisions.

This may be approximate for:

- partial final measures;
- fermatas;
- final tempo changes;
- or shortened endings.

---

## 14. Alignment report

The build writes a report beside the Feedpak:

```text
Artist - Song.feedpak.alignment_report.json
```

The report contains:

- final severity;
- decision mode;
- selected source;
- candidate probabilities;
- event counts;
- source frequency ranges;
- median and 95th-percentile residuals;
- path coverage;
- candidate disagreement;
- minimum and maximum measure stretch;
- repaired measure ranges;
- unresolved severe conditions;
- and warnings.

Example decision information:

```json
{
  "mode": "selected_due_to_disagreement",
  "selected_source": "drums",
  "candidates": [
    {
      "source": "drums",
      "probability": 0.644,
      "median_residual_ms": 19.3,
      "coverage": 1.0
    },
    {
      "source": "bass",
      "probability": 0.356,
      "median_residual_ms": 39.2,
      "coverage": 1.0
    }
  ],
  "agreement": [
    {
      "sources": ["drums", "bass"],
      "median_ms": 6579.4,
      "p95_ms": 19560.5
    }
  ]
}
```

This example demonstrates why low residuals alone are insufficient. Both candidates can match local onsets well while describing globally different song positions.

---

## 15. Vocal analysis

Vocal processing uses three independent layers.

### 15.1 Syllable-level lyrics

WhisperX provides transcription, forced word timing, and speaker diarization.

Words are divided into syllables with Pyphen. Each syllable receives a proportional part of the forced-aligned word interval.

Pitch analysis does not change lyric timing.

### 15.2 Representative pitch

For every syllable containing confident CREPE frames, the builder calculates one representative MIDI pitch using the median confident pitch in that syllable interval.

This produces:

```text
vocal_pitch.json
```

The compatibility pitch note uses the same start and duration as its syllable.

### 15.3 Dynamic pitch contour

Confident CREPE frames are independently preserved in:

```text
vocal_pitch_contour.json
```

This fine-grained contour can represent movement within a syllable more accurately than one MIDI note per syllable.

### 15.4 Removed overlap source

Earlier versions could generate:

- a full-syllable pitch note;
- secondary pitch notes beginning inside that interval;
- and standalone `+` lyric entries.

The current pipeline:

- no longer creates standalone `+` lyric records;
- writes at most one compatibility pitch note per pitched syllable;
- and keeps detailed movement in the pitch contour.

### 15.5 Remaining simultaneous-voice overlaps

If two diarized singers overlap, their notes may still overlap in the single song-level pitch file.

This is reported as a warning because Feedpak v1 provides one song-level discrete pitch track rather than one per speaker.

---

## 16. Vocal layout modes

Use:

```text
--vocal-layout merged
--vocal-layout separated
--vocal-layout both
```

The default is `merged`.

### 16.1 Merged

```powershell
python build_feedpak.py my_song output_folder `
  --vocal-layout merged
```

Writes:

- one merged `lyrics.json`;
- one global `vocal_pitch.json`;
- one global `vocal_pitch_contour.json`.

Use this for current game compatibility.

### 16.2 Separated

```powershell
python build_feedpak.py my_song output_folder `
  --vocal-layout separated
```

Writes:

- one lyric file per detected speaker;
- `lyric_tracks` entries;
- one global pitch file;
- and one global contour file.

Current game versions may not display the separated lyric tracks.

### 16.3 Both

```powershell
python build_feedpak.py my_song output_folder `
  --vocal-layout both
```

Writes:

- merged `lyrics.json` for compatibility;
- one lyric file per detected speaker;
- `lyric_tracks` entries;
- one global pitch file;
- and one global contour file.

This is the recommended preservation mode.

---

## 17. Direct alignment-script usage

### 17.1 Drums only

```powershell
python process_gp_alignment.py song.gp5 drums.ogg `
  --out intermediate_arrangements.json
```

### 17.2 Drums and bass

```powershell
python process_gp_alignment.py song.gp5 drums.ogg `
  --bass-audio bass.ogg `
  --out intermediate_arrangements.json
```

### 17.3 Drums, bass, and piano

```powershell
python process_gp_alignment.py song.gp5 drums.ogg `
  --bass-audio bass.ogg `
  --piano-audio piano.ogg `
  --out intermediate_arrangements.json
```

### 17.4 Bass without drums

```powershell
python process_gp_alignment.py song.gp5 `
  --bass-audio bass.ogg `
  --out intermediate_arrangements.json
```

### 17.5 Piano without drums

```powershell
python process_gp_alignment.py song.gp5 `
  --piano-audio piano.ogg `
  --out intermediate_arrangements.json
```

### 17.6 Lower DTW analysis sample rate

```powershell
python process_gp_alignment.py song.gp5 drums.ogg `
  --sr 16000 `
  --out intermediate_arrangements.json
```

### 17.7 Explicit count-in offset

```powershell
python process_gp_alignment.py song.gp5 drums.ogg `
  --count-in-offset 2.0 `
  --out intermediate_arrangements.json
```

When run through `build_feedpak.py`, the count-in offset is calculated automatically.

---

## 18. Feedpak output

A complete archive can contain:

```text
song.feedpak/
├── manifest.yaml
├── arrangements/
│   ├── guitar.json
│   ├── bass.json
│   └── notation_piano.json
├── stems/
│   ├── full.ogg
│   ├── drums.ogg
│   ├── bass.ogg
│   ├── piano.ogg
│   └── vocals.ogg
├── lyrics.json
├── lyrics_speaker_00.json
├── lyrics_speaker_01.json
├── vocal_pitch.json
├── vocal_pitch_contour.json
├── drum_tab.json
├── song_timeline.json
├── keys.json
└── cover.jpg
```

The diagnostics file is written beside the archive:

```text
output_folder/
├── Artist - Song.feedpak
└── Artist - Song.feedpak.alignment_report.json
```

---

## 19. Troubleshooting

### 19.1 Chart suddenly becomes one bar early or late

This is normally a structural jump rather than gradual tempo drift.

Common causes:

- repeated riffs;
- different repeat expansion;
- a missing or extra measure;
- an alternate chart revision;
- a solo of different length;
- or DTW matching the wrong occurrence of similar material.

The symptom often looks like:

```text
correct
correct
correct
suddenly one bar late
```

If the same measure range repeatedly triggers repairs, compare the chart and recording around that range before changing the alignment code.

### 19.2 No bar or beat markers

Inspect:

```text
song_timeline.json
```

Then build controlled A/B packages with:

```text
--timeline-mode tempos
--timeline-mode beats
--timeline-mode both
```

Some compatible Feedpaks use dense tempo and time-signature data without an explicit beat array. Do not assume that one representation is universally required.

Before testing renderer behavior, confirm that the accepted measure boundaries are monotonic and imply plausible tempos.

### 19.3 Alignment selects the wrong source

Inspect:

- decision mode;
- source probabilities;
- residuals;
- coverage;
- and candidate disagreement.

If `selected_due_to_disagreement` appears, the sources describe materially different timelines. Compare the chart, stems, and suspicious song section.

### 19.4 Alignment repairs the same measures repeatedly

This usually indicates something specific about that song region:

- repeated musical material;
- sparse transients;
- chart-versus-recording disagreement;
- or an alternate arrangement.

Repair prevents corrupt output, but it does not prove that the corrected region exactly matches the performance.

### 19.5 Bass alignment is poor

Bass alignment works best with clearly articulated attacks.

Possible causes:

- long sustains;
- slides and legato;
- kick-drum bleed;
- noisy demixing;
- missing chart notes;
- and differences between the GP bass transcription and recording.

### 19.6 Piano alignment is unavailable

Check that:

- the stem is named `piano`, `keys`, or `keyboard`;
- and the Guitar Pro track name contains `PIANO` or `KEY`.

### 19.7 Wrong vocal language detected

WhisperX detects language from the audio. Distorted, layered, or sparse vocals can produce a wrong result.

A future update should expose an explicit language override. Until then, inspect the detected language in the console and intermediate vocal file.

### 19.8 TorchCodec warning

If Pyannote reports that TorchCodec cannot load:

- verify FFmpeg installation;
- check TorchCodec compatibility with the installed PyTorch version;
- or rely on the in-memory waveform path if the current WhisperX workflow continues successfully.

A warning does not necessarily mean the build failed if audio was already decoded elsewhere.

### 19.9 No vocals in separated mode

Use:

```text
--vocal-layout both
```

This preserves per-speaker files while keeping the merged compatibility pointer.

### 19.10 Pitch overlaps remain

Secondary notes inside one syllable are no longer generated. Remaining overlaps generally represent simultaneous diarized singers sharing the global Feedpak pitch track.

### 19.11 DTW takes too long

Use:

```powershell
python process_gp_alignment.py song.gp5 drums.ogg `
  --sr 16000 `
  --out intermediate_arrangements.json
```

Also remove unusable references rather than calculating candidates from poor stems.

### 19.12 Missing modules

```powershell
pip install --upgrade whisperx torchcrepe librosa
```

### 19.13 Required privilege is not held

Enable Windows Developer Mode or set:

```powershell
$env:HF_HUB_DISABLE_SYMLINKS = "1"
```

---

## 20. Current limitations

### Alignment

- Alignment probabilities are heuristic and not statistically calibrated.
- Strongly disagreeing candidates are not fused, but disagreement still requires manual review.
- Repeated musical material can produce locally convincing but globally incorrect DTW paths.
- Alignment quality depends on chart-to-recording correspondence.
- Missing bars, alternate revisions, and repeat differences may require manual verification.
- Bass and piano candidates use onset timing rather than note-pitch agreement.
- Automatic stem-name recognition is intentionally narrow.
- Piano ties and sustain-pedal semantics are not fully modeled for alignment.
- Compound meters are subdivided by numerator without accent-group metadata.
- Final-measure duration is inferred from the preceding valid measure.
- Timeline repair is conservative and cannot guarantee musical correctness when the underlying chart structure differs.

### Vocals

- Syllable timing is proportionally divided inside word-level alignment windows rather than phoneme-aligned.
- Per-speaker vocal pitch files are not supported by Feedpak v1.
- Simultaneous speakers may overlap in the global pitch file.
- Current game versions may ignore `lyric_tracks`.
- Dynamic contour samples are associated with diarized word intervals, so untranscribed humming or extended notes may be incomplete.
- CPU vocal processing is not yet exposed through a complete set of model, precision, and batch-size options.

### Arrangement content

The current version does not yet extract:

- bends;
- slides;
- hammer-ons and pull-offs;
- fingering;
- chord diagrams and handshapes;
- harmony estimation;
- or amp and effects rigs.

---

## 21. Recommended workflow

1. Start with the cleanest possible source audio.
2. Prepare sample-aligned stems.
3. Confirm that the GP chart and recording represent the same arrangement.
4. Check measure count, repeats, solo length, and ending structure.
5. Build with `--keep-work-dir`.
6. Review console alignment quality.
7. Review the generated alignment report.
8. Inspect any repaired measure ranges manually.
9. Test a known-good chart and recording before using a difficult song as the baseline.
10. Use `--timeline-mode tempos` and `--timeline-mode both` for controlled feedBack compatibility testing.
11. Use `--vocal-layout both` when preserving multiple voices matters.
12. Test early, middle, solo, and ending sections in feedBack.
13. If alignment is poor, compare a rebuild with the questionable reference stem removed.
14. Never use `--allow-severe-alignment` for production without manual verification.

Recommended development command:

```powershell
python build_feedpak.py my_song output_folder `
  --device cuda `
  --hf-token $env:HF_TOKEN `
  --vocal-layout both `
  --timeline-mode both `
  --keep-work-dir
```

Inspect:

```text
intermediate_arrangements.json
intermediate_vocals.json
_feedpak_build/song_timeline.json
_feedpak_build/vocal_pitch.json
_feedpak_build/vocal_pitch_contour.json
Artist - Song.feedpak.alignment_report.json
```

---

## 22. Support information

When reporting an issue, include:

- the complete terminal log;
- the alignment report;
- `metadata.json`;
- the song folder's filenames;
- the Guitar Pro track names;
- the exact command;
- which stems participated in alignment;
- which measures visibly fail;
- whether the failure is gradual or a sudden whole-bar jump;
- Python version;
- installed package versions;
- GPU, VRAM, driver, and CUDA information;
- and whether the chart and recording have been manually compared in the failing region.

Useful commands:

```powershell
python --version
pip list
nvidia-smi
```

Do not share copyrighted audio or Guitar Pro files unless you have permission to distribute them.


## Project configuration and versioning

The builder automatically creates `feedpak-project.yaml` in the song folder on the first run. On later runs it loads the newest project configuration automatically. Explicit CLI arguments always override project values.

When effective project settings change, the previous file is preserved and a new sequential snapshot is written:

```text
feedpak-project.yaml
feedpak-project.v002.yaml
feedpak-project.v003.yaml
```

If settings have not changed, no new file is created. Use `--config path.yaml` to load a specific configuration instead of the newest automatic snapshot.

The project file stores stem names, vocal device/cache settings, anchor file, chunk size, timeline mode, and build behavior. Tokens and one-shot skip switches are deliberately not stored.

## Manual anchor time reference

Anchor files default to the original unpadded source-audio clock:

```json
{
  "version": 1,
  "time_reference": "source",
  "anchors": [
    {"measure": 48, "beat": 1, "audio_time": 118.647, "label": "solo start"}
  ]
}
```

`time_reference` may be `source` or `padded`. If omitted, `source` is assumed. Source times are converted internally by adding only the silence introduced by the current build, not the total detected count-in. The alignment report records source time, converted padded time, and padding added.


### Alignment test modes

Use `--alignment-mode nominal|offset|linear|dtw`. `nominal` keeps the shifted GP clock, `offset` matches the first symbolic and audio onsets without changing tempo, `linear` matches first and last onsets with one constant scale, and `dtw` uses guarded multi-reference alignment. The selected value is stored as `alignment.mode` in the project YAML.


## Verified renderer and vocal timing corrections

- Linear alignment remains the default.
- `song_timeline.json` keeps the accepted aligned times unchanged.
- The same beats and sections are mirrored into the first playable arrangement for historical highway rendering.
- No synthetic measure is inserted at zero.
- Vocal analysis runs on the original vocals stem. At packaging, only physically added padding is added to lyric, per-syllable pitch, and contour timestamps. Existing source silence is never added twice.
- `vocal_pitch.json` remains per-syllable and mirrors `lyrics.json` timing, as required by the karaoke renderer. Detailed samples remain in `vocal_pitch_contour.json`.
- Vocal caches carry the SHA-256 of the exact original stem analyzed. Old hashless caches must be regenerated once.
- The generated manifest declares Feedpak 1.19.0 and uses lowercase stem IDs.

---

## Checkpoint modes added after the original guide

The complete `--alignment-mode` set is:

```text
nominal
offset
linear
dtw
dtw-checkpoint
checkpoint-linear
checkpoint-dtw-diagnostic
checkpoint-dtw-selective
```

`dtw-checkpoint` collects periodic onset, internal silence-transition, and
absolute-chroma evidence without changing timing. `checkpoint-linear` applies a
small, guarded piecewise-linear residual curve when enough trustworthy anchors
survive. `checkpoint-dtw-diagnostic` evaluates anchor-bounded local DTW while
packaging checkpoint-linear timing. `checkpoint-dtw-selective` applies only
segments marked `would_apply` and otherwise returns the exact checkpoint-linear
map. The selective composite is rejected globally if paths overlap, endpoints
are invalid, monotonicity fails, beat stretch leaves 0.96 to 1.04, or measure
stretch leaves 0.98 to 1.02.

Recommended production invocation:

```powershell
python build_feedpak.py my_song output_folder `
  --alignment-mode checkpoint-dtw-selective `
  --checkpoint-measures 4 `
  --checkpoint-search-radius 1.0 `
  --keep-work-dir
```

A zero-segment result is valid and means the tested checkpoint-linear map was
retained.

## Song JSON, GP5 inspection, and strict track allocation

`metadata.json` remains optional. On the first GP-backed run the builder creates
or expands it with `feedpak_project`, inventories all GP5 tracks, fills only
clear name-based allocations, lists all unused tracks for copy/paste editing,
and writes `gp5_inspection.json`.

Validation occurs before count-in padding, WhisperX, CREPE, or alignment.
Configured GP5 names are exact and case-sensitive. Missing, ambiguous, or
conflicting explicit names are blocking errors. The program may suggest a close
name but never changes the configuration automatically.

Example:

```json
{
  "title": "My Song",
  "artist": "My Artist",
  "album": null,
  "year": null,
  "genres": [],
  "feedpak_project": {
    "version": 1,
    "chart": "song.gp5",
    "tracks": {
      "lead_vocal": "Lead Vocals",
      "drums": "Drums",
      "piano": {
        "left": "Piano (LH)",
        "right": "Piano (RH)",
        "combined": null
      },
      "guitars": ["Lead Guitar", "Rhythm Guitar"],
      "bass": ["Bass"],
      "ignored_tracks": ["Backing Vocals", "Tambourine", "Triangle"],
      "overrides": {},
      "unused_tracks": []
    },
    "track_inventory": []
  }
}
```

Unknown names are skipped and never fall through to the fretted parser. An
unusual name can be assigned explicitly:

```json
"overrides": {
  "Saxophone": "lead_vocal",
  "Roger Walters": "bass"
}
```

The current compatibility policy uses one lead-vocal track and one main drum
track. Backing vocals and auxiliary percussion are deliberately ignored. GP
lead-vocal conversion is reserved for the next checkpoint; the current safety
change prevents vocal and unknown tracks from becoming bogus tablature.

`gp5_inspection.json` records track statistics, automatic and final roles,
process/ignore actions, lyric lines, raw lyric `trackChoice`, both one-based and
zero-based candidates, and authored tempo events.

To use a JSON filename other than `metadata.json`:

```powershell
python build_feedpak.py my_song output_folder --song-json song.json
```

The versioned `feedpak-project*.yaml` files continue to store recurring command
options during this transition.

## Corrections to older implementation notes

- Count-in padding prefers the full mix and adds only the missing physical
  silence needed to reach four beats.
- Padded stems are temporarily written as WAV, then recompressed to OGG or FLAC;
  raw padded WAV files are not shipped.
- `vocal_pitch.json` can contain several contiguous stable note blocks inside a
  syllable. Detailed CREPE movement remains in `vocal_pitch_contour.json`.
- The current song-level pitch exporter resolves overlapping speaker notes to a
  single monophonic compatibility stream.
- `drum_tab.json` explicitly declares the complete eight-piece reduced kit.
- `feedpak_version` remains `1.19.0`; project and alignment-report additions do
  not require a wire-format bump.

## Regression tests

```powershell
python -m py_compile build_feedpak.py process_gp_alignment.py process_vocals.py `
  feedpak_common.py checkpoint_dtw.py project_config.py
python -m unittest discover -s tests -v
```
