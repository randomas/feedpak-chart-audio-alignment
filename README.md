# **Feedpak Builder Pipeline**

An automated toolchain to convert raw audio stems and Guitar Pro files (`.gp5`) into a fully compliant [Feedpak](https://got-feedback.github.io/feedpak-spec/) format (`.feedpak` ZIP archive).

**Current v1 Features:**
- ✅ Vocal extraction (WhisperX): lyrics + speaker diarization + word timestamps
- ✅ Pitch tracking (CREPE): discrete MIDI per word + continuous Hz contour
- ✅ Guitar Pro parsing + DTW alignment to real audio
- ✅ Drum tab extraction (reduced 7-piece kit: kick/snare/hihat/tom1/tom2/crash/ride — no note is ever dropped, every GM percussion note resolves to the closest of these)
- ✅ Hand-position anchors
- ✅ Key signature + real tempo/beat timeline (measure boundaries read directly from the GP file)
- ✅ **Automatic 4-beat count-in:** if a stem starts with no real lead-in silence, the pipeline pads all stems with a silent 4-beat count-in (sized to the song's actual BPM) and shifts every chart timestamp to match — audio and chart data stay in sync automatically

---

## **1. Environment & Setup**

### **System Requirements**

- **Windows Developer Mode:** Recommended (Settings → Privacy & Security → For developers → toggle on)
- **CUDA / GPU:** A CUDA-capable NVIDIA GPU is strongly recommended. The scripts default to `--device cuda`.
  - Without GPU, WhisperX and CREPE run ~10× slower on CPU.

### **Step 1: Install FFmpeg via winget (Windows)**

FFmpeg is required for audio I/O:

```powershell
winget install Gyan.FFmpeg
```

Close and reopen PowerShell, then verify:
```powershell
ffmpeg -version
```

### **Step 2: Hugging Face Token & Gated Model Access**

`process_vocals.py` uses WhisperX, which requires the Pyannote speaker diarization model (gated on Hugging Face).

1. Create or log in at [huggingface.co](https://huggingface.co/).
2. **Accept model conditions** at each of these pages (click "Access repository"):
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Generate a **Read** access token via **User Settings → Access Tokens**.
4. Save this token — you'll use it when running the pipeline.

### **Step 3: Create & Activate a Virtual Environment**

```powershell
# Create virtual environment
python -m venv env

# Activate it
.\env\Scripts\Activate.ps1

# If you get a script execution error, run this once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### **Step 4: Install PyTorch (with CUDA) & Dependencies**

Install PyTorch with explicit CUDA support first (standard `pip install torch` installs CPU-only):

```powershell
# PyTorch with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ML models & audio processing
pip install whisperx torchcrepe librosa scipy soundfile pyyaml

# Guitar Pro parsing
pip install pyguitarpro

# Utilities
pip install jsonschema
```

### **Step 5: Verify CUDA**

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected output: `CUDA: True` and your GPU name (e.g., "NVIDIA RTX 3060").

---

## **2. Environment Variables**

Set your Hugging Face token **before running the pipeline**:

**PowerShell:**
```powershell
$env:HF_TOKEN = "hf_your_actual_token_here"
```

**Command Prompt (cmd.exe):**
```cmd
set HF_TOKEN=hf_your_actual_token_here
```

---

## **3. Project Folder Structure**

Create a folder for each song. The folder must contain:

```
my_test_song/
├── song.gp5              (required: Guitar Pro file, .gp5/.gp4/.gp3)
├── vocals.ogg            (optional: vocal stem, required for lyrics/pitch)
├── drums.ogg             (optional: drum stem, required for DTW alignment)
├── bass.ogg              (optional)
├── guitar.ogg            (optional)
├── full.ogg              (optional: complete mix/backing)
├── cover.jpg             (optional: album art)
└── metadata.json         (optional: song metadata)
```

### **Audio Format**

- Supported: `.ogg`, `.wav`, `.flac`
- Recommended: `.ogg` (good compression)
- Sample rate: 44.1 kHz or higher, mono or stereo

### **metadata.json (Optional)**

Create this file in your song folder to populate the manifest. If omitted, defaults to "Unknown Title" / "Unknown Artist":

```json
{
  "title": "My Song",
  "artist": "My Band",
  "album": "Album Name",
  "year": 2024,
  "genres": ["rock"]
}
```

---

## **4. Running the Pipeline**

### **Full Pipeline (Recommended)**

```powershell
python build_feedpak.py my_test_song output_folder --device cuda --hf-token $env:HF_TOKEN
```

**Arguments:**
- `my_test_song` — Song folder (contains stems + `song.gp5`)
- `output_folder` — Where to save the `.feedpak` archive
- `--device cuda` — Use GPU (`cpu` if no CUDA available)
- `--hf-token <token>` — Your Hugging Face token
- `--vocals-stem vocals` — Name of vocal stem file (default: `vocals`)
- `--drums-stem drums` — Name of drum stem file (default: `drums`)
- `--skip-vocals` — Skip vocal processing
- `--skip-gp` — Skip GP parsing + alignment

**Output:**
```
output_folder/
└── My Band - My Song.feedpak
```

### **Step-by-Step Testing (Debugging)**

If you need to debug individual stages:

**Step 1: Vocal Processing**
```powershell
python process_vocals.py my_test_song\vocals.ogg --out my_test_song\intermediate_vocals.json --device cuda --hf-token $env:HF_TOKEN
```

Check `intermediate_vocals.json` for lyrics, pitches, contours per speaker.

**Step 2: GP Alignment + DTW**
```powershell
python process_gp_alignment.py my_test_song\song.gp5 my_test_song\drums.ogg --out my_test_song\intermediate_arrangements.json
```

Check `intermediate_arrangements.json` for warped note times, anchors, drum hits.

**Step 3: Build the Feedpak**
```powershell
python build_feedpak.py my_test_song output_folder --device cuda --hf-token $env:HF_TOKEN
```

---

## **5. Output: The `.feedpak` Archive**

The `.feedpak` is a ZIP containing:

```
song.feedpak/
├── manifest.yaml                      (metadata + file index)
├── stems/
│   ├── vocals.ogg
│   ├── drums.ogg
│   └── ...
├── arrangements/
│   ├── lead.json                      (fretted notes)
│   ├── bass.json
│   └── notation_piano.json            (staff notation if keyboard exists)
├── lyrics.json                        (flat word list with timestamps)
├── vocal_pitch.json                   (discrete MIDI per word)
├── vocal_pitch_contour.json           (continuous Hz contour)
├── drum_tab.json                      (drum hits + piece labels)
├── song_timeline.json                 (tempo + beat timeline)
├── keys.json                          (key signature changes)
└── cover.jpg                          (if provided)
```

All files conform to the [Feedpak v1 schema](https://got-feedback.github.io/feedpak-spec/).

---

## **6. Known Issues & v1 Limitations**

### **Audio Timing (v1)**

**4-beat count-in is automatic.** Before running Script 1/2, the pipeline checks one reference stem (the full mix if present, else `drums.ogg`, else whichever stem it finds first) for leading silence. If that stem starts "cold" (less than ~0.15s of silence before the first transient), it prepends a silent 4-beat count-in — sized to the song's actual tempo, read from the `.gp5` file — to **every** stem uniformly, and shifts every nominal chart timestamp (notes, drum hits, key changes, song timeline, keyboard notation) by the same amount before DTW alignment runs. If a stem already has a real lead-in gap, nothing is padded.

You don't need to do anything for this — it's automatic. The padded stems (not the originals) are what ends up in the final `.feedpak`; your original files in the song folder are never modified.

**DTW Alignment Lag:** If tab notes still appear offset from audio after this, the DTW warp itself may have drifted on a long or tempo-heavy section. Check that `drums.ogg` is a proper isolated drum stem (not the full mix). Try running with `--sr 16000` for faster DTW on long songs.

### **Vocals (v1)**

- **Flat lyrics only:** Currently writes a single `lyrics.json` with all speakers' words merged and time-sorted. No per-speaker attribution in the lyrics file itself (check `intermediate_vocals.json` for speaker info).
- **Per-speaker lyric_tracks:** Will be added in v2 (manifest `lyric_tracks[]` structure exists but is not yet populated by Script 3).

### **Not Yet Extracted**

- Technique fields (slides, bends, hammer-on/pull-off, fingering)
- Chord hand-shapes / diagrams
- Harmony / chord progression estimation
- Rig / amp-effects chains
- Extended drum kits beyond the reduced 7-piece set (kick/snare/hihat/tom1/tom2/crash/ride) — auxiliary percussion (cowbell, tambourine, etc.) folds into the nearest of these rather than getting its own piece

---

## **7. Troubleshooting**

### "Module not found: whisperx / torchcrepe / librosa"

```powershell
pip install --upgrade whisperx torchcrepe librosa
```

### "A required privilege is not held" (Windows, during model download)

Either:
1. Enable Developer Mode (Settings → Privacy & Security → For developers), or
2. Disable symlinking: `$env:HF_HUB_DISABLE_SYMLINKS = "1"`

Then re-run.

### "DiarizationPipeline() got unexpected keyword argument"

Update WhisperX:
```powershell
pip install --upgrade whisperx
```

### DTW Takes Very Long

- Use a proper isolated drum stem (not the full mix)
- Try lower sample rate: `python process_gp_alignment.py song.gp5 drums.ogg --out inter.json --sr 16000`
- For very long songs, consider splitting into sections

### No .gp5 File Found

- Ensure `song.gp5` is in the root of your song folder (not in a subfolder)
- Filename must match exactly (case-sensitive on Linux/macOS)

### Vocals Detected as Wrong Language

- WhisperX detects based on audio content, not file name
- Check the audio itself; detection is usually correct
- Explicit language selection will be added in a future release

---

## **8. Performance Tips**

- **GPU is essential:** Use `--device cuda`. Without it, expect 5–10× slowdown.
- **Sample rate:** Lower `--sr 16000` if DTW is slow.
- **Skip unnecessary stages:** Use `--skip-vocals` or `--skip-gp` to test one part.
- **Batch processing:** Loop over multiple song folders in a script.

---

## **9. References**

- **Feedpak Specification:** https://got-feedback.github.io/feedpak-spec/
- **WhisperX GitHub:** https://github.com/m-bain/whisperx
- **CREPE (PyTorch):** https://github.com/marl/torchcrepe
- **pyguitarpro:** https://github.com/nferretti/guitarpro
- **Librosa:** https://librosa.org/

---

## **10. Support**

If you encounter issues, provide:
1. Full error traceback
2. Your `metadata.json` and folder structure
3. The exact command you ran
4. Output of `python --version` and `pip list`
5. GPU name/driver version (if using CUDA)

---

**Happy transcribing!** 🎸🎤
