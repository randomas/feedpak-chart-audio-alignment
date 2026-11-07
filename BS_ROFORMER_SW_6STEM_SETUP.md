# BS-RoFormer SW six-stem preparation

## Status

BS-RoFormer SW is the current default external separator for FeedPak material preparation. It is run directly on the original full mix and produces six nominal targets:

```text
bass
 drums
 other
 vocals
 guitar
 piano
```

Direct six-stem output was preferred over the earlier vocal-RoFormer followed by HTDemucs workflow in current tests on **All That I Bleed** and **Caught in a Web**. This is an observed project result, not a claim that one separator is universally superior. Keep the older pipeline available as a fallback or comparison when a specific direct-output defect affects downstream MIR evidence.

## Separation policy

- FeedPak does not install or run stem separation.
- Run separation before MuScriptor and FeedPak.
- Keep the original mix unchanged.
- Do not independently trim, pad, normalize, or time-shift stems.
- Preserve intermediate and final outputs.
- Treat target names as model hypotheses, not verified instrument identity.
- Missing or ambiguous audio evidence is neutral and does not invalidate reliable MIDI evidence.
- MIDI remains the primary dense timing source; strict audio evidence may later contribute only a few solid structural anchors.

## Validated local environment

The working development configuration was:

```text
Windows PowerShell
Python 3.13
PyTorch 2.6.0+cu124
torch.version.cuda: 12.4
CUDA available: True
GPU: NVIDIA RTX 3060 12 GB
CUDA Toolkit installed on the machine: 13.5
```

BS-RoFormer SW ran from the existing activated Audio Alignment environment without additional package installation. This does not mean every new machine should reuse its FeedPak environment. If the runner's requirements are not already satisfied, use a separate environment rather than broadly upgrading a working project stack.

The proven route is the cloned runner plus `python inference.py`. It is not currently an `uvx` command. MuScriptor may remain isolated through `uvx`.

## Runner and model

Inference runner:

- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)

Background discussion:

- [BS-RoFormer discussion 45](https://github.com/lucidrains/BS-RoFormer/discussions/45)

Checkpoint:

- [bs_6stem_fixed.ckpt](https://huggingface.co/noblebarkrr/mvsepless_resources/blob/main/bs_roformer/bs_6stem_fixed.ckpt)

Published configuration reference:

- [bs_6stem_config.yaml](https://huggingface.co/noblebarkrr/mvsepless_resources/blob/main/bs_roformer/bs_6stem_config.yaml)

The validated local command used these local filenames:

```text
bs_6stem_fixed.ckpt
bs_6stem_fixed.yaml
```

The published configuration link and validated local filename are not assumed to be interchangeable without inspection. Preserve the configuration that actually accompanied the working checkpoint. The exact runner commit and file hashes were not recorded in the current setup notes; record them before treating another machine as an exact reproduction.

## Suggested directory layout

```text
Stem separation/
├─ audio/
│  └─ input/
├─ models/
│  └─ bs_roformer_sw/
│     ├─ bs_6stem_fixed.ckpt
│     └─ bs_6stem_fixed.yaml
├─ output/
└─ Music-Source-Separation-Training/
```

## Clone the runner

```powershell
git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git
```

Before updating a working clone, record its current state:

```powershell
Push-Location .\Music-Source-Separation-Training
git rev-parse HEAD
git status --short
Pop-Location
```

Do not run a blind `git pull` or broad dependency upgrade on a validated setup.

## Standard inference

Activate the known-compatible environment, enter the runner directory, and run:

```powershell
python inference.py `
  --model_type bs_roformer `
  --config_path "..\models\bs_roformer_sw\bs_6stem_fixed.yaml" `
  --start_check_point "..\models\bs_roformer_sw\bs_6stem_fixed.ckpt" `
  --input_folder "..\audio\input" `
  --store_dir "..\output\bs_roformer_sw" `
  --flac_file
```

Successful loading should report:

```text
Using device: cuda:0
Instruments: ['bass', 'drums', 'other', 'vocals', 'guitar', 'piano']
```

Keep each run's input set and output directory unambiguous. If the input directory contains several songs, verify the runner's output naming before moving stems into FeedPak material folders.

## Optional TTA comparison

```powershell
python inference.py `
  --model_type bs_roformer `
  --config_path "..\models\bs_roformer_sw\bs_6stem_fixed.yaml" `
  --start_check_point "..\models\bs_roformer_sw\bs_6stem_fixed.ckpt" `
  --input_folder "..\audio\input" `
  --store_dir "..\output\bs_roformer_sw_tta" `
  --flac_file `
  --use_tta
```

TTA was described during testing as roughly three times slower. Treat that as an observed estimate, not a guaranteed multiplier. It may improve some material, but the project should select it only when listening and downstream MIR evidence improve. `--bigshifts` was discussed as an additional comparison parameter, but no value is documented here as the validated default.

## Validation

Use `ffprobe` to compare the original and each stem:

```powershell
ffprobe .\audio\input\original.flac
ffprobe .\output\bs_roformer_sw\bass.flac
ffprobe .\output\bs_roformer_sw\drums.flac
ffprobe .\output\bs_roformer_sw\guitar.flac
ffprobe .\output\bs_roformer_sw\piano.flac
ffprobe .\output\bs_roformer_sw\vocals.flac
ffprobe .\output\bs_roformer_sw\other.flac
```

Verify:

- sample rate;
- channel count;
- duration and, where practical, exact frame count;
- unchanged start time and song clock;
- no unintended trimming or padding;
- no silent or malformed output;
- no NaN values or severe clipping;
- approximate reconstruction of the original from the six outputs.

Do not independently normalize stems before FeedPak ingestion. Independent processing can destroy reconstruction and alter source relationships.

An apparent time translation during early testing was caused by Windows Media Player playback. Audacity confirmed that the generated stems were correctly timed.

## Downstream use

Preferred current flow:

```text
Original full mix
→ BS-RoFormer SW six-stem separation
→ validate outputs
→ MuScriptor on useful role stems, initially bass and guitar
→ FeedPak alignment
```

Retain regions without credible transcription or audio evidence. Do not manufacture anchors merely because a nominal stem exists. A nominal `guitar` stem may still contain piano or other leakage, and `other` must not be treated as confirmed guitar.

## Dependency policy

> If it ain't broke, don't fix it.

Do not broadly upgrade Python, PyTorch, CUDA, NumPy, librosa, soundfile, or the runner merely to remove warnings. Any dependency update is a separate controlled activity requiring the unit suite, corpus validation, and listening checks where timing or separation output changes.
