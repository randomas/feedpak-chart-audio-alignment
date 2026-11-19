# BS-RoFormer SW six-stem preparation

BS-RoFormer SW is the current default external separator. Run it directly on the original full mix and retain nominal `bass`, `drums`, `other`, `vocals`, `guitar`, and `piano` outputs. Direct six-stem output was preferred over vocal-RoFormer followed by HTDemucs in current tests on All That I Bleed and Caught in a Web. Keep the older pipeline only as a fallback or comparison for a documented failure.

FeedPak does not install or run the separator. Preserve the original song clock and do not independently trim, pad, normalize, or time-shift stems. Stem names are model hypotheses, not verified instrument identities. Missing audio evidence is neutral.

Validated local setup:

```text
Windows PowerShell
Python 3.13
PyTorch 2.6.0+cu124
CUDA runtime reported by PyTorch: 12.4
GPU: NVIDIA RTX 3060 12 GB
```

Runner: [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)

Background: [BS-RoFormer discussion 45](https://github.com/lucidrains/BS-RoFormer/discussions/45)

Checkpoint: [bs_6stem_fixed.ckpt](https://huggingface.co/noblebarkrr/mvsepless_resources/blob/main/bs_roformer/bs_6stem_fixed.ckpt)

The validated route uses the cloned runner and `python inference.py`, not `uvx`:

```powershell
python inference.py `
  --model_type bs_roformer `
  --config_path "..\models\bs_roformer_sw\bs_6stem_fixed.yaml" `
  --start_check_point "..\models\bs_roformer_sw\bs_6stem_fixed.ckpt" `
  --input_folder "..\audio\input" `
  --store_dir "..\output\bs_roformer_sw" `
  --flac_file
```

Optional TTA comparison adds `--use_tta`. Retain non-TTA output until listening and downstream MIR diagnostics show an improvement.

Validate sample rate, channels, duration, start clock, clipping, silence, malformed output, and approximate six-stem reconstruction. The apparent time translation seen during early testing was a Windows Media Player playback issue; Audacity confirmed correct timing.

Dependency policy: **if it ain't broke, don't fix it**. Upgrade Python, PyTorch, CUDA, or MIR/audio libraries only as a separate controlled task.
