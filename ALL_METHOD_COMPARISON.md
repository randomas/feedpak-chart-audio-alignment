# All-method comparison

`run_all_alignment_methods.ps1` runs every retained method into its own output
folder. This is intentionally expensive, but avoids duplicated neural vocal
analysis:

- With `-IncludeVocals`, the first method runs WhisperX and CREPE.
- Later methods pass `--reuse-vocals`; `build_feedpak.py` reuses the per-song
  `intermediate_vocals.json` only when its stored padded-audio SHA-256 matches.
- Structural audio features use the existing versioned `_alignment_cache`.
- alphaTab schema JSONs and per-song work/caches remain available for subsequent
  methods because the corpus runner uses `--keep-work-dir`.

Example:

```powershell
.\run_all_alignment_methods.ps1 `
  -MaterialsRoot "C:\Users\Matteo\Documents\Guitar\Audio alignment" `
  -OutputRoot "C:\Users\Matteo\Documents\Guitar\Audio alignment\all_method_comparison_outputs" `
  -Only "my_test_song_6_gp" `
  -IncludeVocals `
  -HFToken $env:HF_TOKEN `
  -Device cuda `
  -SkipUnitTests
```

For a faster alignment-only comparison, omit `-IncludeVocals`.
