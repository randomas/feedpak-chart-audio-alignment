# Manual anchor production integration

Replace the files in the project root with the files in this archive, preserving
`tests/` paths. Existing projects without `alignment.manual_anchors` remain
unchanged.

## ATIB test metadata

Add the following to the existing `metadata.json`, using the playback bar and
audio time verified by the charter:

```json
"alignment": {
  "manual_anchor_time_basis": "original",
  "manual_anchors": [
    {
      "id": "full-band-entry",
      "score": {"playback_bar": 49, "beat": 1},
      "audio": {"time": 139.683},
      "label": "Full-band entrance",
      "mode": "search",
      "search": {"score_radius_bars": 2, "audio_radius_seconds": 2.0}
    }
  ]
}
```

Start with `mode: search`. Build normally and inspect
`decision.manual_anchor_quality` in the alignment report. Once the bar, beat,
time basis, and audio time are validated, change the anchor to `mode: exact` and
rebuild.

`original` means the unpadded source recording. The builder adds only the
physical silence introduced during the current build. Use `padded` when the
time was read from the already padded/package audio.
