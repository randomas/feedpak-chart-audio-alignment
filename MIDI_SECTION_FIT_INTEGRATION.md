# MIDI Section Fit integration v13

The cumulative Steps 1-3 are integrated as a diagnostic-only pipeline feature.

## Switches

- PowerShell corpus runner: `-MidiSectionFit`
- Builder and direct alignment CLI: `--midi-section-fit`

The switch reads canonical prepared MuScriptor files, builds expanded
playback-measure windows, selects independent monotonic role paths, and creates
compatible multi-role diagnostic control proposals. It never changes production
timing.

## Outputs

- `<feedpak>.midi_section_fit.json`
- light diagnostics: `<feedpak>.midi_section_fit.light.json`

The light report retains candidate/path totals, role streaks, unsupported
windows, fused regions, conflicts, rejected blocks, normal 0.90-1.10 edge
limits, and the monotonicity certificate.

## Example

```powershell
.\run_all_test_materials.ps1 `
  -MaterialsRoot "C:\Users\Matteo\Documents\Guitar\Audio alignment" `
  -OutputRoot "C:\Users\Matteo\Documents\Guitar\Audio alignment\midi_section_fit_diagnostics" `
  -AlignmentMode "continuous-anchor-structural" `
  -MidiSectionFit `
  -LightDiagnostics `
  -ReuseVocals `
  -Device cuda `
  -SkipUnitTests
```

`-TranscriptionInvestigator` is optional. Both features can run together.

Full suite: 251 passed, 1 skipped.
