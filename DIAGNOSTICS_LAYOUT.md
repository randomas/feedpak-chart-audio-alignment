# Run diagnostics layout

Every `run_all_test_materials.ps1` run now creates a `Diagnostics` directory:

```text
run_YYYYMMDD_HHMMSS/
  Diagnostics/
    test_material_summary.csv
    test_material_summary.json
    my_test_song_6_gp/
      diagnostics_manifest.json
      logs/build.log
      feedpak_sidecars/*.json
      source_ascii/*.json
      source_ascii/feedpak-project*.yaml
      feedpak_ascii/<feedpak-name>/manifest.yaml
      feedpak_ascii/<feedpak-name>/arrangements/*.json
      feedpak_ascii/<feedpak-name>/*.json
```

Files are copied, not moved. Normal feedpak output locations remain unchanged.
Each feedpak has its own extraction directory, so common internal names such as
`manifest.yaml`, `song_timeline.json`, and `keys.json` cannot collide.
Audio stems and other binary feedpak members are deliberately not duplicated.
Diagnostics are collected for failed builds too, preserving partial reports.

## Light diagnostics for analysis and chat sharing

Use the test runner switch:

```powershell
.\run_all_test_materials.ps1 ... -LightDiagnostics
```

This writes a separate `Diagnostics_light` folder. It contains only compact
alignment and transcription skeletons, the run log, and a manifest. It omits
feedpak ASCII extraction, raw candidates, raw score/MIDI events, arrangements,
and repeated source snapshots.

The collector may also be called directly with `--light`.
