# Light Diagnostics v8

## Test runner

```powershell
.\run_all_test_materials.ps1 ... -LightDiagnostics
```

Output is written to `run_<timestamp>/Diagnostics_light/`.

## Direct collector

```bash
python collect_run_diagnostics.py ... --light
```

## Included

- run log
- build/method identity manifest
- compact timing-origin and production-timing decisions
- opening-anchor summary
- incipit consensus
- incipit-role tracking state, streaks, transitions, and release reason
- candidate/path counts
- per-role path span and unsupported-gap counts
- MIDI authority gate and failed predicates
- compact representative controls
- final timing decisions and safety-relevant alignment fields

## Omitted

- raw score and MIDI events
- all local candidates
- full selected-path rows
- feedpak ASCII extraction
- arrangement snapshots
- full score schema
- repeated source and project snapshots
- dense feature arrays and alignment matrices

A real DSMN diagnostic pair was reduced to approximately 45 KB of JSON plus a small manifest.

## Validation

Full test suite: 220 passed, 1 skipped.
