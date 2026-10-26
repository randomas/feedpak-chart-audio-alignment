# Phase 3D: analyse once, materialize many

Apply once:

```powershell
python apply_phase3d_patch.py
```

Use the normal `build_feedpak.py` invocation, adding:

```powershell
--phase3d-variants --phase3d-safe-candidates 2 --phase3d-structural-candidates 1
```

The ordinary feedpak remains the baseline. During the same Python process and before the work directory is removed, Phase 3D reuses the already-computed alignment proposal and structural-search reports, clones the current work tree, applies each candidate timing map, and calls the existing `package_feedpak` function. It does not launch another analysis, call an alternative runner, or reopen and rewrite completed feedpaks.

Outputs are named `.SAFE-01.feedpak`, `.SAFE-02.feedpak`, and `.EXP-01.feedpak`. Structural variants are explicitly experimental. A sibling `.phase3d_variants.json` records the single-pass provenance and all materializations.
