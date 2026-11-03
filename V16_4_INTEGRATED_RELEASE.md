# FeedPak v16.4 integrated release

This cumulative archive is based on v16.3 cache/solo-role authority and includes the prepared-stem cache output-format hotfix.

The three-clock incipit implementation is integrated directly into the active runtime:

- `process_gp_alignment.py` measures actual activity starts for available piano, guitar, bass, and drum role stems.
- `midi_section_fit.py` accepts those role-audio starts and calls `incipit_three_clock.reconcile_opening()` after solo-role annotation.
- `collect_run_diagnostics.py` exposes `three_clock_incipit` in light diagnostics.
- Unit and runtime-wiring tests are included under `tests/`.

No patch script needs to be run after extracting this archive.
