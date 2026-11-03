# v16.6E.2.1 Windows Atomic JSON Write Hotfix

## Failure

Sound of Love completed alignment, MIDI-section materialization, final timing selection, and arrangement warping. The build then failed while replacing `rythm_guitar.json` with its `.tmp` file:

`PermissionError: [WinError 5] Access is denied`

This is a transient Windows destination-file lock, not an alignment failure.

## Fix

`feedpak_common.write_json` retains atomic `os.replace` semantics and now retries `PermissionError` up to eight times with bounded exponential backoff. Non-permission errors are raised immediately. A persistent lock is still surfaced after the bounded retries.

## Scope

The hotfix changes only JSON output robustness. It does not change alignment, diagnostics, phase calibration, materialization, or timing.

## Validation

- transient-lock retry test;
- persistent-lock bounded-failure test;
- integrated suite: **377 passed, 1 skipped**.
