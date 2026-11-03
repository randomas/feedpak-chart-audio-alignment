# v16.6E.1 Process Integration Hotfix

## Failure

The post-materialization phase-qualification call in `process_gp_alignment.py` referenced `playback_measures`, which is not defined in the `process()` scope. All corpus jobs therefore completed MIDI section fitting but failed before output materialization with `NameError`.

## Fix

The call now uses `section_measures`, the in-scope expanded measure list already constructed for MIDI section fit and passed into the section-fit pipeline.

## Scope

This is an integration-only hotfix. It does not alter:

- candidate generation or caches;
- scoring or paths;
- rooted phrase support;
- fusion or materialization;
- phase classification policy;
- final timing.

## Regression

A source-level integration regression test ensures the post-materialization phase call uses `section_measures` and does not reference the undefined `playback_measures` name.

## Validation

Integrated suite: **371 passed, 1 skipped**.
