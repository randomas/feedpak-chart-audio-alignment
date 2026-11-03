# v16.3 cache compatibility and solo-role authority

This cumulative repair restores the canonical persistent candidate-cache API in `midi_section_fit.build_section_fusion_report` while preserving Phase 2b prepared-stem, parsed-chart, and pipeline-performance work. Candidate cache loading and writing are owned by `midi_section_fit.py`; `process_gp_alignment.py` supplies `.feedpak_cache/midi_section_candidates.json` and no longer implements a duplicate report cache.

Supported cache arguments are `candidate_cache_path`, `candidate_cache_rebuild`, and `candidate_cache_disabled`. Exact hits reuse only the candidate report. Path selection, fusion, expected-role annotation, materialization, arrangement generation, and packaging still rerun.

Solo-role corroboration is evaluated against locally active chart roles. A high-confidence control from the only expected role is classified `solo_role_expected_strong`; inactive roles are neutral. Two or more such controls near the first chart onset may validate the acquisition root. An invalid baseline with negative, reversed, or implausibly stretched sampled timing is replaced by an identity-shaped clock anchored to that root; without a root, MIDI materialization is blocked.

Validation: 328 passed, 1 skipped.
