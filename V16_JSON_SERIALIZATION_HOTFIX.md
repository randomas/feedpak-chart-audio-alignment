# v16 JSON serialization hotfix

The v15.1 materialization report copied the mode policy dictionary directly.
Its `tiers` member is a Python `set`, which is valid in memory but cannot be
encoded by the standard JSON encoder. The full build completed alignment and
materialization, then failed while writing `intermediate_arrangements.json`.

The hotfix serializes `tiers` as a deterministically sorted JSON list. No search,
scoring, path, fusion, materialization, or timing values are changed.

A regression test serializes the complete application report with
`json.dumps(..., allow_nan=False)` and checks the permissive tier list.

Validation: 279 passed, 1 skipped.
