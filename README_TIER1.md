# Tier 1 expression implementation

Copy the replacement files into the project root, preserving `alphatab-extractor/`. Run `python -m pytest -q tests/test_expression_encoding.py`, then rebuild normally. Alignment and piano encoding were not changed. The build writes `<feedpak>.expression_report.json`.

Implemented official fields: `sl`, `slu`, `bn`, `bt`, `bnv`, `ho`, `po`, `hm`, `hp`, `pm`, `mt`, `fhm`, `vb`, `ac`, `tp`, `plk`, `slp`, `pkd`, `fg`. Tied continuation attacks are suppressed and sustain is extended.
