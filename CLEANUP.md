# Directory cleanup plan

## Keep active
Root source files, `tests/`, and `alphatab-extractor/` files in this package.

## Archive
Everything under `archive_candidates/` is historical, diagnostic, or a one-shot patch script from the dump. Keep the folder in version control history or a dated ZIP, but remove these files from the active project root.

## Safe generated cleanup
`__pycache__/`, `.pytest_cache/`, `_feedpak_build/`, `*.pyc`, `intermediate_arrangements.json`, and regenerable inspection/diagnostic outputs. Keep `intermediate_vocals.json` only when intentionally used as a matching cache.

```powershell
Remove-Item -Recurse -Force __pycache__, .pytest_cache, _feedpak_build -ErrorAction SilentlyContinue
Remove-Item -Force *.pyc, intermediate_arrangements.json -ErrorAction SilentlyContinue
```
