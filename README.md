# Checkpoint 6.2.1 patch

From the project root:

```powershell
python apply_checkpoint_6_2_1.py
python -m py_compile checkpoint_dtw.py process_gp_alignment.py
python -m unittest discover -s tests -v
```

Copy `tests/test_checkpoint_6_2_1.py` into the project's `tests` directory.
The patch creates `.cp62.bak` backups before changing either source file.

# Selective local DTW production mode

The completed alignment update adds `checkpoint-dtw-selective`. It runs the
same checkpoint-linear baseline and local-DTW acceptance gates as
`checkpoint-dtw-diagnostic`, then applies only segments whose decision is
`would_apply`.

Safety behavior:

- no accepted segments gives an exact checkpoint-linear fallback;
- accepted segment residuals are zero at both endpoints;
- overlapping accepted segments reject the complete local layer;
- the composed warp is validated globally at measure and metric-beat edges;
- any monotonicity or stretch failure rejects the complete local layer;
- diagnostic mode remains timing-neutral.

Production invocation:

```powershell
python build_feedpak.py SONG_FOLDER OUTPUT_FOLDER --alignment-mode checkpoint-dtw-selective
```

The alignment report records `local_dtw` evidence plus
`local_dtw_production`, including applied segments, global validation and the
actual `timing_output`.
