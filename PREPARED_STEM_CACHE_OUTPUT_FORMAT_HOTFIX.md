# Prepared-stem cache output-format hotfix

The cache planner previously assumed every padded stem would become FLAC. The encoder actually prefers OGG and falls back to FLAC only when FFmpeg/Vorbis encoding fails. A successful OGG encode therefore left no planned `Bass.flac`, and manifest construction failed after the expensive audio work had completed.

The cache now fingerprints preparation inputs without predicting an output extension. After encoding or copying, it records each actual artifact basename in the manifest. Cache-hit validation reads those recorded names and verifies that every artifact exists. Cache schema implementation version is incremented to 3, so stale version-2 prepared-stem manifests rebuild automatically.

The regression test forces an OGG result, verifies that `Bass.ogg` is recorded, and confirms that the next invocation is an exact cache hit.
