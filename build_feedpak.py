"""
build_feedpak.py

Script 3 of the feedpak pipeline. Orchestrates process_vocals.py and
process_gp_alignment.py as subprocesses, converts their intermediate
JSON into strict feedpak wire format, writes manifest.yaml, and zips
everything into a .feedpak archive.

v1 scope (see architecture doc §2.4 for full rationale):
  - harmony.json: NOT generated (needs GP chord-diagram extraction or
    audio chord estimation, neither implemented yet).
  - rigs.json / tones: NOT generated. Arrangement JSON omits `tones`
    entirely for v1, so there's nothing that can dangle-reference a
    missing rig id.
  - Technique fields (bends, slides, ho/po, fingering): NOT extracted.
    Arrangement notes are strictly {t, s, f, sus} for v1, per project
    decision to solve timing/sync first.
  - keys.json: generated, deterministically, from GP measure key
    signatures (see process_gp_alignment.py).

Expects, in `song_folder`:
  - exactly one *.gp5 file
  - a `drums.ogg` (or --drums-stem override) stem for DTW reference
  - a `vocals.ogg` (or --vocals-stem override) stem, if the song has vocals
  - any number of other *.ogg/*.wav/*.flac stems (bass, other, full mix...)
  - an optional metadata.json: {"title", "artist", "album", "year", "genres"}

Usage:
    python build_feedpak.py /path/to/song_folder /path/to/output_folder
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import zipfile

import yaml

import feedpak_common as fc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_EXTENSIONS = (".ogg", ".wav", ".flac")
FULL_MIX_STEM_NAMES = {"full", "mix", "song", "master"}


# --------------------------------------------------------------------------
# Metadata / stem discovery
# --------------------------------------------------------------------------

def load_metadata(song_folder):
    path = os.path.join(song_folder, "metadata.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def find_gp_file(song_folder):
    candidates = glob.glob(os.path.join(song_folder, "*.gp5"))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"Warning: multiple .gp5 files found, using {candidates[0]}", file=sys.stderr)
    return candidates[0]


def discover_stems(song_folder, vocals_stem_name, drums_stem_name):
    """
    Returns (stems_list, vocals_path, drums_path). `stems_list` entries
    follow manifest.schema.json's stemEntry shape. The 'full' id is
    reserved for a complete mixdown (spec §5.3) and, per spec, its
    `default` is set false whenever other separated stems also exist.
    """
    audio_files = sorted(
        f for f in os.listdir(song_folder)
        if f.lower().endswith(AUDIO_EXTENSIONS)
    )
    other_stem_count = sum(
        1 for f in audio_files
        if os.path.splitext(f)[0].lower() not in FULL_MIX_STEM_NAMES
    )

    stems = []
    vocals_path = drums_path = None
    for f in audio_files:
        base = os.path.splitext(f)[0]
        is_full_mix = base.lower() in FULL_MIX_STEM_NAMES
        stem_id = "full" if is_full_mix else base
        default_state = not (is_full_mix and other_stem_count > 0)
        stems.append({
            "id": stem_id,
            "file": fc.to_posix_relpath("stems", f),
            "default": default_state,
        })
        if base.lower() == vocals_stem_name.lower():
            vocals_path = os.path.join(song_folder, f)
        if base.lower() == drums_stem_name.lower():
            drums_path = os.path.join(song_folder, f)

    return stems, vocals_path, drums_path


def probe_duration_seconds(audio_path):
    if audio_path is None or not os.path.exists(audio_path):
        return None
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return round(info.frames / info.samplerate, 3)
    except Exception as e:
        print(f"Warning: could not probe duration of {audio_path}: {e}", file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Subprocess orchestration
# --------------------------------------------------------------------------

def run_vocals_script(vocals_path, out_path, device="cuda", hf_token=None):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "process_vocals.py"),
           vocals_path, "--out", out_path, "--device", device]
    if hf_token:
        cmd += ["--hf-token", hf_token]
    fc.log(f"Launching process_vocals.py as a subprocess (device={device})")
    # No capture_output: let the child's own [*] progress logs stream
    # straight through to this terminal in real time.
    subprocess.run(cmd, check=True)
    fc.log("process_vocals.py finished")


def run_gp_script(gp_path, drums_path, out_path, sr=22050):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "process_gp_alignment.py"),
           gp_path, drums_path, "--out", out_path, "--sr", str(sr)]
    fc.log(f"Launching process_gp_alignment.py as a subprocess (sr={sr})")
    subprocess.run(cmd, check=True)
    fc.log("process_gp_alignment.py finished")


# --------------------------------------------------------------------------
# Wire-format conversion
# --------------------------------------------------------------------------

def write_lyric_tracks(vocals_data, work_dir, vocal_stem_id):
    """
    Splits diarized speakers into one lyrics_<speaker>.json per speaker
    (flat {t,d,w} array per lyrics.schema.json) — the schema has no
    per-word speaker field, so this has to be a file-per-speaker split,
    not a merged file. Returns (lyric_track_manifest_entries, files_written).
    """
    entries = []
    files = []
    speakers = vocals_data.get("speakers", {})
    for speaker_id, data in speakers.items():
        words = data.get("words", [])
        if not words:
            continue
        flat = [{"t": w["t"], "d": w["d"], "w": w["w"]} for w in words]
        filename = f"lyrics_{speaker_id.lower()}.json"
        fc.write_json(os.path.join(work_dir, filename), flat)
        files.append(filename)
        entries.append({
            "id": speaker_id.lower(),
            "file": fc.to_posix_relpath(filename),
            "language": "und",
            "kind": "original",
            "stem": vocal_stem_id,
        })
    return entries, files


def write_merged_vocal_pitch(vocals_data, work_dir):
    """
    vocal_pitch.json / vocal_pitch_contour.json are single top-level
    manifest entries — the schemas have no per-speaker field and the
    manifest has no per-speaker list for pitch (unlike lyric_tracks), so
    for multi-vocalist songs all speakers' discrete notes / contour
    samples are merged into one flat, time-sorted array. This is a real
    schema limitation, not an oversight — see architecture doc §2.3.
    """
    all_notes = []
    all_samples = []
    for data in vocals_data.get("speakers", {}).values():
        for w in data.get("words", []):
            if "midi" in w:
                all_notes.append({"t": w["t"], "d": w["d"], "midi": w["midi"]})
        for s in data.get("contour", []):
            all_samples.append(s)

    all_notes.sort(key=lambda n: n["t"])
    all_samples.sort(key=lambda s: s["t"])

    wrote_pitch = wrote_contour = False
    if all_notes:
        fc.write_json(os.path.join(work_dir, "vocal_pitch.json"),
                       {"version": 1, "notes": all_notes})
        wrote_pitch = True
    if all_samples:
        fc.write_json(os.path.join(work_dir, "vocal_pitch_contour.json"),
                       {"version": 1, "samples": all_samples})
        wrote_contour = True
    return wrote_pitch, wrote_contour


def write_arrangement_files(gp_data, work_dir):
    """
    Converts each fretted-instrument entry from intermediate_arrangements
    into a standalone arrangement JSON file. v1: no tones/chords/technique
    fields — see module docstring.
    """
    entries = []
    for track_id, arr in gp_data.get("arrangements", {}).items():
        out = {
            "name": arr["name"],
            "tuning": arr["tuning"],
            "capo": arr["capo"],
            "notes": arr["notes"],
            "chords": arr.get("chords", []),
            "anchors": arr.get("anchors", []),
            "handshapes": arr.get("handshapes", []),
            "templates": arr.get("templates", []),
        }
        fc.drop_empty_lists(out, ["tempos", "phrases"])
        filename = f"{track_id}.json"
        fc.write_json(os.path.join(work_dir, "arrangements", filename), out)
        entry = {
            "id": track_id,
            "name": arr["name"],
            "file": fc.to_posix_relpath("arrangements", filename),
            "tuning": arr["tuning"],
            "capo": arr["capo"],
        }
        if arr.get("tuning_nonstandard_string_count"):
            print(f"Note: {track_id} has a nonstandard string count; "
                  f"tuning offsets are best-effort.", file=sys.stderr)
        entries.append(entry)
    return entries


def write_notation_files(gp_data, work_dir):
    """
    Keyboard tracks get notation.json ONLY (measures/staves), never a
    synthetic flat tab-style file — see architecture doc §2.4.
    """
    entries = []
    for track_id, notation in (gp_data.get("notation") or {}).items():
        filename = f"notation_{track_id}.json"
        fc.write_json(os.path.join(work_dir, "arrangements", filename), notation)
        entries.append({
            "id": track_id,
            "name": track_id.replace("_", " ").title(),
            "notation": fc.to_posix_relpath("arrangements", filename),
        })
    return entries


def write_drum_tab(gp_data, work_dir):
    drum_tab = gp_data.get("drum_tab")
    if not drum_tab or not drum_tab.get("hits"):
        return None
    fc.write_json(os.path.join(work_dir, "drum_tab.json"), drum_tab)
    return "drum_tab.json"


def write_keys(gp_data, work_dir):
    keys = gp_data.get("keys")
    if not keys or not keys.get("events"):
        return None
    fc.write_json(os.path.join(work_dir, "keys.json"), keys)
    return "keys.json"


# --------------------------------------------------------------------------
# Manifest construction
# --------------------------------------------------------------------------

def build_manifest(metadata, duration, arrangements, stems, lyric_tracks,
                    vocal_pitch_file, vocal_pitch_contour_file,
                    drum_tab_file, keys_file, song_timeline_file, cover_file):
    manifest = {
        "title": metadata.get("title", "Unknown Title"),
        "artist": metadata.get("artist", "Unknown Artist"),
        "album": metadata.get("album"),
        "year": metadata.get("year"),
        "genres": metadata.get("genres"),
        "duration": duration if duration is not None else 0.0,
        "arrangements": arrangements,
        "stems": stems,
    }
    if lyric_tracks:
        manifest["lyric_tracks"] = lyric_tracks
    if vocal_pitch_file:
        manifest["vocal_pitch"] = vocal_pitch_file
    if vocal_pitch_contour_file:
        manifest["vocal_pitch_contour"] = vocal_pitch_contour_file
    if drum_tab_file:
        manifest["drum_tab"] = drum_tab_file
    if keys_file:
        manifest["keys"] = keys_file
    if song_timeline_file:
        manifest["song_timeline"] = song_timeline_file
    if cover_file:
        manifest["cover"] = cover_file

    # drop None / empty-string / empty-list values so we never write a key
    # the schema would reject as present-but-empty, and never write nulls
    manifest = {k: v for k, v in manifest.items() if v not in (None, "", [])}
    return manifest


def validate_manifest_paths(manifest):
    """Defensive check: every relpath-typed field is actually a valid relpath."""
    def check(path, label):
        if not fc.is_valid_relpath(path):
            raise ValueError(f"Invalid relpath for {label}: {path!r}")

    for arr in manifest.get("arrangements", []):
        if "file" in arr:
            check(arr["file"], f"arrangements[{arr.get('id')}].file")
        if "notation" in arr:
            check(arr["notation"], f"arrangements[{arr.get('id')}].notation")
        if "drum_tab" in arr:
            check(arr["drum_tab"], f"arrangements[{arr.get('id')}].drum_tab")
        if "capo" in arr and arr["capo"] < 0:
            raise ValueError(f"arrangements[{arr.get('id')}].capo must be >= 0")
    for stem in manifest.get("stems", []):
        check(stem["file"], f"stems[{stem.get('id')}].file")
    for lt in manifest.get("lyric_tracks", []):
        check(lt["file"], f"lyric_tracks[{lt.get('id')}].file")
    for key in ("vocal_pitch", "vocal_pitch_contour", "drum_tab", "keys",
                "song_timeline", "cover"):
        if key in manifest:
            check(manifest[key], key)


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------

def package_feedpak(song_folder, work_dir, manifest, output_path):
    manifest_path = os.path.join(work_dir, "manifest.yaml")
    with open(manifest_path, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, "manifest.yaml")
        for stem in manifest["stems"]:
            src = os.path.join(song_folder, os.path.basename(stem["file"]))
            zf.write(src, stem["file"])
        for root, _, files in os.walk(work_dir):
            for f in files:
                if f == "manifest.yaml":
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, work_dir).replace(os.sep, "/")
                zf.write(full, rel)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build(song_folder, output_folder, vocals_stem_name="vocals", drums_stem_name="drums",
          device="cuda", hf_token=None, skip_vocals=False, skip_gp=False):
    total_steps = 5
    fc.log(f"Building feedpak from: {song_folder}")

    fc.log_step(1, total_steps, "Discovering stems and metadata")
    metadata = load_metadata(song_folder)
    stems, vocals_path, drums_path = discover_stems(song_folder, vocals_stem_name, drums_stem_name)
    if not stems:
        raise RuntimeError(f"No audio stems found in {song_folder}")
    fc.log(f"Found {len(stems)} stem(s): {[s['id'] for s in stems]}", indent=1)
    fc.log(f"Vocals stem: {vocals_path or '(none found)'}", indent=1)
    fc.log(f"Drums stem: {drums_path or '(none found)'}", indent=1)

    full_stem = next((s for s in stems if s["id"] == "full"), None)
    duration_source = os.path.join(song_folder, os.path.basename(full_stem["file"])) \
        if full_stem else os.path.join(song_folder, os.path.basename(stems[0]["file"]))
    duration = probe_duration_seconds(duration_source)
    fc.log(f"Duration: {duration}s (probed from {os.path.basename(duration_source)})", indent=1)

    work_dir = os.path.join(song_folder, "_feedpak_build")
    os.makedirs(os.path.join(work_dir, "arrangements"), exist_ok=True)

    fc.log_step(2, total_steps, "Vocal processing (Script 1)")
    lyric_tracks = []
    vocal_pitch_file = vocal_pitch_contour_file = None
    if vocals_path and not skip_vocals:
        vocals_intermediate = os.path.join(song_folder, "intermediate_vocals.json")
        run_vocals_script(vocals_path, vocals_intermediate, device=device, hf_token=hf_token)
        with open(vocals_intermediate, "r", encoding="utf-8") as f:
            vocals_data = json.load(f)
        vocal_stem_id = next(s["id"] for s in stems
                              if os.path.splitext(os.path.basename(s["file"]))[0].lower() == vocals_stem_name.lower())
        lyric_tracks, _ = write_lyric_tracks(vocals_data, work_dir, vocal_stem_id)
        wrote_pitch, wrote_contour = write_merged_vocal_pitch(vocals_data, work_dir)
        vocal_pitch_file = "vocal_pitch.json" if wrote_pitch else None
        vocal_pitch_contour_file = "vocal_pitch_contour.json" if wrote_contour else None
        fc.log(f"Wrote {len(lyric_tracks)} lyric track(s), "
               f"vocal_pitch={'yes' if wrote_pitch else 'no'}, "
               f"vocal_pitch_contour={'yes' if wrote_contour else 'no'}", indent=1)
    elif skip_vocals:
        fc.log("--skip-vocals set; skipping vocal processing.", indent=1)
    else:
        fc.log(f"No stem matching '{vocals_stem_name}' found; skipping vocal processing.", indent=1)

    fc.log_step(3, total_steps, "GP parsing + DTW alignment (Script 2)")
    arrangements = []
    drum_tab_file = keys_file = song_timeline_file = None
    gp_path = find_gp_file(song_folder)
    if gp_path and drums_path and not skip_gp:
        gp_intermediate = os.path.join(song_folder, "intermediate_arrangements.json")
        run_gp_script(gp_path, drums_path, gp_intermediate)
        with open(gp_intermediate, "r", encoding="utf-8") as f:
            gp_data = json.load(f)
        arrangements += write_arrangement_files(gp_data, work_dir)
        arrangements += write_notation_files(gp_data, work_dir)
        drum_tab_file = write_drum_tab(gp_data, work_dir)
        if drum_tab_file:
            arrangements.append({"id": "drums", "name": "Drums", "type": "drums",
                                  "drum_tab": fc.to_posix_relpath(drum_tab_file)})
        keys_file = write_keys(gp_data, work_dir)
        fc.log(f"Wrote {len(arrangements)} arrangement entr(y/ies), "
               f"drum_tab={'yes' if drum_tab_file else 'no'}, "
               f"keys={'yes' if keys_file else 'no'}", indent=1)
    elif skip_gp:
        fc.log("--skip-gp set; skipping arrangement extraction.", indent=1)
    elif not gp_path:
        fc.log(f"No .gp5 file found in {song_folder}; skipping arrangement extraction.", indent=1)
    elif not drums_path:
        fc.log(f"No stem matching '{drums_stem_name}' found; cannot DTW-align .gp5 data. Skipping.",
               indent=1)

    fc.log_step(4, total_steps, "Building manifest.yaml")
    cover_file = next(
        (f for f in os.listdir(song_folder)
         if f.lower() in ("cover.jpg", "cover.png", "album.jpg", "album.png")),
        None,
    )

    manifest = build_manifest(
        metadata, duration, arrangements, stems, lyric_tracks,
        vocal_pitch_file, vocal_pitch_contour_file, drum_tab_file, keys_file,
        song_timeline_file, cover_file,
    )
    validate_manifest_paths(manifest)
    fc.log(f"Manifest validated: {manifest['title']} — {manifest['artist']}", indent=1)

    fc.log_step(5, total_steps, "Packaging .feedpak")
    os.makedirs(output_folder, exist_ok=True)
    safe_name = f"{manifest['artist']} - {manifest['title']}.feedpak".replace("/", "_")
    output_path = os.path.join(output_folder, safe_name)
    package_feedpak(song_folder, work_dir, manifest, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    fc.log(f"Wrote {output_path} ({size_mb:.1f} MB)", indent=1)

    print(f"\nDone: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build a .feedpak from a song folder.")
    parser.add_argument("song_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--vocals-stem", default="vocals")
    parser.add_argument("--drums-stem", default="drums")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--skip-vocals", action="store_true")
    parser.add_argument("--skip-gp", action="store_true")
    args = parser.parse_args()

    build(args.song_folder, args.output_folder,
          vocals_stem_name=args.vocals_stem, drums_stem_name=args.drums_stem,
          device=args.device, hf_token=args.hf_token,
          skip_vocals=args.skip_vocals, skip_gp=args.skip_gp)


if __name__ == "__main__":
    main()
