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
import process_gp_alignment as pga

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_EXTENSIONS = (".ogg", ".wav", ".flac")
FULL_MIX_STEM_NAMES = {"full", "mix", "song", "master"}
COUNT_IN_BEATS = 4


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
# Count-in padding
# --------------------------------------------------------------------------

def prepare_stems_with_count_in(song_folder, stems, gp_path, work_dir,
                                 min_pad_epsilon=0.005):
    """
    Ensures every stem has a full 4-beat count-in of silence before the
    song starts, padding with only as much silence as is missing —
    top-up, not a binary gate. (An earlier version only padded when
    leading silence was under a small fixed threshold like 0.15s; that
    meant a song with, say, exactly 0.15s of real lead-in was judged
    "already padded" even though it needs a full 4-beat count-in, not a
    fraction of one. Comparing directly against count_in_seconds fixes
    that.)

    All stems are padded identically — they're all cuts of the same
    original recording and have to stay in sync with each other — using
    ONE reference stem's leading silence to decide how much is missing
    (preferring the full mix, else drums, else whatever's first).

    Padded copies are written into work_dir/stems/, never overwriting
    the originals in song_folder. Returns (count_in_seconds, stems_dir,
    padded_paths) where padded_paths maps original stem id -> new path,
    and count_in_seconds is the offset process_gp_alignment.py needs
    (via --count-in-offset) to shift every nominal GP time to match —
    always the FULL 4-beat count-in duration, even when only part of it
    was actually added as new silence, since that's how much every
    chart time needs to move to land after the now-guaranteed-full
    count-in.
    """
    stems_out_dir = os.path.join(work_dir, "stems")
    os.makedirs(stems_out_dir, exist_ok=True)

    reference = next((s for s in stems if s["id"] == "full"), None) \
        or next((s for s in stems if s["id"] == "drums"), None) \
        or stems[0]
    reference_path = os.path.join(song_folder, os.path.basename(reference["file"]))
    leading_silence = fc.detect_leading_silence_seconds(reference_path)

    if gp_path:
        bpm = pga.get_initial_tempo(gp_path)
    else:
        bpm = 120.0
        fc.log("No .gp5 file to read tempo from; assuming 120 BPM for count-in sizing", indent=1)
    count_in_seconds = COUNT_IN_BEATS * (60.0 / bpm)

    pad_seconds = max(0.0, count_in_seconds - leading_silence)
    if pad_seconds < min_pad_epsilon:
        pad_seconds = 0.0

    fc.log(f"Reference stem '{reference['id']}' has {leading_silence:.3f}s of leading "
           f"silence; needs {count_in_seconds:.3f}s for a full {COUNT_IN_BEATS}-beat "
           f"count-in at {bpm:.0f} BPM -> padding {pad_seconds:.3f}s", indent=1)

    padded_paths = {}
    for stem in stems:
        src = os.path.join(song_folder, os.path.basename(stem["file"]))
        src_basename = os.path.basename(stem["file"])
        if pad_seconds:
            # Padded stems are written as WAV regardless of source format
            # — see pad_audio_with_silence()'s docstring for why. Update
            # both the on-disk filename and this stem's manifest `file`
            # entry to match, so the archive and the manifest agree.
            new_basename = os.path.splitext(src_basename)[0] + ".wav"
            dst = os.path.join(stems_out_dir, new_basename)
            fc.pad_audio_with_silence(src, dst, pad_seconds)
            stem["file"] = fc.to_posix_relpath("stems", new_basename)
        else:
            import shutil
            dst = os.path.join(stems_out_dir, src_basename)
            shutil.copyfile(src, dst)
        padded_paths[stem["id"]] = dst

    # The offset Script 2 needs is simply "how much silence precedes the
    # real content after padding" — max(existing, count_in_seconds)
    # covers all three cases uniformly: cold start (existing=0, pad up to
    # count_in_seconds), partial lead-in (top up to count_in_seconds),
    # and lead-in that's already MORE than a full count-in (no padding
    # needed, but the chart still has to shift to match wherever the
    # real content actually starts, not stay at zero).
    count_in_offset = max(leading_silence, count_in_seconds)
    return count_in_offset, stems_out_dir, padded_paths


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


def run_gp_script(gp_path, drums_path, out_path, sr=22050, count_in_offset=0.0):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "process_gp_alignment.py"),
           gp_path, drums_path, "--out", out_path, "--sr", str(sr),
           "--count-in-offset", str(count_in_offset)]
    fc.log(f"Launching process_gp_alignment.py as a subprocess (sr={sr}, "
           f"count_in_offset={count_in_offset:.3f}s)")
    subprocess.run(cmd, check=True)
    fc.log("process_gp_alignment.py finished")


# --------------------------------------------------------------------------
# Wire-format conversion
# --------------------------------------------------------------------------

def write_lyric_tracks(vocals_data, work_dir, vocal_stem_id):
    """
    NOT CALLED in v1 — build() currently uses write_single_merged_lyrics()
    instead (see that function's docstring for why). Kept here, tested,
    and ready for when per-speaker lyric_tracks[] becomes the v2 default.

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


def write_single_merged_lyrics(vocals_data, work_dir):
    """
    For now (v1), write a single merged lyrics.json containing all speakers'
    words in time order. A future version should split per speaker into
    lyric_tracks (manifest §5.5), but for compatibility with existing
    feedpak readers, keep this flat for now.
    """
    all_words = []
    for data in vocals_data.get("speakers", {}).values():
        for w in data.get("words", []):
            all_words.append({"t": w["t"], "d": w["d"], "w": w["w"]})

    all_words.sort(key=lambda n: n["t"])

    if all_words:
        fc.write_json(os.path.join(work_dir, "lyrics.json"), all_words)
        return "lyrics.json"
    return None


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
    """
    Declares the kit explicitly (matches a working reference feedpak's
    drum_tab.json exactly: kick/snare/hh_closed/tom_hi/ride/tom_mid/
    crash_r/tom_floor). Every hit's `p` value is guaranteed to be one of
    these eight — see gm_drum_to_piece() in feedpak_common.py — so the
    kit array here is a complete, closed set, not just "whatever pieces
    happened to appear."
    """
    drum_tab = gp_data.get("drum_tab")
    if not drum_tab or not drum_tab.get("hits"):
        return None
    out = {"version": 1, "name": "Drums", "kit": fc.REDUCED_DRUM_KIT,
           "hits": drum_tab["hits"]}
    fc.write_json(os.path.join(work_dir, "drum_tab.json"), out)
    return "drum_tab.json"


def write_keys(gp_data, work_dir):
    keys = gp_data.get("keys")
    if not keys or not keys.get("events"):
        return None
    fc.write_json(os.path.join(work_dir, "keys.json"), keys)
    return "keys.json"


def write_song_timeline(gp_data, work_dir):
    """
    Writes the real song_timeline computed in process_gp_alignment.py
    (measure-header-derived tempos/time_signatures/beats, then warped to
    real audio the same as every other track) — not a placeholder.
    """
    timeline = gp_data.get("song_timeline")
    if not timeline or not (timeline.get("tempos") or timeline.get("beats")):
        return None
    out = {"version": 1}
    out.update({k: v for k, v in timeline.items() if v})
    fc.write_json(os.path.join(work_dir, "song_timeline.json"), out)
    return "song_timeline.json"


# --------------------------------------------------------------------------
# Manifest construction
# --------------------------------------------------------------------------

def build_manifest(metadata, duration, arrangements, stems, lyrics_file,
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
    if lyrics_file:
        manifest["lyrics"] = lyrics_file
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
    for key in ("lyrics", "vocal_pitch", "vocal_pitch_contour", "drum_tab", "keys",
                "song_timeline", "cover"):
        if key in manifest:
            check(manifest[key], key)


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------

def package_feedpak(work_dir, manifest, output_path):
    """
    Everything under work_dir (arrangements/, stems/, and the top-level
    side-files) gets zipped as-is — stems here are already the padded
    copies written by prepare_stems_with_count_in(), not the originals in
    song_folder, so audio and chart timing stay consistent.
    """
    manifest_path = os.path.join(work_dir, "manifest.yaml")
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, explicit_end=False)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, "manifest.yaml")
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
    total_steps = 6
    fc.log(f"Building feedpak from: {song_folder}")

    fc.log_step(1, total_steps, "Discovering stems and metadata")
    metadata = load_metadata(song_folder)
    stems, vocals_path, drums_path = discover_stems(song_folder, vocals_stem_name, drums_stem_name)
    if not stems:
        raise RuntimeError(f"No audio stems found in {song_folder}")
    fc.log(f"Found {len(stems)} stem(s): {[s['id'] for s in stems]}", indent=1)
    fc.log(f"Vocals stem: {vocals_path or '(none found)'}", indent=1)
    fc.log(f"Drums stem: {drums_path or '(none found)'}", indent=1)

    work_dir = os.path.join(song_folder, "_feedpak_build")
    os.makedirs(os.path.join(work_dir, "arrangements"), exist_ok=True)

    gp_path = find_gp_file(song_folder)

    fc.log_step(2, total_steps, "Count-in check (padding stems if needed)")
    count_in_offset, stems_dir, padded_paths = prepare_stems_with_count_in(
        song_folder, stems, gp_path, work_dir)
    # From here on, use the padded copies for everything — DTW reference,
    # vocal processing input, and duration probing all need to see the
    # same audio that ends up in the archive.
    vocals_path = padded_paths.get(next((s["id"] for s in stems
                                          if os.path.splitext(os.path.basename(s["file"]))[0].lower()
                                          == vocals_stem_name.lower()), None))
    drums_path = padded_paths.get(next((s["id"] for s in stems
                                         if os.path.splitext(os.path.basename(s["file"]))[0].lower()
                                         == drums_stem_name.lower()), None))

    full_stem = next((s for s in stems if s["id"] == "full"), None)
    duration_source = padded_paths.get(full_stem["id"]) if full_stem else padded_paths.get(stems[0]["id"])
    duration = probe_duration_seconds(duration_source)
    fc.log(f"Duration: {duration}s (probed from padded {os.path.basename(duration_source)})", indent=1)

    fc.log_step(3, total_steps, "Vocal processing (Script 1)")
    vocal_pitch_file = vocal_pitch_contour_file = lyrics_file = None
    if vocals_path and not skip_vocals:
        vocals_intermediate = os.path.join(song_folder, "intermediate_vocals.json")
        run_vocals_script(vocals_path, vocals_intermediate, device=device, hf_token=hf_token)
        with open(vocals_intermediate, "r", encoding="utf-8") as f:
            vocals_data = json.load(f)
        # Write merged (single) lyrics file, not per-speaker split
        lyrics_file = write_single_merged_lyrics(vocals_data, work_dir)
        wrote_pitch, wrote_contour = write_merged_vocal_pitch(vocals_data, work_dir)
        vocal_pitch_file = "vocal_pitch.json" if wrote_pitch else None
        vocal_pitch_contour_file = "vocal_pitch_contour.json" if wrote_contour else None
        fc.log(f"Wrote lyrics={'yes' if lyrics_file else 'no'}, "
               f"vocal_pitch={'yes' if wrote_pitch else 'no'}, "
               f"vocal_pitch_contour={'yes' if wrote_contour else 'no'}", indent=1)
    elif skip_vocals:
        fc.log("--skip-vocals set; skipping vocal processing.", indent=1)
    else:
        fc.log(f"No stem matching '{vocals_stem_name}' found; skipping vocal processing.", indent=1)

    fc.log_step(4, total_steps, "GP parsing + DTW alignment (Script 2)")
    arrangements = []
    drum_tab_file = keys_file = song_timeline_file = None
    if gp_path and drums_path and not skip_gp:
        gp_intermediate = os.path.join(song_folder, "intermediate_arrangements.json")
        run_gp_script(gp_path, drums_path, gp_intermediate, count_in_offset=count_in_offset)
        with open(gp_intermediate, "r", encoding="utf-8") as f:
            gp_data = json.load(f)
        arrangements += write_arrangement_files(gp_data, work_dir)
        arrangements += write_notation_files(gp_data, work_dir)
        drum_tab_file = write_drum_tab(gp_data, work_dir)
        if drum_tab_file:
            arrangements.append({"id": "drums", "name": "Drums", "type": "drums",
                                  "drum_tab": fc.to_posix_relpath(drum_tab_file)})
        keys_file = write_keys(gp_data, work_dir)
        song_timeline_file = write_song_timeline(gp_data, work_dir)
        fc.log(f"Wrote {len(arrangements)} arrangement entr(y/ies), "
               f"drum_tab={'yes' if drum_tab_file else 'no'}, "
               f"keys={'yes' if keys_file else 'no'}, "
               f"song_timeline={'yes' if song_timeline_file else 'no'}", indent=1)
    elif skip_gp:
        fc.log("--skip-gp set; skipping arrangement extraction.", indent=1)
    elif not gp_path:
        fc.log(f"No .gp5 file found in {song_folder}; skipping arrangement extraction.", indent=1)
    elif not drums_path:
        fc.log(f"No stem matching '{drums_stem_name}' found; cannot DTW-align .gp5 data. Skipping.",
               indent=1)

    fc.log_step(5, total_steps, "Building manifest.yaml")
    cover_file = next(
        (f for f in os.listdir(song_folder)
         if f.lower() in ("cover.jpg", "cover.png", "album.jpg", "album.png")),
        None,
    )
    if cover_file:
        import shutil
        shutil.copyfile(os.path.join(song_folder, cover_file), os.path.join(work_dir, cover_file))

    manifest = build_manifest(
        metadata, duration, arrangements, stems, lyrics_file,
        vocal_pitch_file, vocal_pitch_contour_file, drum_tab_file, keys_file,
        song_timeline_file, cover_file,
    )
    validate_manifest_paths(manifest)
    fc.log(f"Manifest validated: {manifest['title']} — {manifest['artist']}", indent=1)

    fc.log_step(6, total_steps, "Packaging .feedpak")
    os.makedirs(output_folder, exist_ok=True)
    safe_name = f"{manifest['artist']} - {manifest['title']}.feedpak".replace("/", "_")
    output_path = os.path.join(output_folder, safe_name)
    package_feedpak(work_dir, manifest, output_path)
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
