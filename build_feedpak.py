
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

def compress_padded_stem(wav_path, out_dir):
    """
    Re-encodes a padded WAV stem into a compressed format before it's
    eligible for final packaging — WAV is only safe/necessary as the
    transient output of pad_audio_with_silence() (see its docstring for
    the libsndfile OGG-write crash this avoids); it must never end up in
    the shipped .feedpak, where an uncompressed 3-4 minute stereo 44.1kHz
    stem runs ~35-45MB apiece.

    Tries ffmpeg's own Vorbis encoder first — a completely different
    code path from libsndfile's (which is what crashed), and ffmpeg is
    already a required install for this project (see README setup). If
    ffmpeg isn't on PATH or fails, falls back to FLAC via soundfile:
    still lossless, confirmed not to hit the libsndfile crash (unlike
    Vorbis), and a large size improvement over raw WAV even though it's
    bigger than OGG.

    Deletes wav_path once the replacement is written. Returns the new
    file's path.
    """
    base = os.path.splitext(os.path.basename(wav_path))[0]
    ogg_path = os.path.join(out_dir, base + ".ogg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
             "-c:a", "libvorbis", "-qscale:a", "5", ogg_path],
            check=True,
        )
        os.remove(wav_path)
        return ogg_path
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        fc.log(f"ffmpeg OGG encode unavailable/failed ({e}); "
               f"falling back to FLAC for this stem", indent=1)
        flac_path = os.path.join(out_dir, base + ".flac")
        import soundfile as sf
        data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        sf.write(flac_path, data, sr, format="FLAC")
        os.remove(wav_path)
        return flac_path


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
            # Pad to WAV first (safe, no crash risk — see
            # pad_audio_with_silence()'s docstring), then immediately
            # compress that WAV before it's eligible for packaging.
            wav_basename = os.path.splitext(src_basename)[0] + ".wav"
            wav_path = os.path.join(stems_out_dir, wav_basename)
            fc.pad_audio_with_silence(src, wav_path, pad_seconds)
            dst = compress_padded_stem(wav_path, stems_out_dir)
            stem["file"] = fc.to_posix_relpath("stems", os.path.basename(dst))
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


def run_gp_script(gp_path,drums_path,out_path,sr=22050,count_in_offset=0.0,
                  bass_path=None,piano_path=None,timeline_mode="both",
                  allow_severe_alignment=False):
    cmd=[sys.executable,os.path.join(SCRIPT_DIR,"process_gp_alignment.py"),gp_path]
    if drums_path: cmd.append(drums_path)
    cmd += ["--out",out_path,"--sr",str(sr),"--count-in-offset",str(count_in_offset),
            "--timeline-mode",timeline_mode]
    if bass_path: cmd += ["--bass-audio",bass_path]
    if piano_path: cmd += ["--piano-audio",piano_path]
    if allow_severe_alignment: cmd.append("--allow-severe-alignment")
    fc.log("Launching guarded multi-reference GP/audio alignment")
    subprocess.run(cmd,check=True)


# --------------------------------------------------------------------------
# Wire-format conversion
# --------------------------------------------------------------------------

def write_lyric_tracks(vocals_data,work_dir,vocal_stem_id):
    entries=[]; files=[]; language=vocals_data.get("language") or "und"
    for i,(speaker,data) in enumerate(sorted(vocals_data.get("speakers",{}).items()),1):
        if not data.get("words"): continue
        sid="".join(c if c.isalnum() else "_" for c in speaker.lower()).strip("_") or f"speaker_{i:02d}"
        fn=f"lyrics_{sid}.json"; fc.write_json(os.path.join(work_dir,fn),data["words"])
        entries.append({"id":sid,"file":fn,"language":language,"kind":"original","stem":vocal_stem_id,"name":f"Voice {i}"}); files.append(fn)
    return entries,files

def write_merged_vocal_pitch(vocals_data,work_dir):
    notes=[]; samples=[]
    for data in vocals_data.get("speakers",{}).values():
        notes.extend(data.get("pitch_notes",[])); samples.extend(data.get("contour",[]))
    notes.sort(key=lambda n:(n["t"],n["d"],n["midi"])); samples.sort(key=lambda x:x["t"])
    seen=set(); samples=[x for x in samples if not ((x["t"],x["hz"]) in seen or seen.add((x["t"],x["hz"])))]
    overlaps=0; end=-1.0
    for n in notes:
        if n["t"]<end-0.001: overlaps+=1
        end=max(end,n["t"]+n["d"])
    if notes: fc.write_json(os.path.join(work_dir,"vocal_pitch.json"),{"version":1,"notes":notes})
    if samples: fc.write_json(os.path.join(work_dir,"vocal_pitch_contour.json"),{"version":1,"samples":samples})
    if overlaps: fc.log(f"Vocal pitch warning: {overlaps} simultaneous-voice overlap(s)",indent=1)
    return bool(notes),bool(samples),overlaps

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

def build_manifest(metadata, duration, arrangements, stems, lyrics_file, lyric_tracks,
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
    for track in manifest.get("lyric_tracks", []):
        check(track["file"], f"lyric_tracks[{track.get('id')}].file")
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
          device="cuda",hf_token=None,skip_vocals=False,skip_gp=False,
          keep_work_dir=False,vocal_layout="merged",timeline_mode="both",
          allow_severe_alignment=False):
    if vocal_layout not in ("merged","separated","both"): raise ValueError("invalid vocal_layout")
    if timeline_mode not in ("tempos","beats","both"): raise ValueError("invalid timeline_mode")
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
    bass_path=padded_paths.get(next((s["id"] for s in stems if s["id"].lower()=="bass"),None))
    piano_path=padded_paths.get(next((s["id"] for s in stems if s["id"].lower() in ("piano","keys","keyboard")),None))

    full_stem = next((s for s in stems if s["id"] == "full"), None)
    duration_source = padded_paths.get(full_stem["id"]) if full_stem else padded_paths.get(stems[0]["id"])
    duration = probe_duration_seconds(duration_source)
    fc.log(f"Duration: {duration}s (probed from padded {os.path.basename(duration_source)})", indent=1)

    fc.log_step(3, total_steps, "Vocal processing (Script 1)")
    vocal_pitch_file=vocal_pitch_contour_file=lyrics_file=None
    lyric_tracks=[]
    if vocals_path and not skip_vocals:
        vocals_intermediate=os.path.join(song_folder,"intermediate_vocals.json")
        run_vocals_script(vocals_path,vocals_intermediate,device=device,hf_token=hf_token)
        with open(vocals_intermediate,"r",encoding="utf-8") as f: vocals_data=json.load(f)
        if vocal_layout in ("merged","both"): lyrics_file=write_single_merged_lyrics(vocals_data,work_dir)
        if vocal_layout in ("separated","both"): lyric_tracks,_=write_lyric_tracks(vocals_data,work_dir,vocals_stem_name)
        wrote_pitch,wrote_contour,overlaps=write_merged_vocal_pitch(vocals_data,work_dir)
        vocal_pitch_file="vocal_pitch.json" if wrote_pitch else None
        vocal_pitch_contour_file="vocal_pitch_contour.json" if wrote_contour else None
    elif skip_vocals:
        fc.log("--skip-vocals set; skipping vocal processing.", indent=1)
    else:
        fc.log(f"No stem matching '{vocals_stem_name}' found; skipping vocal processing.", indent=1)

    fc.log_step(4, total_steps, "GP parsing + DTW alignment (Script 2)")
    arrangements = []
    drum_tab_file=keys_file=song_timeline_file=None
    alignment_report=None
    if gp_path and (drums_path or bass_path or piano_path) and not skip_gp:
        gp_intermediate = os.path.join(song_folder, "intermediate_arrangements.json")
        run_gp_script(gp_path,drums_path,gp_intermediate,count_in_offset=count_in_offset,
                      bass_path=bass_path,piano_path=piano_path,timeline_mode=timeline_mode,
                      allow_severe_alignment=allow_severe_alignment)
        with open(gp_intermediate, "r", encoding="utf-8") as f:
            gp_data = json.load(f)
        arrangements += write_arrangement_files(gp_data, work_dir)
        arrangements += write_notation_files(gp_data, work_dir)
        drum_tab_file = write_drum_tab(gp_data, work_dir)
        if drum_tab_file:
            arrangements.append({"id": "drums", "name": "Drums", "type": "drums",
                                  "drum_tab": fc.to_posix_relpath(drum_tab_file)})
        keys_file = write_keys(gp_data, work_dir)
        song_timeline_file=write_song_timeline(gp_data,work_dir)
        alignment_report=gp_data.get("alignment_report")
        fc.log(f"Wrote {len(arrangements)} arrangement entr(y/ies), "
               f"drum_tab={'yes' if drum_tab_file else 'no'}, "
               f"keys={'yes' if keys_file else 'no'}, "
               f"song_timeline={'yes' if song_timeline_file else 'no'}", indent=1)
    elif skip_gp:
        fc.log("--skip-gp set; skipping arrangement extraction.", indent=1)
    elif not gp_path:
        fc.log(f"No .gp5 file found in {song_folder}; skipping arrangement extraction.", indent=1)
    elif not (drums_path or bass_path or piano_path):
        fc.log("No drums, bass, or piano/keys alignment stem found; skipping GP alignment.",indent=1)

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
        metadata,duration,arrangements,stems,lyrics_file,lyric_tracks,
        vocal_pitch_file, vocal_pitch_contour_file, drum_tab_file, keys_file,
        song_timeline_file, cover_file,
    )
    validate_manifest_paths(manifest)
    fc.log(f"Manifest validated: {manifest['title']} — {manifest['artist']}", indent=1)

    fc.log_step(6, total_steps, "Packaging .feedpak")
    os.makedirs(output_folder, exist_ok=True)
    safe_name = f"{manifest['artist']} - {manifest['title']}.feedpak".replace("/", "_")
    output_path = os.path.join(output_folder, safe_name)
    package_feedpak(work_dir,manifest,output_path)
    if alignment_report: fc.write_json(output_path+".alignment_report.json",alignment_report)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    fc.log(f"Wrote {output_path} ({size_mb:.1f} MB)", indent=1)

    print(f"\nDone: {output_path}")

    if not keep_work_dir:
        import shutil
        for leftover in (work_dir,
                         os.path.join(song_folder, "intermediate_vocals.json"),
                         os.path.join(song_folder, "intermediate_arrangements.json")):
            if os.path.isdir(leftover):
                shutil.rmtree(leftover, ignore_errors=True)
            elif os.path.isfile(leftover):
                os.remove(leftover)
        fc.log("Cleaned up working files (pass --keep-work-dir to keep them for debugging)")

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
    parser.add_argument("--vocal-layout",choices=("merged","separated","both"),default="merged")
    parser.add_argument("--timeline-mode",choices=("tempos","beats","both"),default="both",
                        help="Create controlled A/B packages")
    parser.add_argument("--allow-severe-alignment",action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true",
                         help="Keep _feedpak_build/ and intermediate_*.json in the song "
                              "folder after a successful build, for debugging.")
    args = parser.parse_args()

    build(args.song_folder, args.output_folder,
          vocals_stem_name=args.vocals_stem, drums_stem_name=args.drums_stem,
          device=args.device, hf_token=args.hf_token,
          skip_vocals=args.skip_vocals, skip_gp=args.skip_gp,
          keep_work_dir=args.keep_work_dir,vocal_layout=args.vocal_layout,
          timeline_mode=args.timeline_mode,allow_severe_alignment=args.allow_severe_alignment)


if __name__ == "__main__":
    main()
