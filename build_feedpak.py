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
import re
import subprocess
import sys
import zipfile

import yaml

import feedpak_common as fc
import process_gp_alignment as pga
import project_config as pc

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
        stem_id = "full" if is_full_mix else base.lower()
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


def select_stem_path(padded_paths, stem_id, aliases=()):
    """Resolve an alignment source by manifest stem id.

    ``none``/``off`` disables the source without removing it from packaging.
    When no explicit id is supplied, aliases retain the historical automatic
    discovery behavior.
    """
    if stem_id is not None:
        requested=str(stem_id).strip().lower()
        if requested in ("", "none", "off", "disabled"):
            return None
        return padded_paths.get(requested)
    for alias in aliases:
        if alias in padded_paths:
            return padded_paths[alias]
    return None

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

    if gp_path:
        bpm = pga.get_initial_tempo(gp_path)
    else:
        bpm = 120.0
        fc.log("No .gp5 file to read tempo from; assuming 120 BPM for count-in sizing", indent=1)

    # Initial count-in top-up rule, independent of internal silence checkpoints.
    # Measure only the full mix when available, then add only the missing
    # difference needed to reach exactly four beats of total leading space.
    if gp_path:
        bpm = pga.get_initial_tempo(gp_path)
    else:
        bpm = 120.0
        fc.log("No .gp5 file to read tempo from; assuming 120 BPM for count-in sizing", indent=1)
    count_in_seconds = COUNT_IN_BEATS * (60.0 / bpm)
    full_stem = next((stem for stem in stems if str(stem["id"]).lower() == "full"), None)
    if full_stem is not None:
        full_source = os.path.join(song_folder, os.path.basename(full_stem["file"]))
        leading_silence = fc.detect_leading_silence_seconds(full_source)
        reference = "full"
    else:
        # A full-mix-only project is covered above. If no full mix exists,
        # use the earliest non-vocal stem as a conservative recording-start
        # fallback, never a consensus of later instrument entrances.
        observations = []
        for stem in stems:
            stem_id = str(stem["id"]).lower()
            if stem_id == "vocals":
                continue
            source = os.path.join(song_folder, os.path.basename(stem["file"]))
            observations.append((stem_id, fc.detect_leading_silence_seconds(source)))
        if observations:
            reference, leading_silence = min(observations, key=lambda item: item[1])
        else:
            reference, leading_silence = "none", 0.0
    pad_seconds = max(0.0, count_in_seconds - float(leading_silence))
    if pad_seconds < min_pad_epsilon:
        pad_seconds = 0.0
    fc.log(f"Initial count-in reference: {reference}; existing space {leading_silence:.3f}s", indent=1)
    fc.log(f"Four beats at {bpm:.0f} BPM require {count_in_seconds:.3f}s -> "
           f"adding only missing difference {pad_seconds:.3f}s", indent=1)

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
    audio_content_start = float(leading_silence) + float(pad_seconds)
    fc.log(f"Post-padding audio content start: {audio_content_start:.3f}s; chart offset will be derived from the first configured instrumental score onset", indent=1)
    return audio_content_start, pad_seconds, stems_out_dir, padded_paths


# --------------------------------------------------------------------------
# Subprocess orchestration
# --------------------------------------------------------------------------

def run_vocals_script(vocals_path, out_path, device="cuda", hf_token=None, batch_size=16, compute_type=None, language=None):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "process_vocals.py"),
           vocals_path, "--out", out_path, "--device", device]
    cmd += ["--batch-size",str(batch_size)]
    if compute_type: cmd += ["--compute-type",compute_type]
    if language: cmd += ["--language",language]
    if hf_token: cmd += ["--hf-token",hf_token]
    fc.log(f"Launching process_vocals.py as a subprocess (device={device})")
    # No capture_output: let the child's own [*] progress logs stream
    # straight through to this terminal in real time.
    subprocess.run(cmd, check=True)
    fc.log("process_vocals.py finished")



def load_reusable_vocals(path, vocals_path):
    if not path or not os.path.isfile(path): return None
    with open(path,"r",encoding="utf-8") as f: data=json.load(f)
    if (data.get("cache") or {}).get("audio_sha256") == fc.file_sha256(vocals_path):
        if int(data.get("analysis_version", 1)) < 3:
            fc.log("Vocal cache predates contiguous multi-note syllables; rerunning models",indent=1)
            return None
        fc.log(f"Reusing cached vocal analysis: {path}",indent=1); return data
    fc.log("Vocal cache does not match the padded vocal audio; rerunning models",indent=1)
    return None

def run_gp_script(gp_path,drums_path,out_path,sr=22050,count_in_offset=0.0,
                  bass_path=None,piano_path=None,guitar_path=None,full_path=None,
                  timeline_mode="both",
                  allow_severe_alignment=False,anchors_path=None,padding_added=0.0,
                  auto_chunk_measures=16,alignment_mode="dtw",
                  checkpoint_measures=4,checkpoint_search_radius=1.0,project_config_path=None):
    cmd=[sys.executable,os.path.join(SCRIPT_DIR,"process_gp_alignment.py"),gp_path]
    if drums_path: cmd.append(drums_path)
    cmd += ["--out",out_path,"--sr",str(sr),"--count-in-offset",str(count_in_offset),
            "--timeline-mode",timeline_mode]
    if bass_path: cmd += ["--bass-audio",bass_path]
    if piano_path: cmd += ["--piano-audio",piano_path]
    if guitar_path: cmd += ["--guitar-audio",guitar_path]
    if full_path: cmd += ["--full-audio",full_path]
    if allow_severe_alignment: cmd.append("--allow-severe-alignment")
    if anchors_path: cmd += ["--anchors",anchors_path]
    cmd += ["--padding-added",str(padding_added),"--auto-chunk-measures",str(auto_chunk_measures),
            "--alignment-mode",alignment_mode,
            "--checkpoint-measures",str(checkpoint_measures),
            "--checkpoint-search-radius",str(checkpoint_search_radius)]
    if project_config_path: cmd += ["--project-config",project_config_path]
    fc.log("Launching guarded multi-reference GP/audio alignment")
    subprocess.run(cmd,check=True)


# --------------------------------------------------------------------------
# Wire-format conversion
# --------------------------------------------------------------------------

def shift_vocal_analysis(vocals_data,offset):
    """Map source-audio vocal timestamps onto the packaged padded clock."""
    if not offset: return vocals_data
    out=dict(vocals_data); speakers={}
    for speaker,data in (vocals_data.get("speakers") or {}).items():
        item=dict(data)
        item["words"]=[{**x,"t":round(float(x["t"])+offset,4)} for x in data.get("words",[])]
        item["pitch_notes"]=[{**x,"t":round(float(x["t"])+offset,4)} for x in data.get("pitch_notes",[])]
        item["contour"]=[{**x,"t":round(float(x["t"])+offset,4)} for x in data.get("contour",[])]
        speakers[speaker]=item
    out["speakers"]=speakers
    return out


def embed_legacy_timeline_in_first_arrangement(gp_data,work_dir,entries):
    """Mirror beats/sections for highway readers using the legacy hoist path."""
    timeline=gp_data.get("song_timeline") or {}
    first=next((e for e in entries if e.get("file")),None)
    if not first or not timeline.get("beats"): return
    path=os.path.join(work_dir,*first["file"].split("/"))
    with open(path,"r",encoding="utf-8") as f: data=json.load(f)
    data["beats"]=timeline["beats"]
    if timeline.get("sections"): data["sections"]=timeline["sections"]
    fc.write_json(path,data)



def expand_lyrics_with_pitch_continuations(words, pitch_notes, tolerance=0.002):
    """Emit one lyric record per discrete pitch note inside each syllable.

    The first pitch block carries the syllable text and later blocks carry
    ``+``. Unpitched syllables are preserved at their WhisperX timing.
    Pitch blocks are never duplicated between adjacent syllables.
    """
    ordered_words=sorted((dict(w) for w in words),key=lambda x:float(x["t"]))
    ordered_notes=sorted((dict(n) for n in pitch_notes),key=lambda x:float(x["t"]))
    out=[]; used=set()
    for word in ordered_words:
        start=float(word["t"]); end=start+float(word["d"])
        matches=[]
        for i,note in enumerate(ordered_notes):
            if i in used: continue
            ns=float(note["t"]); ne=ns+float(note["d"])
            midpoint=(ns+ne)/2.0
            # Assign each pitch block to exactly one syllable. Midpoint
            # ownership avoids duplicating a block that starts exactly on
            # the boundary between two adjacent syllables.
            if start-tolerance<=midpoint<end-tolerance:
                matches.append((i,note))
        if not matches:
            out.append({"t":round(start,4),"d":round(float(word["d"]),4),"w":word["w"]})
            continue
        for j,(i,note) in enumerate(matches):
            used.add(i)
            out.append({"t":round(float(note["t"]),4),"d":round(float(note["d"]),4),
                        "w":word["w"] if j==0 else "+"})
    out.sort(key=lambda x:(float(x["t"]),0 if x["w"]!="+" else 1))
    return out

def write_lyric_tracks(vocals_data,work_dir,vocal_stem_id):
    entries=[]; files=[]; language=vocals_data.get("language") or "und"
    for i,(speaker,data) in enumerate(sorted(vocals_data.get("speakers",{}).items()),1):
        if not data.get("words"): continue
        sid="".join(c if c.isalnum() else "_" for c in speaker.lower()).strip("_") or f"speaker_{i:02d}"
        expanded=expand_lyrics_with_pitch_continuations(data["words"],data.get("pitch_notes",[]))
        fn=f"lyrics_{sid}.json"; fc.write_json(os.path.join(work_dir,fn),expanded)
        entries.append({"id":sid,"file":fn,"language":language,"kind":"original","stem":vocal_stem_id,"name":f"Voice {i}"}); files.append(fn)
    return entries,files

def _resolve_global_vocal_monophony(notes, minimum_seconds=0.04):
    out=[]; resolved=0
    for raw in sorted(notes,key=lambda n:(float(n["t"]),-float(n.get("confidence",0)))):
        n=dict(raw); n["t"]=float(n["t"]); n["d"]=float(n["d"])
        if n["d"]<=0: continue
        if not out: out.append(n); continue
        p=out[-1]; pe=p["t"]+p["d"]; ne=n["t"]+n["d"]
        if n["t"]>=pe-0.001:
            if n["t"]<pe: n["t"]=pe; n["d"]=ne-pe
            if n["d"]>=minimum_seconds: out.append(n)
            continue
        resolved+=1
        if int(n["midi"])==int(p["midi"]):
            p["d"]=max(pe,ne)-p["t"]
            p["confidence"]=max(float(p.get("confidence",0)),float(n.get("confidence",0)))
        elif float(n.get("confidence",0))>float(p.get("confidence",0)):
            p["d"]=max(0,n["t"]-p["t"])
            if p["d"]<minimum_seconds: out.pop()
            out.append(n)
        else:
            n["t"]=pe; n["d"]=max(0,ne-pe)
            if n["d"]>=minimum_seconds: out.append(n)
    return out,resolved


def write_merged_vocal_pitch(vocals_data,work_dir):
    notes=[]; samples=[]
    for data in vocals_data.get("speakers",{}).values():
        notes.extend(data.get("pitch_notes",[])); samples.extend(data.get("contour",[]))
    samples.sort(key=lambda x:x["t"])
    seen=set(); samples=[x for x in samples if not ((x["t"],x["hz"]) in seen or seen.add((x["t"],x["hz"])))]
    raw=len(notes); notes,resolved=_resolve_global_vocal_monophony(notes)
    wire=[{"t":round(float(n["t"]),4),"d":round(float(n["d"]),4),"midi":int(n["midi"])} for n in notes]
    if wire: fc.write_json(os.path.join(work_dir,"vocal_pitch.json"),{"version":1,"notes":wire})
    if samples: fc.write_json(os.path.join(work_dir,"vocal_pitch_contour.json"),{"version":1,"samples":samples})
    fc.log(f"Vocal pitch coverage: {raw} speaker note(s) -> {len(wire)} monophonic output note(s); {resolved} overlap(s) resolved",indent=1)
    return bool(wire),bool(samples),resolved

def write_single_merged_lyrics(vocals_data, work_dir):
    """
    For now (v1), write a single merged lyrics.json containing all speakers'
    words in time order. A future version should split per speaker into
    lyric_tracks (manifest §5.5), but for compatibility with existing
    feedpak readers, keep this flat for now.
    """
    all_words = []
    for data in vocals_data.get("speakers", {}).values():
        expanded=expand_lyrics_with_pitch_continuations(data.get("words", []),data.get("pitch_notes", []))
        all_words.extend(expanded)

    all_words.sort(key=lambda n: n["t"])

    if all_words:
        fc.write_json(os.path.join(work_dir, "lyrics.json"), all_words)
        return "lyrics.json"
    return None


def validate_vocal_coverage(vocals_data,duration,warning_tail_seconds=45.0):
    words=[]; pitch=[]
    for speaker in (vocals_data.get("speakers") or {}).values():
        words.extend(speaker.get("words") or []); pitch.extend(speaker.get("pitch_notes") or [])
    end=lambda xs:max((float(x.get("t",0))+float(x.get("d",0)) for x in xs),default=None)
    we,pe=end(words),end(pitch); last=max([x for x in (we,pe) if x is not None],default=None)
    tail=None if last is None or duration is None else max(0.0,float(duration)-last)
    threshold=max(float(warning_tail_seconds),.25*float(duration or 0)); warnings=[]
    if not words:warnings.append("no_timed_words")
    if not pitch:warnings.append("no_pitch_notes")
    if tail is not None and tail>threshold:warnings.append("large_unexplained_song_tail_without_vocals")
    if we is not None and pe is not None and abs(we-pe)>30:warnings.append("lyrics_and_pitch_end_times_diverge")
    out={"status":"warning" if warnings else "ok","word_count":len(words),"pitch_note_count":len(pitch),"last_word_end":we,"last_pitch_end":pe,"song_duration":duration,"tail_seconds":tail,"tail_warning_threshold_seconds":threshold,"warnings":warnings}
    fc.log(f"Vocal coverage: words={len(words)}, pitch={len(pitch)}, last={last}, status={out['status']}",indent=1)
    return out

def write_gp_vocals(gp_data,work_dir):
    data=gp_data.get("gp_vocals") or {}
    lyrics=[{"t":x["t"],"d":x["d"],"w":x["w"]} for x in data.get("lyrics",[]) if x.get("d",0)>0]
    pitch=[{"t":x["t"],"d":x["d"],"midi":int(x["midi"])} for x in data.get("pitch_notes",[]) if x.get("d",0)>0]
    if lyrics: fc.write_json(os.path.join(work_dir,"lyrics.json"),lyrics)
    if pitch: fc.write_json(os.path.join(work_dir,"vocal_pitch.json"),{"version":1,"notes":pitch})
    if data:
        fc.write_json(os.path.join(work_dir,"gp_vocal_diagnostics.json"),data.get("diagnostics",{}))
        d=data.get("diagnostics",{}); fc.log(f"GP vocal output: {d.get('bars_written',0)} bar(s) written, {d.get('bars_skipped',0)} skipped",indent=1)
    return ("lyrics.json" if lyrics else None,"vocal_pitch.json" if pitch else None)

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
        if arr.get("type"):
            entry["type"] = arr["type"]
        if arr.get("tuning_nonstandard_string_count"):
            print(f"Note: {track_id} has a nonstandard string count; "
                  f"tuning offsets are best-effort.", file=sys.stderr)
        entries.append(entry)
    return entries


def write_notation_files(gp_data, work_dir, arrangement_entries=None):
    """Write notation and attach it to the matching playable arrangement.

    Piano LH/RH remain independent manifest entries. This avoids the old
    duplicate notation-only entries and guarantees that every piano lane has
    both ``file`` and ``notation``.
    """
    entries = arrangement_entries if arrangement_entries is not None else []
    by_id = {entry.get("id"): entry for entry in entries}
    for track_id, notation in (gp_data.get("notation") or {}).items():
        filename = f"notation_{track_id}.json"
        fc.write_json(os.path.join(work_dir, "arrangements", filename), notation)
        notation_path = fc.to_posix_relpath("arrangements", filename)
        if track_id in by_id:
            by_id[track_id]["notation"] = notation_path
            by_id[track_id]["type"] = "piano"
        else:
            entry = {"id": track_id, "name": track_id.replace("_", " ").title(),
                     "type": "piano", "notation": notation_path}
            entries.append(entry)
            by_id[track_id] = entry
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
        "feedpak_version": "1.19.0",
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



PROJECT_BASE_NAME = "feedpak-project.yaml"
PROJECT_VERSION_RE = re.compile(r"^feedpak-project\.v(\d{3})\.yaml$")


def find_latest_project_file(song_folder, explicit=None):
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(song_folder, explicit)
    if not os.path.isdir(song_folder):
        return None
    found=[]
    base=os.path.join(song_folder,PROJECT_BASE_NAME)
    if os.path.isfile(base): found.append((1,base))
    for name in os.listdir(song_folder):
        match=PROJECT_VERSION_RE.match(name)
        if match: found.append((int(match.group(1)),os.path.join(song_folder,name)))
    return max(found,key=lambda x:x[0])[1] if found else None


def load_project_defaults(path):
    if not path or not os.path.isfile(path): return {}
    with open(path,"r",encoding="utf-8") as f: data=yaml.safe_load(f) or {}
    vocals=data.get("vocals",{}); alignment=data.get("alignment",{}); timeline=data.get("timeline",{}); build=data.get("build",{}); stems=data.get("stems",{})
    return {"vocals_stem":stems.get("vocals"),"drums_stem":stems.get("drums"),
            "bass_alignment_stem":stems.get("bass_alignment"),"piano_alignment_stem":stems.get("piano_alignment"),
            "guitar_alignment_stem":stems.get("guitar_alignment"),
            "device":vocals.get("device"),"vocal_layout":vocals.get("layout"),
            "reuse_vocals":vocals.get("reuse"),"vocals_cache":vocals.get("cache"),
            "vocal_batch_size":vocals.get("batch_size"),"vocal_compute_type":vocals.get("compute_type"),
            "vocal_language":vocals.get("language"),"anchors":alignment.get("anchors"),
            "auto_chunk_measures":alignment.get("max_chunk_measures"),
            "alignment_mode":alignment.get("mode"),
            "checkpoint_measures":alignment.get("checkpoint_measures"),
            "checkpoint_search_radius":alignment.get("checkpoint_search_radius"),
            "allow_severe_alignment":alignment.get("allow_severe"),
            "timeline_mode":timeline.get("mode"),"keep_work_dir":build.get("keep_work_dir")}


def project_document(args):
    return {"version":1,"stems":{"vocals":args.vocals_stem,"drums":args.drums_stem,
                    "bass_alignment":args.bass_alignment_stem,"piano_alignment":args.piano_alignment_stem,
                    "guitar_alignment":args.guitar_alignment_stem},
            "vocals":{"device":args.device,"layout":args.vocal_layout,"reuse":args.reuse_vocals,
                       "cache":args.vocals_cache,"batch_size":args.vocal_batch_size,
                       "compute_type":args.vocal_compute_type,"language":args.vocal_language},
            "alignment":{"mode":args.alignment_mode,"anchors":args.anchors,
                         "max_chunk_measures":args.auto_chunk_measures,
                         "checkpoint_measures":args.checkpoint_measures,
                         "checkpoint_search_radius":args.checkpoint_search_radius,
                         "allow_severe":args.allow_severe_alignment},
            "timeline":{"mode":args.timeline_mode},
            "build":{"keep_work_dir":args.keep_work_dir}}


def write_project_version(song_folder, document, current_path=None):
    current=None
    if current_path and os.path.isfile(current_path):
        with open(current_path,"r",encoding="utf-8") as f: current=yaml.safe_load(f) or {}
    if current == document:
        return current_path
    versions=[1]
    for name in os.listdir(song_folder):
        match=PROJECT_VERSION_RE.match(name)
        if match: versions.append(int(match.group(1)))
    if not current_path:
        path=os.path.join(song_folder,PROJECT_BASE_NAME)
    else:
        path=os.path.join(song_folder,f"feedpak-project.v{max(versions)+1:03d}.yaml")
    with open(path,"w",encoding="utf-8",newline="\n") as f:
        yaml.safe_dump(document,f,sort_keys=False,allow_unicode=True)
    fc.log(f"Project configuration saved: {path}",indent=1)
    return path

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build(song_folder, output_folder, vocals_stem_name="vocals", drums_stem_name="drums",
          device="cuda",hf_token=None,skip_vocals=False,skip_gp=False,
          keep_work_dir=False,vocal_layout="merged",timeline_mode="both",
          allow_severe_alignment=False,reuse_vocals=False,vocals_cache=None,
          vocal_batch_size=16,vocal_compute_type=None,vocal_language=None,
          anchors_path=None,auto_chunk_measures=16,alignment_mode="dtw",
          checkpoint_measures=4,checkpoint_search_radius=1.0,song_json_path=None,
          bass_alignment_stem=None,piano_alignment_stem=None,guitar_alignment_stem=None):
    if vocal_layout not in ("merged","separated","both"): raise ValueError("invalid vocal_layout")
    if timeline_mode not in ("tempos","beats","both"): raise ValueError("invalid timeline_mode")
    total_steps=7
    fc.log(f"Building feedpak from: {song_folder}")
    gp_path=find_gp_file(song_folder); metadata_path=song_json_path or os.path.join(song_folder,"metadata.json")
    fc.log_step(0,total_steps,"Inspecting GP5 and validating song configuration")
    gp_vocal_capability={"classification":"no_vocal_material","direct_gp_lyrics_supported":False,"reason":"GP unavailable"}
    if gp_path and not skip_gp:
        song=pga.guitarpro.parse(gp_path); inventory=pc.inventory_song(song); metadata,created=pc.ensure_song_json(metadata_path,os.path.basename(gp_path),inventory); validation=pc.validate_song_config(metadata,inventory)
        inspection_path=os.path.join(song_folder,"gp5_inspection.json"); fc.write_json(inspection_path,pc.inspect_song(song,gp_path,metadata,inventory,validation))
        if created: fc.log(f"Created song configuration: {metadata_path}",indent=1)
        fc.log(f"Wrote GP5 inspection: {inspection_path}",indent=1)
        if not validation["valid"]:
            for error in validation["errors"]: fc.log("CONFIG ERROR: "+json.dumps(error,ensure_ascii=False),indent=1)
            raise RuntimeError("Song configuration does not match GP5 tracks; no audio processing was started")
        roles=pc.resolved_roles(metadata,inventory)
        gp_vocal_capability=pc.classify_vocal_capability(song,inventory,roles)
        inspection=pc.inspect_song(song,gp_path,metadata,inventory,validation); inspection["vocal_capability"]=gp_vocal_capability; fc.write_json(inspection_path,inspection)
        for item in inventory: fc.log(f"Track {item['index']}: '{item['name']}' -> {roles[item['name']]}",indent=1)
        fc.log(f"GP vocal capability: {gp_vocal_capability['classification']} ({gp_vocal_capability['reason']})",indent=1)
    else: metadata=load_metadata(song_folder)
    gp_vocal_configured=bool(((metadata.get("feedpak_project") or {}).get("tracks") or {}).get("lead_vocal"))
    gp_vocal_production=bool(gp_vocal_configured and gp_vocal_capability.get("direct_gp_lyrics_supported"))
    fc.log_step(1,total_steps,"Discovering stems and metadata")
    stems, vocals_path, drums_path = discover_stems(song_folder, vocals_stem_name, drums_stem_name)
    original_vocals_path=vocals_path
    if not stems:
        raise RuntimeError(f"No audio stems found in {song_folder}")
    fc.log(f"Found {len(stems)} stem(s): {[s['id'] for s in stems]}", indent=1)
    fc.log(f"Vocals stem: {vocals_path or '(none found)'}", indent=1)
    fc.log(f"Drums stem: {drums_path or '(none found)'}", indent=1)

    work_dir = os.path.join(song_folder, "_feedpak_build")
    os.makedirs(os.path.join(work_dir, "arrangements"), exist_ok=True)

    fc.log_step(2, total_steps, "Count-in check (padding stems if needed)")
    count_in_offset, padding_added, stems_dir, padded_paths = prepare_stems_with_count_in(
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
    bass_path=select_stem_path(padded_paths,bass_alignment_stem,("bass",))
    piano_path=select_stem_path(padded_paths,piano_alignment_stem,("piano","keys","keyboard"))
    guitar_path=select_stem_path(padded_paths,guitar_alignment_stem,("guitar","rhythm_guitar","lead_guitar"))
    fc.log(f"Alignment stem routing: drums={os.path.basename(drums_path) if drums_path else 'off'}, "
           f"bass={os.path.basename(bass_path) if bass_path else 'off'}, "
           f"piano={os.path.basename(piano_path) if piano_path else 'off'}, "
           f"guitar={os.path.basename(guitar_path) if guitar_path else 'off'}",indent=1)

    full_stem = next((s for s in stems if s["id"] == "full"), None)
    full_path = padded_paths.get(full_stem["id"]) if full_stem else None
    duration_source = full_path if full_path else padded_paths.get(stems[0]["id"])
    duration = probe_duration_seconds(duration_source)
    fc.log(f"Duration: {duration}s (probed from padded {os.path.basename(duration_source)})", indent=1)

    fc.log_step(3, total_steps, "Vocal processing (Script 1)")
    vocal_pitch_file=vocal_pitch_contour_file=lyrics_file=None
    lyric_tracks=[]
    if gp_vocal_production:
        fc.log("Configured GP lead vocal has synchronized lyric timing; deferring to GP interpreter",indent=1)
    elif gp_vocal_configured:
        fc.log("WARNING: Configured GP lead vocal cannot provide synchronized lyrics",indent=1); fc.log(f"GP classification: {gp_vocal_capability['classification']}; {gp_vocal_capability['reason']}",indent=2); fc.log("Falling back to WhisperX/CREPE",indent=2)
    if not gp_vocal_production and vocals_path and not skip_vocals:
        vocals_analysis_path=original_vocals_path or vocals_path
        vocals_intermediate=vocals_cache or os.path.join(song_folder,"intermediate_vocals.json")
        vocals_data=load_reusable_vocals(vocals_intermediate,vocals_analysis_path) if (reuse_vocals or os.path.isfile(vocals_intermediate)) else None
        if vocals_data is None:
            run_vocals_script(vocals_analysis_path,vocals_intermediate,device=device,hf_token=hf_token,
                              batch_size=vocal_batch_size,compute_type=vocal_compute_type,
                              language=vocal_language)
            with open(vocals_intermediate,"r",encoding="utf-8") as f: vocals_data=json.load(f)
        vocals_data=shift_vocal_analysis(vocals_data,padding_added)
        fc.write_json(os.path.join(work_dir,"vocal_coverage_report.json"),validate_vocal_coverage(vocals_data,duration))
        if vocal_layout in ("merged","both"): lyrics_file=write_single_merged_lyrics(vocals_data,work_dir)
        if vocal_layout in ("separated","both"): lyric_tracks,_=write_lyric_tracks(vocals_data,work_dir,vocals_stem_name)
        wrote_pitch,wrote_contour,overlaps=write_merged_vocal_pitch(vocals_data,work_dir)
        vocal_pitch_file="vocal_pitch.json" if wrote_pitch else None
        vocal_pitch_contour_file="vocal_pitch_contour.json" if wrote_contour else None
    elif not gp_vocal_production and skip_vocals:
        fc.log("--skip-vocals set; skipping audio vocal fallback.", indent=1)
    elif not gp_vocal_production and not vocals_path:
        fc.log(f"WARNING: No stem matching '{vocals_stem_name}' found; reliable vocals omitted.", indent=1)

    fc.log_step(4, total_steps, "GP parsing + DTW alignment (Script 2)")
    arrangements = []
    drum_tab_file=keys_file=song_timeline_file=None
    alignment_report=None
    if gp_path and (drums_path or bass_path or piano_path) and not skip_gp:
        gp_intermediate = os.path.join(song_folder, "intermediate_arrangements.json")
        run_gp_script(gp_path,drums_path,gp_intermediate,count_in_offset=count_in_offset,
                      bass_path=bass_path,piano_path=piano_path,guitar_path=guitar_path,
                      full_path=full_path,timeline_mode=timeline_mode,
                      allow_severe_alignment=allow_severe_alignment,
                      anchors_path=anchors_path,padding_added=padding_added,
                      auto_chunk_measures=auto_chunk_measures,alignment_mode=alignment_mode,
                      checkpoint_measures=checkpoint_measures,
                      checkpoint_search_radius=checkpoint_search_radius,project_config_path=metadata_path if gp_path else None)
        with open(gp_intermediate, "r", encoding="utf-8") as f:
            gp_data = json.load(f)
        if gp_vocal_production:
            lyrics_file,vocal_pitch_file=write_gp_vocals(gp_data,work_dir)
            vocal_pitch_contour_file=None
            lyric_tracks=[]
        arrangements += write_arrangement_files(gp_data, work_dir)
        arrangements = write_notation_files(gp_data, work_dir, arrangements)
        drum_tab_file = write_drum_tab(gp_data, work_dir)
        if drum_tab_file:
            arrangements.append({"id": "drums", "name": "Drums", "type": "drums",
                                  "drum_tab": fc.to_posix_relpath(drum_tab_file)})
        keys_file = write_keys(gp_data, work_dir)
        song_timeline_file=write_song_timeline(gp_data,work_dir)
        embed_legacy_timeline_in_first_arrangement(gp_data,work_dir,arrangements)
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
    if alignment_report:
        report_path=output_path+".alignment_report.json"
        fc.write_json(report_path,alignment_report)
        if not os.path.isfile(report_path): raise RuntimeError(f"Alignment report write failed: {report_path}")
        fc.log(f"Wrote alignment report: {report_path}",indent=1)
    else:
        fc.log("No alignment report returned by process_gp_alignment.py",indent=1)
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
    # First pass finds song folder and optional explicit project file.
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument("song_folder"); pre.add_argument("output_folder")
    pre.add_argument("--config",default=None)
    known,_=pre.parse_known_args()
    project_path=find_latest_project_file(known.song_folder,known.config)
    defaults=load_project_defaults(project_path)

    parser=argparse.ArgumentParser(description="Build a .feedpak from a song folder.")
    parser.add_argument("song_folder"); parser.add_argument("output_folder"); parser.add_argument("--config",default=known.config)
    parser.add_argument("--song-json",default=None)
    parser.add_argument("--vocals-stem",default=defaults.get("vocals_stem") or "vocals")
    parser.add_argument("--drums-stem",default=defaults.get("drums_stem") or "drums")
    parser.add_argument("--bass-alignment-stem",default=defaults.get("bass_alignment_stem"),help="Stem id used as bass alignment evidence; use 'none' to disable")
    parser.add_argument("--piano-alignment-stem",default=defaults.get("piano_alignment_stem"),help="Stem id used as piano alignment evidence; e.g. guitar if that file is piano-dominant")
    parser.add_argument("--guitar-alignment-stem",default=defaults.get("guitar_alignment_stem"),help="Stem id used as guitar alignment evidence; use 'none' to disable")
    parser.add_argument("--device",default=defaults.get("device") or "cuda")
    parser.add_argument("--hf-token",default=None)
    parser.add_argument("--skip-vocals",action="store_true"); parser.add_argument("--skip-gp",action="store_true")
    parser.add_argument("--vocal-layout",choices=("merged","separated","both"),default=defaults.get("vocal_layout") or "merged")
    parser.add_argument("--timeline-mode",choices=("tempos","beats","both"),default=defaults.get("timeline_mode") or "both")
    parser.add_argument("--allow-severe-alignment",action="store_true",default=bool(defaults.get("allow_severe_alignment",False)))
    parser.add_argument("--alignment-mode",choices=("nominal","offset","linear","dtw","dtw-checkpoint","checkpoint-linear","checkpoint-dtw-diagnostic","checkpoint-dtw-selective"),
                        default=defaults.get("alignment_mode") or "dtw")
    parser.add_argument("--anchors",default=defaults.get("anchors"))
    parser.add_argument("--auto-chunk-measures",type=int,default=defaults.get("auto_chunk_measures") or 16)
    parser.add_argument("--checkpoint-measures",type=int,default=defaults.get("checkpoint_measures") or 4)
    parser.add_argument("--checkpoint-search-radius",type=float,default=defaults.get("checkpoint_search_radius") or 1.0)
    parser.add_argument("--reuse-vocals",action="store_true",default=bool(defaults.get("reuse_vocals",False)))
    parser.add_argument("--vocals-cache",default=defaults.get("vocals_cache"))
    parser.add_argument("--vocal-batch-size",type=int,default=defaults.get("vocal_batch_size") or 16)
    parser.add_argument("--vocal-compute-type",default=defaults.get("vocal_compute_type"))
    parser.add_argument("--vocal-language",default=defaults.get("vocal_language"))
    parser.add_argument("--keep-work-dir",action="store_true",default=bool(defaults.get("keep_work_dir",False)))
    args=parser.parse_args()

    # Explicit CLI values have already overridden project-derived parser defaults.
    project_path=write_project_version(args.song_folder,project_document(args),project_path)
    anchors=args.anchors
    if anchors and not os.path.isabs(anchors): anchors=os.path.join(args.song_folder,anchors)
    cache=args.vocals_cache
    if cache and not os.path.isabs(cache): cache=os.path.join(args.song_folder,cache)
    build(args.song_folder,args.output_folder,vocals_stem_name=args.vocals_stem,
          drums_stem_name=args.drums_stem,device=args.device,hf_token=args.hf_token,
          skip_vocals=args.skip_vocals,skip_gp=args.skip_gp,keep_work_dir=args.keep_work_dir,
          vocal_layout=args.vocal_layout,timeline_mode=args.timeline_mode,
          allow_severe_alignment=args.allow_severe_alignment,reuse_vocals=args.reuse_vocals,
          vocals_cache=cache,vocal_batch_size=args.vocal_batch_size,
          vocal_compute_type=args.vocal_compute_type,vocal_language=args.vocal_language,
          anchors_path=anchors,auto_chunk_measures=args.auto_chunk_measures,
          alignment_mode=args.alignment_mode,
          checkpoint_measures=args.checkpoint_measures,
          checkpoint_search_radius=args.checkpoint_search_radius,song_json_path=(args.song_json if not args.song_json or os.path.isabs(args.song_json) else os.path.join(args.song_folder,args.song_json)),
          bass_alignment_stem=args.bass_alignment_stem,piano_alignment_stem=args.piano_alignment_stem,
          guitar_alignment_stem=args.guitar_alignment_stem)


if __name__ == "__main__":
    main()















