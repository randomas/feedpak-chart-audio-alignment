"""
process_gp_alignment.py

Script 2 of the feedpak pipeline. Parses a .gp5 file, builds a tick->second
map from the file's own tempo data, extracts fretted/drum/keyboard tracks,
then time-warps every nominal (GP-clock) timestamp onto the real audio's
clock via frequency-filtered DTW against the drum stem.

v1 scope note: technique fields (bends, slides, hammer-on/pull-off,
fingering) are intentionally NOT extracted yet — timing/sync correctness
is the current priority (per project decision). Fretted-note output is
limited to {t, s, f, sus}. This keeps Script 2 simpler and easier to
validate against real audio before technique notation is layered on.

Usage:
    python process_gp_alignment.py song.gp5 drums.ogg --out intermediate_arrangements.json
"""
import argparse
import json

import numpy as np

import feedpak_common as fc

try:
    import guitarpro
except ImportError:
    guitarpro = None

try:
    import librosa
    from scipy.interpolate import interp1d
except ImportError:
    librosa = None
    interp1d = None


SUSTAIN_MIN_SECONDS = 0.05


# --------------------------------------------------------------------------
# Tempo map extraction
# --------------------------------------------------------------------------

def build_tempo_events(song):
    """
    Walk every beat in every track looking for a mixTableChange.tempo
    (GP stores tempo changes on beats, not measure headers). Returns a
    sorted, deduplicated list of (tick, bpm) starting with (0, song.tempo).

    Only one track's beats need scanning in principle (tempo changes are
    song-wide), but different tracks can have different beat grids, so we
    scan the first track that has measures and trust GP authoring tools
    to keep tempo meta-events consistent across tracks.
    """
    events = [(0, song.tempo)]
    track = next((t for t in song.tracks if t.measures), None)
    if track is None:
        return events

    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                mtc = getattr(beat.effect, "mixTableChange", None)
                if mtc is not None and mtc.tempo is not None:
                    tick = beat.start if beat.start is not None else measure.header.start
                    events.append((tick, mtc.tempo.value))

    events.sort(key=lambda e: e[0])
    deduped = [events[0]]
    for tick, bpm in events[1:]:
        if tick == deduped[-1][0]:
            deduped[-1] = (tick, bpm)
        else:
            deduped.append((tick, bpm))
    return deduped


def build_key_signature_events(song, tempo_events, first_tick):
    """
    Deterministic keys.json events, straight from GP measure headers'
    keySignature (accidentals, is_minor) — no audio analysis needed.
    Only emits an event when the key actually changes.
    """
    events = []
    last_key = None
    for header in song.measureHeaders:
        ks = header.keySignature
        accidentals, is_minor = ks.value[0], bool(ks.value[1])
        name = fc.key_signature_to_name(accidentals, is_minor)
        if name is None or name == last_key:
            continue
        last_key = name
        t_gp = fc.tick_to_seconds(header.start - first_tick, tempo_events)
        events.append({"t_gp": t_gp, "key": name})
    return events


# --------------------------------------------------------------------------
# Fretted-track parsing (guitar / bass)
# --------------------------------------------------------------------------

def parse_fretted_track(track, tempo_events, first_tick):
    string_count = len(track.strings)
    notes = []

    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                if beat.start is None:
                    continue
                beat_t = fc.tick_to_seconds(beat.start - first_tick, tempo_events)
                dur_ticks = beat.duration.time
                beat_end_t = fc.tick_to_seconds(
                    beat.start - first_tick + dur_ticks, tempo_events
                )
                sustain = beat_end_t - beat_t

                for note in beat.notes:
                    s = fc.remap_string_index(note.string, string_count)
                    entry = {"t_gp": beat_t, "s": s, "f": note.value}
                    fc.omit_if_negligible(entry, "sus", sustain, SUSTAIN_MIN_SECONDS)
                    notes.append(entry)

    notes.sort(key=lambda n: (n["t_gp"], n["s"]))

    absolute_midi = sorted(
        (gs.number, gs.value) for gs in track.strings
    )
    # track.strings numbers are 1-based highest-first, same convention as
    # notes; sort ascending by string number then reverse to get low->high
    absolute_midi_low_to_high = [v for _, v in sorted(absolute_midi, key=lambda x: -x[0])]

    anchors = compute_anchors(notes)

    return {
        "name": track.name or ("Bass" if track.channel and track.channel.instrument in (33, 34, 35, 36, 37, 38, 39) else "Guitar"),
        "string_count": string_count,
        "absolute_tuning_midi": absolute_midi_low_to_high,
        "capo": getattr(track, "offset", 0) or 0,
        "notes": notes,
        "anchors": anchors,
    }


def compute_anchors(notes, window_seconds=2.0):
    """
    Hand-position anchors, computed dynamically from played notes since
    there's no native GP marker for this (ported from a prior converter's
    tested heuristic — see architecture doc §2.2a).
    """
    from collections import Counter

    fretted = [n for n in notes if n["f"] > 0]
    if not fretted:
        return [{"time": 0.0, "fret": 1, "width": 4}]

    grouped = {}
    for n in fretted:
        grouped.setdefault(n["t_gp"], []).append(n)

    timestamps = sorted(grouped.keys())
    anchors = []
    last_fret = None
    last_width = None
    start_idx = 0

    while start_idx < len(timestamps):
        window_start = timestamps[start_idx]
        window_end = window_start + window_seconds

        window_frets = []
        chord_found = False
        chord_lowest = chord_highest = None

        end_idx = start_idx
        while end_idx < len(timestamps) and timestamps[end_idx] <= window_end:
            simultaneous = grouped[timestamps[end_idx]]
            if len(simultaneous) > 1:
                chord_found = True
                chord_lowest = min(n["f"] for n in simultaneous)
                chord_highest = max(n["f"] for n in simultaneous)
                break
            window_frets.append(simultaneous[0]["f"])
            end_idx += 1

        if chord_found:
            target_fret = chord_lowest
            target_width = max(4, chord_highest - chord_lowest + 1)
            start_idx = end_idx + 1
        else:
            if not window_frets:
                start_idx = end_idx + 1
                continue
            counts = Counter(window_frets)
            top_count = counts.most_common(1)[0][1]
            target_fret = min(f for f, c in counts.items() if c == top_count)
            close_frets = [f for f in window_frets if abs(f - target_fret) <= 4]
            target_width = max(4, max(close_frets) - min(close_frets) + 1) if close_frets else 4
            start_idx = end_idx

        if target_fret != last_fret or target_width != last_width:
            anchors.append({"time": window_start, "fret": target_fret, "width": target_width})
            last_fret, last_width = target_fret, target_width

    return anchors if anchors else [{"time": 0.0, "fret": 1, "width": 4}]


# --------------------------------------------------------------------------
# Drum-track parsing
# --------------------------------------------------------------------------

def parse_drum_track(track, tempo_events, first_tick):
    hits = []
    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                if beat.start is None:
                    continue
                beat_t = fc.tick_to_seconds(beat.start - first_tick, tempo_events)
                for note in beat.notes:
                    piece = fc.gm_drum_to_piece(note.value)
                    if piece is None:
                        continue
                    hits.append({
                        "t_gp": beat_t,
                        "p": piece,
                        "v": max(1, min(127, note.velocity)),
                    })
    hits.sort(key=lambda h: h["t_gp"])
    return hits


# --------------------------------------------------------------------------
# Keyboard-track -> notation.json
# --------------------------------------------------------------------------

MIDDLE_C = 60


def parse_keyboard_track(track, tempo_events, first_tick):
    """
    Splits notes into right-hand (>= middle C) / left-hand (< middle C)
    staves by pitch threshold, matching the convention used in the
    project's example notation_keys.json. This is a simplification —
    a real player's hand split doesn't always follow a fixed pitch
    threshold — flagged as a v2 improvement, not a blocker for v1.
    """
    measures_out = []
    last_ts = None

    for idx, measure in enumerate(track.measures, start=1):
        header = measure.header
        m_t = fc.tick_to_seconds(header.start - first_tick, tempo_events)

        rh_beats, lh_beats = [], []
        for voice in measure.voices:
            for beat in voice.beats:
                if beat.start is None or not beat.notes:
                    continue
                beat_t = fc.tick_to_seconds(beat.start - first_tick, tempo_events)
                dur_index = beat.duration.index
                dur_value = 2 ** dur_index if dur_index else 1
                dur_value = min(32, max(1, dur_value))
                midis = [n.value for n in beat.notes]
                beat_entry = {"t": beat_t, "dur": dur_value,
                              "notes": [{"midi": m} for m in midis]}
                if all(m >= MIDDLE_C for m in midis):
                    rh_beats.append(beat_entry)
                elif all(m < MIDDLE_C for m in midis):
                    lh_beats.append(beat_entry)
                else:
                    # mixed chord spanning the split point: put it on rh,
                    # simplest correct-but-imperfect choice for v1
                    rh_beats.append(beat_entry)

        m_out = {"idx": idx, "t": m_t}
        ts = (header.timeSignature.numerator, header.timeSignature.denominator.value) \
            if header.timeSignature else None
        if ts and ts != last_ts:
            m_out["ts"] = list(ts)
            last_ts = ts
        staves = {}
        if rh_beats:
            staves["rh"] = {"voices": [{"v": 1, "beats": rh_beats}]}
        if lh_beats:
            staves["lh"] = {"voices": [{"v": 1, "beats": lh_beats}]}
        if staves:
            m_out["staves"] = staves
        measures_out.append(m_out)

    return {
        "version": 1,
        "instrument": "piano",
        "staves": [
            {"id": "rh", "clef": "G2", "label": "Right Hand"},
            {"id": "lh", "clef": "F4", "label": "Left Hand"},
        ],
        "measures": measures_out,
    }


# --------------------------------------------------------------------------
# DTW alignment
# --------------------------------------------------------------------------

def synthesize_click_track(hit_times, sr, duration_s, freq=180.0, decay_s=0.04):
    """
    Short exponentially-decaying clicks, not sustained tones — a sine
    tone has no sharp transient and onset_strength localizes poorly
    against it (see architecture doc §2.2 step 4).
    """
    n_samples = max(1, int(duration_s * sr))
    audio = np.zeros(n_samples, dtype=np.float32)
    click_n = max(1, int(decay_s * sr))
    t = np.arange(click_n) / sr
    click = np.sin(2 * np.pi * freq * t) * np.exp(-t / (decay_s / 5.0))
    for ht in hit_times:
        start = int(ht * sr)
        if start >= n_samples:
            continue
        end = min(n_samples, start + click_n)
        audio[start:end] += click[: end - start]
    return audio


def compute_warp_function(gp_drum_hits, drums_audio_path, sr=22050, band_rad=0.25):
    """
    Returns a function t_gp -> t_real built from DTW between a synthetic
    click track (from GP's nominal drum timing) and onset-strength of the
    real drum stem. Falls back to identity if librosa/scipy aren't
    available or there are no drum hits to align against.
    """
    if librosa is None or interp1d is None or not gp_drum_hits:
        return lambda t: t

    with fc.timed_step("Loading drum stem audio", indent=2):
        y, actual_sr = librosa.load(drums_audio_path, sr=sr, mono=True)
    duration_s = len(y) / actual_sr
    fc.log(f"{duration_s:.1f}s of audio at {actual_sr}Hz", indent=2)

    hit_times = [h["t_gp"] for h in gp_drum_hits]
    synth = synthesize_click_track(hit_times, actual_sr, duration_s)

    with fc.timed_step("Computing onset-strength envelopes", indent=2):
        onset_real = librosa.onset.onset_strength(y=y, sr=actual_sr)
        onset_synth = librosa.onset.onset_strength(y=synth, sr=actual_sr)

    with fc.timed_step("Running DTW", indent=2):
        _, wp = librosa.sequence.dtw(
            X=onset_synth[np.newaxis, :],
            Y=onset_real[np.newaxis, :],
            global_constraints=True,
            band_rad=band_rad,
        )
    # wp is ordered from the end backward; frame indices for synth (col 0)
    # and real (col 1)
    frames_synth = wp[::-1, 0]
    frames_real = wp[::-1, 1]
    times_synth = librosa.frames_to_time(frames_synth, sr=actual_sr)
    times_real = librosa.frames_to_time(frames_real, sr=actual_sr)

    x, y_vals = fc.dedupe_monotonic_path(list(times_synth), list(times_real))
    fc.log(f"Warp path: {len(wp)} DTW points -> {len(x)} unique time anchors", indent=2)
    if len(x) < 2:
        fc.log("Not enough unique warp points — falling back to identity mapping", indent=2)
        return lambda t: t

    warp = interp1d(x, y_vals, kind="linear", bounds_error=False,
                     fill_value=(y_vals[0], y_vals[-1]))

    def warp_fn(t_gp):
        return float(warp(t_gp))

    return warp_fn


def apply_warp(entries, warp_fn, time_key="t_gp", out_key="t"):
    out = []
    for e in entries:
        e2 = dict(e)
        e2[out_key] = round(warp_fn(e2.pop(time_key)), 4)
        out.append(e2)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def process(gp_path, drums_audio_path, sr=22050):
    if guitarpro is None:
        raise RuntimeError("pyguitarpro is required: pip install pyguitarpro")

    fc.log(f"Processing GP file: {gp_path}")
    fc.log_step(1, 5, "Parsing .gp5 structure")
    with fc.timed_step("guitarpro.parse()", indent=1):
        song = guitarpro.parse(gp_path)
    tempo_events = build_tempo_events(song)
    first_tick = song.measureHeaders[0].start if song.measureHeaders else 0
    fc.log(f"{len(song.tracks)} track(s), {len(song.measureHeaders)} measure(s), "
           f"{len(tempo_events)} tempo segment(s) (starting {tempo_events[0][1]:.0f} BPM)",
           indent=1)

    fc.log_step(2, 5, "Routing tracks (fretted / drum / keyboard)")
    fretted_tracks = {}
    drum_hits_gp = []
    notation_tracks = {}

    for track in song.tracks:
        if not track.measures:
            continue
        if track.isPercussionTrack:
            fc.log(f"'{track.name}' -> drum track", indent=1)
            drum_hits_gp = parse_drum_track(track, tempo_events, first_tick)
            fc.log(f"{len(drum_hits_gp)} drum hits parsed", indent=2)
        elif _looks_like_keyboard(track):
            fc.log(f"'{track.name}' -> keyboard/notation track", indent=1)
            notation_tracks[_safe_track_id(track)] = parse_keyboard_track(
                track, tempo_events, first_tick
            )
        else:
            fc.log(f"'{track.name}' -> fretted track", indent=1)
            data = parse_fretted_track(track, tempo_events, first_tick)
            fretted_tracks[_safe_track_id(track)] = data
            fc.log(f"{len(data['notes'])} notes, {len(data['anchors'])} anchors", indent=2)

    fc.log_step(3, 5, "Reading key signatures")
    key_events_gp = build_key_signature_events(song, tempo_events, first_tick)
    fc.log(f"{len(key_events_gp)} key change event(s)", indent=1)

    fc.log_step(4, 5, "DTW-aligning nominal GP timing to real audio")
    if drum_hits_gp:
        fc.log(f"Reference: {drums_audio_path}", indent=1)
        with fc.timed_step("Computing DTW warp function", indent=1):
            warp_fn = compute_warp_function(drum_hits_gp, drums_audio_path, sr=sr)
    else:
        fc.log("No drum hits parsed from the GP file — skipping DTW, "
               "using identity mapping (nominal GP timing unchanged).", indent=1)
        warp_fn = lambda t: t

    drum_hits = apply_warp(drum_hits_gp, warp_fn)
    key_events = apply_warp(key_events_gp, warp_fn)

    fc.log_step(5, 5, "Applying warp to all tracks")
    arrangements_out = {}
    for track_id, data in fretted_tracks.items():
        warped_notes = apply_warp(data["notes"], warp_fn)
        warped_anchors = [
            {"time": round(warp_fn(a["time"]), 4), "fret": a["fret"], "width": a["width"]}
            for a in data["anchors"]
        ]
        tuning_offsets, nonstandard = fc.tuning_offsets_from_absolute(
            data["absolute_tuning_midi"], is_bass="bass" in data["name"].lower()
        )
        if nonstandard:
            fc.log(f"'{data['name']}': nonstandard string count, tuning offsets "
                   f"are best-effort", indent=1)
        arrangements_out[track_id] = {
            "name": data["name"],
            "tuning": tuning_offsets,
            "tuning_nonstandard_string_count": nonstandard,
            "capo": max(0, data["capo"]),
            "notes": warped_notes,
            "chords": [],
            "anchors": warped_anchors,
            "handshapes": [],
            "templates": [],
        }
        fc.log(f"'{data['name']}' warped ({len(warped_notes)} notes)", indent=1)

    result = {
        "arrangements": arrangements_out,
        "drum_tab": {"version": 1, "hits": drum_hits} if drum_hits else None,
        "notation": notation_tracks if notation_tracks else None,
        "keys": {"version": 1, "events": [{"t": e["t"], "key": e["key"]} for e in key_events]}
                 if key_events else None,
    }
    fc.log(f"Done: {len(arrangements_out)} fretted arrangement(s), "
           f"{len(drum_hits)} drum hits, {len(notation_tracks)} notation track(s)")
    return result


def _looks_like_keyboard(track):
    name = (track.name or "").upper()
    return "KEY" in name or "PIANO" in name


def _safe_track_id(track):
    name = (track.name or f"track{track.number}").strip().lower()
    return "".join(c if c.isalnum() else "_" for c in name).strip("_") or f"track{track.number}"


def main():
    parser = argparse.ArgumentParser(description="Parse a .gp5 file and align it to real audio via DTW.")
    parser.add_argument("gp_file", help="Path to the .gp5 file")
    parser.add_argument("drums_audio", help="Path to the isolated drums stem (e.g. drums.ogg)")
    parser.add_argument("--out", default="intermediate_arrangements.json")
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    result = process(args.gp_file, args.drums_audio, sr=args.sr)
    fc.write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
