"""Adapter for schema-v4 JSON emitted by alphatab-extractor/extract-score.mjs.

This module deliberately does not replace the PyGuitarPro path. It converts a
modern .gp extraction into the nominal products consumed by the existing
alignment and Feedpak writing stages.
"""
from __future__ import annotations

from collections import Counter
import json
import os
import re
import subprocess

import feedpak_common as fc

SUPPORTED_SCHEMA = 4


def ensure_json(score_path, extractor=None, node="node", output_path=None):
    score_path = os.fspath(score_path)
    if score_path.lower().endswith(".json"):
        return score_path
    if not extractor:
        raise ValueError(
            "alphaTab parsing requires --alphatab-extractor for a .gp input, "
            "or a pre-extracted --score-json"
        )
    output_path = output_path or os.path.splitext(score_path)[0] + ".alphatab.json"
    subprocess.run([node, os.fspath(extractor), score_path, output_path], check=True)
    return output_path


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if int(data.get("schema_version", -1)) != SUPPORTED_SCHEMA:
        raise ValueError(
            f"Unsupported alphaTab score schema {data.get('schema_version')!r}; "
            f"expected {SUPPORTED_SCHEMA}"
        )
    if not (data.get("playback") or {}).get("available"):
        raise ValueError("alphaTab extraction has no playback timeline")
    if int((data.get("source") or {}).get("ppq", 0)) <= 0:
        raise ValueError("alphaTab extraction has no valid PPQ")
    return data


def guess_role(track):
    n = (track.get("name") or "").lower()
    if any(x in n for x in ("drum", "kit", "percussion", "tambourine", "triangle")):
        return "drums"
    if any(x in n for x in ("vocal", "voice", "singer")):
        return "lead_vocal" if "back" not in n else "ignore"
    if any(x in n for x in ("piano", "keyboard", "keys")):
        if any(x in n for x in ("lh", "left")):
            return "piano_left"
        if any(x in n for x in ("rh", "right")):
            return "piano_right"
        return "piano_combined"
    if "bass" in n:
        return "bass"
    if any(x in n for x in ("guitar", "lead", "rhythm")):
        return "guitar"
    return "ignore"


def inventory(data):
    return [
        {
            "index": t["index"],
            "name": t.get("name") or f"Track {t['index'] + 1}",
            "automatic_role": guess_role(t),
        }
        for t in data.get("tracks", [])
    ]


def resolved_roles(data, project_config_path=None):
    roles = {x["name"]: x["automatic_role"] for x in inventory(data)}
    if not project_config_path or not os.path.isfile(project_config_path):
        return roles
    with open(project_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    tracks = ((cfg.get("feedpak_project") or {}).get("tracks") or {})

    def assign(name, role):
        if isinstance(name, str) and name in roles:
            roles[name] = role

    for name in tracks.get("unused_tracks") or []:
        assign(name, "unsupported")
    for name in tracks.get("ignored_tracks") or []:
        assign(name, "ignored")
    for role in ("lead_vocal", "drums"):
        assign(tracks.get(role), role)
    piano = tracks.get("piano") or {}
    if isinstance(piano, dict):
        for hand in ("left", "right", "combined"):
            assign(piano.get(hand), "piano_" + hand)
    for name in tracks.get("guitars") or []:
        assign(name, "guitar")
    for name in tracks.get("bass") or []:
        assign(name, "bass")
    overrides = tracks.get("overrides") or {}
    if isinstance(overrides, dict):
        for name, role in overrides.items():
            if isinstance(role, str):
                assign(name, role)
    return roles


def _tempo_events(data):
    initial = float((data.get("metadata") or {}).get("tempo") or 120.0)
    raw = sorted(
        (int(x.get("tick", 0)), float(x["tempo"]))
        for x in data["playback"].get("tempo_events", [])
        if x.get("tempo")
    )
    if not raw or raw[0][0] != 0:
        raw.insert(0, (0, initial))
    out = []
    for tick, bpm in raw:
        if out and tick == out[-1][0]:
            out[-1] = (tick, bpm)
        elif not out or bpm != out[-1][1]:
            out.append((tick, bpm))
    return out


def _seconds(tick, tempo, ppq):
    return fc.tick_to_seconds(float(tick), tempo, ppq)


def _safe_id(name, index):
    stem = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return stem or f"track_{index + 1}"


def _drum_midi(event):
    percussion = event.get("percussion") or {}
    value = percussion.get("output_midi_number")
    if value is None:
        return None
    return int(value)


def products(data, project_config_path=None):
    ppq = int(data["source"]["ppq"])
    tempo = _tempo_events(data)
    roles = resolved_roles(data, project_config_path)
    tracks = {int(t["index"]): t for t in data.get("tracks", [])}

    notes_by_id = {}
    for track in tracks.values():
        for staff in track.get("staves", []):
            for bar in staff.get("bars", []):
                for voice in bar.get("voices", []):
                    for beat in voice.get("beats", []):
                        for note in beat.get("notes", []):
                            notes_by_id[note.get("id")] = note

    fretted = {}
    drums = []
    notation = {}
    gp_vocals = None
    bass_on, guitar_on, piano_on = [], [], []
    bass_pitch, guitar_pitch, piano_pitch = [], [], []
    unresolved_drums = Counter()

    occurrences = data["playback"].get("beat_occurrences", [])
    pnotes = data["playback"].get("playback_notes", [])
    by_track = {i: [] for i in tracks}
    for note in pnotes:
        by_track.setdefault(int(note.get("track_index", -1)), []).append(note)

    for idx, track in tracks.items():
        name = track.get("name") or f"Track {idx + 1}"
        role = roles.get(name, "ignore")
        tid = _safe_id(name, idx)
        events = by_track.get(idx, [])

        if role in ("guitar", "bass"):
            props = (track.get("staves") or [{}])[0].get("properties", {})
            tuning = [int(x) for x in (props.get("tuning") or [])]
            out = []
            for event in events:
                start = _seconds(event["absolute_start_tick"], tempo, ppq)
                end = _seconds(
                    event["absolute_start_tick"] + event.get("duration_ticks", 0),
                    tempo,
                    ppq,
                )
                string = event.get("string")
                fret = event.get("fret")
                if string is None or fret is None:
                    continue
                # alphaTab strings are one-based in the same low-to-high order
                # used by Feedpak's zero-based lanes. Do not mirror by count.
                string_index = int(string) - 1
                if string_index < 0:
                    continue
                out.append(
                    {
                        "t_gp": start,
                        "s": string_index,
                        "f": int(fret),
                        # Keep an ordinary nominal sustain in the common model.
                        # Expression/tie handling can refine this later.
                        "sus": max(0.0, end - start),
                    }
                )
                onset = {"t_gp": start}
                pitch = {"t_gp": start, "midi": int(event["pitch_midi"])}
                (bass_on if role == "bass" else guitar_on).append(onset)
                (bass_pitch if role == "bass" else guitar_pitch).append(pitch)
            fretted[tid] = {
                "name": name,
                "absolute_tuning_midi": tuning,
                "capo": int(props.get("capo") or 0),
                "notes": out,
                "anchors": [],
            }

        elif role == "drums":
            for event in events:
                midi_number = _drum_midi(event)
                if midi_number is None:
                    raw = (event.get("percussion") or {}).get("articulation_reference")
                    unresolved_drums[raw] += 1
                    continue
                drums.append(
                    {
                        "t_gp": _seconds(event["absolute_start_tick"], tempo, ppq),
                        "p": fc.gm_drum_to_piece(midi_number),
                    }
                )

        elif role in ("piano_left", "piano_right", "piano_combined"):
            hand = {
                "piano_left": "left",
                "piano_right": "right",
                "piano_combined": "combined",
            }[role]
            measures = []
            visits = {
                x["playback_index"]: x
                for x in data["playback"].get("master_bar_visits", [])
            }
            grouped = {}
            for event in events:
                grouped.setdefault(int(event["playback_master_bar_index"]), []).append(event)
            for pi, visit in visits.items():
                evs = grouped.get(pi, [])
                staves = {}
                staff_ids = ["lh"] if hand == "left" else ["rh"] if hand == "right" else ["rh", "lh"]
                for staff_id in staff_ids:
                    staves[staff_id] = {"voices": []}
                voice_beats = {key: [] for key in staves}
                for event in evs:
                    midi = int(event["pitch_midi"])
                    staff_id = "lh" if hand == "left" else "rh" if hand == "right" else ("rh" if midi >= 60 else "lh")
                    start = _seconds(event["absolute_start_tick"], tempo, ppq)
                    end = _seconds(event["absolute_start_tick"] + event.get("duration_ticks", 0), tempo, ppq)
                    voice_beats[staff_id].append({"t_gp": start, "notes": [{"midi": midi, "end_gp": end}]})
                    piano_on.append({"t_gp": start})
                    piano_pitch.append({"t_gp": start, "midi": midi})
                for staff_id, beats_for_staff in voice_beats.items():
                    if beats_for_staff:
                        staves[staff_id]["voices"] = [{"beats": beats_for_staff}]
                measures.append({"idx": pi + 1, "t_gp": _seconds(visit["start_tick"], tempo, ppq), "staves": staves})
            notation[tid] = {"version": 1, "source_track": name, "hand": hand, "measures": measures}

        elif role == "lead_vocal":
            lyrics, pitch_notes = [], []
            for occurrence in occurrences:
                if int(occurrence.get("track_index", -1)) != idx:
                    continue
                start = _seconds(occurrence["absolute_start_tick"], tempo, ppq)
                end = _seconds(occurrence["absolute_start_tick"] + occurrence.get("playback_duration_ticks", 0), tempo, ppq)
                fragments = occurrence.get("lyric_fragments") or []
                if fragments:
                    lyrics.append({"t_gp": start, "end_gp": end, "w": fragments[0]["text"]})
                for note_id in occurrence.get("note_ids") or []:
                    note = notes_by_id.get(note_id) or {}
                    midi = (note.get("pitch") or {}).get("midi")
                    if midi is not None:
                        pitch_notes.append({"t_gp": start, "end_gp": end, "midi": int(midi)})
            if lyrics or pitch_notes:
                gp_vocals = {
                    "lyrics": lyrics,
                    "pitch_notes": pitch_notes,
                    "diagnostics": {
                        "source": "alphatab_schema4",
                        "track": name,
                        "lyric_fragments": len(lyrics),
                        "pitch_notes": len(pitch_notes),
                    },
                }

    if unresolved_drums:
        details = ", ".join(
            f"{key!r}={count}" for key, count in sorted(unresolved_drums.items(), key=lambda item: str(item[0]))
        )
        fc.log(f"WARNING: unresolved alphaTab percussion articulations skipped: {details}", indent=1)

    bars = data["playback"].get("master_bar_visits", [])
    master = data.get("master_bars", [])
    time_sigs, beats = [], []
    last = None
    for visit in bars:
        mb = master[int(visit["source_master_bar_index"])]
        ts = [int(mb["meter"]["numerator"]), int(mb["meter"]["denominator"])]
        t = _seconds(visit["start_tick"], tempo, ppq)
        if ts != last:
            time_sigs.append({"t_gp": t, "ts": ts})
            last = ts
        beats.append({"t_gp": t, "measure": int(visit["playback_index"]) + 1, "ts_num": ts[0]})

    keys, last_key = [], None
    for visit in bars:
        mb = master[int(visit["source_master_bar_index"])]
        acc = int((mb.get("key") or {}).get("signature") or 0)
        minor = str((mb.get("key") or {}).get("type")) in ("1", "Minor")
        key = fc.key_signature_to_name(acc, minor)
        if key and key != last_key:
            keys.append({"t_gp": _seconds(visit["start_tick"], tempo, ppq), "key": key})
            last_key = key

    score_onsets = [x["t_gp"] for x in bass_on + guitar_on + piano_on + drums]
    return {
        "tempo_events": tempo,
        "role_by_name": roles,
        "fretted_tracks": fretted,
        "drum_hits_gp": drums,
        "bass_alignment_events": bass_on,
        "piano_alignment_events": piano_on,
        "guitar_alignment_events": guitar_on,
        "bass_pitch_events": bass_pitch,
        "guitar_pitch_events": guitar_pitch,
        "piano_pitch_events": piano_pitch,
        "notation_tracks": notation,
        "gp_vocals": gp_vocals,
        "key_events_gp": keys,
        "song_timeline_gp": {"time_signatures": time_sigs, "beats": beats},
        "score_content_time": min(score_onsets) if score_onsets else 0.0,
        "track_count": len(tracks),
        "measure_count": len(bars),
        "percussion_diagnostics": {
            "resolved_hits": len(drums),
            "unresolved_articulations": dict(unresolved_drums),
        },
    }
