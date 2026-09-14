#!/usr/bin/env python3
"""Diagnose alphaTab score-clock, tempo placement, sparse sections and alignment drift.

Usage:
  python diagnose_score_clock.py SCORE.alphatab.json INTERMEDIATE.json ALIGNMENT_REPORT.json --out-prefix score_clock

Outputs:
  <prefix>.summary.json
  <prefix>.bars.csv
  <prefix>.tempo.csv
  <prefix>.checkpoints.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows):
    rows = list(rows)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_tempo_events(events):
    out = []
    for event in sorted(events, key=lambda x: (float(x.get("tick", 0)), float(x.get("tempo", 0)))):
        tick = int(event["tick"])
        tempo = float(event["tempo"])
        if out and out[-1]["tick"] == tick:
            out[-1] = {"tick": tick, "tempo": tempo}
        elif not out or out[-1]["tempo"] != tempo:
            out.append({"tick": tick, "tempo": tempo})
    return out


def seconds_at_tick(tick, tempo_events, ppq):
    events = normalize_tempo_events(tempo_events)
    if not events:
        return tick * 60.0 / (120.0 * ppq)
    total = 0.0
    previous_tick = events[0]["tick"]
    tempo = events[0]["tempo"]
    if previous_tick > 0:
        total += previous_tick * 60.0 / (tempo * ppq)
    for event in events[1:]:
        if tick <= event["tick"]:
            break
        total += (event["tick"] - previous_tick) * 60.0 / (tempo * ppq)
        previous_tick = event["tick"]
        tempo = event["tempo"]
    total += max(0, tick - previous_tick) * 60.0 / (tempo * ppq)
    return total


def active_tempo(tick, events, default=120.0):
    normalized = normalize_tempo_events(events)
    ticks = [x["tick"] for x in normalized]
    index = bisect_right(ticks, tick) - 1
    return normalized[index]["tempo"] if index >= 0 else default


def expected_tempo_events(score):
    visits = score["playback"]["master_bar_visits"]
    bars = score["master_bars"]
    result = []
    current = float(score.get("metadata", {}).get("tempo") or 120.0)
    for visit in visits:
        source = int(visit["source_master_bar_index"])
        bar = bars[source]
        automations = bar.get("tempo_automations") or ([] if not bar.get("tempo_automation") else [bar["tempo_automation"]])
        for auto in automations:
            if auto and auto.get("value") is not None:
                ratio = float(auto.get("ratio_position") or 0.0)
                if ratio > 1.0:
                    ratio /= 100.0
                tick = int(round(float(visit["start_tick"]) + ratio * float(visit.get("duration_ticks") or 0)))
                current = float(auto["value"])
                result.append({
                    "tick": tick,
                    "tempo": current,
                    "playback_bar": int(visit["playback_index"]) + 1,
                    "source_bar": source + 1,
                    "section": (bar.get("section") or {}).get("text") or "",
                })
    if not result or result[0]["tick"] != 0:
        result.insert(0, {"tick": 0, "tempo": current, "playback_bar": 1, "source_bar": 1, "section": ""})
    dedup = []
    for item in sorted(result, key=lambda x: x["tick"]):
        if dedup and item["tick"] == dedup[-1]["tick"]:
            dedup[-1] = item
        elif not dedup or item["tempo"] != dedup[-1]["tempo"]:
            dedup.append(item)
    return dedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("score_json")
    parser.add_argument("intermediate_json")
    parser.add_argument("alignment_report_json")
    parser.add_argument("--out-prefix", default="score_clock_diagnostic")
    args = parser.parse_args()

    score = load(args.score_json)
    intermediate = load(args.intermediate_json)
    report = load(args.alignment_report_json)
    ppq = int(score.get("source", {}).get("ppq") or 960)
    visits = score["playback"]["master_bar_visits"]
    bars = score["master_bars"]
    emitted = normalize_tempo_events(score["playback"].get("tempo_events", []))
    expected = expected_tempo_events(score)

    # Index performed notes and lyric-bearing beats by playback bar.
    note_counts = defaultdict(Counter)
    for note in score["playback"].get("playback_notes", []):
        note_counts[int(note.get("playback_master_bar_index", -1))][int(note.get("track_index", -1))] += 1
    lyric_counts = Counter()
    for beat in score["playback"].get("beat_occurrences", []):
        lyric_counts[int(beat.get("playback_master_bar_index", -1))] += len(beat.get("lyric_fragments") or [])
    track_names = {int(t["index"]): t.get("name") or f"track_{t['index']}" for t in score.get("tracks", [])}

    decision = report.get("decision", {})
    anchor = decision.get("opening_anchor", {})
    scale = float(decision.get("scale") or 1.0)
    score_anchor = float(anchor.get("score_anchor") or 0.0)
    audio_anchor = float(anchor.get("audio_anchor") or 0.0)

    bar_rows = []
    for visit in visits:
        pindex = int(visit["playback_index"])
        source = int(visit["source_master_bar_index"])
        tick = int(visit["start_tick"])
        bar = bars[source]
        counts = note_counts[pindex]
        authored_seconds = seconds_at_tick(tick, expected, ppq)
        emitted_seconds = seconds_at_tick(tick, emitted, ppq)
        nominal_from_report_clock = emitted_seconds
        mapped = audio_anchor + scale * (nominal_from_report_clock - score_anchor)
        row = {
            "playback_bar": pindex + 1,
            "source_bar": source + 1,
            "occurrence": int(visit.get("occurrence") or 1),
            "start_tick": tick,
            "duration_ticks": int(visit.get("duration_ticks") or 0),
            "section": (bar.get("section") or {}).get("text") or "",
            "authored_tempo": active_tempo(tick, expected, score.get("metadata", {}).get("tempo") or 120),
            "emitted_tempo": active_tempo(tick, emitted, score.get("metadata", {}).get("tempo") or 120),
            "authored_nominal_seconds": round(authored_seconds, 6),
            "emitted_nominal_seconds": round(emitted_seconds, 6),
            "clock_difference_seconds": round(emitted_seconds - authored_seconds, 6),
            "globally_mapped_seconds": round(mapped, 6),
            "lyric_fragments": lyric_counts[pindex],
            "total_notes": sum(counts.values()),
        }
        for track_index, count in counts.items():
            row[f"notes_{track_names.get(track_index, track_index)}"] = count
        bar_rows.append(row)

    tempo_rows = []
    for item in expected:
        matching = [x for x in emitted if x["tempo"] == item["tempo"]]
        nearest = min(matching, key=lambda x: abs(x["tick"] - item["tick"])) if matching else None
        actual_tick = nearest["tick"] if nearest else None
        delta = actual_tick - item["tick"] if actual_tick is not None else None
        tempo_rows.append({
            **item,
            "expected_tick": item["tick"],
            "emitted_matching_tick": actual_tick,
            "tick_error": delta,
            "bar_start_tick": visits[item["playback_bar"] - 1]["start_tick"],
            "double_offset_signature": bool(actual_tick is not None and item["tick"] != 0 and actual_tick != item["tick"] and actual_tick == item["tick"] + visits[item["playback_bar"] - 1]["start_tick"]),
        })

    checkpoints = []
    for cp in decision.get("checkpoint_diagnostics", {}).get("checkpoints", []):
        checkpoints.append({
            "measure": cp.get("measure"),
            "nominal_time": cp.get("nominal_time"),
            "predicted_time": cp.get("predicted_time"),
            "matched_events": cp.get("matched_events"),
            "median_correction_seconds": cp.get("median_correction_seconds"),
            "mad_seconds": cp.get("mad_seconds"),
            "accepted": cp.get("accepted"),
        })

    suspicious_gaps = []
    for previous, current in zip(bar_rows, bar_rows[1:]):
        if current["authored_nominal_seconds"] - previous["authored_nominal_seconds"] > 4.0:
            suspicious_gaps.append({"after_bar": previous["playback_bar"], "before_bar": current["playback_bar"], "seconds": round(current["authored_nominal_seconds"] - previous["authored_nominal_seconds"], 6)})

    summary = {
        "ppq": ppq,
        "written_bars": len(bars),
        "playback_bars": len(visits),
        "expected_tempo_changes": expected,
        "emitted_tempo_changes": emitted,
        "tempo_placement_errors": [x for x in tempo_rows if x["tick_error"] not in (0, None)],
        "double_offset_suspected": any(x["double_offset_signature"] for x in tempo_rows),
        "final_authored_seconds": seconds_at_tick(int(visits[-1]["end_tick"]), expected, ppq),
        "final_emitted_seconds": seconds_at_tick(int(visits[-1]["end_tick"]), emitted, ppq),
        "global_scale": scale,
        "opening_score_anchor": score_anchor,
        "opening_audio_anchor": audio_anchor,
        "instrument_scales": {x.get("source"): x.get("measured_scale") for x in decision.get("instrument_scale", {}).get("candidates", [])},
        "suspicious_bar_gaps": suspicious_gaps,
        "notes": "Compare authored_nominal_seconds with emitted_nominal_seconds. A tempo event whose emitted tick equals expected tick plus the bar start tick indicates an absolute tick was added twice.",
    }

    prefix = Path(args.out_prefix)
    Path(str(prefix) + ".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(str(prefix) + ".bars.csv", bar_rows)
    write_csv(str(prefix) + ".tempo.csv", tempo_rows)
    write_csv(str(prefix) + ".checkpoints.csv", checkpoints)

    print(f"Wrote {prefix}.summary.json")
    print(f"Wrote {prefix}.bars.csv")
    print(f"Wrote {prefix}.tempo.csv")
    print(f"Wrote {prefix}.checkpoints.csv")
    print(f"Expected tempo changes: {[(x['tick'], x['tempo']) for x in expected]}")
    print(f"Emitted tempo changes: {[(x['tick'], x['tempo']) for x in emitted]}")
    print(f"Double-offset suspected: {summary['double_offset_suspected']}")
    print(f"Final authored clock: {summary['final_authored_seconds']:.3f}s")
    print(f"Final emitted clock: {summary['final_emitted_seconds']:.3f}s")


if __name__ == "__main__":
    main()
