"""v16.7B1 diagnostic hybrid MIDI-audio control proposals.

MIDI section geometry remains the baseline. Existing reconciled audio evidence
may confirm it or propose a bounded local correction, but this module never
materializes a warp or changes MIDI authority.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping
import math

VERSION = "v16.7B1.2A-audio-qualified-occurrence-resolution-v1"
MIN_CORRECTION_SECONDS = 0.08
MAX_CORRECTION_SECONDS = 0.25
MIN_AUDIO_CONFIDENCE = 0.85
MAX_INFLUENCE_SECONDS = 2.0


def _num(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _regions(report):
    fusion = (report or {}).get("fusion_report") or {}
    applied = ((report or {}).get("application_report") or {}).get("applied_regions") or []
    rows = []
    for index, region in enumerate(fusion.get("regions") or [], 1):
        start, end = _num(region.get("chart_start")), _num(region.get("chart_end"))
        if start is None or end is None or end <= start:
            continue
        region_id = region.get("region_id") or f"region-{index}"
        survived = any(
            _num(x.get("chart_start")) is not None and _num(x.get("chart_end")) is not None
            and max(start, _num(x.get("chart_start"))) <= min(end, _num(x.get("chart_end")))
            for x in applied
        )
        rows.append({"region_id": region_id, "chart_start": start, "chart_end": end,
                     "controls": deepcopy(region.get("controls") or []), "survived": survived})
    return rows


def _clean_audio_clocks(event):
    clocks = []
    for row in event.get("audio_clocks") or []:
        chart = _num(row.get("chart_time"))
        mapped = _num(row.get("mapped_audio_time"))
        if mapped is None:
            mapped = _num(row.get("padded_audio_time"))
        confidence = _num(row.get("confidence"))
        limitations = list(row.get("limitations") or [])
        if chart is None or mapped is None:
            continue
        clocks.append({"event_id": row.get("event_id"), "chart_time": chart,
                       "mapped_audio_time": mapped, "confidence": confidence,
                       "limitations": limitations, "source": row.get("source")})
    return clocks


def _candidate(event, region):
    if event.get("status") != "exact_cross_modal_onset":
        return None
    candidates = []
    midi_clocks = event.get("midi_clocks") or []
    for audio in _clean_audio_clocks(event):
        if (audio["limitations"] or audio["confidence"] is None or
                audio["confidence"] < MIN_AUDIO_CONFIDENCE or
                not audio.get("event_id") or not audio.get("source")):
            continue
        nearest = None
        for midi in midi_clocks:
            score = _num(midi.get("chart_time")); target = _num(midi.get("mapped_audio_time"))
            if score is None or target is None:
                continue
            distance = abs(score - audio["chart_time"])
            if nearest is None or distance < nearest[0]:
                nearest = (distance, score, target, midi.get("event_id"))
        if nearest is None or nearest[0] > MAX_CORRECTION_SECONDS:
            continue
        delta = audio["mapped_audio_time"] - nearest[2]
        if MIN_CORRECTION_SECONDS <= abs(delta) <= MAX_CORRECTION_SECONDS:
            candidates.append((abs(delta), audio, nearest, delta))
    if not candidates:
        return None
    _, audio, nearest, delta = sorted(candidates, key=lambda x: (-x[0], x[1]["chart_time"]))[0]
    chart_time = min(region["chart_end"], max(region["chart_start"], audio["chart_time"]))
    return {"chart_time": chart_time, "proposed_audio_time": audio["mapped_audio_time"],
            "baseline_midi_audio_time": nearest[2], "delta_seconds": delta,
            "audio_event_id": audio["event_id"], "midi_event_id": nearest[3],
            "audio_source": audio["source"], "audio_confidence": audio["confidence"],
            "influence_chart_start": max(region["chart_start"], chart_time - MAX_INFLUENCE_SECONDS),
            "influence_chart_end": min(region["chart_end"], chart_time + MAX_INFLUENCE_SECONDS)}


def build_proposal(*, evidence_contribution_ledger: Mapping[str, Any], midi_section_fit_report=None,
                   selected_production_timing=None) -> dict[str, Any]:
    events = list((evidence_contribution_ledger or {}).get("events") or [])
    proposals = []
    covered_event_ids = set()
    for region in _regions(midi_section_fit_report):
        local = [e for e in events if region["chart_start"] - 1e-6 <= (_num(e.get("chart_time")) or -1e99) <= region["chart_end"] + 1e-6]
        for event in local:
            covered_event_ids.add(event.get("event_id"))
        exact = [e for e in local if e.get("status") == "exact_cross_modal_onset"]
        phrase = [e for e in local if e.get("status") == "same_phrase_cross_modal_support"]
        correction = next((c for c in (_candidate(e, region) for e in exact) if c), None)
        if correction:
            classification = "audio_corrected_midi_geometry"
            decision = "propose_bounded_local_control"
            reasons = ["exact cross-modal onset", "audio-to-MIDI delta is inside the 0.08-0.25 second correction band",
                       "control influence is clipped to two seconds and to the MIDI region"]
        elif exact or phrase:
            classification = "audio_confirmed_midi_geometry"
            decision = "retain_midi_geometry"
            reasons = ["audio corroborates the MIDI region", "no qualified material correction is required"]
        else:
            classification = "midi_only_retained_geometry"
            decision = "retain_midi_geometry"
            reasons = ["no qualified audio control exists in this MIDI region", "missing audio evidence is timing-neutral"]
        proposals.append({"proposal_id": f"hybrid-{len(proposals)+1}", "region_id": region["region_id"],
            "chart_start": region["chart_start"], "chart_end": region["chart_end"],
            "classification": classification, "decision": decision,
            "midi_survived_materialization": region["survived"], "midi_control_count": len(region["controls"]),
            "supporting_event_ids": [e.get("event_id") for e in local], "proposed_control": correction,
            "downstream_warp_authorized": False, "reasons": reasons})
    for event in events:
        if event.get("event_id") in covered_event_ids or event.get("status") != "audio_only_coverage":
            continue
        proposals.append({"proposal_id": f"hybrid-{len(proposals)+1}", "region_id": None,
            "chart_start": _num(event.get("chart_time")), "chart_end": _num(event.get("chart_time")),
            "classification": "unresolved_audio_only_evidence", "decision": "no_intervention",
            "midi_survived_materialization": False, "midi_control_count": 0,
            "supporting_event_ids": [event.get("event_id")], "proposed_control": None,
            "downstream_warp_authorized": False,
            "reasons": ["audio evidence has no containing MIDI region", "isolated audio evidence cannot authorize downstream warping"]})
    counts = {}
    for row in proposals:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    return {"version": VERSION, "diagnostic_only": True, "proposal_only": True,
        "timing_changes_applied": False, "anchors_created": False, "midi_authority_changed": False,
        "selected_production_timing": deepcopy(selected_production_timing or {}),
        "policy": {"midi_is_primary_geometry": True, "min_audio_confidence": MIN_AUDIO_CONFIDENCE,
            "correction_delta_seconds": [MIN_CORRECTION_SECONDS, MAX_CORRECTION_SECONDS],
            "max_local_influence_seconds": MAX_INFLUENCE_SECONDS,
            "unsupported_gaps_are_timing_neutral": True, "downstream_extrapolation_allowed": False},
        "proposals": proposals,
        "summary": {"proposal_count": len(proposals), "classification_counts": counts,
            "bounded_local_control_count": sum(p["proposed_control"] is not None for p in proposals),
            "timing_changes_applied": 0}}
