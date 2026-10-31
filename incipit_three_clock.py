"""Three-clock opening acquisition: chart sequence, transcription MIDI, role audio."""
from __future__ import annotations
from typing import Mapping

MAX_AUDIO_CHART_DISAGREEMENT=0.75
MAX_TRANSCRIPTION_LAG=4.0
MIN_OPENING_CONTROLS=2


def first_chart_onset(chart_events_by_role,role):
    values=[]
    for row in chart_events_by_role.get(role) or []:
        for key in ("t_gp","time","score_time","start_time","t"):
            if isinstance(row,Mapping) and row.get(key) is not None:
                values.append(float(row[key]));break
    return min(values) if values else None


def independent_candidate_count(region):
    ids=set()
    for control in region.get("controls") or []:
        for source in control.get("provenance") or []:
            role=source.get("role");window=source.get("window_id");streak=source.get("streak_id")
            if window is not None:ids.add((role,window))
            elif streak is not None:ids.add((role,streak))
    return len(ids)


def reconcile_opening(fusion_report,chart_events_by_role,role_audio_starts):
    """Audio establishes when; chart-MIDI establishes phrase identity."""
    regions=sorted(fusion_report.get("regions") or [],key=lambda r:float(r.get("chart_start",0)))
    diagnostics={"status":"no_opening_candidate","clock_policy":"audio_when_midi_identity",
                 "candidates":[]}
    for region in regions:
        roles=sorted({role for c in region.get("controls") or [] for role in c.get("roles") or []})
        if len(roles)!=1:continue
        role=roles[0];chart_start=first_chart_onset(chart_events_by_role,role);audio_start=role_audio_starts.get(role)
        if chart_start is None or audio_start is None:continue
        midi_start=float(region.get("midi_start",0));candidate_count=independent_candidate_count(region)
        audio_chart_delta=float(audio_start)-float(chart_start)
        transcription_lag=midi_start-float(audio_start)
        row={"region_id":region.get("region_id"),"role":role,"chart_start":chart_start,
             "audio_start":float(audio_start),"midi_start":midi_start,
             "audio_chart_delta_seconds":audio_chart_delta,
             "transcription_lag_seconds":transcription_lag,
             "independent_candidate_count":candidate_count}
        diagnostics["candidates"].append(row)
        near_opening=float(region.get("chart_start",999999))<=chart_start+3.0
        valid=(near_opening and len(region.get("controls") or [])>=MIN_OPENING_CONTROLS and
               candidate_count>=2 and abs(audio_chart_delta)<=MAX_AUDIO_CHART_DISAGREEMENT and
               0.0<=transcription_lag<=MAX_TRANSCRIPTION_LAG)
        if not valid:continue
        # Correct transcription clock only for this opening region. The symbolic
        # occurrence is retained, while actual audio supplies the root time.
        for control in region.get("controls") or []:
            control["raw_transcription_midi_time"]=float(control["audio_time"])
            control["audio_time"]=float(control["audio_time"])-transcription_lag
            control["incipit_audio_verified"]=True
        region["raw_transcription_midi_start"]=midi_start
        region["raw_transcription_midi_end"]=float(region.get("midi_end",midi_start))
        region["midi_start"]=midi_start-transcription_lag
        region["midi_end"]=float(region["midi_end"])-transcription_lag
        region["acquisition_root"]={"validated":True,"authority":"three_clock_role_incipit",
            "role":role,"score_time":chart_start,"audio_time":float(audio_start),
            "offset_seconds":audio_chart_delta,"transcription_lag_seconds":transcription_lag,
            "independent_candidate_count":candidate_count}
        diagnostics.update({"status":"validated","selected_region_id":region.get("region_id"),
            "role":role,"root_offset_seconds":audio_chart_delta,
            "transcription_lag_seconds":transcription_lag})
        break
    fusion_report["three_clock_incipit"]=diagnostics
    return fusion_report
