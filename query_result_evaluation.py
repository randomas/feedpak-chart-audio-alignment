"""v16.7Q2 diagnostic evaluation of Q1 strict-audio query results.

Q2 evaluates support type, physical-source independence, and bridge geometry to
nearby fused MIDI controls.  It remains diagnostic-only and cannot create an
anchor, change timing, materialize a proposal, or alter MIDI authority.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping
import math

VERSION="v16.7Q2.2-query-localization-v1"
SAFE_STRETCH=(0.9,1.1)
MIN_DISTINCT_CHART_SPAN_SECONDS=0.25
COINCIDENT_AUDIO_TOLERANCE_SECONDS=0.10


def _num(v):
    if v is None or isinstance(v,bool):return None
    try:v=float(v)
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) else None


def _midi_controls(midi_report: Mapping[str,Any] | None) -> list[dict[str,Any]]:
    rows=[];fusion=(midi_report or {}).get("fusion_report") or {}
    for region in fusion.get("regions") or []:
        for control in region.get("controls") or []:
            chart=_num(control.get("score_time",control.get("chart_time")))
            audio=_num(control.get("audio_time",control.get("midi_time")))
            if chart is not None and audio is not None:
                rows.append({"chart_time":chart,"audio_time":audio,"region_id":region.get("region_id"),
                    "roles":list(control.get("roles") or []),"classification":control.get("classification")})
    rows.sort(key=lambda x:(x["chart_time"],x["audio_time"]))
    groups=[]
    for row in rows:
        if groups and abs(row["chart_time"]-groups[-1][-1]["chart_time"])<MIN_DISTINCT_CHART_SPAN_SECONDS:
            groups[-1].append(row)
        else:groups.append([row])
    deduped=[]
    for group in groups:
        # Prefer the strongest cross-role control, then the latest item for stable behavior.
        selected=max(group,key=lambda x:(len(x.get("roles") or []),x["chart_time"],x["audio_time"]))
        item=dict(selected);item["collapsed_control_count"]=len(group)
        if len(group)>1:item["collapsed_controls"]=[dict(x) for x in group]
        deduped.append(item)
    return deduped


def _representative_audio(answer):
    support=answer.get("selected_support") or []
    usable=[]
    for row in support:
        clocks=row.get("clocks") or {};chart=_num(clocks.get("chart_time"))
        audio=_num(clocks.get("padded_audio_time"))
        if audio is None:audio=_num(clocks.get("mapped_audio_time"))
        if chart is not None and audio is not None:
            usable.append((float(row.get("confidence") or 0),-abs(float(row.get("residual_seconds") or 0)),row,chart,audio))
    if not usable:return None
    _,_,row,chart,audio=max(usable,key=lambda x:(x[0],x[1]))
    return {"evidence_id":row.get("evidence_id"),"source":row.get("source"),
        "physical_source_id":row.get("physical_source_id"),"chart_time":chart,"audio_time":audio,
        "confidence":row.get("confidence"),"residual_seconds":row.get("residual_seconds")}


def _bridge_geometry(point,controls,query_type=None):
    if point is None:return {"status":"unavailable","reason":"strict_support_has_no_complete_audio_point"}
    chart=point["chart_time"];audio=point["audio_time"]
    coincident=[x for x in controls if abs(x["chart_time"]-chart)<MIN_DISTINCT_CHART_SPAN_SECONDS]
    if coincident:
        control=min(coincident,key=lambda x:abs(x["chart_time"]-chart))
        audio_delta=audio-control["audio_time"]
        agrees=abs(audio_delta)<=COINCIDENT_AUDIO_TOLERANCE_SECONDS
        return {"status":"coincident_with_midi_control","reason":"direct_boundary_corroboration" if agrees else "coincident_audio_midi_disagreement",
            "minimum_distinct_chart_span_seconds":MIN_DISTINCT_CHART_SPAN_SECONDS,
            "coincident_audio_tolerance_seconds":COINCIDENT_AUDIO_TOLERANCE_SECONDS,
            "coincident_control":control,"chart_delta_seconds":chart-control["chart_time"],
            "audio_delta_seconds":audio_delta,"direct_boundary_corroboration":agrees,"edges":[]}
    before=max((x for x in controls if x["chart_time"]<=chart-MIN_DISTINCT_CHART_SPAN_SECONDS),key=lambda x:x["chart_time"],default=None)
    after=min((x for x in controls if x["chart_time"]>=chart+MIN_DISTINCT_CHART_SPAN_SECONDS),key=lambda x:x["chart_time"],default=None)
    def edge(a,b,label):
        if a is None or b is None:return None
        dc=b["chart_time"]-a["chart_time"];da=b["audio_time"]-a["audio_time"]
        if dc<MIN_DISTINCT_CHART_SPAN_SECONDS:
            return {"edge":label,"status":"insufficient_geometric_separation","chart_delta_seconds":dc,
                "audio_delta_seconds":da,"minimum_distinct_chart_span_seconds":MIN_DISTINCT_CHART_SPAN_SECONDS}
        stretch=da/dc
        return {"edge":label,"status":"evaluated","chart_delta_seconds":dc,"audio_delta_seconds":da,
            "stretch":stretch,"monotonic":da>0,"within_safe_stretch":da>0 and SAFE_STRETCH[0]<=stretch<=SAFE_STRETCH[1]}
    p={"chart_time":chart,"audio_time":audio}
    edges=[x for x in (edge(before,p,"previous_midi_to_audio"),edge(p,after,"audio_to_next_midi")) if x]
    if not edges:return {"status":"insufficient_geometric_separation","reason":"no_distinct_neighboring_midi_control",
        "previous_control":before,"next_control":after,"edges":[],"minimum_distinct_chart_span_seconds":MIN_DISTINCT_CHART_SPAN_SECONDS}
    evaluated=[x for x in edges if x.get("status")=="evaluated"]
    if not evaluated:return {"status":"insufficient_geometric_separation","reason":"all_neighboring_controls_too_close",
        "previous_control":before,"next_control":after,"edges":edges,"minimum_distinct_chart_span_seconds":MIN_DISTINCT_CHART_SPAN_SECONDS}
    monotonic=all(x.get("monotonic") for x in evaluated);safe=monotonic and all(x.get("within_safe_stretch") for x in evaluated)
    one_sided=len(evaluated)==1
    if not monotonic:status="conflict"
    elif not safe:status="out_of_policy"
    elif one_sided and query_type=="opening_root_uncertain" and after is not None:status="safe_one_sided_opening_bridge"
    elif one_sided and query_type=="terminal_mismatch" and before is not None:status="safe_one_sided_terminal_bridge"
    elif one_sided:status="insufficient_geometric_separation"
    else:status="safe_two_sided_gap_bridge"
    return {"status":status,"safe_stretch_range":list(SAFE_STRETCH),
        "minimum_distinct_chart_span_seconds":MIN_DISTINCT_CHART_SPAN_SECONDS,
        "previous_control":before,"next_control":after,"edges":edges,"monotonic":monotonic,
        "within_safe_stretch":safe,"one_sided":one_sided}


def evaluate_query(row: Mapping[str,Any], controls: list[dict[str,Any]]) -> dict[str,Any]:
    answer=row.get("answer") or {};status=answer.get("status");point=_representative_audio(answer)
    query_type=row.get("query_type")
    geometry=_bridge_geometry(point,controls,query_type) if status=="strict_audio_support_available" else {"status":"not_evaluated","reason":"strict_support_unavailable"}
    geometry_status=geometry.get("status")
    if status=="insufficient_query_localization":classification="insufficient_query_localization"
    elif status!="strict_audio_support_available":classification="insufficient_strict_audio_evidence"
    elif geometry_status=="conflict":classification="audio_midi_conflict"
    elif geometry_status=="out_of_policy":classification="evidence_too_isolated_to_use"
    elif geometry_status=="insufficient_geometric_separation":classification="insufficient_geometric_separation"
    elif geometry_status=="coincident_with_midi_control":
        classification="direct_boundary_corroboration" if geometry.get("direct_boundary_corroboration") else "audio_midi_conflict"
    elif query_type in {"materialization_gap_boundary","opening_root_uncertain","terminal_mismatch"}:
        classification="audio_supported_midi_void"
    elif query_type in {"rejected_region_boundary","persistent_phase_transition","competing_occurrence_boundary"}:
        classification="existing_midi_boundary_corroboration"
    else:classification="diagnostic_support_only"
    safe_bridge=geometry_status in {"safe_two_sided_gap_bridge","safe_one_sided_opening_bridge","safe_one_sided_terminal_bridge"}
    future_candidate=(classification=="audio_supported_midi_void" and safe_bridge and
        int(answer.get("independent_physical_source_count") or 0)>=int(row.get("minimum_independent_physical_sources") or 1))
    return {"query_id":row.get("query_id"),"query_type":query_type,"chart_start":row.get("chart_start"),
        "chart_end":row.get("chart_end"),"reason":row.get("reason"),
        "bounded_query_status":row.get("bounded_query_status"),"q1_status":status,
        "shared_evidence_usage":deepcopy(row.get("shared_evidence_usage") or {}),
        "shared_evidence_query_count":row.get("shared_evidence_query_count",0),
        "classification":classification,"independent_physical_source_count":answer.get("independent_physical_source_count",0),
        "alternative_count":answer.get("alternative_count",0),"representative_audio_support":point,
        "bridge_geometry":geometry,"future_audio_only_anchor_candidate":future_candidate,
        "anchors_created":False,"timing_changes_applied":False,"midi_authority_changed":False}


def build_report(*, strict_query_report: Mapping[str,Any], midi_section_fit_report=None) -> dict[str,Any]:
    controls=_midi_controls(midi_section_fit_report);evaluations=[evaluate_query(x,controls) for x in strict_query_report.get("queries") or []]
    classes={};geometry={}
    for row in evaluations:
        classes[row["classification"]]=classes.get(row["classification"],0)+1
        state=row["bridge_geometry"].get("status");geometry[state]=geometry.get(state,0)+1
    return {"version":VERSION,"diagnostic_only":True,"detectors_executed":False,"anchors_created":False,
        "timing_changes_applied":False,"midi_authority_changed":False,"safe_stretch_range":list(SAFE_STRETCH),
        "minimum_distinct_chart_span_seconds":MIN_DISTINCT_CHART_SPAN_SECONDS,
        "coincident_audio_tolerance_seconds":COINCIDENT_AUDIO_TOLERANCE_SECONDS,
        "midi_control_count":len(controls),"evaluations":evaluations,
        "summary":{"evaluation_count":len(evaluations),"classification_counts":classes,
            "bridge_geometry_counts":geometry,"future_audio_only_anchor_candidate_count":sum(x["future_audio_only_anchor_candidate"] for x in evaluations)},
        "notes":["Future anchor candidates are diagnostic labels only.",
                 "Trusted MIDI controls are never moved by Q2.",
                 "Out-of-policy or non-monotonic bridge geometry cannot become a future anchor candidate."]}
