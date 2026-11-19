"""v16.7Q1 strict audio queries at MIDI/alignment uncertainty boundaries.

Queries are generated from existing diagnostic reports and answered only from
R4-qualified observations.  This module never executes detectors, creates
anchors, changes timing, materializes a proposal, or changes MIDI authority.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping, Sequence
import math
import audio_evidence_qualification as aeq

VERSION="v16.7Q2.2-query-localization-v1"
POINT_QUERY_RADIUS_SECONDS=0.5
QUERY_TYPES={"opening_root_uncertain","competing_occurrence_boundary","persistent_phase_transition",
    "materialization_gap_boundary","rejected_region_boundary","terminal_mismatch"}


def _num(v):
    if v is None or isinstance(v,bool):return None
    try:v=float(v)
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) else None


def _add(rows,seen,query_type,start=None,end=None,reason=None,origin=None,details=None):
    start=_num(start);end=_num(end)
    if start is not None and end is not None and end<start:start,end=end,start
    key=(query_type,None if start is None else round(start,6),None if end is None else round(end,6),str(reason))
    if key in seen:return
    seen.add(key);rows.append({"query_id":f"strict-audio-query-{len(rows)+1:04d}","query_type":query_type,
        "chart_start":start,"chart_end":end,"reason":reason,"origin_report":origin,
        "origin_details":deepcopy(details or {}),"requested_qualifications":["near_baseline_support","compatible_displaced_candidate"],
        "minimum_independent_physical_sources":1})


def generate_queries(*, midi_section_fit_report: Mapping[str,Any] | None=None,
                     alignment_report: Mapping[str,Any] | None=None,
                     chart_start: float | None=None, chart_end: float | None=None) -> list[dict[str,Any]]:
    midi=dict(midi_section_fit_report or {});alignment=dict(alignment_report or {});rows=[];seen=set()
    fusion=midi.get("fusion_report") or {};application=midi.get("application_report") or {}
    phase=midi.get("phase_migration_qualification") or fusion.get("phase_migration_qualification") or {}
    incipit=phase.get("three_clock_incipit") or fusion.get("three_clock_incipit") or {}
    incipit_status=str(incipit.get("status") or incipit.get("acquisition_status") or "")
    if incipit and any(token in incipit_status.casefold() for token in ("unvalidated","unrooted","uncertain","missing","unavailable","unresolved")):
        horizon=_num(incipit.get("opening_horizon_seconds")) or 8.0
        _add(rows,seen,"opening_root_uncertain",chart_start or 0.0,horizon,"incipit_not_validated","midi_section_fit.phase_migration",incipit)

    # Selected-path unsupported windows and fused unsupported points.
    path=midi.get("path_report") or {}
    for role,report in (path.get("roles") or {}).items():
        for gap in report.get("unsupported_windows") or []:
            window=gap.get("window") or gap
            _add(rows,seen,"materialization_gap_boundary",window.get("chart_start"),window.get("chart_end"),
                 gap.get("reason") or "unsupported_midi_window","midi_section_fit.path_report",{"role":role,**dict(gap)})
    for point in fusion.get("unsupported_points") or []:
        t=_num(point.get("chart_time"))
        start=None if t is None else t-POINT_QUERY_RADIUS_SECONDS
        end=None if t is None else t+POINT_QUERY_RADIUS_SECONDS
        _add(rows,seen,"materialization_gap_boundary",start,end,point.get("reason") or "no_role_support",
             "midi_section_fit.fusion_report",point)

    for region in application.get("rejected_regions") or []:
        _add(rows,seen,"rejected_region_boundary",region.get("chart_start"),region.get("chart_end"),
             region.get("reason") or region.get("rejection_reason") or "materialization_rejected",
             "midi_section_fit.application_report",region)

    diagnostics=application.get("production_phase_diagnostics") or fusion.get("production_phase_diagnostics") or {}
    for item in diagnostics.get("review_points") or []:
        kind=str(item.get("kind") or "")
        if "phase" in kind or "drift" in kind:
            center=_num(item.get("chart_time"));start=item.get("chart_start");end=item.get("chart_end")
            if start is None and center is not None:start=center-POINT_QUERY_RADIUS_SECONDS
            if end is None and center is not None:end=center+POINT_QUERY_RADIUS_SECONDS
            _add(rows,seen,"persistent_phase_transition",start,end,kind,"midi_section_fit.production_phase_diagnostics",item)
    for item in phase.get("review_points") or []:
        kind=str(item.get("kind") or "phase_migration_review")
        q="competing_occurrence_boundary" if "occurrence" in kind else "persistent_phase_transition"
        center=_num(item.get("chart_time"));start=item.get("chart_start");end=item.get("chart_end")
        if start is None and center is not None:start=center-POINT_QUERY_RADIUS_SECONDS
        if end is None and center is not None:end=center+POINT_QUERY_RADIUS_SECONDS
        _add(rows,seen,q,start,end,kind,"midi_section_fit.phase_migration_qualification",item)

    terminal=diagnostics.get("terminal") or {}
    tail=_num(terminal.get("unsupported_terminal_tail_seconds"))
    if tail is not None and tail>0.5:
        end=_num(chart_end) or _num(terminal.get("terminal_chart_time"))
        start=None if end is None else end-tail
        _add(rows,seen,"terminal_mismatch",start,end,"unsupported_terminal_tail",
             "midi_section_fit.production_phase_diagnostics",terminal)

    # Alignment warnings are query origins only; they do not become evidence.
    for warning in alignment.get("warnings") or []:
        kind=str(warning.get("kind") or "")
        if kind in {"repaired_timeline","unresolved_timeline"}:
            center=_num(warning.get("chart_time"));start=warning.get("chart_start");end=warning.get("chart_end")
            if start is None and center is not None:start=center-POINT_QUERY_RADIUS_SECONDS
            if end is None and center is not None:end=center+POINT_QUERY_RADIUS_SECONDS
            _add(rows,seen,"persistent_phase_transition",start,end,kind,"alignment_report",warning)
    for row in rows:
        if row.get("chart_start") is None and row.get("chart_end") is None:
            row["bounded_query_status"]="unbounded_origin_missing_chart_location"
        else:row["bounded_query_status"]="bounded"
    return rows


def answer_queries(qualification_report: Mapping[str,Any], queries: Sequence[Mapping[str,Any]]) -> list[dict[str,Any]]:
    answers=[];shared_usage={}
    for query in queries:
        if query.get("bounded_query_status")!="bounded":
            answer={"version":VERSION,"query_type":query["query_type"],"chart_start":query.get("chart_start"),
                "chart_end":query.get("chart_end"),"status":"insufficient_query_localization","diagnostic_only":True,
                "timing_changes_applied":False,"midi_authority_changed":False,"independent_physical_source_count":0,
                "selected_support":[],"alternative_count":0,
                "notes":["The originating diagnostic supplied no bounded chart location; global evidence was not selected."]}
        else:
            answer=aeq.answer_strict_query(qualification_report,query["query_type"],query.get("chart_start"),query.get("chart_end"))
        row={**deepcopy(dict(query)),"answer":answer};answers.append(row)
        for evidence in answer.get("selected_support") or []:
            eid=evidence.get("evidence_id")
            if eid:shared_usage.setdefault(eid,[]).append(query.get("query_id"))
    for row in answers:
        selected=[x.get("evidence_id") for x in (row.get("answer") or {}).get("selected_support") or [] if x.get("evidence_id")]
        usage={eid:shared_usage[eid] for eid in selected if len(shared_usage.get(eid,[]))>1}
        row["shared_evidence_usage"]=usage
        row["shared_evidence_query_count"]=len({qid for ids in usage.values() for qid in ids})
    return answers


def build_report(*, qualification_report: Mapping[str,Any], midi_section_fit_report=None,
                 alignment_report=None, chart_start=None, chart_end=None) -> dict[str,Any]:
    queries=generate_queries(midi_section_fit_report=midi_section_fit_report,alignment_report=alignment_report,
                             chart_start=chart_start,chart_end=chart_end)
    answers=answer_queries(qualification_report,queries);status_counts={}
    for row in answers:
        status=row["answer"]["status"];status_counts[status]=status_counts.get(status,0)+1
    return {"version":VERSION,"diagnostic_only":True,"detectors_executed":False,"anchors_created":False,
        "timing_changes_applied":False,"midi_authority_changed":False,"queries":answers,
        "summary":{"query_count":len(answers),"answer_status_counts":status_counts,
            "strict_support_query_count":status_counts.get("strict_audio_support_available",0),
            "insufficient_query_count":status_counts.get("insufficient_strict_audio_evidence",0),
            "insufficient_localization_count":status_counts.get("insufficient_query_localization",0),
            "bounded_query_count":sum(x.get("bounded_query_status")=="bounded" for x in answers),
            "unbounded_query_count":sum(x.get("bounded_query_status")!="bounded" for x in answers),
            "shared_evidence_id_count":len({eid for x in answers for eid in x.get("shared_evidence_usage",{})})},
        "notes":["Queries originate in existing MIDI/alignment uncertainty diagnostics.",
                 "Audio answers cannot move trusted MIDI or create timing controls in Q1."]}
