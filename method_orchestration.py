"""v16.8-lite method orchestration and global manual-anchor invariants."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import math

VERSION="v16.8-lite.2-reporting-corrections-v1"
PREPARED_MIDI_MODES={"off","auto","diagnostic","force-trusted"}
COMPATIBLE_AUDIO_METHODS={"dtw-checkpoint","checkpoint-linear","checkpoint-dtw-diagnostic",
    "checkpoint-dtw-selective","continuous-anchor-linear","continuous-anchor-dtw",
    "continuous-anchor-structural","bar-landmark-regularized"}
ANCHOR_TOLERANCE_SECONDS=.001


def validate_prepared_midi_mode(mode,alignment_mode):
    mode=str(mode or "auto").casefold()
    if mode not in PREPARED_MIDI_MODES:raise ValueError(f"invalid prepared MIDI mode: {mode}")
    compatible=alignment_mode in COMPATIBLE_AUDIO_METHODS
    if mode in {"auto","force-trusted"} and not compatible:
        return mode,False,"selected_method_not_compatible_with_continuous_midi"
    return mode,compatible,None


def exact_anchors(manual_report):
    rows=((manual_report or {}).get("quality") or {}).get("anchors") or []
    return [x for x in rows if str(x.get("mode","exact")).casefold()=="exact"]


def verify_manual_anchors(warp_fn,manual_report,method_id,tolerance=ANCHOR_TOLERANCE_SECONDS):
    anchors=exact_anchors(manual_report);errors=[]
    for row in anchors:
        chart=float(row["nominal_time"]);target=float(row["padded_audio_time"]);mapped=float(warp_fn(chart));error=abs(mapped-target)
        errors.append({"anchor_id":row.get("id"),"chart_time":chart,"target_audio_time":target,
            "mapped_audio_time":mapped,"error_ms":error*1000.0,"satisfied":error<=tolerance})
    maximum=max((x["error_ms"] for x in errors),default=0.0)
    status="not_supplied" if not anchors else ("satisfied" if all(x["satisfied"] for x in errors) else "materialization_failed")
    return {"manual_anchor_count":len(anchors),"manual_anchor_status":status,
        "maximum_manual_anchor_error_ms":maximum,"manual_anchor_conflicts":[x for x in errors if not x["satisfied"]],
        "verification_tolerance_ms":tolerance*1000.0}


def check_simple_method_compatibility(base_warp,manual_report,method_id,tolerance=ANCHOR_TOLERANCE_SECONDS):
    anchors=exact_anchors(manual_report)
    if not anchors:return {"status":"not_supplied","reason":None}
    if method_id not in {"nominal","offset"}:return {"status":"compatible","reason":None}
    errors=[abs(float(base_warp(float(x["nominal_time"])))-float(x["padded_audio_time"])) for x in anchors]
    if max(errors,default=0.0)<=tolerance:return {"status":"satisfied","reason":None}
    reason="nominal_timing_cannot_satisfy_anchor" if method_id=="nominal" else "anchors_require_nonuniform_or_different_offset"
    return {"status":"incompatible_method","reason":reason,"maximum_error_ms":max(errors)*1000.0}


def _midi_comparison(transcription_report,midi_section_report):
    continuous=((transcription_report or {}).get("second_pass") or transcription_report or {})
    csummary=continuous.get("summary") or {};ssummary=(midi_section_report or {}).get("summary") or {}
    c_available=bool(continuous) and (csummary.get("selected_monotonic_candidates",0)>0 or csummary.get("second_pass_selected_monotonic_candidates",0)>0)
    s_available=bool(midi_section_report) and bool((midi_section_report.get("fusion_report") or {}).get("regions"))
    if c_available and s_available:status="both_midi_methods_available"
    elif c_available:status="continuous_midi_only"
    elif s_available:status="section_midi_only"
    else:status="no_trusted_midi"
    return {"status":status,"continuous_midi_available":c_available,"section_midi_available":s_available,
        "continuous_midi_summary":csummary,"midi_section_summary":ssummary,
        "timing_map_averaged":False,"diagnostic_only":True}


def _num(value):
    if value is None or isinstance(value,bool): return None
    try: value=float(value)
    except (TypeError,ValueError): return None
    return value if math.isfinite(value) else None


def _add_review(rows,seen,*,kind,chart_time=None,chart_end=None,severity="review",message,source=None):
    t=_num(chart_time);end=_num(chart_end)
    key=(kind,None if t is None else round(t,3),None if end is None else round(end,3),message)
    if key in seen:return
    seen.add(key);rows.append({"type":kind,"chart_time":t,"chart_end":end,"severity":severity,
        "message":message,"source":source})


def _component_versions(**reports):
    return {name:(report or {}).get("version") for name,report in reports.items() if report is not None}


def _opening_status(decision,strict_query_report,occurrence_report):
    explicit=(decision or {}).get("opening_anchor") or {}
    if explicit.get("status"):return explicit["status"]
    for query in (strict_query_report or {}).get("queries") or []:
        if query.get("query_type")=="opening_root_uncertain":
            status=((query.get("answer") or {}).get("status") or query.get("status"))
            return "supported" if status=="strict_audio_support_available" else "uncertain"
    for row in (occurrence_report or {}).get("qualifications") or []:
        if (_num(row.get("chart_start")) or 999)>12:continue
        outcome=row.get("occurrence_identity_support")
        if outcome=="phrase_and_occurrence_supported":return "occurrence_supported"
        if outcome in {"phrase_supported_occurrence_ambiguous","phrase_supported_alternative_occurrence_favored"}:return "uncertain"
    return "not_reported"


def _measure_stretch_groups(warnings):
    rows=[]
    for raw in warnings or []:
        if not isinstance(raw,Mapping) or raw.get("kind")!="measure_stretch":continue
        try:measure=int(raw.get("measure"));ratio=float(raw.get("ratio"))
        except (TypeError,ValueError):continue
        if not math.isfinite(ratio):continue
        direction="low" if ratio<0.9 else "high" if ratio>1.1 else None
        if direction:rows.append({"measure":measure,"ratio":ratio,"direction":direction})
    rows.sort(key=lambda x:x["measure"]);groups=[]
    for row in rows:
        if groups and row["direction"]==groups[-1]["direction"] and row["measure"]<=groups[-1]["end_measure"]+1:
            groups[-1]["end_measure"]=row["measure"];groups[-1]["ratios"].append(row["ratio"])
        else:groups.append({"start_measure":row["measure"],"end_measure":row["measure"],"direction":row["direction"],"ratios":[row["ratio"]]})
    for g in groups:
        vals=g.pop("ratios");g["minimum_ratio"]=min(vals);g["maximum_ratio"]=max(vals);g["warning_count"]=len(vals)
    return groups


def _alignment_report_warnings(alignment_report):
    warnings=(alignment_report or {}).get("warnings") or [];phase=[];other=[];reviews=[];seen=set()
    groups=_measure_stretch_groups(warnings)
    for g in groups:
        label=(f"Measures {g['start_measure']}-{g['end_measure']}" if g["start_measure"]!=g["end_measure"] else f"Measure {g['start_measure']}")
        message=f"{label}: local stretch {g['minimum_ratio']:.4f}..{g['maximum_ratio']:.4f} outside 0.9-1.1"
        phase.append({"type":"excessive_local_stretch",**g,"message":message})
        _add_review(reviews,seen,kind="measure_stretch",severity="warning",message=message,source="alignment_report")
    for raw in warnings:
        if not isinstance(raw,Mapping) or raw.get("kind")=="measure_stretch":continue
        item=dict(raw);other.append(item);message=item.get("message") or item.get("reason") or item.get("kind") or "alignment warning"
        _add_review(reviews,seen,kind=str(item.get("kind") or "alignment_warning"),chart_time=item.get("chart_time"),
            chart_end=item.get("chart_end"),severity=str(item.get("level") or "warning"),message=message,source="alignment_report")
    return phase,other,reviews,groups

def _aggregate_warnings(*,decision,midi_section_fit_report,occurrence_report,comparative_report,strict_query_report):
    occurrence=[];phase=[];reviews=[];seen=set();manual=False
    # Repeated-occurrence uncertainty and comparative conflicts.
    for row in (occurrence_report or {}).get("qualifications") or []:
        cls=row.get("occurrence_identity_support")
        if cls in {"phrase_supported_occurrence_ambiguous","phrase_supported_alternative_occurrence_favored"}:
            warning={"window_id":row.get("window_id"),"role":row.get("role"),"classification":cls,
                "chart_start":row.get("chart_start"),"chart_end":row.get("chart_end")}
            occurrence.append(warning);manual=True
            _add_review(reviews,seen,kind="occurrence",chart_time=row.get("chart_start"),chart_end=row.get("chart_end"),
                severity="manual_review",message=cls,source="audio_qualified_occurrence_resolution")
    for row in (comparative_report or {}).get("pair_evaluations") or []:
        cls=row.get("pair_classification")
        if cls in {"comparative_sources_conflict","comparative_entry_only","comparative_exit_only"}:
            occurrence.append({"window_id":row.get("window_id"),"role":row.get("role"),"classification":cls})
    # MIDI section materialization, phase migration and unsupported regions.
    m=midi_section_fit_report or {};app=m.get("application_report") or {};summary=m.get("summary") or {}
    status=app.get("status") or summary.get("materialization_status")
    if status in {"materialization_blocked","blocked"}:
        phase.append({"type":"midi_materialization_blocked","status":status});manual=True
    for key in ("rejected_regions","rejected_blocks"):
        for row in app.get(key) or []:
            _add_review(reviews,seen,kind="midi_rejected_region",chart_time=row.get("chart_start"),chart_end=row.get("chart_end"),
                message="MIDI section rejected",source="midi_section_fit")
    for row in ((m.get("fusion_report") or {}).get("rejected_regions") or []):
        _add_review(reviews,seen,kind="midi_rejected_region",chart_time=row.get("chart_start"),chart_end=row.get("chart_end"),
            message="Fused MIDI region rejected",source="midi_section_fit")
    for row in (m.get("phase_review_points") or summary.get("phase_review_points") or []):
        msg=row.get("reason") or row.get("classification") or "cumulative phase movement"
        phase.append({"type":"phase_review","chart_time":row.get("chart_time") or row.get("chart_start"),"message":msg})
        _add_review(reviews,seen,kind="phase",chart_time=row.get("chart_time") or row.get("chart_start"),
            chart_end=row.get("chart_end"),severity="review",message=msg,source="midi_section_fit")
    for row in (app.get("unsupported_intervals") or (m.get("fusion_report") or {}).get("unsupported_intervals") or []):
        _add_review(reviews,seen,kind="unsupported_interval",chart_time=row.get("chart_start") or row.get("start"),
            chart_end=row.get("chart_end") or row.get("end"),message="Timing interval is not MIDI-validated",source="midi_section_fit")
    # Existing quality/measure-stretch diagnostics in the alignment decision.
    for key in ("measure_stretch_warnings","alignment_warnings","warnings"):
        for raw in (decision or {}).get(key) or []:
            row=raw if isinstance(raw,dict) else {"message":str(raw)}
            msg=row.get("message") or row.get("reason") or row.get("classification") or key
            phase.append({"type":key,"chart_time":row.get("chart_time"),"message":msg})
            _add_review(reviews,seen,kind="stretch_or_alignment",chart_time=row.get("chart_time"),
                chart_end=row.get("chart_end"),message=msg,source="alignment_report")
    for query in (strict_query_report or {}).get("queries") or []:
        answer=query.get("answer") or {};status=answer.get("status") or query.get("status")
        if status in {"audio_midi_conflict","strict_audio_midi_conflict"}:
            _add_review(reviews,seen,kind="audio_midi_conflict",chart_time=query.get("chart_start"),chart_end=query.get("chart_end"),
                message="Strict audio and MIDI evidence conflict",source="strict_audio_queries")
    return occurrence,phase,reviews,manual,status


def build_result_envelope(*,alignment_mode,decision,prepared_midi_mode,manual_anchor_verification,
                          alignment_report=None,transcription_report=None,midi_section_fit_report=None,output_path=None,
                          occurrence_report=None,comparative_report=None,strict_query_report=None,
                          targeted_boundary_report=None,hybrid_report=None):
    timing=str((decision or {}).get("timing_output") or (decision or {}).get("mode") or alignment_mode)
    primary="midi" if "midi-section" in timing else "audio"
    method_id=("continuous-midi-landmark" if "trusted-midi" in timing or "forced-midi" in timing else
               "midi-section-fit-permissive" if "midi-section-fit-permissive" in timing else
               "midi-section-fit-conservative" if "midi-section-fit" in timing else alignment_mode)
    comparison=_midi_comparison(transcription_report,midi_section_fit_report)
    anchor_status=manual_anchor_verification.get("manual_anchor_status")
    occurrence,phase,reviews,manual_recommended,section_status=_aggregate_warnings(
        decision=decision,midi_section_fit_report=midi_section_fit_report,occurrence_report=occurrence_report,
        comparative_report=comparative_report,strict_query_report=strict_query_report)
    report_phase,alignment_warnings,report_reviews,stretch_groups=_alignment_report_warnings(alignment_report)
    phase.extend(report_phase);reviews.extend(report_reviews)
    opening=_opening_status(decision,strict_query_report,occurrence_report)
    if opening=="uncertain":
        manual_recommended=True
        _add_review(reviews,set((x["type"],x.get("chart_time"),x.get("chart_end"),x["message"]) for x in reviews),
            kind="opening",chart_time=0,chart_end=12,severity="manual_review",
            message="Opening root remains uncertain",source="orchestration")
    if anchor_status not in {"not_supplied","satisfied"}:status="failed"
    elif section_status in {"materialization_blocked","blocked"}:status="partial"
    elif occurrence or phase or reviews or opening in {"uncertain","not_reported"}:status="complete_with_warnings"
    else:status="complete"
    comparison["component_versions"]=_component_versions(
        transcription_investigator=transcription_report,midi_section_fit=midi_section_fit_report,
        occurrence_resolution=occurrence_report,comparative_evidence=comparative_report,
        targeted_boundary=targeted_boundary_report,strict_audio_queries=strict_query_report,
        hybrid_control=hybrid_report)
    return {"schema_version":3,"orchestration_version":"v16.8-lite.2-reporting-corrections-v1","method_id":method_id,
        "method_version":timing,"status":status,"primary_evidence":primary,
        "prepared_midi_mode":prepared_midi_mode,"continuous_midi_status":comparison["status"],
        "midi_section_status":"selected" if method_id.startswith("midi-section-fit") else ("available" if comparison["section_midi_available"] else "not_available"),
        **manual_anchor_verification,"opening_root_status":opening,
        "materialization_status":"failed" if status=="failed" else ("partial" if status=="partial" else "complete"),
        "occurrence_warnings":occurrence,"phase_warnings":phase,"alignment_warnings":alignment_warnings,
        "measure_stretch_warning_ranges":stretch_groups,"warning_count":len(occurrence)+len(phase)+len(alignment_warnings),
        "manual_anchor_recommended":bool(manual_recommended and anchor_status=="not_supplied"),
        "listening_review_points":reviews,"feedpak_path":output_path,"midi_method_comparison":comparison}
