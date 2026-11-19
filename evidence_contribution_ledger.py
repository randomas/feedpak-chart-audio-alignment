"""v16.7Q3 cross-modal evidence contribution ledger.

The ledger explains what chart, audio, and prepared MIDI detected; where each
observation contributed; and why evidence did or did not affect production.
It is diagnostic-only and does not rerun detectors, select controls, alter
warps, materialize proposals, or change authority.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping
import math

VERSION="v16.7B1.2A.2-qualification-schema-normalization-v1"
EVENT_CLUSTER_SECONDS=0.75
CROSS_MODAL_MATCH_SECONDS=1.50


def _num(v):
    if v is None or isinstance(v,bool):return None
    try:v=float(v)
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) else None


def _audio_rows(bundle):
    rows=[]
    for e in (bundle or {}).get("evidence") or []:
        clocks=e.get("clocks") or {};chart=_num(clocks.get("chart_time"))
        if chart is None:continue
        etype=e.get("evidence_type")
        contributed=etype in {"retained_local_audio_hypothesis","retained_structural_occurrence_cluster","accepted_instrument_pattern_consensus"}
        rows.append({"modality":"audio","event_id":e.get("evidence_id"),"chart_time":chart,
            "mapped_audio_time":_num(clocks.get("mapped_audio_time")),"padded_audio_time":_num(clocks.get("padded_audio_time")),
            "source":e.get("source"),"physical_source_id":e.get("physical_source_id") or e.get("source"),
            "nominal_role":e.get("nominal_role"),"observed_signal_type":e.get("observed_signal_type"),
            "detector_module":e.get("detector_module"),"detector_method":e.get("detector_method"),
            "detector_version":e.get("detector_version"),"provenance":deepcopy(e.get("provenance") or {}),
            "source_lineage_id":(e.get("provenance") or {}).get("source_lineage_id") or e.get("physical_source_id") or e.get("source"),
            "evidence_type":etype,"confidence":_num(e.get("confidence")),
            "residual_seconds":_num(e.get("residual_seconds")),"contributed_to_diagnostic_path":contributed,
            "limitations":list(e.get("limitations") or []),"authority":e.get("authority","diagnostic")})
    return rows


def _coverage_rows(coverage_reports):
    rows=[]
    for role,report in dict(coverage_reports or {}).items():
        gp=(report or {}).get("gp") or {};audio=(report or {}).get("audio") or {};comparison=(report or {}).get("comparison") or {}
        chart=_num(gp.get("first_note_time"));padded=_num(audio.get("first_sustained_activity"))
        if chart is None or padded is None:continue
        rows.append({"modality":"audio","event_id":f"coverage-{role}-incipit","chart_time":chart,
            "mapped_audio_time":padded,"padded_audio_time":padded,"source_audio_time":_num(audio.get("source_first_sustained_activity")),
            "source":role,"evidence_type":"track_audio_coverage_incipit","detector_event_type":"sustained_activity_incipit",
            "confidence":_num((report or {}).get("confidence")),"residual_seconds":padded-chart,
            "contributed_to_diagnostic_path":True,"limitations":[comparison.get("classification")] if comparison.get("classification") else [],
            "authority":"diagnostic","provenance":{"origin_report":"decision.guitar_coverage_diagnostic"}})
    return rows


def _midi_rows(report):
    rows=[];fusion=(report or {}).get("fusion_report") or {}
    applied_regions=list(((report or {}).get("application_report") or {}).get("applied_regions") or [])
    applied_ids={str(x.get("source_region_id") or x.get("region_id") or "").split("-trim")[0] for x in applied_regions}
    def applied(chart,region_id=None):
        interval_rows=[x for x in applied_regions if _num(x.get("chart_start")) is not None and _num(x.get("chart_end")) is not None]
        if interval_rows:
            return any(_num(x.get("chart_start"))-1e-6<=chart<=_num(x.get("chart_end"))+1e-6 for x in interval_rows)
        return str(region_id or "").split("-trim")[0] in applied_ids
    for region in fusion.get("regions") or []:
        rid=region.get("region_id");base=str(rid or "").split("-trim")[0]
        for i,c in enumerate(region.get("controls") or []):
            chart=_num(c.get("score_time",c.get("chart_time")))
            if chart is None:continue
            rows.append({"modality":"midi","event_id":f"{rid or 'region'}-control-{i+1}","chart_time":chart,
                "mapped_audio_time":_num(c.get("audio_time",c.get("midi_time"))),"source":"+".join(c.get("roles") or []),
                "roles":list(c.get("roles") or []),"classification":c.get("classification"),"confidence":_num(c.get("confidence")),
                "region_id":rid,"region_chart_start":_num(region.get("chart_start")),"region_chart_end":_num(region.get("chart_end")),
                "contributed_to_midi_region":True,"survived_materialization":applied(chart,rid)})
    return rows


def _audio_production_controls(decision):
    rows=[];linear=(decision or {}).get("continuous_anchor_linear") or {}
    controls=linear.get("controls") or linear.get("selected_controls") or []
    for i,c in enumerate(controls):
        chart=_num(c.get("score_time",c.get("chart_time",c.get("t_gp"))))
        if chart is None:continue
        rows.append({"modality":"audio_production_control","event_id":f"audio-production-control-{i+1}","chart_time":chart,
            "mapped_audio_time":_num(c.get("audio_time",c.get("time"))),"source":c.get("source"),
            "contributed_to_production_warp":bool(linear.get("timing_changes_applied")),"control":deepcopy(c)})
    return rows


def _cluster(rows):
    rows=sorted(rows,key=lambda x:x["chart_time"]);clusters=[]
    for row in rows:
        if clusters and row["chart_time"]-clusters[-1][-1]["chart_time"]<=EVENT_CLUSTER_SECONDS:clusters[-1].append(row)
        else:clusters.append([row])
    return clusters


def _nearest(rows,t,limit=CROSS_MODAL_MATCH_SECONDS):
    candidates=[x for x in rows if abs(x["chart_time"]-t)<=limit]
    return sorted(candidates,key=lambda x:(abs(x["chart_time"]-t),-(x.get("confidence") or 0)))


def _region_support(midi_rows,audio_row):
    t=audio_row["chart_time"]
    return [x for x in midi_rows if x.get("region_chart_start") is not None and x.get("region_chart_end") is not None and
            x["region_chart_start"]-1e-6<=t<=x["region_chart_end"]+1e-6]


def _relation(audio_rows,midi_rows):
    if not audio_rows or not midi_rows:return None
    exact=[]
    for a in audio_rows:
        for m in midi_rows:
            if abs(a["chart_time"]-m["chart_time"])<=0.25:
                aa=_num(a.get("mapped_audio_time") or a.get("padded_audio_time"));ma=_num(m.get("mapped_audio_time"))
                if aa is not None and ma is not None and abs(aa-ma)<=0.25:exact.append((a,m))
    if exact:return "exact_cross_modal_onset"
    if any(_region_support(midi_rows,a) for a in audio_rows):return "same_phrase_cross_modal_support"
    return "cross_modal_structural_corroboration"


def build_ledger(*, audio_evidence_bundle:Mapping[str,Any], midi_section_fit_report=None,
                 alignment_decision=None, production_timing=None, query_result_evaluation=None,
                 coverage_reports=None, audio_evidence_qualification=None,
                 strict_audio_queries=None) -> dict[str,Any]:
    audio=_audio_rows(audio_evidence_bundle)+_coverage_rows(coverage_reports)
    qualification_by_id={str(x.get("evidence_id")):x for x in (audio_evidence_qualification or {}).get("qualifications") or [] if x.get("evidence_id")}
    query_by_evidence={}
    for evaluation in (query_result_evaluation or {}).get("evaluations") or []:
        point=evaluation.get("representative_audio_support") or {}; evidence_id=point.get("evidence_id")
        if evidence_id:
            query_by_evidence.setdefault(str(evidence_id),[]).append({"query_id":evaluation.get("query_id"),
                "classification":evaluation.get("classification"),"bounded_query_status":evaluation.get("bounded_query_status"),
                "bridge_geometry":deepcopy(evaluation.get("bridge_geometry") or {}),
                "future_audio_only_anchor_candidate":bool(evaluation.get("future_audio_only_anchor_candidate"))})
    for row in audio:
        q=qualification_by_id.get(str(row.get("event_id"))) or {}
        strict=bool(q.get("strict_eligible"))
        detector_value=q.get("detector_identity_complete")
        physical_value=q.get("physical_source_identity_complete")
        detector_status=q.get("detector_identity_status")
        physical_status=q.get("physical_source_identity_status")
        if detector_value is None and strict:
            detector_value=True; detector_status="inherited_from_strict_qualification"
        if physical_value is None and strict:
            physical_value=True; physical_status="inherited_from_strict_qualification"
        row.update({"strict_eligible":strict,"qualification":q.get("qualification"),
            "qualification_reasons":list(q.get("strict_ineligibility_reasons") or q.get("reasons") or []),
            "clock_complete":bool(q.get("clock_complete")),
            "detector_identity_complete":bool(detector_value),"detector_identity_status":detector_status or "missing",
            "physical_source_identity_complete":bool(physical_value),"physical_source_identity_status":physical_status or "missing",
            "completeness_inherited_from_strict":bool(q.get("completeness_inherited_from_strict")) or
                strict and (detector_status=="inherited_from_strict_qualification" or physical_status=="inherited_from_strict_qualification"),
            "qualification_schema_missing_fields":[name for name,value in (("detector_identity_complete",q.get("detector_identity_complete")),
                ("physical_source_identity_complete",q.get("physical_source_identity_complete"))) if value is None],
            "query_links":deepcopy(query_by_evidence.get(str(row.get("event_id"))) or [])})
        row["query_localized"]=any(x.get("bounded_query_status")=="localized" for x in row["query_links"])
        row["source_independence_group"]=row.get("physical_source_id") or row.get("source_lineage_id") or row.get("event_id")
    midi=_midi_rows(midi_section_fit_report);prod=_audio_production_controls(alignment_decision)
    seeds=_cluster(audio+midi+prod);events=[]
    materialization=((midi_section_fit_report or {}).get("application_report") or {})
    blocked=str(materialization.get("status") or "").casefold() in {"materialization_blocked","blocked"} or not bool(materialization.get("global_monotonicity_validated",True))
    for idx,group in enumerate(seeds,1):
        center=sum(x["chart_time"] for x in group)/len(group)
        audio_matches=_nearest(audio,center);midi_matches=_nearest(midi,center);prod_matches=_nearest(prod,center)
        # Reconcile an audio event with the MIDI region that contains it, even if
        # the nearest retained control is more than the point-match window away.
        for a in list(audio_matches):
            for m in _region_support(midi,a):
                if m not in midi_matches:midi_matches.append(m)
        if not audio_matches and not midi_matches:continue
        modalities=[]
        if audio_matches:modalities.append("audio")
        if midi_matches:modalities.append("midi")
        chart_detected=True
        audio_used=any(x.get("contributed_to_diagnostic_path") for x in audio_matches)
        midi_region=any(x.get("contributed_to_midi_region") for x in midi_matches)
        midi_applied=any(x.get("survived_materialization") for x in midi_matches)
        production_audio=any(x.get("contributed_to_production_warp") for x in prod_matches)
        if audio_matches and midi_matches:
            status=_relation(audio_matches,midi_matches)
        elif audio_matches:status="audio_only_coverage"
        else:status="midi_only_coverage"
        if midi_applied:effect="midi_materialized_to_production"
        elif production_audio:effect="audio_structural_warp_contributor"
        elif midi_region and blocked:effect="midi_region_contributor_materialization_blocked_elsewhere"
        elif midi_region:effect="midi_region_contributor_not_materialized"
        elif audio_used:effect="audio_diagnostic_contributor"
        else:effect="observed_not_selected"
        reasons=[]
        if blocked and midi_region and not midi_applied:reasons.append("global_midi_materialization_blocked_elsewhere")
        if audio_matches and midi_matches:reasons.append("audio_and_midi_detect_same_chart_neighborhood")
        if status=="same_phrase_cross_modal_support":reasons.append("audio_event_falls_inside_midi_region")
        if status=="exact_cross_modal_onset":reasons.append("chart_and_audio_onset_agree_within_0.25_seconds")
        events.append({"ledger_event_id":f"contribution-event-{idx:04d}","chart_time":round(center,6),
            "chart_interval":[round(min(x["chart_time"] for x in group),6),round(max(x["chart_time"] for x in group),6)],
            "chart_detected":chart_detected,"modalities":modalities,"status":status,"production_effect":effect,
            "audio_contributed_to_structural_analysis":audio_used,"audio_contributed_to_production_warp":production_audio,
            "midi_contributed_to_region":midi_region,"midi_survived_materialization":midi_applied,
            "materialization_blocked_elsewhere":blocked and midi_region and not midi_applied,"reasons":reasons,
            "audio_evidence_ids":[x["event_id"] for x in audio_matches],
            "audio_clocks":[{"event_id":x.get("event_id"),"evidence_id":x.get("event_id"),"chart_time":x.get("chart_time"),
                "mapped_audio_time":x.get("mapped_audio_time"),"padded_audio_time":x.get("padded_audio_time"),
                "source_audio_time":x.get("source_audio_time"),"source":x.get("source"),
                "physical_source_id":x.get("physical_source_id"),"nominal_role":x.get("nominal_role"),
                "observed_signal_type":x.get("observed_signal_type"),"detector_module":x.get("detector_module"),
                "detector_method":x.get("detector_method"),"detector_version":x.get("detector_version"),
                "detector_event_type":x.get("detector_event_type") or x.get("evidence_type"),
                "source_lineage_id":x.get("source_lineage_id"),"source_independence_group":x.get("source_independence_group"),
                "confidence":x.get("confidence"),"limitations":x.get("limitations") or [],"strict_eligible":x.get("strict_eligible",False),
                "qualification":x.get("qualification"),"qualification_reasons":x.get("qualification_reasons") or [],
                "clock_complete":x.get("clock_complete",False),"detector_identity_complete":x.get("detector_identity_complete",False),"detector_identity_status":x.get("detector_identity_status"),
                "physical_source_identity_complete":x.get("physical_source_identity_complete",False),"physical_source_identity_status":x.get("physical_source_identity_status"),
                "completeness_inherited_from_strict":x.get("completeness_inherited_from_strict",False),
                "qualification_schema_missing_fields":x.get("qualification_schema_missing_fields") or [],"query_links":x.get("query_links") or [],
                "query_localized":x.get("query_localized",False),"provenance":deepcopy(x.get("provenance") or {})} for x in audio_matches],
            "midi_clocks":[{"control_id":x["event_id"],"chart_time":x.get("chart_time"),"mapped_audio_time":x.get("mapped_audio_time"),
                "region_id":x.get("region_id"),"region_chart_start":x.get("region_chart_start"),"region_chart_end":x.get("region_chart_end")} for x in midi_matches],
            "midi_control_ids":[x["event_id"] for x in midi_matches],
            "audio_production_control_ids":[x["event_id"] for x in prod_matches],
            "audio_sources":sorted({str(x.get("source")) for x in audio_matches if x.get("source")}),
            "midi_roles":sorted({r for x in midi_matches for r in x.get("roles",[])})})
    counts={};effects={}
    for e in events:
        counts[e["status"]]=counts.get(e["status"],0)+1;effects[e["production_effect"]]=effects.get(e["production_effect"],0)+1
    audio_clock_rows=[x for e in events for x in e.get("audio_clocks") or []]
    strict_clock_rows=[x for x in audio_clock_rows if x.get("strict_eligible")]
    contradictory=[x for x in strict_clock_rows if not x.get("clock_complete") or
        not x.get("detector_identity_complete") or not x.get("physical_source_identity_complete")]
    inherited=[x for x in strict_clock_rows if x.get("completeness_inherited_from_strict")]
    schema_missing=[x for x in audio_clock_rows if x.get("qualification_schema_missing_fields")]
    return {"version":VERSION,"diagnostic_only":True,"detectors_executed":False,"anchors_created":False,
        "timing_changes_applied":False,"midi_authority_changed":False,"cluster_seconds":EVENT_CLUSTER_SECONDS,
        "cross_modal_match_seconds":CROSS_MODAL_MATCH_SECONDS,"selected_production_timing":deepcopy(production_timing or {}),
        "events":events,"summary":{"event_count":len(events),"status_counts":counts,"production_effect_counts":effects,
            "cross_modal_event_count":sum(counts.get(x,0) for x in ("cross_modal_structural_corroboration","exact_cross_modal_onset","same_phrase_cross_modal_support")),
            "exact_cross_modal_onset_count":counts.get("exact_cross_modal_onset",0),
            "same_phrase_cross_modal_support_count":counts.get("same_phrase_cross_modal_support",0),
            "audio_only_event_count":counts.get("audio_only_coverage",0),"midi_only_event_count":counts.get("midi_only_coverage",0),
            "materialization_blocked_elsewhere_count":sum(e["materialization_blocked_elsewhere"] for e in events),
            "strict_audio_clock_count":len(strict_clock_rows),
            "explicitly_complete_strict_audio_clock_count":sum(not x.get("completeness_inherited_from_strict") for x in strict_clock_rows),
            "inherited_complete_strict_audio_clock_count":len(inherited),
            "contradictory_strict_audio_clock_count":len(contradictory),
            "qualification_schema_missing_audio_clock_count":len(schema_missing)},
        "notes":["Failure to qualify as a future audio-only anchor does not erase prior evidence contribution.",
                 "Local MIDI contribution is reported separately from global materialization success.",
                 "Events are explanatory chart-time neighborhoods, not new timing controls.",
                 "Strict qualification component states are normalized; inherited completeness is explicit and contradictory states are counted."]}
