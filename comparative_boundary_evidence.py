"""v16.7B1.2A.5.2 comparative evidence recovery and triage.

Recovers candidate-relative value from non-strict but well-clocked observations.
Comparative evidence can rank occurrences and triage acquisition; it can never
create controls, anchors, MIDI authority, path changes, or timing changes.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Mapping
import math

VERSION="v16.7B1.2A.5.2-comparative-evidence-recovery-v1"
COMPARATIVE_CONFIDENCE_FLOOR=0.40
STRICT_CONFIDENCE=0.70
SHARP_SIGMA=0.20
ACTIVITY_SIGMA=0.40
STRUCTURAL_SIGMA=0.60
MIN_SCORE_MARGIN=0.20
CORRELATED_BONUS_CAP=0.15

HARD_LIMITATIONS={"detector_failed","invalid_clock","clock_incomplete","wrong_source",
    "catastrophic_contamination","non_monotonic_geometry"}


def _num(v):
    if v is None or isinstance(v,bool):return None
    try:v=float(v)
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) else None


def _sigma(row):
    text=" ".join(str(row.get(k) or "") for k in ("observed_signal_type","evidence_type","detector_method","detector_module")).casefold()
    if any(x in text for x in ("onset","transient","beat","attack")):return SHARP_SIGMA
    if any(x in text for x in ("activity","silence","restart","coverage","offset")):return ACTIVITY_SIGMA
    return STRUCTURAL_SIGMA


def _quality_weight(confidence,limitations):
    c=max(COMPARATIVE_CONFIDENCE_FLOOR,min(1.0,confidence))
    penalty=max(0.45,1.0-0.12*len(limitations))
    return c*penalty


def _kernel(residual,sigma):return math.exp(-(residual*residual)/(2*sigma*sigma))


def observations(bundle,qualification):
    q={str(x.get("evidence_id")):x for x in (qualification or {}).get("qualifications") or [] if x.get("evidence_id")}
    rows=[]
    for e in (bundle or {}).get("evidence") or []:
        eid=e.get("evidence_id");clocks=e.get("clocks") or {};qt=q.get(str(eid)) or {}
        audio=_num(clocks.get("mapped_audio_time"));audio=_num(clocks.get("padded_audio_time")) if audio is None else audio
        chart=_num(clocks.get("chart_time"));confidence=_num(e.get("confidence"));limitations=list(e.get("limitations") or [])
        source=e.get("physical_source_id") or e.get("source");family=e.get("detector_module") or e.get("detector_method") or "unknown_detector"
        hard=[x for x in limitations if str(x).casefold() in HARD_LIMITATIONS]
        clock_complete=bool(qt.get("clock_complete",chart is not None and audio is not None))
        source_known=bool(source);valid_time=audio is not None and chart is not None
        strict=bool(qt.get("strict_eligible"))
        if strict:level="strict"
        elif valid_time and clock_complete and source_known and confidence is not None and confidence>=COMPARATIVE_CONFIDENCE_FLOOR and not hard:level="comparative"
        elif valid_time:level="contextual"
        else:level="excluded"
        rows.append({"evidence_id":eid,"chart_time":chart,"audio_time":audio,"confidence":confidence,"limitations":limitations,
            "hard_exclusions":hard,"source":e.get("source"),"physical_source_id":source,"source_group":source or eid,
            "detector_family":family,"nominal_role":e.get("nominal_role"),"observed_signal_type":e.get("observed_signal_type"),
            "evidence_type":e.get("evidence_type"),"strict_eligible":strict,"evidence_level":level,
            "strict_failure_reasons":list(qt.get("strict_ineligibility_reasons") or qt.get("reasons") or []),
            "comparative_eligible":level in {"strict","comparative"},"timing_authority":strict,
            "sigma_seconds":_sigma(e)})
    return rows


def _request_scores(request,obs):
    candidates=request.get("candidate_occurrence_audio_times") or []
    rows=[]
    for c in candidates:
        target=_num(c.get("audio_time"));score=0.0;groups={};raw=[]
        if target is not None:
            for e in obs:
                if not e["comparative_eligible"] or e["audio_time"] is None:continue
                residual=abs(e["audio_time"]-target)
                if residual>5.0:continue
                value=_quality_weight(e["confidence"],e["limitations"])*_kernel(residual,e["sigma_seconds"])
                item={"evidence_id":e["evidence_id"],"residual_seconds":residual,"weighted_support":value,
                    "evidence_level":e["evidence_level"],"source_group":e["source_group"],"detector_family":e["detector_family"],
                    "confidence":e["confidence"],"strict_failure_reasons":e["strict_failure_reasons"]}
                raw.append(item);key=(e["source_group"],e["detector_family"],request.get("boundary"))
                if key not in groups or value>groups[key]["weighted_support"]:groups[key]=item
        primary=sum(x["weighted_support"] for x in groups.values())
        correlated=max(0.0,sum(x["weighted_support"] for x in raw)-primary)
        score=primary+min(CORRELATED_BONUS_CAP,correlated)
        rows.append({"occurrence_rank":c.get("occurrence_rank"),"candidate_audio_time":target,
            "comparative_score":score,"independent_group_count":len(groups),"raw_evidence_count":len(raw),
            "strict_group_count":sum(x["evidence_level"]=="strict" for x in groups.values()),
            "comparative_group_count":sum(x["evidence_level"]=="comparative" for x in groups.values()),
            "best_supporting_evidence":sorted(groups.values(),key=lambda x:x["weighted_support"],reverse=True)[:8]})
    rows.sort(key=lambda x:x["comparative_score"],reverse=True)
    best=rows[0] if rows else None;runner=rows[1] if len(rows)>1 else None
    margin=None if best is None else best["comparative_score"]-(runner["comparative_score"] if runner else 0.0)
    if best is None or best["comparative_score"]<=0:classification="genuine_evidence_gap"
    elif margin is not None and margin>=MIN_SCORE_MARGIN:classification="comparative_occurrence_supported"
    else:classification="comparative_support_shared"
    return {"boundary_query_id":request.get("boundary_query_id"),"window_id":request.get("window_id"),"role":request.get("role"),
        "boundary":request.get("boundary"),"candidate_scores":rows,"favored_occurrence_rank":None if best is None else best["occurrence_rank"],
        "selected_vs_runner_up_margin":margin,"comparative_classification":classification,
        "detector_execution_recommended":classification in {"genuine_evidence_gap","comparative_support_shared"}}


def _pair(rows):
    by={}
    for r in rows:by.setdefault((r["window_id"],r["role"]),{})[r["boundary"]]=r
    out=[]
    for (wid,role),p in by.items():
        entry=p.get("entry");exit=p.get("exit");er=None if entry is None else entry.get("favored_occurrence_rank");xr=None if exit is None else exit.get("favored_occurrence_rank")
        ec=None if entry is None else entry.get("comparative_classification");xc=None if exit is None else exit.get("comparative_classification")
        if ec=="comparative_occurrence_supported" and xc=="comparative_occurrence_supported" and er==xr:klass="comparative_boundary_pair_supported"
        elif ec=="comparative_occurrence_supported" and xc=="comparative_occurrence_supported" and er!=xr:klass="comparative_sources_conflict"
        elif ec=="comparative_occurrence_supported":klass="comparative_entry_only"
        elif xc=="comparative_occurrence_supported":klass="comparative_exit_only"
        elif ec=="genuine_evidence_gap" and xc=="genuine_evidence_gap":klass="genuine_evidence_gap"
        else:klass="comparative_support_shared"
        out.append({"window_id":wid,"role":role,"entry_favored_occurrence_rank":er,"exit_favored_occurrence_rank":xr,
            "pair_classification":klass,"path_reselection_authorized":False,
            "detector_execution_recommended":klass in {"genuine_evidence_gap","comparative_support_shared","comparative_sources_conflict"}})
    return out


def build_report(*,occurrence_report:Mapping[str,Any],audio_evidence_bundle:Mapping[str,Any],audio_evidence_qualification:Mapping[str,Any]|None=None)->dict[str,Any]:
    obs=observations(audio_evidence_bundle,audio_evidence_qualification);evaluations=[_request_scores(q,obs) for q in occurrence_report.get("targeted_boundary_queries") or []];pairs=_pair(evaluations)
    counts={};pair_counts={}
    for x in evaluations:counts[x["comparative_classification"]]=counts.get(x["comparative_classification"],0)+1
    for x in pairs:pair_counts[x["pair_classification"]]=pair_counts.get(x["pair_classification"],0)+1
    return {"version":VERSION,"diagnostic_only":True,"detectors_executed":False,"audio_decoded":False,
        "candidate_scores_changed":False,"path_reselected":False,"controls_created":False,"anchors_created":False,
        "midi_authority_changed":False,"timing_changes_applied":False,
        "policy":{"comparative_confidence_floor":COMPARATIVE_CONFIDENCE_FLOOR,"strict_confidence":STRICT_CONFIDENCE,
            "minimum_score_margin":MIN_SCORE_MARGIN,"correlated_bonus_cap":CORRELATED_BONUS_CAP,
            "sigma_seconds":{"sharp":SHARP_SIGMA,"activity":ACTIVITY_SIGMA,"structural":STRUCTURAL_SIGMA},
            "comparative_evidence_has_timing_authority":False},
        "evidence_inventory":obs,"boundary_evaluations":evaluations,"pair_evaluations":pairs,
        "triage":{"detector_execution_windows":[{"window_id":x["window_id"],"role":x["role"],"reason":x["pair_classification"]} for x in pairs if x["detector_execution_recommended"]],
            "comparative_path_review_windows":[{"window_id":x["window_id"],"role":x["role"],"classification":x["pair_classification"]} for x in pairs if x["pair_classification"] in {"comparative_boundary_pair_supported","comparative_entry_only","comparative_exit_only"}]},
        "summary":{"evidence_level_counts":{k:sum(x["evidence_level"]==k for x in obs) for k in ("strict","comparative","contextual","excluded")},
            "boundary_classification_counts":counts,"pair_classification_counts":pair_counts,
            "comparative_boundary_pair_count":pair_counts.get("comparative_boundary_pair_supported",0),
            "detector_execution_window_count":sum(x["detector_execution_recommended"] for x in pairs),
            "timing_changes_applied":0}}
