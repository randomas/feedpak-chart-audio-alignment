"""v16.7B1.2A.5 targeted bounded acquisition from existing detector products.

The module plans tight/context requests for unresolved occurrence boundaries,
merges overlapping requests, scans existing detector observations without
whole-song recomputation, and caches deterministic request results. It remains
fully diagnostic and never changes a path or timing map.
"""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping
import json, math

VERSION="v16.7B1.2A.5-targeted-boundary-evidence-acquisition-v1"
TIGHT_RADIUS_SECONDS=1.0
CONTEXT_RADIUS_SECONDS=5.0
EXACT_SECONDS=0.10
COMPATIBLE_SECONDS=0.50
MIN_CONFIDENCE=0.70


def _num(v):
    if v is None or isinstance(v,bool):return None
    try:v=float(v)
    except (TypeError,ValueError):return None
    return v if math.isfinite(v) else None


def _family(row):
    return row.get("detector_module") or row.get("detector_method") or row.get("detector_event_type") or row.get("adapter_name") or "unknown_detector"


def _source(row):
    return row.get("physical_source_id") or row.get("source") or "unknown_source"


def _bundle_observations(bundle,qualification):
    q={str(x.get("evidence_id")):x for x in (qualification or {}).get("qualifications") or [] if x.get("evidence_id")}
    rows=[]
    for e in (bundle or {}).get("evidence") or []:
        eid=e.get("evidence_id"); clocks=e.get("clocks") or {}; qual=q.get(str(eid)) or {}
        chart=_num(clocks.get("chart_time")); audio=_num(clocks.get("mapped_audio_time"))
        if audio is None:audio=_num(clocks.get("padded_audio_time"))
        confidence=_num(e.get("confidence")); limitations=list(e.get("limitations") or [])
        if chart is None or audio is None:continue
        rows.append({"evidence_id":eid,"chart_time":chart,"audio_time":audio,"confidence":confidence,
            "source":e.get("source"),"physical_source_id":e.get("physical_source_id") or e.get("source"),
            "source_independence_group":e.get("physical_source_id") or e.get("source") or eid,
            "detector_family":_family(e),"nominal_role":e.get("nominal_role"),
            "observed_signal_type":e.get("observed_signal_type"),"limitations":limitations,
            "strict_eligible":bool(qual.get("strict_eligible")),"qualification":qual.get("qualification"),
            "provenance":deepcopy(e.get("provenance") or {})})
    return rows


def _request_key(request):
    payload={k:request.get(k) for k in ("boundary","role","candidate_audio_time","tight_radius_seconds","context_radius_seconds","detector_families")}
    return sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def plan_requests(occurrence_report:Mapping[str,Any])->list[dict[str,Any]]:
    requests=[]
    for query in occurrence_report.get("targeted_boundary_queries") or []:
        for candidate in query.get("candidate_occurrence_audio_times") or []:
            target=_num(candidate.get("audio_time"))
            if target is None:continue
            row={"request_id":f"target-{len(requests)+1}","boundary_query_id":query.get("boundary_query_id"),
                "window_id":query.get("window_id"),"role":query.get("role"),"boundary":query.get("boundary"),
                "occurrence_rank":candidate.get("occurrence_rank"),"candidate_audio_time":target,
                "tight_radius_seconds":TIGHT_RADIUS_SECONDS,"context_radius_seconds":CONTEXT_RADIUS_SECONDS,
                "detector_families":["continuous_anchors","structural_search","beatmap_evidence","instrument_evidence","coverage"],
                "detectors_executed":False,"acquisition_mode":"existing_detector_products_targeted_scan"}
            row["cache_key"]=_request_key(row);requests.append(row)
    return requests


def merge_requests(requests):
    rows=sorted(requests,key=lambda x:(x["role"],x["candidate_audio_time"]));merged=[]
    for r in rows:
        start=r["candidate_audio_time"]-r["context_radius_seconds"];end=r["candidate_audio_time"]+r["context_radius_seconds"]
        if merged and merged[-1]["role"]==r["role"] and start<=merged[-1]["audio_end"]:
            m=merged[-1];m["audio_end"]=max(m["audio_end"],end);m["request_ids"].append(r["request_id"]);m["cache_keys"].append(r["cache_key"])
        else:merged.append({"merged_request_id":f"merged-{len(merged)+1}","role":r["role"],"audio_start":start,"audio_end":end,
            "request_ids":[r["request_id"]],"cache_keys":[r["cache_key"]],"detectors_executed":False})
    return merged


def _evaluate(request,observations):
    target=request["candidate_audio_time"];tight=[];context=[]
    for obs in observations:
        residual=abs(obs["audio_time"]-target);row={**obs,"residual_seconds":residual}
        if residual<=request["context_radius_seconds"]:context.append(row)
        if residual<=request["tight_radius_seconds"]:tight.append(row)
    strict=[x for x in tight if x["strict_eligible"] and x["confidence"] is not None and x["confidence"]>=MIN_CONFIDENCE and not x["limitations"]]
    groups={}
    for x in strict:
        key=(x["source_independence_group"],x["detector_family"],request["boundary"])
        if key not in groups or x["residual_seconds"]<groups[key]["residual_seconds"]:groups[key]=x
    independent=sorted(groups.values(),key=lambda x:x["residual_seconds"])
    best=independent[0]["residual_seconds"] if independent else None
    if best is None:status="targeted_evidence_still_insufficient"
    elif best<=EXACT_SECONDS:status="targeted_exact_boundary_support_found"
    elif best<=COMPATIBLE_SECONDS:status="targeted_compatible_boundary_support_found"
    else:status="targeted_context_only_support_found"
    role=str(request.get("role") or "").casefold()
    return {**request,"acquisition_status":status,"best_residual_seconds":best,
        "raw_strict_observation_count":len(strict),"independent_support_count":len(independent),
        "independent_physical_source_count":len({x["source_independence_group"] for x in independent}),
        "independent_detector_family_count":len({x["detector_family"] for x in independent}),
        "same_role_support_count":sum(str(x.get("nominal_role") or "").casefold()==role for x in independent),
        "cross_role_support_count":sum(str(x.get("nominal_role") or "").casefold()!=role for x in independent),
        "supporting_evidence_ids":[x["evidence_id"] for x in independent],"tight_observation_count":len(tight),
        "context_observation_count":len(context),"results_cached":False}


def _pair(evaluations):
    by={}
    for e in evaluations:by.setdefault((e["window_id"],e["role"],e["occurrence_rank"]),{})[e["boundary"]]=e
    rows=[]
    for (wid,role,rank),parts in by.items():
        entry=parts.get("entry");exit=parts.get("exit")
        eok=entry and entry["acquisition_status"] in {"targeted_exact_boundary_support_found","targeted_compatible_boundary_support_found"}
        xok=exit and exit["acquisition_status"] in {"targeted_exact_boundary_support_found","targeted_compatible_boundary_support_found"}
        if eok and xok:klass="targeted_boundary_pair_supported"
        elif eok:klass="targeted_entry_only_support"
        elif xok:klass="targeted_exit_only_support"
        else:klass="targeted_evidence_still_insufficient"
        shared=bool(eok and xok and set(entry["supporting_evidence_ids"])&set(exit["supporting_evidence_ids"]))
        rows.append({"window_id":wid,"role":role,"occurrence_rank":rank,"targeted_pair_classification":klass,
            "entry_request_id":None if entry is None else entry["request_id"],"exit_request_id":None if exit is None else exit["request_id"],
            "shared_evidence_across_boundaries":shared,"path_reselection_authorized":False})
    return rows


def build_report(*,occurrence_report:Mapping[str,Any],audio_evidence_bundle:Mapping[str,Any],
                 audio_evidence_qualification:Mapping[str,Any]|None=None)->dict[str,Any]:
    requests=plan_requests(occurrence_report);observations=_bundle_observations(audio_evidence_bundle,audio_evidence_qualification)
    evaluations=[_evaluate(r,observations) for r in requests];pairs=_pair(evaluations);counts={}
    for x in evaluations:counts[x["acquisition_status"]]=counts.get(x["acquisition_status"],0)+1
    pair_counts={}
    for x in pairs:pair_counts[x["targeted_pair_classification"]]=pair_counts.get(x["targeted_pair_classification"],0)+1
    return {"version":VERSION,"diagnostic_only":True,"detectors_executed":False,"whole_song_recomputation":False,
        "candidate_scores_changed":False,"path_reselected":False,"controls_created":False,"anchors_created":False,
        "midi_authority_changed":False,"timing_changes_applied":False,
        "policy":{"tight_radius_seconds":TIGHT_RADIUS_SECONDS,"context_radius_seconds":CONTEXT_RADIUS_SECONDS,
            "exact_seconds":EXACT_SECONDS,"compatible_seconds":COMPATIBLE_SECONDS,"minimum_confidence":MIN_CONFIDENCE,
            "independent_vote_key":"physical_source_group + detector_family + boundary"},
        "requests":requests,"merged_requests":merge_requests(requests),"evaluations":evaluations,"pair_evaluations":pairs,
        "summary":{"request_count":len(requests),"merged_request_count":len(merge_requests(requests)),
            "acquisition_status_counts":counts,"pair_classification_counts":pair_counts,
            "targeted_boundary_pair_count":pair_counts.get("targeted_boundary_pair_supported",0),
            "cacheable_request_count":len(requests),"detectors_executed":False,"timing_changes_applied":0}}
