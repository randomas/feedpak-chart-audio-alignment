"""v16.7B1.2A.3 independent occurrence evidence and boundary coverage.

Diagnostic-only. Phrase identity, occurrence identity, independent support, and
regional consistency remain separate. No score, path, control, anchor, or timing
is changed.
"""
from __future__ import annotations
from typing import Any, Mapping
import math

VERSION="v16.7B1.2A.4-existing-evidence-boundary-query-evaluation-v1"
EXACT_SECONDS=0.10
COMPATIBLE_SECONDS=0.35
MIN_CONFIDENCE=0.70
DISCRIMINATIVE_MARGIN_SECONDS=0.15
BOUNDARY_CONTEXT_SECONDS=1.0


def _num(value):
    if value is None or isinstance(value,bool): return None
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) else None


def _selected_by_window(report):
    selected={}
    for role,row in (((report or {}).get("path_report") or {}).get("roles") or {}).items():
        for streak in row.get("streaks") or []:
            for candidate in streak.get("selected_candidates") or streak.get("candidates") or []:
                if candidate.get("window_id"):selected[(str(role),str(candidate["window_id"]))]=candidate
    return selected


def _family(clock):
    return clock.get("detector_module") or clock.get("detector_method") or clock.get("detector_event_type") or "unknown_detector"


def _audio_observations(ledger):
    rows=[];seen=set()
    for event in (ledger or {}).get("events") or []:
        event_chart=_num(event.get("chart_time"))
        for clock in event.get("audio_clocks") or []:
            chart=_num(clock.get("chart_time")); chart=event_chart if chart is None else chart
            audio=_num(clock.get("mapped_audio_time")); audio=_num(clock.get("padded_audio_time")) if audio is None else audio
            confidence=_num(clock.get("confidence")); source=clock.get("physical_source_id") or clock.get("source")
            event_id=clock.get("event_id") or clock.get("evidence_id") or event.get("event_id")
            limitations=list(clock.get("limitations") or [])
            key=(event_id,chart,audio,source)
            if chart is None or audio is None or key in seen:continue
            seen.add(key)
            strict=bool(clock.get("strict_eligible")); clock_ok=bool(clock.get("clock_complete"))
            detector_ok=bool(clock.get("detector_identity_complete")); physical_ok=bool(clock.get("physical_source_identity_complete"))
            complete=bool(event_id and source and confidence is not None and not limitations and strict and clock_ok and detector_ok and physical_ok)
            rows.append({"event_id":event_id,"chart_time":chart,"audio_time":audio,"confidence":confidence,
                "source":source,"physical_source_id":clock.get("physical_source_id") or source,
                "source_independence_group":clock.get("source_independence_group") or source or event_id,
                "detector_family":_family(clock),"nominal_role":clock.get("nominal_role"),
                "observed_signal_type":clock.get("observed_signal_type"),"limitations":limitations,
                "strict_eligible":strict,"strict_identity_complete":complete and confidence>=MIN_CONFIDENCE,
                "query_links":list(clock.get("query_links") or []),"query_localized":bool(clock.get("query_localized")),
                "ledger_status":event.get("status")})
    return rows


def _support(candidate,observations,role):
    start=_num(candidate.get("chart_start"));end=_num(candidate.get("chart_end"));stretch=_num(candidate.get("chart_stretch"));translation=_num(candidate.get("translation_seconds"))
    if None in (start,end,stretch,translation):return {"support_class":"unavailable","best_residual_seconds":None,"supporting_observations":[],"strict_support_count":0}
    matches=[]
    for obs in observations:
        if not start-1e-6<=obs["chart_time"]<=end+1e-6:continue
        predicted=translation+stretch*obs["chart_time"]
        residual=abs(obs["audio_time"]-predicted)
        if residual<=COMPATIBLE_SECONDS:matches.append({**obs,"predicted_audio_time":predicted,"residual_seconds":residual,
            "support_role":"same_role" if str(obs.get("nominal_role") or "").casefold()==str(role).casefold() else "cross_role"})
    matches.sort(key=lambda x:(x["residual_seconds"],-(x["confidence"] or -1)))
    strict=[x for x in matches if x["strict_identity_complete"]]
    groups={}
    for row in strict:
        key=(row["source_independence_group"],row["detector_family"])
        if key not in groups or row["residual_seconds"]<groups[key]["residual_seconds"]:groups[key]=row
    independent=sorted(groups.values(),key=lambda x:x["residual_seconds"])
    best=(independent or matches or [None])[0]
    if best is None:klass="no_local_audio_support"
    elif not independent:klass="unqualified_phrase_support"
    elif best["residual_seconds"]<=EXACT_SECONDS:klass="strict_exact_support"
    else:klass="strict_compatible_support"
    physical={x["source_independence_group"] for x in independent};families={x["detector_family"] for x in independent}
    return {"support_class":klass,"best_residual_seconds":None if best is None else best["residual_seconds"],
        "supporting_observations":matches[:8],"raw_strict_observation_count":len(strict),"strict_support_count":len(independent),
        "independent_physical_source_count":len(physical),"independent_detector_family_count":len(families),
        "same_role_support_count":sum(x["support_role"]=="same_role" for x in independent),
        "cross_role_support_count":sum(x["support_role"]=="cross_role" for x in independent),
        "full_mix_support_count":sum(str(x.get("source") or "").casefold() in {"mix","full_mix","mixture"} for x in independent),
        "best_residual_by_independence_group":{str(k[0]):v["residual_seconds"] for k,v in groups.items()},
        "observation_count":len(matches)}


def _same_candidate(a,b):
    return bool(a and b and a.get("window_id")==b.get("window_id") and abs((_num(a.get("midi_start")) or 0)-(_num(b.get("midi_start")) or 0))<1e-6)


def _boundary_queries(row):
    if row["production_status"]=="occurrence_supported":return []
    queries=[]
    for boundary,chart in (("entry",row.get("chart_start")),("exit",row.get("chart_end"))):
        if chart is None:continue
        queries.append({"boundary_query_id":f"{row['qualification_id']}-{boundary}","window_id":row["window_id"],"role":row["role"],
            "boundary":boundary,"chart_time":chart,"chart_search_start":chart-BOUNDARY_CONTEXT_SECONDS,
            "chart_search_end":chart+BOUNDARY_CONTEXT_SECONDS,"candidate_occurrence_audio_times":[{"occurrence_rank":x.get("occurrence_rank"),
                "audio_time":None if x.get("midi_start") is None else x.get("midi_start")+(chart-row["chart_start"])} for x in row["candidate_occurrences"]],
            "purpose":"distinguish repeated occurrence using entry/exit and neighboring structural context","detectors_executed":False})
    return queries


def _regional(rows):
    result=[]
    by_role={}
    for row in rows:by_role.setdefault(row["role"],[]).append(row)
    for role,items in by_role.items():
        items.sort(key=lambda x:(x.get("chart_start") or -1,x["window_id"]));run=[]
        def flush():
            if not run:return
            supported=[x for x in run if x["production_status"]=="occurrence_supported"]
            disputed=[x for x in run if x["production_status"]=="occurrence_disputed"]
            if disputed and supported:klass="conflicting_neighbor_support"
            elif len(supported)>=2:klass="neighbor_consistent_support"
            elif len(supported)==1:klass="isolated_window_support"
            else:klass="insufficient_regional_support"
            result.append({"regional_id":f"regional-{len(result)+1}","role":role,"chart_start":run[0]["chart_start"],"chart_end":run[-1]["chart_end"],
                "window_ids":[x["window_id"] for x in run],"regional_support_class":klass,
                "supported_window_count":len(supported),"disputed_window_count":len(disputed),"path_reselection_authorized":False})
        for item in items:
            if run and item["chart_start"]>(run[-1]["chart_end"] or 0)+BOUNDARY_CONTEXT_SECONDS:flush();run=[]
            run.append(item)
        flush()
    return result


def build_report(*,midi_section_fit_report:Mapping[str,Any]|None,evidence_contribution_ledger:Mapping[str,Any]|None)->dict[str,Any]:
    report=midi_section_fit_report or {};observations=_audio_observations(evidence_contribution_ledger);selected=_selected_by_window(report);rows=[]
    for role,rr in (((report.get("candidate_report") or {}).get("roles") or {}).items()):
        for window in rr.get("windows") or []:
            candidates=list(window.get("candidates") or [])
            if len(candidates)<2:continue
            wid=str(window.get("window_id") or candidates[0].get("window_id") or "");sel=selected.get((str(role),wid)) or candidates[0];scored=[]
            for c in candidates:
                scored.append({"occurrence_rank":(c.get("phase_diagnostics") or {}).get("occurrence_rank"),"midi_start":_num(c.get("midi_start")),
                    "midi_end":_num(c.get("midi_end")),"content_match":_num(c.get("combined_score")),
                    "phase_distinctiveness":(c.get("phase_diagnostics") or {}).get("phase_distinctiveness"),"selected_by_midi_path":_same_candidate(c,sel),**_support(c,observations,role)})
            sr=next((x for x in scored if x["selected_by_midi_path"]),scored[0]);alts=[x for x in scored if x is not sr and x["strict_support_count"]]
            best_alt=min(alts,key=lambda x:x["best_residual_seconds"]) if alts else None;sr_res=sr["best_residual_seconds"] if sr["strict_support_count"] else None;ar=None if best_alt is None else best_alt["best_residual_seconds"]
            if sr_res is not None and (ar is None or sr_res+DISCRIMINATIVE_MARGIN_SECONDS<ar):outcome="phrase_and_occurrence_supported";status="occurrence_supported"
            elif ar is not None and (sr_res is None or ar+DISCRIMINATIVE_MARGIN_SECONDS<sr_res):outcome="phrase_supported_alternative_occurrence_favored";status="occurrence_disputed"
            elif sr_res is not None or ar is not None:outcome="phrase_supported_occurrence_ambiguous";status="occurrence_disputed"
            elif any(x["observation_count"] for x in scored):outcome="phrase_supported_occurrence_unqualified";status="occurrence_unresolved"
            else:outcome="insufficient_audio_for_occurrence_resolution";status="occurrence_unresolved"
            row={"qualification_id":f"occurrence-{len(rows)+1}","role":role,"window_id":wid,
                "chart_start":_num(window.get("nominal_start") if window.get("nominal_start") is not None else sel.get("chart_start")),
                "chart_end":_num(window.get("nominal_end") if window.get("nominal_end") is not None else sel.get("chart_end")),
                "phrase_identity_support":"supported" if any(x["observation_count"] for x in scored) else "not_observed",
                "occurrence_identity_support":outcome,"production_status":status,"selected_occurrence_rank":sr.get("occurrence_rank"),
                "selected_midi_start":sr.get("midi_start"),"favored_alternative_rank":None if best_alt is None else best_alt.get("occurrence_rank"),
                "favored_alternative_midi_start":None if best_alt is None else best_alt.get("midi_start"),"candidate_occurrences":scored,"timing_changes_applied":False}
            rows.append(row)
    boundary=[q for r in rows for q in _boundary_queries(r)];regional=_regional(rows);counts={}
    for r in rows:counts[r["occurrence_identity_support"]]=counts.get(r["occurrence_identity_support"],0)+1
    return {"version":VERSION,"diagnostic_only":True,"timing_changes_applied":False,"candidate_scores_changed":False,"path_reselected":False,"anchors_created":False,"midi_authority_changed":False,
        "policy":{"exact_seconds":EXACT_SECONDS,"compatible_seconds":COMPATIBLE_SECONDS,"minimum_confidence":MIN_CONFIDENCE,
            "discriminative_margin_seconds":DISCRIMINATIVE_MARGIN_SECONDS,"boundary_context_seconds":BOUNDARY_CONTEXT_SECONDS,
            "independent_vote_key":"physical_source_group + detector_family","phrase_identity_is_not_occurrence_identity":True},
        "qualifications":rows,"regional_consolidation":regional,"targeted_boundary_queries":boundary,
        "summary":{"repeated_window_count":len(rows),"classification_counts":counts,
            "strict_audio_observation_count":sum(x["strict_identity_complete"] for x in observations),
            "audio_observation_count":len(observations),"regional_count":len(regional),
            "neighbor_consistent_region_count":sum(x["regional_support_class"]=="neighbor_consistent_support" for x in regional),
            "isolated_supported_region_count":sum(x["regional_support_class"]=="isolated_window_support" for x in regional),
            "targeted_boundary_query_count":len(boundary),"timing_changes_applied":0}}

# v16.7B1.2A.4 existing-evidence boundary-query evaluation.
BOUNDARY_EXACT_SECONDS=0.10
BOUNDARY_COMPATIBLE_SECONDS=0.50


def _boundary_candidate_support(query, observations):
    rows=[]
    for candidate in query.get("candidate_occurrence_audio_times") or []:
        target=_num(candidate.get("audio_time")); raw=[]
        if target is not None:
            for obs in observations:
                if not obs.get("strict_identity_complete"): continue
                residual=abs(obs["audio_time"]-target)
                if residual<=BOUNDARY_COMPATIBLE_SECONDS:
                    raw.append({**obs,"target_audio_time":target,"residual_seconds":residual})
        raw.sort(key=lambda x:(x["residual_seconds"],-(x.get("confidence") or -1)))
        groups={}
        for item in raw:
            key=(item["source_independence_group"],item["detector_family"])
            if key not in groups or item["residual_seconds"]<groups[key]["residual_seconds"]:groups[key]=item
        independent=sorted(groups.values(),key=lambda x:x["residual_seconds"])
        best=None if not independent else independent[0]["residual_seconds"]
        if best is None:klass="no_boundary_support"
        elif best<=BOUNDARY_EXACT_SECONDS:klass="exact_boundary_support"
        else:klass="compatible_boundary_support"
        rows.append({"occurrence_rank":candidate.get("occurrence_rank"),"target_audio_time":target,
            "support_class":klass,"best_residual_seconds":best,"raw_strict_observation_count":len(raw),
            "independent_support_count":len(independent),
            "independent_physical_source_count":len({x["source_independence_group"] for x in independent}),
            "independent_detector_family_count":len({x["detector_family"] for x in independent}),
            "same_role_boundary_support":sum(str(x.get("nominal_role") or "").casefold()==str(query.get("role") or "").casefold() for x in independent),
            "cross_role_boundary_support":sum(str(x.get("nominal_role") or "").casefold()!=str(query.get("role") or "").casefold() for x in independent),
            "full_mix_boundary_support":sum(str(x.get("source") or "").casefold() in {"mix","full_mix","mixture"} for x in independent),
            "supporting_observations":independent[:8]})
    supported=[x for x in rows if x["independent_support_count"]]
    selected=next((x for x in rows if x.get("occurrence_rank")==query.get("selected_occurrence_rank")),None)
    best=min(supported,key=lambda x:x["best_residual_seconds"]) if supported else None
    if not supported:classification="insufficient_boundary_evidence"
    elif len(supported)>1:
        ordered=sorted(supported,key=lambda x:x["best_residual_seconds"])
        classification="shared_boundary_support" if ordered[1]["best_residual_seconds"]-ordered[0]["best_residual_seconds"]<DISCRIMINATIVE_MARGIN_SECONDS else (
            "selected_boundary_supported" if best is selected else "alternative_boundary_supported")
    else:classification="selected_boundary_supported" if best is selected else "alternative_boundary_supported"
    return {**query,"candidate_boundary_support":rows,"boundary_classification":classification,
        "favored_occurrence_rank":None if best is None else best.get("occurrence_rank"),"timing_changes_applied":False}


def _boundary_pairs(qualifications, evaluations):
    by_window={}
    for row in evaluations:by_window.setdefault((row["role"],row["window_id"]),{})[row["boundary"]]=row
    results=[]
    for q in qualifications:
        pair=by_window.get((q["role"],q["window_id"]),{});entry=pair.get("entry");exit_=pair.get("exit")
        er=None if entry is None else entry.get("favored_occurrence_rank");xr=None if exit_ is None else exit_.get("favored_occurrence_rank")
        selected=q.get("selected_occurrence_rank")
        if er is None and xr is None:klass="insufficient_boundary_evidence"
        elif er is not None and xr is not None and er==xr==selected:klass="selected_occurrence_boundary_pair_supported"
        elif er is not None and xr is not None and er==xr and er!=selected:klass="alternative_occurrence_boundary_pair_supported"
        elif er is not None and xr is not None and er!=xr:klass="conflicting_boundary_support"
        elif er is not None:klass="entry_only_support"
        else:klass="exit_only_support"
        entry_groups={x.get("source_independence_group") for c in (entry or {}).get("candidate_boundary_support",[]) for x in c.get("supporting_observations",[])}
        exit_groups={x.get("source_independence_group") for c in (exit_ or {}).get("candidate_boundary_support",[]) for x in c.get("supporting_observations",[])}
        results.append({"boundary_pair_id":f"boundary-pair-{len(results)+1}","role":q["role"],"window_id":q["window_id"],
            "chart_start":q.get("chart_start"),"chart_end":q.get("chart_end"),"selected_occurrence_rank":selected,
            "entry_classification":None if entry is None else entry.get("boundary_classification"),
            "exit_classification":None if exit_ is None else exit_.get("boundary_classification"),
            "entry_favored_occurrence_rank":er,"exit_favored_occurrence_rank":xr,
            "boundary_pair_classification":klass,"shared_source_across_boundaries":bool(entry_groups&exit_groups),
            "entry_independent_source_count":len(entry_groups),"exit_independent_source_count":len(exit_groups),
            "path_reselection_authorized":False,"timing_changes_applied":False})
    return results


_build_report_a3=build_report
def build_report(*,midi_section_fit_report:Mapping[str,Any]|None,evidence_contribution_ledger:Mapping[str,Any]|None)->dict[str,Any]:
    report=_build_report_a3(midi_section_fit_report=midi_section_fit_report,evidence_contribution_ledger=evidence_contribution_ledger)
    observations=_audio_observations(evidence_contribution_ledger)
    selected={(x["role"],x["window_id"]):x.get("selected_occurrence_rank") for x in report["qualifications"]}
    queries=[]
    query_specs=list(report.get("targeted_boundary_queries") or [])
    existing={(x.get("role"),x.get("window_id"),x.get("boundary")) for x in query_specs}
    for row in report["qualifications"]:
        candidate_times=row.get("candidate_occurrences") or []
        for boundary,chart in (("entry",row.get("chart_start")),("exit",row.get("chart_end"))):
            key=(row["role"],row["window_id"],boundary)
            if key in existing or chart is None:continue
            query_specs.append({"boundary_query_id":f"{row['qualification_id']}-{boundary}",
                "window_id":row["window_id"],"role":row["role"],"boundary":boundary,"chart_time":chart,
                "chart_search_start":chart-BOUNDARY_CONTEXT_SECONDS,"chart_search_end":chart+BOUNDARY_CONTEXT_SECONDS,
                "candidate_occurrence_audio_times":[{"occurrence_rank":x.get("occurrence_rank"),
                    "audio_time":None if x.get("midi_start") is None else x.get("midi_start")+(chart-row["chart_start"])} for x in candidate_times],
                "purpose":"evaluate entry/exit support for every repeated occurrence hypothesis",
                "detectors_executed":False})
    for query in query_specs:
        q=dict(query);q["selected_occurrence_rank"]=selected.get((q["role"],q["window_id"]));queries.append(_boundary_candidate_support(q,observations))
    pairs=_boundary_pairs(report["qualifications"],queries);counts={}
    for row in pairs:counts[row["boundary_pair_classification"]]=counts.get(row["boundary_pair_classification"],0)+1
    report["version"]="v16.7B1.2A.4-existing-evidence-boundary-query-evaluation-v1"
    report["boundary_query_evaluations"]=queries;report["boundary_pair_evaluations"]=pairs
    report["summary"]["boundary_query_evaluation_count"]=len(queries)
    report["summary"]["boundary_pair_classification_counts"]=counts
    report["summary"]["discriminative_boundary_pair_count"]=sum(counts.get(x,0) for x in (
        "selected_occurrence_boundary_pair_supported","alternative_occurrence_boundary_pair_supported"))
    report["policy"].update({"boundary_exact_seconds":BOUNDARY_EXACT_SECONDS,"boundary_compatible_seconds":BOUNDARY_COMPATIBLE_SECONDS,
        "boundary_pair_required_before_path_reconsideration":True})
    return report
