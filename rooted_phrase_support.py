"""v16.6D rooted phrase-ordinal support for section candidate selection.

This module supplements the existing section matcher. It never creates phrase-
derived timing controls and never deletes candidates. Eligible rooted phrase runs
apply bounded score adjustments to already existing section candidates, after
which the normal monotonic path and fusion stages are rerun.
"""
from __future__ import annotations
import copy
from typing import Any, Mapping

VERSION="v16.6D-rooted-phrase-ordinal-support"


def _predict(candidate,t):
    a=float(candidate["chart_start"]);b=float(candidate["chart_end"])
    ma=float(candidate["midi_start"]);mb=float(candidate["midi_end"])
    if b<=a:return ma
    return ma+(float(t)-a)*(mb-ma)/(b-a)


def _root_offset(fusion_report):
    incipit=fusion_report.get("three_clock_incipit") or {}
    solo=fusion_report.get("solo_role_authority") or {}
    rooted=incipit.get("status")=="validated" or bool(solo.get("validated_root_region_ids"))
    controls=[x for r in fusion_report.get("regions") or [] for x in (r.get("controls") or [])]
    if not rooted or not controls:return None
    controls.sort(key=lambda x:float(x["score_time"]))
    first=controls[0]
    return float(first["audio_time"])-float(first["score_time"])


def _eligible_runs(structural,root_offset,max_root_disagreement=.75):
    if root_offset is None:return [],[]
    runs={x["run_id"]:x for x in structural.get("phrase_repeat_runs") or []}
    rows={x["run_id"]:x for x in structural.get("phrase_run_correspondences") or []}
    selected={x.get("selected_run_id") for x in structural.get("overlapping_hypothesis_clusters") or []}
    eligible=[];rejected=[]
    for proposal in structural.get("structural_anchor_proposals") or []:
        if proposal.get("proposal_type")!="phrase_run_ordinal_constraint":continue
        run_id=proposal.get("run_id");run=runs.get(run_id);row=rows.get(run_id)
        reason=None
        if not run or not row:reason="missing_run_detail"
        elif selected and run_id not in selected:reason="not_selected_cluster_representative"
        elif row.get("classification")!="matching_phrase_run":reason="occurrence_count_not_exact"
        elif float(row.get("boundary_context_score") or 0)<.45:reason="boundary_context_too_weak"
        elif float(row.get("joint_confidence") or 0)<.76:reason="joint_confidence_too_weak"
        elif len(run.get("chart_occurrences") or [])!=len(row.get("midi_occurrences") or []):reason="ordinal_count_mismatch"
        else:
            chart0=float(run["chart_occurrences"][0]["start"]);midi0=float(row["midi_occurrences"][0]["start"])
            disagreement=abs((midi0-chart0)-root_offset)
            if disagreement>max_root_disagreement:reason="phrase_origin_disagrees_with_validated_root"
        if reason:rejected.append({"run_id":run_id,"reason":reason})
        else:eligible.append({"run":run,"correspondence":row,"proposal":proposal})
    return eligible,rejected


def apply(candidate_report:Mapping[str,Any],structural:Mapping[str,Any],fusion_report:Mapping[str,Any],
          compatible_bonus=.22,incompatible_penalty=.10,tolerance_seconds=.35):
    """Return a rescored copy plus a complete authority report."""
    root_offset=_root_offset(fusion_report);eligible,rejected=_eligible_runs(structural,root_offset)
    result=copy.deepcopy(candidate_report);adjusted=0;compatible=0;incompatible=0;affected=set()
    for item in eligible:
        run=item["run"];corr=item["correspondence"];role=run["role"]
        pairs=list(zip(run.get("chart_occurrences") or [],corr.get("midi_occurrences") or []))
        for window in ((result.get("roles") or {}).get(role) or {}).get("windows") or []:
            for candidate in window.get("candidates") or []:
                covered=[(c,m) for c,m in pairs if float(candidate["chart_start"])-1e-9<=float(c["start"])<=float(candidate["chart_end"])+1e-9]
                if not covered:continue
                errors=[abs(_predict(candidate,float(c["start"]))-float(m["start"])) for c,m in covered]
                maximum=max(errors);before=float(candidate["combined_score"])
                if maximum<=tolerance_seconds:
                    delta=compatible_bonus*(1-maximum/max(tolerance_seconds,1e-9));compatible+=1
                    status="phrase_ordinal_compatible"
                else:
                    delta=-min(incompatible_penalty,incompatible_penalty*(maximum/tolerance_seconds-1));incompatible+=1
                    status="phrase_ordinal_incompatible"
                candidate["combined_score"]=before+delta
                candidate["rooted_phrase_support"]={"run_id":run["run_id"],"status":status,
                    "maximum_ordinal_error_seconds":maximum,"score_before":before,"score_adjustment":delta,
                    "score_after":candidate["combined_score"],"root_offset_seconds":root_offset}
                adjusted+=1;affected.add(role)
    authority={"version":VERSION,"enabled":True,"root_validated":root_offset is not None,
        "root_offset_seconds":root_offset,"eligible_run_count":len(eligible),"eligible_run_ids":[x["run"]["run_id"] for x in eligible],
        "rejected_runs":rejected,"adjusted_candidate_count":adjusted,"compatible_candidate_count":compatible,
        "incompatible_candidate_count":incompatible,"affected_roles":sorted(affected),
        "candidate_generation_changed":False,"candidates_deleted":False,"phrase_controls_created":False,
        "section_logic_preserved":True,"timing_may_change_only_through_normal_path_reselection":bool(adjusted),
        "policy":{"compatible_bonus":compatible_bonus,"incompatible_penalty":incompatible_penalty,
                  "ordinal_tolerance_seconds":tolerance_seconds,"maximum_root_disagreement_seconds":.75}}
    return result,authority
