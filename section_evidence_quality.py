"""v16.6C.1 calibrated diagnostic-only section evidence quality summarizer."""
from __future__ import annotations
from collections import Counter
from typing import Any, Mapping

VERSION="v16.6C.1-section-evidence-quality-calibration"
QUALITY_SCORE={"pre_roll":0.72,"terminal_tail":0.65,"unsupported":0.08,"conflicting":0.22,
               "structurally_uncertain":0.34,"weak":0.42,"average":0.59,"good":0.76,"strong":0.91}

def _f(value,default=0.0):
    try:return float(value)
    except (TypeError,ValueError):return default

def _overlap(a,b,c,d):return max(0.0,min(b,d)-max(a,c))

def _structural(report):
    fusion=report.get("fusion_report") or {}
    return fusion.get("structural_repeat_diagnostics") or report.get("structural_repeat_report") or {}

def _phase(report):return (report.get("fusion_report") or {}).get("production_phase_diagnostics") or {}

def _timeline_end(report):
    inventory=_structural(report).get("playback_structure_inventory") or []
    if inventory:return max(_f(x.get("end")) for x in inventory)
    return _f((_phase(report).get("terminal") or {}).get("chart_terminal_time"),0.0)

def _first_content(report):
    inventory=_structural(report).get("playback_structure_inventory") or []
    active=[_f(x.get("start")) for x in inventory if x.get("active_roles") and not x.get("empty")]
    opening=_phase(report).get("opening") or {}
    candidates=active+([_f(opening.get("first_fused_control_chart_time"))] if opening.get("first_fused_control_chart_time") is not None else [])
    return min(candidates) if candidates else 0.0

def _last_content(report):
    inventory=_structural(report).get("playback_structure_inventory") or []
    active=[_f(x.get("end")) for x in inventory if x.get("active_roles") and not x.get("empty")]
    return max(active) if active else _timeline_end(report)

def _boundaries(report):
    fusion=report.get("fusion_report") or {};phase=_phase(report);structural=_structural(report)
    app=report.get("application_report") or {};end=_timeline_end(report)
    points={0.0,end,_first_content(report),_last_content(report)}
    for region in fusion.get("regions") or []:points.update((_f(region.get("chart_start")),_f(region.get("chart_end"))))
    for review in phase.get("review_points") or []:
        # Long drift spans create a boundary at onset and exit only; they are not
        # copied as a warning into every overlapping subsection.
        for key in ("chart_time","chart_start","chart_end"):
            if review.get(key) is not None:points.add(_f(review[key]))
    for run in structural.get("phrase_repeat_runs") or []:points.update((_f(run.get("chart_start")),_f(run.get("chart_end"))))
    for region in app.get("applied_regions") or []:points.update((_f(region.get("chart_start")),_f(region.get("chart_end"))))
    for gap in app.get("transitions") or []:
        for key in ("chart_start","chart_end","from_score_time","to_score_time"):
            if gap.get(key) is not None:points.add(_f(gap[key]))
    points=sorted(x for x in points if 0<=x<=end)
    merged=[]
    for x in points:
        if merged and x-merged[-1]<2.0 and x not in (0.0,end):continue
        merged.append(x)
    if not merged or merged[-1]!=end:merged.append(end)
    return merged

def _applied(app,start,end):
    rows=[x for x in app.get("applied_regions") or [] if _overlap(start,end,_f(x.get("chart_start")),_f(x.get("chart_end")))>0]
    return rows

def _active_content(report,start,end):
    rows=_structural(report).get("playback_structure_inventory") or []
    return any(x.get("active_roles") and not x.get("empty") and _overlap(start,end,_f(x.get("start")),_f(x.get("end")))>0 for x in rows)

def _review_events(report,start,end):
    events=[]
    for row in _phase(report).get("review_points") or []:
        t=_f(row.get("chart_time",row.get("chart_start")),-1)
        # A diagnostic episode is assigned to the section containing its onset,
        # not every section overlapped by its broad chart span.
        if start<=t<end:events.append(row)
    return events

def _phrase_rows(report,start,end):
    out=[]
    for row in _structural(report).get("phrase_run_correspondences") or []:
        span=row.get("chart_span") or [0,0]
        if _overlap(start,end,_f(span[0]),_f(span[1]))>0:out.append(row)
    return out

def _selected_conflict(fusion,start,end,applied,strong_ratio):
    conflicts=[x for x in fusion.get("role_conflicts") or [] if start<=_f(x.get("score_time",x.get("chart_time")),-1)<end]
    if not conflicts:return None
    # Alternative clusters are usually already rejected by fusion. With a strong
    # applied path they are a minor uncertainty, not a production conflict.
    if applied and strong_ratio>=.60:return "rejected_alternative_clusters"
    if applied:return "dominant_selected_cluster_with_alternatives"
    return "selected_path_unresolved_conflict"

def _section(report,start,end,index):
    fusion=report.get("fusion_report") or {};app=report.get("application_report") or {}
    opening=_phase(report).get("opening") or {};first=_first_content(report);last=_last_content(report)
    active=_active_content(report,start,end)
    applied=_applied(app,start,end);positive=[];risks=[];score=.46;hard=None;review=None
    semantic="validated_evidence"
    if end<=first+.05:
        semantic="pre_roll";score=.72;positive.append("intentional pre-roll before playable chart content")
    elif start>=last-.05 and not active:
        semantic="terminal_tail";score=.65;positive.append("terminal tail after playable chart content")
    regions=[x for x in fusion.get("regions") or [] if _overlap(start,end,_f(x.get("chart_start")),_f(x.get("chart_end")))>0]
    controls=sum(int(x.get("control_count") or len(x.get("controls") or [])) for x in regions)
    strong=sum(int(x.get("strong_control_count") or 0) for x in regions);strong_ratio=strong/max(1,controls)
    if semantic not in ("pre_roll","terminal_tail"):
        if regions:
            score+=.15;positive.append("coherent monotonic section path")
            if strong_ratio>=.75:score+=.15;positive.append("high strong-control ratio")
            elif strong_ratio>=.40:score+=.08;positive.append("usable strong-control support")
        elif active:
            score-=.30;risks.append("playable chart content has no fused MIDI section");hard="unsupported";semantic="unsupported_playable_content"
        else:
            score-=.08;semantic="baseline_shaped_gap";risks.append("interval is shaped by the selected baseline")
        if applied:
            score+=.16;positive.append("selected production path applies this section")
            semantic="validated_evidence" if strong_ratio>=.60 else "provisional_evidence"
        elif app and active:
            score-=.10;risks.append("production timing uses baseline shape or a recovery gap")
            semantic="baseline_shaped_gap"
        if start<=first<end:
            if opening.get("validated_acquisition_root"):
                score+=.14;positive.append("validated acquisition root")
            elif opening.get("status")=="unrooted_opening":
                score-=.24;risks.append("opening is not tied to a validated acquisition root");hard="structurally_uncertain"
        conflict=_selected_conflict(fusion,start,end,bool(applied),strong_ratio)
        if conflict=="rejected_alternative_clusters":
            score-=.025;risks.append("alternative role hypotheses were rejected by fusion")
        elif conflict=="dominant_selected_cluster_with_alternatives":
            score-=.07;risks.append("selected path dominates competing role hypotheses")
        elif conflict=="selected_path_unresolved_conflict":
            score-=.22;risks.append("selected production phase remains unresolved across roles");hard="conflicting"
        phrases=_phrase_rows(report,start,end)
        if any(x.get("classification")=="matching_phrase_run" and _f(x.get("boundary_context_score"))>=.45 for x in phrases):
            score+=.09;positive.append("bounded phrase repetition supports occurrence identity")
        if any(x.get("classification") in ("multiple_possible_run_instances","additional_midi_occurrences") for x in phrases):
            score-=.055;risks.append("phrase has multiple plausible MIDI instances")
        if any(x.get("classification") in ("phrase_count_mismatch","possible_extra_chart_or_missing_midi_phrase") for x in phrases):
            score-=.09;risks.append("chart and MIDI phrase counts remain unresolved")
        # Region-wide phase evolution is retained as context without turning
        # every subsection into a duplicate review event. Risk scales with the
        # selected path's strong-control ratio and root status.
        phase_regions=[x for x in _phase(report).get("regions") or []
                       if _overlap(start,end,_f(x.get("chart_start")),_f(x.get("chart_end")))>0]
        for phase_region in phase_regions:
            trend=phase_region.get("cumulative_offset_trend") or {}
            beats=abs(_f(trend.get("beats"),0.0))
            if beats>=4.0:
                penalty=.08 if strong_ratio>=.75 else .18
                score-=penalty;risks.append("large regional phase evolution requires structural support")
            elif beats>=1.0 and strong_ratio<.75:
                score-=.12;risks.append("regional phase evolution has limited strong-control support")
            elif beats>=2.0 and opening.get("status")=="unrooted_opening":
                score-=.10;risks.append("unrooted regional phase evolution remains provisional")
        events=_review_events(report,start,end)
        for row in events:
            severity=int(row.get("severity_score") or 1);kind=row.get("kind");t=_f(row.get("chart_time",row.get("chart_start")),start)
            if kind=="unrooted_opening":
                score-=.16;hard="structurally_uncertain"
            elif kind=="gradual_region_drift":
                score-=min(.15,.025*severity);risks.append("cumulative phase movement begins in this section")
            elif kind=="persistent_phase_change":
                score-=min(.12,.025*severity);risks.append("persistent production phase change begins here")
            if review is None or severity>review[0]:review=(severity,t)
    score=max(0.0,min(1.0,score))
    if semantic=="pre_roll":quality="pre_roll"
    elif semantic=="terminal_tail":quality="terminal_tail"
    elif hard=="unsupported":quality="unsupported"
    elif hard=="conflicting":quality="conflicting"
    elif hard=="structurally_uncertain":quality="structurally_uncertain"
    elif score>=.84:quality="strong"
    elif score>=.68:quality="good"
    elif score>=.50:quality="average"
    else:quality="weak"
    risks=list(dict.fromkeys(risks));positive=list(dict.fromkeys(positive))
    if positive and risks:summary=f"{positive[0].capitalize()}, but {risks[0]}."
    elif positive:summary=f"{positive[0].capitalize()} with no leading production-path warning."
    else:summary=(risks[0].capitalize()+".") if risks else "Limited diagnostic evidence is available."
    use="applied" if applied else "baseline_or_unsupported"
    return {"section_index":index,"section_start":round(start,6),"section_end":round(end,6),
      "section_name":"intro" if index==1 and semantic!="pre_roll" else None,"semantic_type":semantic,
      "quality":quality,"confidence":round(score,3),"summary":summary,"positive_factors":positive[:3],
      "risk_factors":risks[:3],"production_use":use,"production_method":"midi-section-fit" if applied else "selected-baseline-shape",
      "production_provenance":sorted({str(x.get("source_region_id") or x.get("region_id") or "midi-section") for x in applied}),
      "primary_warning":risks[0] if risks else None,"suggested_review_time":None if review is None else round(review[1],6),
      "diagnostic_only":True}

def _merge(rows):
    merged=[]
    for row in rows:
        if merged and row["quality"]==merged[-1]["quality"] and row["semantic_type"]==merged[-1]["semantic_type"] and row["production_use"]==merged[-1]["production_use"] and row["section_end"]-merged[-1]["section_start"]<=40:
            prev=merged[-1];prev["section_end"]=row["section_end"];prev["confidence"]=round((prev["confidence"]+row["confidence"])/2,3)
            prev["positive_factors"]=list(dict.fromkeys(prev["positive_factors"]+row["positive_factors"]))[:3]
            prev["risk_factors"]=list(dict.fromkeys(prev["risk_factors"]+row["risk_factors"]))[:3]
            prev["primary_warning"]=prev["risk_factors"][0] if prev["risk_factors"] else None
            if prev["suggested_review_time"] is None:prev["suggested_review_time"]=row["suggested_review_time"]
        else:merged.append(dict(row))
    for i,row in enumerate(merged,1):row["section_index"]=i
    return merged

def build(report:Mapping[str,Any]):
    boundaries=_boundaries(report);rows=[_section(report,a,b,i) for i,(a,b) in enumerate(zip(boundaries,boundaries[1:]),1) if b-a>=.25]
    rows=_merge(rows)
    musical=[x for x in rows if x["semantic_type"] not in ("pre_roll","terminal_tail")]
    duration=sum(x["section_end"]-x["section_start"] for x in musical) or 1.0
    distribution=Counter()
    for row in musical:distribution[row["quality"]]+=row["section_end"]-row["section_start"]
    percentages={k:round(100*v/duration,1) for k,v in distribution.items()}
    weighted=sum(QUALITY_SCORE[x["quality"]]*(x["section_end"]-x["section_start"]) for x in musical)/duration if musical else 0
    overall="strong" if weighted>=.82 else "good" if weighted>=.67 else "average" if weighted>=.50 else "weak"
    review_values=[]
    for row in rows:
        t=row.get("suggested_review_time")
        if t is not None and t not in review_values:review_values.append(t)
    review_values=review_values[:8]
    # Ignore pre-roll/terminal and rank playable production concerns first.
    candidates=[x for x in musical if x.get("primary_warning")]
    candidates.sort(key=lambda x:(0 if x["production_use"]=="applied" else 1,x["confidence"],-(x["section_end"]-x["section_start"])))
    concern=candidates[0]["primary_warning"] if candidates else None
    return {"version":VERSION,"diagnostic_only":True,"selection_behavior_changed":False,"timing_changes_applied":False,
      "overall_quality":overall,"overall_confidence":round(weighted,3),"quality_distribution_percent":percentages,
      "primary_concern":concern,"suggested_review_points":review_values,"section_count":len(rows),"sections":rows,
      "notes":["Evidence quality describes the selected production path; rejected alternative hypotheses are not hard conflicts.",
               "Pre-roll and terminal tails are excluded from musical evidence percentages.",
               "The summary does not accept, reject, prune, or retime alignment evidence."]}
