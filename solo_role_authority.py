"""Expected-role-aware solo authority and acquisition-root safety."""
from __future__ import annotations
from typing import Callable, Mapping
import math

MIN_SOLO_CONTROL_CONFIDENCE=.60
ROOT_OFFSET_MAX_SECONDS=12.0
ROOT_STRETCH_MIN=.70
ROOT_STRETCH_MAX=1.30


def _time(row):
    for key in ("t_gp","time","score_time","start_time","t"):
        if isinstance(row,Mapping) and row.get(key) is not None:return float(row[key])
    return None


def _end(row,start):
    for key in ("end_gp","end_time","score_end"):
        if row.get(key) is not None:return float(row[key])
    return start+max(0.0,float(row.get("duration",row.get("sus",0.0)) or 0.0))


def expected_roles(chart_events_by_role,start,end):
    found=[]
    for role,events in chart_events_by_role.items():
        for event in events or []:
            t=_time(event)
            if t is not None and t<=end and _end(event,t)>=start:
                found.append(str(role));break
    return sorted(set(found))


def annotate_fusion_report(fusion_report,chart_events_by_role):
    onsets=[_time(x) for events in chart_events_by_role.values() for x in (events or [])]
    onsets=[x for x in onsets if x is not None]
    first_onset=min(onsets) if onsets else None;promoted=0
    for region in fusion_report.get("regions") or []:
        controls=region.get("controls") or []
        for i,control in enumerate(controls):
            t=float(control["score_time"])
            left=(float(controls[i-1]["score_time"])+t)/2 if i else t-1.5
            right=(t+float(controls[i+1]["score_time"]))/2 if i+1<len(controls) else t+1.5
            expected=expected_roles(chart_events_by_role,left,right)
            supporting=sorted(set(control.get("roles") or []))
            supported=sorted(set(expected)&set(supporting))
            control.update(expected_roles=expected,supporting_expected_roles=supported,
                expected_role_support_ratio=len(supported)/len(expected) if expected else 0.0)
            if (len(expected)==1 and supported==expected and
                    float(control.get("confidence",0.0))>=MIN_SOLO_CONTROL_CONFIDENCE):
                control["classification"]="solo_role_expected_strong"
                control["solo_role_authority_basis"]="only_expected_observable_role"
                promoted+=1
        region["strong_control_count"]=sum(x.get("classification")!="single_role_provisional" for x in controls)
        solo=[x for x in controls if x.get("classification")=="solo_role_expected_strong"]
        chart_span=float(region.get("chart_end",0))-float(region.get("chart_start",0))
        midi_span=float(region.get("midi_end",0))-float(region.get("midi_start",0))
        stretch=midi_span/chart_span if chart_span>0 else None
        offset=float(region.get("midi_start",0))-float(region.get("chart_start",0))
        near=first_onset is not None and float(region.get("chart_start",math.inf))<=first_onset+3.0
        valid=bool(near and len(solo)>=2 and abs(offset)<=ROOT_OFFSET_MAX_SECONDS and stretch is not None and
                   ROOT_STRETCH_MIN<=stretch<=ROOT_STRETCH_MAX and float(region.get("midi_start",-1))>=0)
        region["acquisition_root"]={"validated":valid,"authority":"solo_role_expected_strong" if valid else None,
            "first_chart_onset":first_onset,"score_time":region.get("chart_start"),"audio_time":region.get("midi_start"),
            "offset_seconds":offset,"stretch":stretch,"solo_strong_control_count":len(solo)}
    fusion_report["solo_role_authority"]={"promoted_control_count":promoted,"first_chart_onset":first_onset,
        "validated_root_region_ids":[r.get("region_id") for r in fusion_report.get("regions") or []
                                     if (r.get("acquisition_root") or {}).get("validated")]}
    return fusion_report


def safe_materialization_baseline(fusion_report,baseline:Callable[[float],float]):
    regions=sorted(fusion_report.get("regions") or [],key=lambda r:float(r.get("chart_start",0)))
    samples=sorted({float(c["score_time"]) for r in regions for c in (r.get("controls") or [])})
    try:mapped=[float(baseline(t)) for t in samples]
    except Exception:mapped=[]
    slopes=[(b2-b1)/(t2-t1) for t1,t2,b1,b2 in zip(samples,samples[1:],mapped,mapped[1:]) if t2>t1]
    negative=any(x<0 for x in mapped);reversed_=any(b<=a for a,b in zip(mapped,mapped[1:]))
    implausible=any(x<.50 or x>1.50 for x in slopes)
    valid=bool(samples and len(mapped)==len(samples) and not negative and not reversed_ and not implausible)
    report={"valid":valid,"negative_mapped_time":negative,"nonmonotonic":reversed_,
        "implausible_stretch":implausible,"sample_count":len(samples),"fallback":"none"}
    if valid:return baseline,report
    root=next((r for r in regions if (r.get("acquisition_root") or {}).get("validated")),None)
    if root:
        offset=float(root["midi_start"])-float(root["chart_start"])
        report.update(valid=True,original_baseline_valid=False,fallback="acquisition_root_identity_clock",
                      root_region_id=root.get("region_id"),root_offset_seconds=offset)
        return lambda t:float(t)+offset,report
    report["blocked_reason"]="invalid_baseline_without_acquisition_root"
    return baseline,report
