"""Stage 3C.1-3C.3 diagnostic alignment proposal generation.

Generates baseline, boundary-aware regional offset/scale alternatives, and
source-fused piecewise-linear proposals. Every proposal maps the complete
playback downbeat timeline so boundary transitions are validated explicitly.
Nothing here modifies the canonical beatmap or packaged timing.
"""
from __future__ import annotations

from statistics import median
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = 2
SAFE_STRETCH = (0.9, 1.1)
MIN_FIT_OBSERVATIONS = 4
MIN_REGION_SPAN_SECONDS = 8.0
MIN_OFFSET_MS = 40.0
MIN_SCALE_SLOPE_MS_PER_SECOND = 0.35
BOUNDARY_TAPER_MEASURES = 2
MIN_PIECEWISE_CONTROLS = 3
MAX_PIECEWISE_CONTROLS = 8
MIN_CONTROL_SPACING_MEASURES = 2
BOUNDARY_WARNING_MS = 60.0
BOUNDARY_ERROR_MS = 180.0
PRACTICAL_IMPROVEMENT_MS = 20.0


def _number(value: Any, default: float | None = None) -> float | None:
    try: return float(value)
    except (TypeError, ValueError): return default


def _downbeats(beatmap: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [p for p in beatmap.get("beats") or [] if p.get("structural_role") == "downbeat"]


def _region_samples(region: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [s for s in region.get("samples") or []
            if _number(s.get("audio_time")) is not None and _number(s.get("residual_ms")) is not None]


def _mad(values: Sequence[float]) -> float | None:
    if not values: return None
    centre=median(values)
    return median(abs(v-centre) for v in values)


def _theil_sen(samples: Sequence[Mapping[str, Any]]) -> tuple[float | None,float | None]:
    points=sorted((_number(s["audio_time"],0.0),_number(s["residual_ms"],0.0)) for s in samples)
    slopes=[(y2-y1)/(x2-x1) for i,(x1,y1) in enumerate(points) for x2,y2 in points[i+1:] if x2>x1]
    if not slopes: return None,None
    slope=median(slopes)
    return slope,median(y-slope*x for x,y in points)


def _fit_metrics(samples: Sequence[Mapping[str, Any]], correction_ms: Callable[[float,int],float]) -> dict[str,Any]:
    before=[abs(_number(s["residual_ms"],0.0)) for s in samples]
    after=[abs(_number(s["residual_ms"],0.0)-correction_ms(_number(s["audio_time"],0.0),int(s.get("playback_measure",0) or 0))) for s in samples]
    raw_before=[_number(s["residual_ms"],0.0) for s in samples]
    raw_after=[_number(s["residual_ms"],0.0)-correction_ms(_number(s["audio_time"],0.0),int(s.get("playback_measure",0) or 0)) for s in samples]
    return {"observation_count":len(samples),
            "baseline_median_absolute_residual_ms":None if not before else round(median(before),3),
            "proposal_median_absolute_residual_ms":None if not after else round(median(after),3),
            "baseline_residual_mad_ms":None if not raw_before else round(_mad(raw_before),3),
            "proposal_residual_mad_ms":None if not raw_after else round(_mad(raw_after),3),
            "median_absolute_improvement_ms":None if not before else round(median(before)-median(after),3)}


def _proposal_id(region:Mapping[str,Any],suffix:str)->str:
    return f"{region.get('region_id','region')}.{suffix}"


def _measure_to_time(points:Sequence[Mapping[str,Any]])->dict[int,float]:
    return {int(p["identity"]["playback_measure"]):float(p["timing"]["audio_time"]) for p in points}


def _linear_interpolate(controls:Sequence[tuple[float,float]],x:float)->float:
    controls=sorted(controls)
    if not controls:return 0.0
    if x<=controls[0][0]:return controls[0][1]
    if x>=controls[-1][0]:return controls[-1][1]
    for (x1,y1),(x2,y2) in zip(controls,controls[1:]):
        if x1<=x<=x2:
            ratio=(x-x1)/(x2-x1) if x2>x1 else 0.0
            return y1+ratio*(y2-y1)
    return 0.0


def _taper_factor(measure:int,start:int,end:int,width:int=BOUNDARY_TAPER_MEASURES)->float:
    if measure<start or measure>end:return 0.0
    if width<=0:return 1.0
    left=min(1.0,max(0.0,(measure-start+1)/(width+1)))
    right=min(1.0,max(0.0,(end-measure+1)/(width+1)))
    return min(left,right,1.0)


def _full_mapping(beatmap:Mapping[str,Any], correction_for_point:Callable[[Mapping[str,Any]],float])->list[dict[str,Any]]:
    output=[]
    for p in _downbeats(beatmap):
        base=float(p["timing"]["audio_time"]); correction=float(correction_for_point(p))
        output.append({"beat_id":p["identity"]["beat_id"],"playback_measure":int(p["identity"]["playback_measure"]),
                       "baseline_audio_time":base,"proposed_audio_time":round(base+correction/1000.0,6),
                       "correction_ms":round(correction,3)})
    return output


def baseline_proposal(beatmap:Mapping[str,Any])->dict[str,Any]:
    points=_downbeats(beatmap)
    return {"proposal_id":"global.baseline","method":"retain_baseline","status":"diagnostic_only",
            "timing_changes_applied":False,
            "region":{"start_playback_measure":points[0]["identity"]["playback_measure"] if points else None,
                      "end_playback_measure":points[-1]["identity"]["playback_measure"] if points else None},
            "controls":[],"complexity":{"control_count":0,"regional_boundary_count":0},
            "mapped_downbeats":_full_mapping(beatmap,lambda _:0.0),"validation":{},"evidence_fit":{}}


def _regional_correction(region:Mapping[str,Any], raw:Callable[[float],float])->Callable[[float,int],float]:
    start=int(region["start_playback_measure"]);end=int(region["end_playback_measure"])
    return lambda audio_time,measure: raw(audio_time)*_taper_factor(measure,start,end)


def offset_proposal(beatmap:Mapping[str,Any],region:Mapping[str,Any])->dict[str,Any]|None:
    samples=_region_samples(region)
    if len(samples)<MIN_FIT_OBSERVATIONS:return None
    offset=median([_number(s["residual_ms"],0.0) for s in samples])
    if abs(offset)<MIN_OFFSET_MS:return None
    start=int(region["start_playback_measure"]);end=int(region["end_playback_measure"])
    correction=_regional_correction(region,lambda _:offset)
    return {"proposal_id":_proposal_id(region,"offset_tapered01"),"method":"regional_constant_offset",
            "status":"diagnostic_only","timing_changes_applied":False,
            "region":{"region_id":region.get("region_id"),"start_playback_measure":start,"end_playback_measure":end},
            "controls":[{"kind":"soft_regional_offset","offset_ms":round(offset,3),"taper_measures":BOUNDARY_TAPER_MEASURES}],
            "complexity":{"control_count":1,"regional_boundary_count":2,"taper_measure_count":BOUNDARY_TAPER_MEASURES*2},
            "mapped_downbeats":_full_mapping(beatmap,lambda p:correction(float(p["timing"]["audio_time"]),int(p["identity"]["playback_measure"]))),
            "evidence_fit":_fit_metrics(samples,correction),"validation":{}}


def scale_proposal(beatmap:Mapping[str,Any],region:Mapping[str,Any])->dict[str,Any]|None:
    samples=_region_samples(region)
    if len(samples)<MIN_FIT_OBSERVATIONS:return None
    times=[_number(s["audio_time"],0.0) for s in samples]
    if max(times)-min(times)<MIN_REGION_SPAN_SECONDS:return None
    slope,intercept=_theil_sen(samples)
    if slope is None or abs(slope)<MIN_SCALE_SLOPE_MS_PER_SECOND:return None
    start=int(region["start_playback_measure"]);end=int(region["end_playback_measure"])
    raw=lambda t:intercept+slope*t
    correction=_regional_correction(region,raw)
    return {"proposal_id":_proposal_id(region,"scale_tapered01"),"method":"regional_linear_scale",
            "status":"diagnostic_only","timing_changes_applied":False,
            "region":{"region_id":region.get("region_id"),"start_playback_measure":start,"end_playback_measure":end},
            "controls":[{"kind":"soft_linear_fit","residual_slope_ms_per_second":round(slope,6),"taper_measures":BOUNDARY_TAPER_MEASURES}],
            "diagnostic_residual_slope_ms_per_second":round(slope,6),"implied_scale":round(1+slope/1000.0,9),
            "complexity":{"control_count":2,"regional_boundary_count":2,"taper_measure_count":BOUNDARY_TAPER_MEASURES*2},
            "mapped_downbeats":_full_mapping(beatmap,lambda p:correction(float(p["timing"]["audio_time"]),int(p["identity"]["playback_measure"]))),
            "evidence_fit":_fit_metrics(samples,correction),"validation":{}}


def _select_piecewise_controls(region:Mapping[str,Any],fusion:Mapping[str,Any],points:Sequence[Mapping[str,Any]])->list[dict[str,Any]]:
    start=int(region["start_playback_measure"]);end=int(region["end_playback_measure"])
    hypotheses=[h for h in fusion.get("hypotheses") or [] if h.get("accepted_hypothesis")
                and start<=int(h["playback_measure"])<=end and _number(h.get("residual_ms")) is not None]
    hypotheses.sort(key=lambda h:(int(h["playback_measure"]),-float(h.get("fusion_score",0))))
    selected=[]
    for h in hypotheses:
        m=int(h["playback_measure"])
        if selected and m-int(selected[-1]["playback_measure"])<MIN_CONTROL_SPACING_MEASURES:
            if float(h.get("fusion_score",0))>float(selected[-1].get("fusion_score",0)):selected[-1]=h
            continue
        selected.append(h)
    if len(selected)>MAX_PIECEWISE_CONTROLS:
        indices=[round(i*(len(selected)-1)/(MAX_PIECEWISE_CONTROLS-1)) for i in range(MAX_PIECEWISE_CONTROLS)]
        selected=[selected[i] for i in sorted(set(indices))]
    return selected


def piecewise_proposal(beatmap:Mapping[str,Any],region:Mapping[str,Any],fusion:Mapping[str,Any])->dict[str,Any]|None:
    points=_downbeats(beatmap); controls=_select_piecewise_controls(region,fusion,points)
    if len(controls)<MIN_PIECEWISE_CONTROLS:return None
    start=int(region["start_playback_measure"]);end=int(region["end_playback_measure"])
    time_by_measure=_measure_to_time(points)
    interior=[(time_by_measure[int(h["playback_measure"])],float(h["residual_ms"])) for h in controls if int(h["playback_measure"]) in time_by_measure]
    if len(interior)<MIN_PIECEWISE_CONTROLS:return None
    before=max([m for m in time_by_measure if m<start],default=start)
    after=min([m for m in time_by_measure if m>end],default=end)
    interpolation=[(time_by_measure[before],0.0)]+interior+[(time_by_measure[after],0.0)]
    interpolation=sorted(dict(interpolation).items())
    correction=lambda t,m:_linear_interpolate(interpolation,t) if before<=m<=after else 0.0
    samples=_region_samples(region)
    return {"proposal_id":_proposal_id(region,"piecewise_fused01"),"method":"source_fused_piecewise_linear",
            "status":"diagnostic_only","timing_changes_applied":False,
            "region":{"region_id":region.get("region_id"),"start_playback_measure":start,"end_playback_measure":end,
                      "transition_start_measure":before,"transition_end_measure":after},
            "controls":[{"kind":"fused_soft_control","playback_measure":int(h["playback_measure"]),
                         "residual_ms":round(float(h["residual_ms"]),3),"roles":list(h.get("roles") or []),
                         "fusion_score":h.get("fusion_score")} for h in controls],
            "complexity":{"control_count":len(controls),"regional_boundary_count":2,"piecewise_segment_count":len(interpolation)-1},
            "mapped_downbeats":_full_mapping(beatmap,lambda p:correction(float(p["timing"]["audio_time"]),int(p["identity"]["playback_measure"]))),
            "evidence_fit":_fit_metrics(samples,correction),"validation":{}}


def validate_proposal(proposal:Mapping[str,Any],safe_stretch=SAFE_STRETCH)->dict[str,Any]:
    points=list(proposal.get("mapped_downbeats") or []);errors=[];warnings=[];stretches=[];boundary_intervals=[]
    region=proposal.get("region") or {};start=region.get("start_playback_measure");end=region.get("end_playback_measure")
    for previous,current in zip(points,points[1:]):
        base=float(current["baseline_audio_time"])-float(previous["baseline_audio_time"])
        proposed=float(current["proposed_audio_time"])-float(previous["proposed_audio_time"])
        if proposed<=0:
            errors.append({"kind":"non_monotonic_proposal","playback_measure":current["playback_measure"]});continue
        if base>0:
            stretch=proposed/base;stretches.append(stretch)
            is_boundary=(start is not None and current["playback_measure"]==start) or (end is not None and previous["playback_measure"]==end)
            if is_boundary:boundary_intervals.append({"from_measure":previous["playback_measure"],"to_measure":current["playback_measure"],"stretch":round(stretch,6)})
            if not safe_stretch[0]<=stretch<=safe_stretch[1]:warnings.append({"kind":"stretch_outside_safe_range","playback_measure":current["playback_measure"],"stretch":round(stretch,6),"boundary_interval":is_boundary})
    corrections=[float(p.get("correction_ms",0)) for p in points]
    max_jump=max([abs(b-a) for a,b in zip(corrections,corrections[1:])] or [0.0])
    if max_jump>=BOUNDARY_ERROR_MS:errors.append({"kind":"abrupt_correction_jump","maximum_adjacent_jump_ms":round(max_jump,3)})
    elif max_jump>=BOUNDARY_WARNING_MS:warnings.append({"kind":"large_correction_jump","maximum_adjacent_jump_ms":round(max_jump,3)})
    return {"status":"error" if errors else "warning" if warnings else "ok","safe_stretch_range":list(safe_stretch),
            "error_count":len(errors),"warning_count":len(warnings),"errors":errors,"warnings":warnings,
            "minimum_local_stretch":None if not stretches else round(min(stretches),6),
            "maximum_local_stretch":None if not stretches else round(max(stretches),6),
            "maximum_adjacent_correction_jump_ms":round(max_jump,3),"boundary_intervals":boundary_intervals}


def _proposal_score(proposal:Mapping[str,Any])->dict[str,Any]:
    fit=proposal.get("evidence_fit") or {};validation=proposal.get("validation") or {};complexity=proposal.get("complexity") or {}
    improvement=_number(fit.get("median_absolute_improvement_ms"),0.0) or 0.0
    base_mad=_number(fit.get("baseline_residual_mad_ms"),0.0) or 0.0;prop_mad=_number(fit.get("proposal_residual_mad_ms"),0.0) or 0.0
    dispersion_penalty=max(0.0,prop_mad-base_mad)
    control_penalty=5.0*int(complexity.get("control_count",0));warning_penalty=15.0*int(validation.get("warning_count",0));error_penalty=1000.0*int(validation.get("error_count",0))
    jump_penalty=max(0.0,float(validation.get("maximum_adjacent_correction_jump_ms",0))-BOUNDARY_WARNING_MS)*0.25
    total=improvement-dispersion_penalty-control_penalty-warning_penalty-error_penalty-jump_penalty
    return {"fit_improvement_ms":round(improvement,3),"dispersion_penalty":round(dispersion_penalty,3),
            "control_complexity_penalty":round(control_penalty,3),"validation_warning_penalty":round(warning_penalty,3),
            "validation_error_penalty":round(error_penalty,3),"boundary_jump_penalty":round(jump_penalty,3),"total":round(total,3)}


def _song_level_eligibility(proposal:Mapping[str,Any],fusion:Mapping[str,Any]|None)->str:
    if proposal["method"]=="retain_baseline":return "retain_baseline"
    summary=(fusion or {}).get("summary") or {};total=int(summary.get("playback_measure_count",0) or 0);accepted=int(summary.get("accepted_hypothesis_count",0) or 0)
    coverage=accepted/total if total else 0.0
    if proposal["validation"]["status"]=="error":return "rejected_by_safety_validation"
    if coverage<0.20:return "partial_local_improvement_only"
    if int(summary.get("long_gap_count",0) or 0)>0:return "underconstrained_by_coverage_gaps"
    return "carry_forward_for_piecewise_validation"


def generate(beatmap:Mapping[str,Any],evidence_report:Mapping[str,Any],fusion_report:Mapping[str,Any]|None=None)->dict[str,Any]:
    proposals=[baseline_proposal(beatmap)]
    for region in evidence_report.get("regions") or []:
        for factory in (offset_proposal,scale_proposal):
            p=factory(beatmap,region)
            if p:proposals.append(p)
        if fusion_report:
            p=piecewise_proposal(beatmap,region,fusion_report)
            if p:proposals.append(p)
    for p in proposals:
        p["validation"]=validate_proposal(p);p["comparison_score"]=_proposal_score(p);p["song_level_eligibility"]=_song_level_eligibility(p,fusion_report)
        p["local_recommendation"]="carry_forward" if p["comparison_score"]["total"]>0 and p["validation"]["status"]!="error" else "retain_baseline_or_reject"
    ranked=sorted(proposals,key=lambda p:p["comparison_score"]["total"],reverse=True)
    eligible=[p for p in ranked if p["local_recommendation"]=="carry_forward" and p["song_level_eligibility"] not in ("rejected_by_safety_validation","partial_local_improvement_only")]
    best=eligible[0] if eligible else next(p for p in proposals if p["method"]=="retain_baseline")
    return {"schema":"feedpak.alignment.proposals","schema_version":SCHEMA_VERSION,"phase":"3C.3","timing_changes_applied":False,
            "safe_stretch_range":list(SAFE_STRETCH),"proposal_count":len(proposals),"baseline_proposal_id":"global.baseline",
            "recommended_proposal_id":best["proposal_id"],"recommendation":"prefer_alternative_diagnostically" if best["method"]!="retain_baseline" else "retain_baseline",
            "fusion_context_summary":dict((fusion_report or {}).get("summary") or {}),"proposals":proposals,
            "ranking":[{"proposal_id":p["proposal_id"],"method":p["method"],"score":p["comparison_score"],"song_level_eligibility":p["song_level_eligibility"]} for p in ranked],
            "notes":["Diagnostic proposals only; canonical timing is unchanged.","Full-timeline mappings make regional boundary transitions explicit.","Piecewise controls come only from accepted fused hypotheses."]}
