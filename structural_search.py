"""Stage 3C.4.1 general structural-hypothesis refinement.

This module performs wide whole-measure correspondence search, estimates source
quality and dependence without trusting nominal stem labels, clusters overlapping
window hits into displacement families, and emits correspondence hypotheses only.
It never creates a temporary warp and never changes canonical timing.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Callable, Mapping, Sequence
import math

import numpy as np
import instrument_evidence as ie

try:
    import soundfile as sf
except ImportError:
    sf = None

SCHEMA_VERSION = 3
WINDOW_SIZES = (4, 8)
MEASURE_SHIFTS = tuple(i for i in range(-8, 9) if i != 0)
MIN_WINDOW_SCORE = 0.56
MIN_SCORE_IMPROVEMENT = 0.08
MIN_VALID_MEASURES_PER_SOURCE = 2
MAX_RAW_HYPOTHESES_PER_GAP = 12
MAX_CLUSTERS_PER_GAP = 4
SHIFT_CLUSTER_TOLERANCE_MEASURES = 1
MIN_CLUSTER_WINDOWS = 2
MIN_UNIQUENESS_MARGIN = 0.04
HIGH_REDUNDANCY = 0.82
MODERATE_REDUNDANCY = 0.65
MIN_EFFECTIVE_SOURCE_GROUPS = 1.65
NEAR_MISS_LIMIT_PER_REASON = 5
REGIONAL_DEPENDENCE_PADDING_SECONDS = 1.0
COMMON_FEATURE_HOP_SECONDS = 0.02
COMMON_SIMILARITY_MAX_LAG_SECONDS = 0.10
REGIONAL_SIMILARITY_CACHE_LIMIT = 4096
PERSISTENCE_WINDOW_MEASURES = 4
PERSISTENCE_STEP_MEASURES = 2
PERSISTENCE_CONTEXT_WINDOWS = 3
MIN_PERSISTENT_WINDOWS = 2
MIN_PERSISTENT_RATIO = 0.50
BOUNDARY_SEARCH_RADIUS_MEASURES = 4

STRUCTURAL_CONTROL_PENALTY = 20.0


def _number(value: Any, default: float = 0.0) -> float:
    try: return float(value)
    except (TypeError, ValueError): return default


def _event_times(events) -> list[float]:
    output=[]
    for event in events or []:
        if isinstance(event,(int,float)): output.append(float(event)); continue
        for key in ("t_gp","time","t"):
            if key in event: output.append(float(event[key])); break
    return sorted(set(output))


def _measure_rows(downbeats:Sequence[Mapping[str,Any]])->list[dict[str,Any]]:
    rows=[]
    for i,db in enumerate(downbeats):
        start=float(db["t_gp"])
        if i+1<len(downbeats): end=float(downbeats[i+1]["t_gp"])
        elif i: end=start+(start-float(downbeats[i-1]["t_gp"]))
        else: continue
        rows.append({"index":i,"measure":int(db.get("measure",i+1)),"start":start,"end":end})
    return rows


def _window_starts(gap:Mapping[str,Any],rows:Sequence[Mapping[str,Any]],size:int)->list[int]:
    start=int(gap["start_playback_measure"]); end=int(gap["end_playback_measure"])
    candidates=[int(r["measure"]) for r in rows if start<=int(r["measure"]) and int(r["measure"])+size-1<=end]
    if len(candidates)<=4: return candidates
    positions=[0,round((len(candidates)-1)/3),round(2*(len(candidates)-1)/3),len(candidates)-1]
    return [candidates[i] for i in sorted(set(positions))]


def _resample(values:np.ndarray,length:int)->np.ndarray:
    values=np.asarray(values,dtype=float)
    if length<=0:return np.zeros(0)
    if len(values)==length:return values
    if len(values)<2:return np.full(length,float(values[0]) if len(values) else 0.0)
    return np.interp(np.linspace(0,len(values)-1,length),np.arange(len(values)),values)


def _robust_similarity(a:np.ndarray,b:np.ndarray)->float:
    length=min(len(a),len(b))
    if length<8:return 0.0
    a=_resample(a,length); b=_resample(b,length)
    a=a-np.median(a); b=b-np.median(b)
    divisor=float(np.linalg.norm(a)*np.linalg.norm(b))
    if divisor<=1e-12:return 0.0
    return max(0.0,min(1.0,float(np.dot(a,b)/divisor)))


def _feature_slice(feature:Mapping[str,Any],start:float,end:float)->np.ndarray:
    times=np.asarray(feature.get("times", []),dtype=float); novelty=np.asarray(feature.get("novelty", []),dtype=float)
    if not len(times) or not len(novelty):return np.zeros(0)
    mask=(times>=start)&(times<=end)
    return novelty[mask]



def _common_source_feature(audio_path, target_sr=8000):
    """Source-neutral envelopes used only for usability/dependence estimation."""
    if not audio_path or sf is None:
        return {"status":"unavailable","reason":"missing_audio_or_soundfile"}
    try:
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    except Exception as exc:
        return {"status":"unavailable","reason":f"audio_read_failed:{type(exc).__name__}"}
    audio=np.asarray(audio,dtype=float)
    if audio.ndim>1: audio=np.mean(audio,axis=1)
    if not len(audio) or sr<=0: return {"status":"unavailable","reason":"empty_audio"}
    if sr!=target_sr:
        length=max(1,round(len(audio)*target_sr/sr)); audio=_resample(audio,length); sr=target_sr
    hop=max(1,round(COMMON_FEATURE_HOP_SECONDS*sr)); frame=max(hop*2,round(.04*sr))
    count=max(0,1+(len(audio)-frame)//hop)
    if count<8:return {"status":"unavailable","reason":"insufficient_frames"}
    rms=[]; flux=[]; low=[]; previous=None
    window=np.hanning(frame)
    cutoff=max(2,round(250*frame/sr))
    for i in range(count):
        segment=audio[i*hop:i*hop+frame]*window
        spectrum=np.abs(np.fft.rfft(segment))
        rms.append(float(np.sqrt(np.mean(segment*segment))))
        flux.append(0.0 if previous is None else float(np.mean(np.maximum(0,spectrum-previous))))
        low.append(float(np.mean(spectrum[:cutoff])))
        previous=spectrum
    def norm(values):
        values=np.asarray(values,dtype=float); scale=np.percentile(values,95) if len(values) else 0
        return values/(scale+1e-12)
    return {"status":"ok","times":np.arange(count)*hop/sr+frame/(2*sr),
            "envelopes":{"rms":norm(rms),"broadband_flux":norm(flux),"low_frequency_energy":norm(low)}}

def _lag_tolerant_similarity(left_values,right_values,hop_seconds=COMMON_FEATURE_HOP_SECONDS):
    left_values=np.asarray(left_values,dtype=float);right_values=np.asarray(right_values,dtype=float)
    length=min(len(left_values),len(right_values))
    if length<8:return 0.0
    left_values=_resample(left_values,length);right_values=_resample(right_values,length)
    max_lag=max(0,round(COMMON_SIMILARITY_MAX_LAG_SECONDS/max(hop_seconds,1e-6)))
    scores=[]
    for lag in range(-max_lag,max_lag+1):
        if lag<0:a=left_values[-lag:];b=right_values[:len(a)]
        elif lag>0:a=left_values[:-lag];b=right_values[lag:]
        else:a=left_values;b=right_values
        if len(a)>=8:scores.append(_robust_similarity(a,b))
    return max(scores) if scores else 0.0


def _common_similarity_details(left,right,start=None,end=None):
    if left.get("status")!="ok" or right.get("status")!="ok":return None
    descriptor_scores={}
    for key in sorted(set(left["envelopes"])&set(right["envelopes"])):
        lt=np.asarray(left["times"]);rt=np.asarray(right["times"])
        la=np.asarray(left["envelopes"][key]);ra=np.asarray(right["envelopes"][key])
        if start is not None and end is not None:
            lo=min(start,end);hi=max(start,end)
            la=la[(lt>=lo)&(lt<=hi)];ra=ra[(rt>=lo)&(rt<=hi)]
        descriptor_scores[key]=round(_lag_tolerant_similarity(la,ra),6)
    if not descriptor_scores:return None
    ordered=sorted(descriptor_scores.values(),reverse=True)
    # Dependence is a warning/soft independence weight. Preserve a strong shared
    # descriptor without allowing one channel alone to prove contamination.
    combined=ordered[0] if len(ordered)==1 else 0.65*ordered[0]+0.35*ordered[1]
    return {"combined_similarity":round(float(combined),6),"descriptor_scores":descriptor_scores,
            "max_lag_seconds":COMMON_SIMILARITY_MAX_LAG_SECONDS}


def _common_similarity(left,right,start=None,end=None):
    details=_common_similarity_details(left,right,start,end)
    return None if details is None else details["combined_similarity"]

def _regional_quality(base_quality,prepared,source_start,source_end,target_start=None,target_end=None,cache=None):
    # Backward-compatible 3C.4.2 call: one interval is used for both
    # windows. Production 3C.4.2a passes separate source/target bounds.
    if target_start is None: target_start=source_start
    if target_end is None: target_end=source_end
    quality={"sources":base_quality.get("sources",{}),"pairwise_similarity":[],"dependence_groups":[]}
    roles=sorted(prepared);adjacency=defaultdict(set)
    source_bounds=(min(source_start,source_end)-REGIONAL_DEPENDENCE_PADDING_SECONDS,max(source_start,source_end)+REGIONAL_DEPENDENCE_PADDING_SECONDS)
    target_bounds=(min(target_start,target_end)-REGIONAL_DEPENDENCE_PADDING_SECONDS,max(target_start,target_end)+REGIONAL_DEPENDENCE_PADDING_SECONDS)
    for i,left in enumerate(roles):
        for right in roles[i+1:]:
            cache_key=(left,right,round(source_bounds[0],3),round(source_bounds[1],3),round(target_bounds[0],3),round(target_bounds[1],3))
            details=None if cache is None else cache.get(cache_key)
            if details is None:
                source_details=_common_similarity_details(prepared[left].get("common_feature",{}),prepared[right].get("common_feature",{}),*source_bounds)
                target_details=_common_similarity_details(prepared[left].get("common_feature",{}),prepared[right].get("common_feature",{}),*target_bounds)
                available=[x for x in (source_details,target_details) if x is not None]
                if available:
                    sim=max(x["combined_similarity"] for x in available)
                    details={"combined_similarity":sim,"source_window":source_details,"target_window":target_details}
                else:
                    sim=_pair_similarity(base_quality,left,right)
                    details={"combined_similarity":sim,"source_window":None,"target_window":None,"fallback":"whole_song"}
                if cache is not None and len(cache)<REGIONAL_SIMILARITY_CACHE_LIMIT:cache[cache_key]=details
            sim=float(details["combined_similarity"])
            level="high" if sim>=HIGH_REDUNDANCY else "moderate" if sim>=MODERATE_REDUNDANCY else "low"
            if sim>=HIGH_REDUNDANCY:adjacency[left].add(right);adjacency[right].add(left)
            quality["pairwise_similarity"].append({"sources":[left,right],"observed_similarity":round(sim,6),"dependence":level,
                "scope":"regional_common_feature","source_window":details.get("source_window"),"target_window":details.get("target_window"),
                "interpretation":"soft_independence_weight_not_instrument_identity"})
    seen=set()
    for role in roles:
        if role in seen:continue
        stack=[role];group=[]
        while stack:
            item=stack.pop()
            if item in seen:continue
            seen.add(item);group.append(item);stack.extend(adjacency[item]-seen)
        quality["dependence_groups"].append(sorted(group))
    return quality

def source_quality(prepared:Mapping[str,Mapping[str,Any]],warp_fn:Callable[[float],float],rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    """Estimate source usability and pairwise dependence from observed features.

    The declared role is retained as metadata only. Independence is derived from
    observed novelty similarity, so contaminated or duplicated stems are not
    counted as fully independent witnesses.
    """
    if not rows:return {"sources":{},"pairwise_similarity":[],"dependence_groups":[]}
    audio_start=float(warp_fn(rows[0]["start"])); audio_end=float(warp_fn(rows[-1]["end"]))
    sources={}
    for role,data in prepared.items():
        feature=data["feature"]; segment=_feature_slice(feature,audio_start,audio_end)
        activity=float(np.mean(segment>np.percentile(segment,65))) if len(segment)>=8 else 0.0
        dynamic=float(np.percentile(segment,90)-np.percentile(segment,10)) if len(segment)>=8 else 0.0
        usability=max(0.0,min(1.0,0.55*min(1.0,dynamic)+0.45*min(1.0,activity/0.35)))
        sources[role]={"declared_role":role,"feature_family":feature.get("feature_family"),
                       "event_count":len(data["events"]),"activity_ratio":round(activity,6),
                       "novelty_dynamic_range":round(dynamic,6),"source_usability":round(usability,6),
                       "role_identity_status":"unverified_declared_label"}
    pairs=[]; adjacency=defaultdict(set)
    roles=sorted(prepared)
    for i,left in enumerate(roles):
        for right in roles[i+1:]:
            sim=_common_similarity(prepared[left].get("common_feature",{}),prepared[right].get("common_feature",{}),audio_start,audio_end)
            comparison="whole_song_common_feature"
            if sim is None:
                a=_feature_slice(prepared[left]["feature"],audio_start,audio_end)
                b=_feature_slice(prepared[right]["feature"],audio_start,audio_end)
                sim=_robust_similarity(a,b);comparison="role_feature_fallback"
            level="high" if sim>=HIGH_REDUNDANCY else "moderate" if sim>=MODERATE_REDUNDANCY else "low"
            if sim>=HIGH_REDUNDANCY: adjacency[left].add(right); adjacency[right].add(left)
            pairs.append({"sources":[left,right],"observed_similarity":round(sim,6),"dependence":level,"scope":comparison})
    seen=set();groups=[]
    for role in roles:
        if role in seen:continue
        stack=[role];group=[]
        while stack:
            item=stack.pop()
            if item in seen:continue
            seen.add(item);group.append(item);stack.extend(adjacency[item]-seen)
        groups.append(sorted(group))
    return {"sources":sources,"pairwise_similarity":pairs,"dependence_groups":groups,
            "note":"Nominal stem labels are not treated as verified instrument identity."}


def _pair_similarity(quality:Mapping[str,Any],left:str,right:str)->float:
    for pair in quality.get("pairwise_similarity") or []:
        if set(pair.get("sources") or [])=={left,right}:return float(pair.get("observed_similarity",0))
    return 0.0


def _effective_source_groups(roles:Sequence[str],quality:Mapping[str,Any])->float:
    if not roles:return 0.0
    ordered=list(dict.fromkeys(roles));effective=1.0
    for i,role in enumerate(ordered[1:],1):
        redundancy=max((_pair_similarity(quality,role,prior) for prior in ordered[:i]),default=0.0)
        effective+=max(0.0,1.0-redundancy)
    return effective


def _score_measure(role,row,events,feature,warp_fn,target_row):
    source_start=float(row["start"]); target_start=float(target_row["start"])
    shift=float(warp_fn(target_start))-float(warp_fn(source_start))
    shifted=lambda t:float(warp_fn(t))+shift
    result=ie.measure_candidates(role,source_start,float(row["end"]),events,shifted,feature,int(row["measure"]))
    eligible=[c for c in result.get("candidates") or [] if c.get("locally_eligible") and not c.get("corridor_saturated")]
    if not eligible:return {"valid":False,"reason":result.get("reason") or "no_eligible_candidate"}
    best=max(eligible,key=lambda c:float(c.get("calibrated_confidence",0)))
    return {"valid":True,"score":float(best.get("calibrated_confidence",0)),"residual_ms":float(best.get("residual_ms",0)),
            "local_rank":best.get("rank"),"target_audio_shift_seconds":round(shift,6)}


def _evaluate_window(role,start,size,shift,rows_by_measure,events,feature,warp_fn):
    results=[]
    for offset in range(size):
        source=rows_by_measure.get(start+offset);target=rows_by_measure.get(start+offset+shift)
        if source is None or target is None:continue
        item=_score_measure(role,source,events,feature,warp_fn,target)
        if item.get("valid"):results.append({"source_measure":start+offset,"target_measure":start+offset+shift,**item})
    scores=[r["score"] for r in results]
    return {"role":role,"valid_measure_count":len(results),"window_score":0.0 if not scores else median(scores),
            "minimum_score":0.0 if not scores else min(scores),"measure_results":results}


def _combine_sources(evaluations,baselines,quality):
    valid=[e for e in evaluations if int(e.get("valid_measure_count",0))>=MIN_VALID_MEASURES_PER_SOURCE]
    roles=sorted(e["role"] for e in valid);effective=_effective_source_groups(roles,quality)
    if not valid:return {"accepted":False,"reason":"insufficient_valid_source_count","source_count":0,"effective_source_groups":0.0,"roles":[]}
    scores=[float(e["window_score"]) for e in valid]
    improvements=[float(e["window_score"])-float((baselines.get(e["role"]) or {}).get("window_score",0)) for e in valid]
    combined=median(scores);improvement=median(improvements)
    if combined<MIN_WINDOW_SCORE:reason="below_absolute_match_score"
    elif improvement<MIN_SCORE_IMPROVEMENT:reason="below_baseline_improvement"
    elif effective<MIN_EFFECTIVE_SOURCE_GROUPS:reason="insufficient_effective_independence"
    else:reason=None
    return {"accepted":reason is None,"reason":reason,"source_count":len(valid),"roles":roles,
            "effective_source_groups":round(effective,6),"combined_score":round(combined,6),
            "median_improvement_over_local_baseline":round(improvement,6),"source_evaluations":list(valid)}




def _combine_roles(evaluations, baselines):
    """Compatibility wrapper for the original Stage 3C.4 public helper.

    Production 3C.4.1 calls ``_combine_sources`` with measured source-dependence
    information. Older tests do not provide that information, so this wrapper
    treats distinct supplied roles as independent, matching the former API.
    """
    roles = sorted({item.get("role") for item in evaluations if item.get("role")})
    quality = {
        "pairwise_similarity": [
            {"sources": [left, right], "observed_similarity": 0.0}
            for index, left in enumerate(roles)
            for right in roles[index + 1:]
        ]
    }
    result = _combine_sources(evaluations, baselines, quality)
    # Preserve the legacy field name while retaining the new diagnostics.
    result["role_count"] = result.get("source_count", 0)
    return result


def _cluster_hypotheses(raw:Sequence[Mapping[str,Any]])->list[dict[str,Any]]:
    clusters=[]
    for hypothesis in sorted(raw,key=lambda h:(h["source_start_measure"],h["measure_shift"])):
        match=None
        for cluster in clusters:
            overlapping=not (hypothesis["source_end_measure"]<cluster["source_start_measure"]-1 or hypothesis["source_start_measure"]>cluster["source_end_measure"]+1)
            nearby=abs(hypothesis["measure_shift"]-cluster["representative_measure_shift"])<=SHIFT_CLUSTER_TOLERANCE_MEASURES
            if overlapping and nearby:match=cluster;break
        if match is None:
            match={"members":[],"source_start_measure":hypothesis["source_start_measure"],"source_end_measure":hypothesis["source_end_measure"],
                   "representative_measure_shift":hypothesis["measure_shift"]};clusters.append(match)
        match["members"].append(dict(hypothesis));match["source_start_measure"]=min(match["source_start_measure"],hypothesis["source_start_measure"])
        match["source_end_measure"]=max(match["source_end_measure"],hypothesis["source_end_measure"])
        match["representative_measure_shift"]=round(median([m["measure_shift"] for m in match["members"]]))
    output=[]
    for index,c in enumerate(clusters,1):
        members=c["members"];scores=sorted((float(m["combined_score"]) for m in members),reverse=True)
        best=scores[0];second=scores[1] if len(scores)>1 else 0.0
        roles=sorted({r for m in members for r in m.get("roles") or []})
        shifts=[int(m["measure_shift"]) for m in members]
        output.append({"cluster_id":f"structural_cluster_{index:03d}","type":"structural_correspondence_family",
                       "source_start_measure":c["source_start_measure"],"source_end_measure":c["source_end_measure"],
                       "representative_measure_shift":c["representative_measure_shift"],"measure_shift_range":[min(shifts),max(shifts)],
                       "median_audio_shift_seconds":round(median([float(m["audio_shift_seconds"]) for m in members]),6),
                       "supporting_window_count":len(members),"window_sizes":sorted({int(m["window_size"]) for m in members}),
                       "supporting_roles":roles,"effective_source_groups":round(max(float(m.get("effective_source_groups",0)) for m in members),6),
                       "best_score":round(best,6),"second_member_score":round(second,6),
                       "member_margin":round(best-second,6),"median_baseline_improvement":round(median([float(m["median_improvement_over_local_baseline"]) for m in members]),6),
                       "persistence_status":"multi_window" if len(members)>=MIN_CLUSTER_WINDOWS else "single_window",
                       "uniqueness_status":"distinctive_within_cluster" if best-second>=MIN_UNIQUENESS_MARGIN else "ambiguous_within_cluster",
                       "members":members})
    return output


def _cluster_overlap(left,right):
    intersection=max(0,min(left["source_end_measure"],right["source_end_measure"])-max(left["source_start_measure"],right["source_start_measure"])+1)
    smaller=min(left["source_end_measure"]-left["source_start_measure"]+1,right["source_end_measure"]-right["source_start_measure"]+1)
    return intersection/smaller if smaller else 0.0

def _rank_competing_clusters(clusters):
    for cluster in clusters:
        competitors=[other for other in clusters if other is not cluster and _cluster_overlap(cluster,other)>=0.5 and abs(other["representative_measure_shift"]-cluster["representative_measure_shift"])>SHIFT_CLUSTER_TOLERANCE_MEASURES]
        cluster["shift_score_margin"]=cluster.pop("member_margin",0.0)
        cluster["shift_precision"]="fine" if cluster["shift_score_margin"]>=MIN_UNIQUENESS_MARGIN else "coarse"
        if not competitors:
            cluster["competing_cluster_margin"]=None;cluster["occurrence_uniqueness"]="uniqueness_not_established"
        else:
            competitor=max(float(c["best_score"]) for c in competitors);margin=float(cluster["best_score"])-competitor
            cluster["competing_cluster_margin"]=round(margin,6)
            cluster["occurrence_uniqueness"]="locally_distinctive" if margin>=MIN_UNIQUENESS_MARGIN else "ambiguous_repeated_occurrence"
        cluster.pop("uniqueness_status",None)
    return list(clusters)

def _fuzzy_ramp(value,low,high):
    if value is None:return 0.0
    if high<=low:return 1.0 if value>=high else 0.0
    return round(max(0.0,min(1.0,(float(value)-low)/(high-low))),6)


def _sequence_window_score(start,size,shift,prepared,rows_by_measure,warp_fn,base_quality,cache):
    maximum=max(rows_by_measure) if rows_by_measure else 0
    minimum=min(rows_by_measure) if rows_by_measure else 0
    if start<minimum or start+size-1>maximum or start+shift<minimum or start+size-1+shift>maximum:return None
    baselines={role:_evaluate_window(role,start,size,0,rows_by_measure,data["events"],data["feature"],warp_fn) for role,data in prepared.items()}
    evaluations=[_evaluate_window(role,start,size,shift,rows_by_measure,data["events"],data["feature"],warp_fn) for role,data in prepared.items()]
    source=rows_by_measure[start];source_end=rows_by_measure[start+size-1];target=rows_by_measure[start+shift];target_end=rows_by_measure[start+shift+size-1]
    regional=_regional_quality(base_quality,prepared,float(warp_fn(source["start"])),float(warp_fn(source_end["end"])),
        float(warp_fn(target["start"])),float(warp_fn(target_end["end"])),cache)
    combined=_combine_sources(evaluations,baselines,regional)
    # Sequence diagnostics keep strong one-source observations visible. Source
    # independence affects confidence, not whether a window is observed.
    if combined.get("combined_score") is None:return None
    return {"start_measure":start,"end_measure":start+size-1,"measure_shift":shift,
        "combined_score":combined.get("combined_score"),"improvement":combined.get("median_improvement_over_local_baseline"),
        "effective_source_groups":combined.get("effective_source_groups"),"roles":combined.get("roles",[]),
        "formal_gate_passed":bool(combined.get("accepted")),"rejection_reason":combined.get("reason")}


def _validate_structural_cluster(cluster,all_clusters,prepared,rows_by_measure,warp_fn,base_quality,cache):
    shift=int(cluster["representative_measure_shift"]);size=PERSISTENCE_WINDOW_MEASURES
    competitors={0,shift}
    for other in all_clusters:
        if other is cluster:continue
        if _cluster_overlap(cluster,other)>=0.5:competitors.add(int(other["representative_measure_shift"]))
    start=max(min(rows_by_measure),int(cluster["source_start_measure"])-PERSISTENCE_CONTEXT_WINDOWS*PERSISTENCE_STEP_MEASURES)
    end=min(max(rows_by_measure)-size+1,int(cluster["source_end_measure"])+PERSISTENCE_CONTEXT_WINDOWS*PERSISTENCE_STEP_MEASURES)
    starts=list(range(start,end+1,PERSISTENCE_STEP_MEASURES))
    sequence=[]
    for window_start in starts:
        scores=[]
        for candidate_shift in sorted(competitors):
            record=_sequence_window_score(window_start,size,candidate_shift,prepared,rows_by_measure,warp_fn,base_quality,cache)
            if record:scores.append(record)
        if not scores:continue
        scores.sort(key=lambda x:(_number(x.get("combined_score")),_number(x.get("improvement"))),reverse=True)
        winner=scores[0];family=[x for x in scores if abs(int(x["measure_shift"])-shift)<=SHIFT_CLUSTER_TOLERANCE_MEASURES]
        family_best=max(family,key=lambda x:_number(x.get("combined_score"))) if family else None
        baseline=next((x for x in scores if x["measure_shift"]==0),None)
        sequence.append({"window_start_measure":window_start,"window_end_measure":window_start+size-1,
            "winning_shift":winner["measure_shift"],"winning_score":winner["combined_score"],
            "family_score":None if family_best is None else family_best["combined_score"],
            "family_improvement":None if family_best is None else family_best["improvement"],
            "baseline_score":None if baseline is None else baseline["combined_score"],"evaluated_shifts":scores})
    favors=[item for item in sequence if item["family_score"] is not None and int(item["winning_shift"])!=0 and abs(int(item["winning_shift"])-shift)<=SHIFT_CLUSTER_TOLERANCE_MEASURES and _number(item["family_improvement"])>0]
    longest=0;current=0;previous=None
    for item in sequence:
        favorable=item in favors
        if favorable and (previous is None or item["window_start_measure"]-previous==PERSISTENCE_STEP_MEASURES):current+=1
        elif favorable:current=1
        else:current=0
        longest=max(longest,current);previous=item["window_start_measure"]
    ratio=len(favors)/len(sequence) if sequence else 0.0
    improvements=[_number(x["family_improvement"]) for x in favors]
    # Boundary candidates are evaluated from the observed sequence, not assumed
    # to coincide with the original discovery window.
    boundary_candidates=[]
    centre=int(cluster["source_start_measure"])
    for boundary in range(max(min(rows_by_measure),centre-BOUNDARY_SEARCH_RADIUS_MEASURES),min(max(rows_by_measure),centre+BOUNDARY_SEARCH_RADIUS_MEASURES)+1):
        before=[x for x in sequence if x["window_end_measure"]<boundary and x["family_improvement"] is not None]
        after=[x for x in sequence if x["window_start_measure"]>=boundary and x["family_improvement"] is not None]
        if not before or not after:continue
        pre=median([_number(x["family_improvement"]) for x in before]);post=median([_number(x["family_improvement"]) for x in after])
        boundary_candidates.append({"boundary_measure":boundary,"pre_shift_improvement":round(pre,6),"post_shift_improvement":round(post,6),"boundary_contrast":round(post-pre,6)})
    boundary_candidates.sort(key=lambda x:x["boundary_contrast"],reverse=True)
    best_boundary=boundary_candidates[0] if boundary_candidates else None
    if not sequence:classification="insufficient_sequence_coverage"
    elif longest>=MIN_PERSISTENT_WINDOWS and ratio>=MIN_PERSISTENT_RATIO and best_boundary and best_boundary["boundary_contrast"]>0:classification="persistent_displacement_candidate"
    elif longest>=MIN_PERSISTENT_WINDOWS and ratio>=MIN_PERSISTENT_RATIO:classification="persistent_correspondence_without_boundary"
    elif len(competitors)>2 and ratio<MIN_PERSISTENT_RATIO:classification="ambiguous_repeated_material"
    elif len(favors)<=1:classification="isolated_correspondence"
    else:classification="contradictory_shift_sequence"
    correspondence=_fuzzy_ramp(cluster.get("best_score"),0.50,0.85)
    persistence=_fuzzy_ramp(ratio,0.20,0.80)
    distinction=_fuzzy_ramp(cluster.get("competing_cluster_margin"),0.0,0.12) if cluster.get("competing_cluster_margin") is not None else 0.25
    independence=_fuzzy_ramp(cluster.get("effective_source_groups"),1.0,2.5)
    boundary=_fuzzy_ramp(None if best_boundary is None else best_boundary.get("boundary_contrast"),0.0,0.25)
    priority=round(0.30*correspondence+0.25*persistence+0.15*distinction+0.15*independence+0.15*boundary,6)
    # Anchor suitability deliberately remains conservative and shadow-only.
    suitability=round(min(correspondence,persistence,max(distinction,0.25),max(boundary,0.15))*independence,6)
    return {"status":"diagnostic_only","classification":classification,"representative_measure_shift":shift,
        "evaluated_window_count":len(sequence),"favorable_window_count":len(favors),"persistent_support_ratio":round(ratio,6),
        "longest_favorable_run_windows":longest,"longest_favorable_run_measures":0 if longest==0 else size+(longest-1)*PERSISTENCE_STEP_MEASURES,
        "median_persistent_improvement":None if not improvements else round(median(improvements),6),
        "minimum_persistent_improvement":None if not improvements else round(min(improvements),6),
        "competing_shifts":sorted(competitors),"sequence":sequence,"best_boundary":best_boundary,"boundary_candidates":boundary_candidates[:5],
        "fuzzy_assessment":{"shadow_mode":True,"correspondence_strength":correspondence,"persistence_strength":persistence,
            "occurrence_distinction":distinction,"source_independence":independence,"boundary_support":boundary,
            "structural_validation_priority":priority,"automatic_anchor_suitability":suitability,
            "note":"Fuzzy values prioritize validation only and do not alter crisp decisions or timing."}}


def analyse(role_inputs,warp_fn,nominal_downbeats,fusion_report,sr=22050):
    rows=_measure_rows(nominal_downbeats);rows_by_measure={r["measure"]:r for r in rows}
    prepared={}
    for role,values in role_inputs.items():
        if not values:continue
        events=_event_times(values.get("events") or []);feature=ie.extract_role_feature(values.get("audio_path"),role,sr=sr)
        if events and feature.get("status")=="ok":prepared[role]={"events":events,"feature":feature,"audio_path":values.get("audio_path"),"common_feature":_common_source_feature(values.get("audio_path"))}
    quality=source_quality(prepared,warp_fn,rows)
    raw=[];attempts=0;rejections=defaultdict(int);near_misses=defaultdict(list);regional_similarity_cache={}
    for gap in fusion_report.get("coverage_gaps") or []:
        if not gap.get("long_gap"):continue
        gap_raw=[]
        for size in WINDOW_SIZES:
            for start in _window_starts(gap,rows,size):
                baselines={role:_evaluate_window(role,start,size,0,rows_by_measure,data["events"],data["feature"],warp_fn) for role,data in prepared.items()}
                candidates=[]
                for shift in MEASURE_SHIFTS:
                    if not rows_by_measure or start+shift<min(rows_by_measure) or start+size-1+shift>max(rows_by_measure):continue
                    evals=[_evaluate_window(role,start,size,shift,rows_by_measure,data["events"],data["feature"],warp_fn) for role,data in prepared.items()]
                    source_row=rows_by_measure[start];target_row=rows_by_measure[start+shift];source_end_row=rows_by_measure[start+size-1];target_end_row=rows_by_measure[start+shift+size-1]
                    regional_quality=_regional_quality(quality,prepared,float(warp_fn(source_row["start"])),float(warp_fn(source_end_row["end"])),
                        float(warp_fn(target_row["start"])),float(warp_fn(target_end_row["end"])),regional_similarity_cache)
                    combined=_combine_sources(evals,baselines,regional_quality);attempts+=1
                    if not combined.get("accepted"):
                        reason=combined.get("reason") or "rejected"; rejections[reason]+=1
                        item={"source_start_measure":start,"source_end_measure":start+size-1,"window_size":size,"measure_shift":shift,
                              "combined_score":combined.get("combined_score"),"median_improvement_over_local_baseline":combined.get("median_improvement_over_local_baseline"),
                              "effective_source_groups":combined.get("effective_source_groups"),"roles":combined.get("roles",[]),"rejected_by":reason}
                        if item["roles"] or item["combined_score"] is not None:
                            bucket=near_misses[reason];bucket.append(item);bucket.sort(key=lambda x:(_number(x.get("combined_score")),_number(x.get("median_improvement_over_local_baseline"))),reverse=True);del bucket[NEAR_MISS_LIMIT_PER_REASON:]
                        continue
                    source=rows_by_measure[start];target=rows_by_measure[start+shift]
                    audio_shift=float(warp_fn(float(target["start"])))-float(warp_fn(float(source["start"])))
                    candidates.append({"hypothesis_id":f"gap{gap['start_playback_measure']}-{gap['end_playback_measure']}.m{start}.w{size}.shift{shift:+d}",
                                       "type":"whole_measure_correspondence","source_start_measure":start,"source_end_measure":start+size-1,
                                       "target_start_measure":start+shift,"target_end_measure":start+shift+size-1,"window_size":size,
                                       "measure_shift":shift,"audio_shift_seconds":round(audio_shift,6),"gap":dict(gap),"regional_source_dependence":regional_quality.get("pairwise_similarity",[]),**combined})
                candidates.sort(key=lambda h:(h["combined_score"],h["median_improvement_over_local_baseline"]),reverse=True)
                if candidates:
                    best=candidates[0]["combined_score"]
                    for c in candidates:
                        c["window_uniqueness_margin"]=round(best-(candidates[1]["combined_score"] if len(candidates)>1 and c is candidates[0] else candidates[0]["combined_score"]),6) if len(candidates)>1 else round(best,6)
                    gap_raw.extend(candidates[:3])
        gap_raw.sort(key=lambda h:(h["combined_score"],h["median_improvement_over_local_baseline"]),reverse=True)
        raw.extend(gap_raw[:MAX_RAW_HYPOTHESES_PER_GAP])
    clusters=_rank_competing_clusters(_cluster_hypotheses(raw))
    by_gap=defaultdict(list)
    for c in clusters:
        member=c["members"][0];gap=member["gap"];by_gap[(gap["start_playback_measure"],gap["end_playback_measure"])].append(c)
    retained=[]
    for items in by_gap.values():
        items.sort(key=lambda c:(c["persistence_status"]=="multi_window",c["occurrence_uniqueness"]=="locally_distinctive",c["best_score"],c["median_baseline_improvement"]),reverse=True)
        retained.extend(items[:MAX_CLUSTERS_PER_GAP])
    for cluster in retained:
        cluster["sequence_validation"]=_validate_structural_cluster(cluster,retained,prepared,rows_by_measure,warp_fn,quality,regional_similarity_cache)
    return {"schema":"feedpak.structural_search","schema_version":SCHEMA_VERSION,"phase":"3C.4.3","timing_changes_applied":False,
            "policy":{"window_sizes_measures":list(WINDOW_SIZES),"measure_shifts":list(MEASURE_SHIFTS),
                      "source_labels_are_unverified":True,"dependence_aware_support":True,"minimum_effective_source_groups":MIN_EFFECTIVE_SOURCE_GROUPS,
                      "note":"Correspondence families only. No temporary warp or structural anchor is generated."},
            "prepared_roles":sorted(prepared),"source_quality":quality,"attempt_count":attempts,"rejected_candidate_summary":dict(rejections),"near_miss_candidates":{k:v for k,v in near_misses.items()},
            "raw_hypothesis_count":len(raw),"hypotheses":raw,"clusters":retained,
            "summary":{"long_gap_count":sum(bool(g.get("long_gap")) for g in fusion_report.get("coverage_gaps") or []),
                       "raw_hypothesis_count":len(raw),"retained_cluster_count":len(retained),"multi_window_cluster_count":sum(c["persistence_status"]=="multi_window" for c in retained),
                       "unique_cluster_count":sum(c["occurrence_uniqueness"]=="locally_distinctive" for c in retained),"prepared_source_count":len(prepared)}}


def append_to_alignment_proposals(proposal_report,structural_report,beatmap):
    proposals=proposal_report.setdefault("proposals",[])
    # Backward-compatible handling for synthetic Stage 3C.4 unit-test records.
    # Production 3C.4.1 reports always contain the explicit ``clusters`` key.
    if "clusters" not in structural_report and structural_report.get("hypotheses"):
        for hypothesis in structural_report.get("hypotheses") or []:
            proposals.append({"proposal_id":hypothesis.get("hypothesis_id","legacy_structural")+".proposal",
                              "method":"structural_measure_shift","status":"diagnostic_only","timing_changes_applied":False,
                              "region":{"start_playback_measure":hypothesis.get("source_start_measure"),"end_playback_measure":hypothesis.get("source_end_measure")},
                              "controls":[],"structural_hypothesis":hypothesis,
                              "complexity":{"control_count":0,"structural_jump_count":0},
                              "evidence_fit":{},"validation":{"status":"requires_two_sided_structural_validation","error_count":0,"warning_count":1},
                              "comparison_score":{"total":0.0},"local_recommendation":"carry_forward_structural_search",
                              "song_level_eligibility":"structural_search_required"})
    for cluster in structural_report.get("clusters") or []:
        score=max(0.0,100*float(cluster["median_baseline_improvement"])-STRUCTURAL_CONTROL_PENALTY)
        proposals.append({"proposal_id":cluster["cluster_id"]+".correspondence","method":"structural_correspondence_family",
                          "status":"diagnostic_correspondence_only","timing_changes_applied":False,
                          "region":{"start_playback_measure":cluster["source_start_measure"],"end_playback_measure":cluster["source_end_measure"]},
                          "controls":[],"structural_cluster":cluster,"complexity":{"control_count":0,"structural_jump_count":0},
                          "evidence_fit":{"observation_count":cluster["supporting_window_count"],"structural_window_score":cluster["best_score"],
                                          "structural_improvement":cluster["median_baseline_improvement"]},
                          "validation":{"status":"requires_persistence_and_two_sided_validation","error_count":0,"warning_count":1,
                                        "warnings":[{"kind":"correspondence_is_not_a_timing_map"}]},
                          "comparison_score":{"fit_improvement_ms":round(100*cluster["median_baseline_improvement"],3),
                                              "control_complexity_penalty":STRUCTURAL_CONTROL_PENALTY,"total":round(score,3)},
                          "local_recommendation":"carry_forward_two_sided_review" if (cluster.get("sequence_validation") or {}).get("classification")=="persistent_displacement_candidate" else "retain_as_diagnostic_correspondence",
                          "song_level_eligibility":"structural_validation_required"})
    legacy_structural_input = "clusters" not in structural_report and bool(structural_report.get("hypotheses"))
    proposal_report["phase"]="3C.4" if legacy_structural_input else "3C.4.3"
    proposal_report["schema_version"]=3 if legacy_structural_input else 7
    proposal_report["structural_search_summary"]=dict(structural_report.get("summary") or {});proposal_report["proposal_count"]=len(proposals)
    ranked=sorted(proposals,key=lambda p:float((p.get("comparison_score") or {}).get("total",0)),reverse=True)
    proposal_report["ranking"]=[{"proposal_id":p["proposal_id"],"method":p["method"],"score":p.get("comparison_score",{}),"song_level_eligibility":p.get("song_level_eligibility")} for p in ranked]
    structural_methods={"structural_correspondence_family","structural_measure_shift"}
    proposal_report["timing_proposal_ranking"]=[item for item in proposal_report["ranking"] if item["method"] not in structural_methods]
    proposal_report["structural_correspondence_ranking"]=[item for item in proposal_report["ranking"] if item["method"] in structural_methods]
    proposal_report["recommended_proposal_id"]="global.baseline";proposal_report["recommendation"]="retain_baseline"
    proposal_report.setdefault("notes",[]).append("Structural outputs are clustered correspondence families, not temporary warps. Baseline remains recommended.")
    return proposal_report
