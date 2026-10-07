"""Chart-guided continuous structural anchor discovery.

Candidate generation is driven by the configured score. For each entrance or
restart after at least one complete local measure of actual score silence, the
corresponding isolated stem is inspected around the baseline-predicted time.
Strong evidence on that stem is sufficient by itself. This module remains
strictly diagnostic: it never modifies a production timing warp.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import os
import hashlib
import numpy as np
try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


@dataclass(frozen=True)
class ContinuousAnchorConfig:
    audio_search_radius_seconds: float = 6.0
    audio_pre_window_seconds: float = 1.0
    audio_post_window_seconds: float = 0.45
    minimum_transition_confidence: float = 0.58
    strong_single_stem_confidence: float = 0.72
    maximum_matches_per_score_event: int = 5
    minimum_measure_fraction: float = 1.0
    dense_candidates_enabled: bool = True
    dense_measure_stride: int = 2
    dense_minimum_events_per_measure: int = 2
    dense_minimum_spacing_seconds: float = 1.0
    dense_minimum_score_confidence: float = 0.38
    dense_maximum_candidates_per_source: int = 96
    dense_production_enabled: bool = False
    fingerprint_context_measures: float = 1.0
    fingerprint_bins: int = 48
    minimum_rhythm_similarity: float = 0.48
    minimum_chroma_similarity: float = 0.50
    minimum_joint_identity: float = 0.52
    minimum_posterior: float = 0.12
    minimum_absolute_margin: float = 0.025
    minimum_score_ratio: float = 1.08
    temporal_prior_local_sigma_seconds: float = 0.45
    temporal_prior_wide_sigma_seconds: float = 2.50
    temporal_prior_wide_weight: float = 0.12
    candidate_nms_seconds: float = 0.30
    feature_cache_version: int = 2
    hop_length: int = 256


def _time(event):
    return float(event.get("t_gp", event.get("time", event)) if isinstance(event, dict) else event)


def _end(event):
    if not isinstance(event, dict):
        return float(event)
    start = _time(event)
    if event.get("end_gp") is not None:
        return max(start, float(event["end_gp"]))
    for key in ("sus", "duration", "d"):
        if event.get(key) is not None:
            return max(start, start + float(event[key]))
    return start


def _measures(measure_downbeats):
    rows = sorted(measure_downbeats or [], key=lambda x: float(x["t_gp"]))
    out = []
    for index, row in enumerate(rows):
        start = float(row["t_gp"])
        end = float(rows[index + 1]["t_gp"]) if index + 1 < len(rows) else None
        if end is None or end <= start:
            continue
        out.append({
            "start": start, "end": end, "duration": end - start,
            "playback_bar": int(row["measure"]),
            "source_bar": int(row.get("source_bar", row["measure"])),
            "occurrence": int(row.get("occurrence", 1)),
            "ts_num": int(row.get("ts_num", 4)),
        })
    return out


def _measure_at(value, measures):
    if not measures:
        return None
    index = int(np.searchsorted([x["start"] for x in measures], float(value), side="right") - 1)
    return measures[max(0, min(index, len(measures) - 1))]


def _position(value, measures):
    row = _measure_at(value, measures)
    if not row:
        return {}
    beat = 1.0 + (float(value) - row["start"]) / row["duration"] * max(1, row["ts_num"])
    return {
        "playback_bar": row["playback_bar"], "source_bar": row["source_bar"],
        "occurrence": row["occurrence"], "beat": round(beat, 4),
    }


def _sounding_regions(events, merge_tolerance=0.025):
    intervals = sorted((_time(x), _end(x)) for x in (events or []))
    regions = []
    for start, end in intervals:
        if not regions or start > regions[-1][1] + merge_tolerance:
            regions.append([start, end])
        else:
            regions[-1][1] = max(regions[-1][1], end)
    return regions


def _dense_measure_candidates(source, events, measures, cfg):
    if not cfg.dense_candidates_enabled or not measures:
        return []
    onsets = sorted({_time(x) for x in (events or [])})
    if not onsets:
        return []
    rows=[]; signatures=[]
    for measure in measures:
        local=[t for t in onsets if measure["start"] <= t < measure["end"]]
        duration=max(measure["duration"],1e-9)
        offsets=tuple(round((t-measure["start"])/duration*8)/8 for t in local[:8])
        signature=(min(len(local),8),offsets)
        rows.append((measure,local,signature))
        if local: signatures.append(signature)
    frequencies={x:signatures.count(x) for x in set(signatures)}
    selected=[]; last_time=float("-inf"); stride=max(1,int(cfg.dense_measure_stride))
    for index,(measure,local,signature) in enumerate(rows):
        if index % stride or len(local) < max(1,int(cfg.dense_minimum_events_per_measure)): continue
        score_time=local[0]
        if score_time-last_time < cfg.dense_minimum_spacing_seconds: continue
        previous=max((t for t in onsets if t < score_time),default=None)
        gap=0.0 if previous is None else score_time-previous
        density=min(1.0,len(local)/6.0)
        downbeat=max(0.0,1.0-(score_time-measure["start"])/max(measure["duration"]*.35,1e-9))
        gap_support=min(1.0,gap/max(measure["duration"]*.25,1e-9))
        uniqueness=1.0/max(1,frequencies.get(signature,1))
        confidence=.35*density+.25*downbeat+.20*gap_support+.20*uniqueness
        if confidence < cfg.dense_minimum_score_confidence: continue
        selected.append({"id":f"{source}-dense-measure-{measure['playback_bar']}","source":source,
            "kind":"dense_active_measure_landmark","score_time":round(score_time,6),
            "score_region_end":round(local[-1],6),"previous_sounding_end":None if previous is None else round(previous,6),
            "score_silence_seconds":None if previous is None else round(gap,6),"required_silence_seconds":None,
            "score_confidence":round(confidence,4),"score_evidence":{"events_in_measure":len(local),
            "downbeat_proximity":round(downbeat,4),"preceding_gap_support":round(gap_support,4),
            "rhythmic_pattern_uniqueness":round(uniqueness,4),"measure_stride":stride},
            "score_position":_position(score_time,measures)})
        last_time=score_time
        if len(selected)>=int(cfg.dense_maximum_candidates_per_source): break
    return selected


def build_score_candidates(score_events, measure_downbeats, tempo_bpm=None, config=None):
    cfg=config or ContinuousAnchorConfig(); measures=_measures(measure_downbeats)
    candidates=[]; structural_count=0; dense_count=0
    for source,events in sorted((score_events or {}).items()):
        if source == "full" or not events: continue
        regions=_sounding_regions(events); structural_times=[]
        for index,(start,end) in enumerate(regions):
            previous_end=regions[index-1][1] if index else None
            silence=None if previous_end is None else start-previous_end
            local=_measure_at(start,measures)
            required=((local or {}).get("duration") or (60.0/max(float(tempo_bpm or 120),1e-9)*4.0))*cfg.minimum_measure_fraction
            first=index==0
            if not first and silence+1e-9 < required: continue
            kind="instrument_entrance" if first else "restart_after_measure_silence"
            candidates.append({"id":f"{source}-{kind}-{index+1}","source":source,"kind":kind,
                "score_time":round(start,6),"score_region_end":round(end,6),
                "previous_sounding_end":None if previous_end is None else round(previous_end,6),
                "score_silence_seconds":None if silence is None else round(silence,6),
                "required_silence_seconds":round(required,6),"score_confidence":1.0,
                "score_evidence":{"release_aware_structural_event":True},"score_position":_position(start,measures)})
            structural_times.append(start); structural_count+=1
        for item in _dense_measure_candidates(source,events,measures,cfg):
            if any(abs(float(item["score_time"])-t)<=.15 for t in structural_times): continue
            candidates.append(item); dense_count+=1
    candidates.sort(key=lambda x:(x["score_time"],x["source"],x["kind"]))
    return {"version":3,"status":"diagnostic_only","config":asdict(cfg),"candidate_count":len(candidates),
            "structural_candidate_count":structural_count,"dense_candidate_count":dense_count,"candidates":candidates}

def _feature_cache_path(audio_path, sr, hop, cache_dir, version):
    if not cache_dir or not audio_path:
        return None
    digest=hashlib.sha256()
    with open(audio_path,"rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):
            digest.update(chunk)
    token="|".join((digest.hexdigest(),str(sr),str(hop),str(version)))
    name=hashlib.sha256(token.encode("utf-8")).hexdigest()+".npz"
    os.makedirs(cache_dir,exist_ok=True)
    return os.path.join(cache_dir,name)


def _features(audio_path, sr, hop, cache_dir=None, cache_version=2):
    """Load one stem once and cache rhythm, energy, onset and chroma features."""
    if librosa is None or not audio_path:
        return None
    cache_path=_feature_cache_path(audio_path,sr,hop,cache_dir,cache_version)
    if cache_path and os.path.isfile(cache_path):
        try:
            data=np.load(cache_path,allow_pickle=False)
            return {"rms":data["rms"],"onset":data["onset"],"times":data["times"],
                    "chroma":data["chroma"],"chroma_times":data["chroma_times"],
                    "sr":float(data["sr"]),"duration":float(data["duration"]),
                    "cache":"disk_hit","cache_path":cache_path}
        except Exception:
            pass
    y,actual_sr=librosa.load(audio_path,sr=sr,mono=True)
    if not len(y):
        return None
    rms=librosa.feature.rms(y=y,hop_length=hop)[0]
    onset=librosa.onset.onset_strength(y=y,sr=actual_sr,hop_length=hop)
    count=min(len(rms),len(onset))
    times=librosa.frames_to_time(np.arange(count),sr=actual_sr,hop_length=hop)
    harmonic=librosa.effects.harmonic(y,margin=3.0)
    chroma=librosa.feature.chroma_cqt(y=harmonic,sr=actual_sr,hop_length=hop)
    chroma_times=librosa.frames_to_time(np.arange(chroma.shape[1]),sr=actual_sr,hop_length=hop)
    chroma=chroma/np.maximum(np.linalg.norm(chroma,axis=0,keepdims=True),1e-9)
    result={"rms":rms[:count],"onset":onset[:count],"times":times,"chroma":chroma,
            "chroma_times":chroma_times,"sr":actual_sr,"duration":len(y)/actual_sr,
            "cache":"computed","cache_path":cache_path}
    if cache_path:
        temporary=cache_path+".tmp.npz"
        np.savez_compressed(temporary,rms=result["rms"],onset=result["onset"],times=times,
                            chroma=chroma,chroma_times=chroma_times,sr=np.asarray(actual_sr),
                            duration=np.asarray(result["duration"]))
        os.replace(temporary,cache_path)
    return result


def _cosine(a,b):
    a=np.asarray(a,dtype=float).ravel(); b=np.asarray(b,dtype=float).ravel()
    if len(a)!=len(b) or not len(a): return 0.0
    den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return 0.0 if den<=1e-12 else float(np.clip(np.dot(a,b)/den,0,1))


def _resample(values,old_times,new_times):
    values=np.asarray(values,dtype=float)
    if values.ndim==1:
        return np.interp(new_times,old_times,values,left=0,right=0)
    return np.vstack([np.interp(new_times,old_times,row,left=0,right=0) for row in values])


def build_score_fingerprint(events,center,half_window,bins=48):
    """Build synchronized rhythm and sustained pitch-class fingerprints."""
    start=float(center)-float(half_window); end=float(center)+float(half_window)
    edges=np.linspace(start,end,int(bins)+1); frame_times=(edges[:-1]+edges[1:])/2
    rhythm=np.zeros(int(bins)); chroma=np.zeros((12,int(bins)))
    pitched=0
    for event in events or []:
        t=_time(event); finish=max(t,_end(event)); midi=event.get("midi") if isinstance(event,dict) else None
        if start<=t<end:
            j=min(int(bins)-1,max(0,int((t-start)/max(end-start,1e-9)*bins)))
            rhythm[j]+=1.0
        if midi is not None and finish>=start and t<=end:
            pitched+=1
            active=(frame_times>=max(start,t)) & (frame_times<=min(end,max(finish,t+(end-start)/bins)))
            chroma[int(midi)%12,active]+=1.0
    if np.max(rhythm)>0: rhythm/=np.max(rhythm)
    chroma=chroma/np.maximum(np.linalg.norm(chroma,axis=0,keepdims=True),1e-9)
    return {"start":start,"end":end,"frame_times":frame_times,"rhythm":rhythm,
            "chroma":chroma,"pitched_event_count":pitched,"event_count":int(sum(rhythm>0))}


def _temporal_prior(residual,cfg):
    local=math.exp(-.5*(residual/max(cfg.temporal_prior_local_sigma_seconds,1e-9))**2)
    wide=math.exp(-.5*(residual/max(cfg.temporal_prior_wide_sigma_seconds,1e-9))**2)
    w=float(cfg.temporal_prior_wide_weight)
    return (1-w)*local+w*wide


def inspect_dense_fingerprint(features,score_events,item,prediction,measure_duration,cfg):
    """Search broadly, but validate every dense match with rhythm AND chroma."""
    if features is None or item.get("source")=="drums": return []
    half=max(.75,min(8.0,float(measure_duration)*float(cfg.fingerprint_context_measures)))
    score=build_score_fingerprint(score_events,float(item["score_time"]),half,cfg.fingerprint_bins)
    if score["pitched_event_count"]<2 or score["event_count"]<2: return []
    candidate_times=features["times"][(features["times"]>=prediction-cfg.audio_search_radius_seconds) &
                                      (features["times"]<=prediction+cfg.audio_search_radius_seconds)]
    if not len(candidate_times): return []
    step=max(1,int(round(.04/(cfg.hop_length/features["sr"]))))
    rows=[]
    score_grid=np.linspace(-half,half,int(cfg.fingerprint_bins))
    for candidate_time in candidate_times[::step]:
        audio_grid=float(candidate_time)+score_grid
        rhythm=_resample(features["onset"],features["times"],audio_grid)
        if np.max(rhythm)>0: rhythm=rhythm/np.max(rhythm)
        chroma=_resample(features["chroma"],features["chroma_times"],audio_grid)
        chroma=chroma/np.maximum(np.linalg.norm(chroma,axis=0,keepdims=True),1e-9)
        rhythm_similarity=_cosine(score["rhythm"],rhythm)
        chroma_similarity=_cosine(score["chroma"],chroma)
        joint=math.sqrt(max(0.0,rhythm_similarity*chroma_similarity))
        residual=float(candidate_time)-float(prediction); prior=_temporal_prior(residual,cfg)
        posterior=joint*prior*float(item.get("score_confidence",1.0))
        rows.append({"time":float(candidate_time),"residual_seconds":residual,
                     "rhythm_similarity":rhythm_similarity,"chroma_similarity":chroma_similarity,
                     "joint_identity":joint,"temporal_prior":prior,"posterior":posterior})
    rows.sort(key=lambda x:(-x["posterior"],abs(x["residual_seconds"])))
    distinct=[]
    for row in rows:
        if all(abs(row["time"]-x["time"])>=cfg.candidate_nms_seconds for x in distinct):
            distinct.append(row)
        if len(distinct)>=max(2,cfg.maximum_matches_per_score_event): break
    if not distinct: return []
    best=distinct[0]; second=distinct[1] if len(distinct)>1 else None
    second_score=0.0 if second is None else second["posterior"]
    margin=best["posterior"]-second_score
    ratio=best["posterior"]/max(second_score,1e-9)
    passed=(best["rhythm_similarity"]>=cfg.minimum_rhythm_similarity and
            best["chroma_similarity"]>=cfg.minimum_chroma_similarity and
            best["joint_identity"]>=cfg.minimum_joint_identity and
            best["posterior"]>=cfg.minimum_posterior and
            margin>=cfg.minimum_absolute_margin and ratio>=cfg.minimum_score_ratio)
    for row in distinct:
        row.update({"confidence":round(row["posterior"],4),"raw_audio_confidence":round(row["joint_identity"],4),
                    "best_second_margin":round(margin,6),"best_second_ratio":round(ratio,6),
                    "joint_validated":bool(passed and row is best),"matching_method":"rhythm_chroma_temporal_prior",
                    "fingerprint_half_window_seconds":round(half,4),
                    "score_pitched_event_count":score["pitched_event_count"]})
        for key in ("time","residual_seconds","rhythm_similarity","chroma_similarity","joint_identity","temporal_prior","posterior"):
            row[key]=round(float(row[key]),6)
    return [distinct[0]] if passed else []

def _candidate_at(features, index, cfg):
    rms, onset, times = features["rms"], features["onset"], features["times"]
    t = float(times[index])
    before = (times >= t - cfg.audio_pre_window_seconds) & (times < t - .03)
    after = (times >= t) & (times <= t + cfg.audio_post_window_seconds)
    if not np.any(before) or not np.any(after):
        return None
    base = max(float(np.percentile(rms, 20)), 1e-9)
    active = max(float(np.percentile(rms, 70)), base)
    before_level = float(np.median(rms[before])); after_level = float(np.median(rms[after]))
    quiet = float(np.clip(1 - (before_level - base) / max(active - base, 1e-9), 0, 1))
    rise = float(np.clip((after_level - before_level) / max(active - base, 1e-9), 0, 1))
    attack = float(np.clip(float(np.max(onset[after])) / max(float(np.percentile(onset, 65)), 1e-9), 0, 1))
    confidence = .42 * quiet + .38 * rise + .20 * attack
    return {"time": round(t, 6), "confidence": round(confidence, 4),
            "quiet_before": round(quiet, 4), "activity_rise": round(rise, 4),
            "onset_support": round(attack, 4), "before_rms": round(before_level, 8),
            "after_rms": round(after_level, 8)}


def inspect_chart_event(features, prediction, source, cfg):
    """Search only the chart-requested local window on the corresponding stem."""
    if features is None:
        return []
    times, rms = features["times"], features["rms"]
    lo = max(0.0, prediction - cfg.audio_search_radius_seconds)
    hi = min(features["duration"], prediction + cfg.audio_search_radius_seconds)
    window = (times >= lo) & (times <= hi)
    floor = float(np.percentile(rms, 20)); active = float(np.percentile(rms, 70))
    threshold = max(floor * 2.0, active * .10, 1e-7)
    active_mask = rms >= threshold
    width = max(1, int(round(.08 * features["sr"] / cfg.hop_length)))
    sustained = np.convolve(active_mask.astype(int), np.ones(width, dtype=int), mode="same") >= max(1, int(math.ceil(width * .60)))
    starts = np.where(window & sustained & ~np.r_[False, sustained[:-1]])[0]
    rows = []
    for index in starts:
        evidence = _candidate_at(features, int(index), cfg)
        if evidence and evidence["confidence"] >= cfg.minimum_transition_confidence:
            evidence.update({"source": source, "kind": "silence_to_activity",
                             "residual_seconds": round(evidence["time"] - prediction, 6)})
            rows.append(evidence)
    rows.sort(key=lambda x: (-x["confidence"], abs(x["residual_seconds"])))
    return rows[:cfg.maximum_matches_per_score_event]


def diagnose_continuous_anchors(score_events,audio_paths,baseline,measure_downbeats,
                                tempo_bpm,sr=22050,config=None,feature_cache_dir=None):
    """Drive local search from chart events with joint rhythm/chroma validation."""
    cfg=config or ContinuousAnchorConfig()
    score=build_score_candidates(score_events,measure_downbeats,tempo_bpm,cfg)
    def load_features(path):
        try:
            return _features(path,sr,cfg.hop_length,feature_cache_dir,cfg.feature_cache_version)
        except TypeError:
            # Compatibility with lightweight three-argument feature stubs in tests and adapters.
            return _features(path,sr,cfg.hop_length)
    cache={source:load_features(path) for source,path in (audio_paths or {}).items() if source!="full"}
    full_features=load_features((audio_paths or {}).get("full"))
    measures=_measures(measure_downbeats); matches=[]; joint_passed=0
    for item in score["candidates"]:
        prediction=float(baseline(float(item["score_time"]))); source=item["source"]
        if item["kind"]=="dense_active_measure_landmark":
            local=_measure_at(float(item["score_time"]),measures)
            duration=(local or {}).get("duration") or (240.0/max(float(tempo_bpm or 120),1e-9))
            candidates=inspect_dense_fingerprint(cache.get(source),score_events.get(source,[]),item,prediction,duration,cfg)
            evidence_source=source
            # Do not fall back to the full mix for pitched dense identity. Bleed creates false chroma matches.
            joint_passed+=int(bool(candidates))
        else:
            candidates=inspect_chart_event(cache.get(source),prediction,source,cfg); evidence_source=source
            if not candidates and full_features is not None:
                candidates=inspect_chart_event(full_features,prediction,"full",cfg); evidence_source="full"
        score_confidence=float(item.get("score_confidence",1.0))
        for candidate in candidates:
            candidate.setdefault("raw_audio_confidence",candidate.get("confidence"))
            candidate["score_confidence"]=round(score_confidence,4)
            candidate["corresponding_stem"]=source; candidate["evidence_source"]=evidence_source
            candidate["single_stem_sufficient"]=bool(evidence_source==source and
                item["kind"]=="restart_after_measure_silence" and
                candidate["confidence"]>=cfg.strong_single_stem_confidence)
        matches.append({"score_event_id":item["id"],"source":source,"kind":item["kind"],
            "score_time":item["score_time"],"score_position":item["score_position"],
            "score_confidence":item.get("score_confidence",1.0),"score_evidence":item.get("score_evidence",{}),
            "baseline_prediction":round(prediction,6),"candidate_count":len(candidates),
            "candidates":candidates,"best_candidate":candidates[0] if candidates else None})
    retained=sum(x["candidate_count"] for x in matches); matched=sum(1 for x in matches if x["candidate_count"])
    feature_cache={k:(None if v is None else v.get("cache")) for k,v in cache.items()}
    summary={"score_candidates":score["candidate_count"],"structural_score_candidates":score.get("structural_candidate_count",0),
        "dense_score_candidates":score.get("dense_candidate_count",0),"joint_validated_dense_candidates":joint_passed,
        "locally_searched_score_events":len(matches),"retained_audio_hypotheses":retained,
        "matched_score_events":matched,"strong_single_stem_restarts":sum(1 for x in matches for c in x["candidates"] if c.get("single_stem_sufficient")),
        "feature_cache":feature_cache}
    return {"version":4,"status":"diagnostic_only","timing_changes_applied":False,
            "search_strategy":"chart_first_corresponding_stem_local_search","config":asdict(cfg),
            "score":score,"matches":matches,"matching":{"matches":matches},"summary":summary,
            "notes":["Dense pitched candidates require both rhythm and chroma agreement.",
                     "A wide search is retained, with probability decaying away from the baseline prediction.",
                     "Dense production remains disabled until corpus listening validates the joint matcher."]}

@dataclass(frozen=True)
class AnchorPathConfig:
    simultaneous_score_tolerance_seconds: float = 0.15
    simultaneous_audio_tolerance_seconds: float = 0.18
    minimum_segment_scale: float = 0.70
    maximum_segment_scale: float = 1.35
    preferred_scale_tolerance: float = 0.08
    scale_change_soft_tolerance: float = 0.05
    candidate_reuse_tolerance_seconds: float = 0.10
    skip_penalty: float = 0.55
    corresponding_stem_bonus: float = 0.18
    single_stem_restart_bonus: float = 0.12
    multi_source_bonus_per_source: float = 0.08
    maximum_multi_source_bonus: float = 0.24
    residual_penalty_per_search_radius: float = 0.15
    scale_deviation_penalty: float = 1.40
    scale_change_penalty: float = 1.00
    alternatives_per_cluster: int = 8


def _cluster_score_matches(matches, tolerance):
    clusters=[]
    for match in sorted(matches or [], key=lambda x:(float(x['score_time']),x.get('source',''))):
        score_time=float(match['score_time'])
        if clusters and score_time-clusters[-1]['maximum_score_time']<=tolerance:
            cluster=clusters[-1]
        else:
            cluster={'index':len(clusters),'matches':[],'score_times':[],
                     'minimum_score_time':score_time,'maximum_score_time':score_time}
            clusters.append(cluster)
        cluster['matches'].append(match); cluster['score_times'].append(score_time)
        cluster['minimum_score_time']=min(cluster['minimum_score_time'],score_time)
        cluster['maximum_score_time']=max(cluster['maximum_score_time'],score_time)
    for cluster in clusters:
        cluster['score_time']=float(np.median(cluster['score_times']))
        cluster['sources']=sorted({x.get('source') for x in cluster['matches'] if x.get('source')})
        cluster['score_event_ids']=[x.get('score_event_id') for x in cluster['matches']]
        cluster['score_positions']=[x.get('score_position',{}) for x in cluster['matches']]
    return clusters


def _candidate_groups_for_cluster(cluster, cfg, search_radius):
    observations=[]
    for match in cluster['matches']:
        for candidate in match.get('candidates',[]):
            observations.append({**candidate,
                'score_event_id':match.get('score_event_id'),
                'score_source':match.get('source'),
                'score_kind':match.get('kind')})
    observations.sort(key=lambda x:float(x['time']))
    groups=[]
    for observation in observations:
        value=float(observation['time'])
        if groups and value-groups[-1]['maximum_time']<=cfg.simultaneous_audio_tolerance_seconds:
            group=groups[-1]
        else:
            group={'observations':[],'times':[],'minimum_time':value,'maximum_time':value}
            groups.append(group)
        group['observations'].append(observation); group['times'].append(value)
        group['minimum_time']=min(group['minimum_time'],value)
        group['maximum_time']=max(group['maximum_time'],value)
    states=[]
    for group in groups:
        # At most one observation per score source contributes confidence, so
        # several nearby alternatives from one detector cannot inflate support.
        best_by_source={}
        for observation in group['observations']:
            source=observation.get('score_source') or observation.get('corresponding_stem')
            if source not in best_by_source or float(observation.get('confidence',0))>float(best_by_source[source].get('confidence',0)):
                best_by_source[source]=observation
        used=list(best_by_source.values())
        audio_time=float(np.median([float(x['time']) for x in used]))
        confidence=float(np.mean([float(x.get('confidence',0)) for x in used])) if used else 0.0
        matching_sources=sum(1 for x in used if x.get('evidence_source')==x.get('score_source'))
        single=any(bool(x.get('single_stem_sufficient')) for x in used)
        multi_bonus=min(cfg.maximum_multi_source_bonus,
                        cfg.multi_source_bonus_per_source*max(0,len(used)-1))
        residual=float(np.median([float(x.get('residual_seconds',0)) for x in used])) if used else 0.0
        emission=(confidence+cfg.corresponding_stem_bonus*matching_sources+multi_bonus+
                  (cfg.single_stem_restart_bonus if single else 0.0)-
                  cfg.residual_penalty_per_search_radius*abs(residual)/max(float(search_radius),1e-9))
        states.append({'kind':'candidate','audio_time':audio_time,'emission_score':float(emission),
            'confidence':confidence,'residual_seconds':residual,
            'supporting_score_sources':sorted(best_by_source),
            'supporting_source_count':len(best_by_source),
            'corresponding_stem_observation_count':matching_sources,
            'single_stem_sufficient':single,'observations':used})
    states.sort(key=lambda x:(-x['emission_score'],abs(x['residual_seconds'])))
    return states[:max(1,int(cfg.alternatives_per_cluster))]


def _manual_nodes(manual_anchor_quality):
    anchors=(manual_anchor_quality or {}).get('anchors',[])
    return sorted([{'score_time':float(x['nominal_time']),
                    'audio_time':float(x['padded_audio_time']),
                    'id':x.get('id'),'kind':'manual_exact'}
                   for x in anchors if x.get('mode')=='exact'],key=lambda x:x['score_time'])


def _segment_scale(a,b):
    score_delta=float(b['score_time'])-float(a['score_time'])
    audio_delta=float(b['audio_time'])-float(a['audio_time'])
    return audio_delta/score_delta if score_delta>0 else None


def preview_monotonic_anchor_path(continuous_report, baseline, manual_anchor_quality=None,
                                  config=None):
    """Select a coherent diagnostic anchor chain with dynamic programming.

    This is deliberately provider-agnostic and song-agnostic. It accepts any
    sparse structural candidates, allows events to be skipped, treats exact
    manual anchors as immutable constraints, prevents non-simultaneous reuse of
    one audio event, and penalizes rather than forbids ordinary tempo changes.
    The returned path is never applied to production timing.
    """
    cfg=config or AnchorPathConfig()
    matches=((continuous_report or {}).get('matching') or {}).get('matches',[])
    search_radius=float((((continuous_report or {}).get('score_candidates') or {}).get('config') or {}).get(
        'audio_search_radius_seconds',6.0))
    clusters=_cluster_score_matches(matches,cfg.simultaneous_score_tolerance_seconds)
    manuals=_manual_nodes(manual_anchor_quality)
    # Assign each cluster to a hard-constraint interval. A candidate may never
    # cross a manual anchor in either score or audio time.
    boundaries=[{'score_time':float('-inf'),'audio_time':float('-inf'),'id':'-infinity'}]+manuals+[
        {'score_time':float('inf'),'audio_time':float('inf'),'id':'+infinity'}]
    selected=[]; intervals=[]; rejected=[]
    for left,right in zip(boundaries,boundaries[1:]):
        local=[x for x in clusters if left['score_time']<x['score_time']<right['score_time']]
        layers=[]
        for cluster in local:
            states=_candidate_groups_for_cluster(cluster,cfg,search_radius)
            states.append({'kind':'skip','audio_time':None,'emission_score':-cfg.skip_penalty})
            layers.append((cluster,states))
        if not layers:
            continue
        # State record: score, predecessor index, last real anchor, prior scale.
        previous=[{'score':0.0,'last_anchor':None if not math.isfinite(left['score_time']) else left,
                   'prior_scale':None,'path':[]}]
        for layer_index,(cluster,states) in enumerate(layers):
            current=[]
            for state in states:
                best=None
                for prior in previous:
                    total=float(prior['score'])+float(state['emission_score'])
                    last=prior['last_anchor']; prior_scale=prior['prior_scale']
                    if state['kind']=='skip':
                        candidate={'score':total,'last_anchor':last,'prior_scale':prior_scale,
                                   'path':prior['path']+[(cluster,state)]}
                    else:
                        node={'score_time':cluster['score_time'],'audio_time':float(state['audio_time'])}
                        if last is not None:
                            if node['audio_time']<=float(last['audio_time'])+cfg.candidate_reuse_tolerance_seconds:
                                continue
                            scale=_segment_scale(last,node)
                            if scale is None or not cfg.minimum_segment_scale<=scale<=cfg.maximum_segment_scale:
                                continue
                            total-=cfg.scale_deviation_penalty*max(0.0,abs(scale-1.0)-cfg.preferred_scale_tolerance)
                            if prior_scale is not None:
                                total-=cfg.scale_change_penalty*max(0.0,abs(scale-prior_scale)-cfg.scale_change_soft_tolerance)
                        else:
                            scale=None
                        # If a right manual boundary exists, this candidate must
                        # leave a positive and plausible segment to it.
                        if math.isfinite(right['score_time']):
                            future=_segment_scale(node,right)
                            if future is None or not cfg.minimum_segment_scale<=future<=cfg.maximum_segment_scale:
                                continue
                        candidate={'score':total,'last_anchor':node,'prior_scale':scale or prior_scale,
                                   'path':prior['path']+[(cluster,state)]}
                    if best is None or candidate['score']>best['score']:
                        best=candidate
                if best is not None:
                    current.append(best)
            if not current:
                # A fully skippable layer should make this unreachable, but keep
                # diagnostics robust against malformed external candidate data.
                rejected.append({'cluster_index':cluster['index'],'reason':'no_valid_state'})
                current=previous
            current.sort(key=lambda x:x['score'],reverse=True)
            previous=current[:max(16,len(states)*2)]
        winner=max(previous,key=lambda x:x['score'])
        runner=sorted((x['score'] for x in previous),reverse=True)[1] if len(previous)>1 else None
        interval_selected=[]
        for cluster,state in winner['path']:
            if state['kind']=='skip':
                rejected.append({'cluster_index':cluster['index'],'score_time':cluster['score_time'],
                                 'sources':cluster['sources'],'reason':'skipped_by_global_path'})
                continue
            interval_selected.append({'cluster_index':cluster['index'],'score_time':cluster['score_time'],
                'audio_time':round(float(state['audio_time']),6),'sources':cluster['sources'],
                'score_event_ids':cluster['score_event_ids'],'score_positions':cluster['score_positions'],
                'confidence':round(float(state['confidence']),4),
                'emission_score':round(float(state['emission_score']),6),
                'supporting_score_sources':state['supporting_score_sources'],
                'supporting_source_count':state['supporting_source_count'],
                'single_stem_sufficient':state['single_stem_sufficient'],
                'residual_seconds':round(float(state['residual_seconds']),6),
                'observations':state['observations']})
        selected.extend(interval_selected)
        intervals.append({'left_constraint':left.get('id'),'right_constraint':right.get('id'),
                          'cluster_count':len(local),'selected_count':len(interval_selected),
                          'best_path_score':round(float(winner['score']),6),
                          'runner_up_score':None if runner is None else round(float(runner),6),
                          'confidence_margin':None if runner is None else round(float(winner['score']-runner),6)})
    all_nodes=manuals+[{'score_time':x['score_time'],'audio_time':x['audio_time'],
                        'id':f"auto-cluster-{x['cluster_index']}",'kind':'automatic_preview'} for x in selected]
    all_nodes.sort(key=lambda x:x['score_time'])
    segments=[]; warnings=[]
    for a,b in zip(all_nodes,all_nodes[1:]):
        scale=_segment_scale(a,b)
        status='plausible' if scale is not None and cfg.minimum_segment_scale<=scale<=cfg.maximum_segment_scale else 'invalid'
        if status!='plausible': warnings.append('selected_path_contains_invalid_segment')
        segments.append({'from':a.get('id'),'to':b.get('id'),
                         'score_duration_seconds':round(b['score_time']-a['score_time'],6),
                         'audio_duration_seconds':round(b['audio_time']-a['audio_time'],6),
                         'implied_scale':None if scale is None else round(scale,8),'status':status})
    monotonic=all(b['score_time']>a['score_time'] and b['audio_time']>a['audio_time']
                  for a,b in zip(all_nodes,all_nodes[1:]))
    selected_ids={x['cluster_index'] for x in selected}
    return {'version':1,'status':'diagnostic_only','timing_changes_applied':False,
            'method':'dynamic_programming_monotonic_anchor_path_preview',
            'config':asdict(cfg),'manual_hard_constraint_count':len(manuals),
            'score_cluster_count':len(clusters),'selected_automatic_anchor_count':len(selected),
            'skipped_cluster_count':len(clusters)-len(selected_ids),
            'selected_anchors':selected,'manual_constraints':manuals,'segments':segments,
            'intervals':intervals,'rejected_or_skipped':rejected,
            'validation':{'monotonic':bool(monotonic),'warnings':sorted(set(warnings))},
            'notes':['Preview only; no production warp was changed.',
                     'Event clustering and penalties are generic and configurable.',
                     'Strong corresponding-stem evidence may stand alone; multi-source agreement improves emission score.']}

@dataclass(frozen=True)
class ContinuousLinearWarpConfig:
    minimum_segment_scale: float = 0.70
    maximum_segment_scale: float = 1.35
    minimum_control_spacing_seconds: float = 0.05
    maximum_measure_stretch: float = 1.35
    minimum_measure_stretch: float = 0.70
    hold_last_residual_after_final_anchor: bool = True


def build_continuous_anchor_linear_warp(
    baseline,
    path_preview,
    measure_downbeats,
    manual_anchor_quality=None,
    apply=False,
    config=None,
    force_invalid=False,
):
    """Build a guarded piecewise-linear map through the preview path.

    Automatic controls come only from the globally monotonic path preview.
    Exact manual anchors are immutable controls and supersede an automatic
    control at the same score position. Before the first control the baseline
    is retained. After the last control, the final residual is held constant,
    preserving the baseline slope instead of extrapolating the final short
    anchor segment indefinitely.

    ``apply=False`` returns the baseline unchanged plus a complete A/B report.
    ``apply=True`` returns the proposal only when every validation gate passes;
    otherwise it falls back to the exact baseline.
    """
    cfg = config or ContinuousLinearWarpConfig()
    beats = sorted(measure_downbeats or [], key=lambda x: float(x["t_gp"]))
    selected = list((path_preview or {}).get("selected_anchors") or [])
    manuals = [
        x for x in ((manual_anchor_quality or {}).get("anchors") or [])
        if x.get("mode") == "exact"
    ]

    raw = []
    for row in selected:
        raw.append({
            "score_time": float(row["score_time"]),
            "audio_time": float(row["audio_time"]),
            "kind": "automatic_continuous_anchor",
            "id": f"auto-cluster-{row.get('cluster_index')}",
            "sources": list(row.get("sources") or []),
            "confidence": row.get("confidence"),
            "supporting_source_count": row.get("supporting_source_count", 0),
        })
    for row in manuals:
        raw.append({
            "score_time": float(row["nominal_time"]),
            "audio_time": float(row["padded_audio_time"]),
            "kind": "manual_exact",
            "id": row.get("id"),
            "sources": ["manual"],
            "confidence": "manual_exact",
            "supporting_source_count": 1,
        })

    # Manual points win when an automatic point resolves to the same event.
    raw.sort(key=lambda x: (x["score_time"], 0 if x["kind"] == "manual_exact" else 1))
    controls = []
    for point in raw:
        if controls and abs(point["score_time"] - controls[-1]["score_time"]) < cfg.minimum_control_spacing_seconds:
            if point["kind"] == "manual_exact":
                controls[-1] = point
            continue
        controls.append(point)

    warnings = []
    path_valid = bool((path_preview or {}).get("validation", {}).get("monotonic", False))
    if not path_valid:
        warnings.append("preview_path_not_monotonic")
    if not controls:
        warnings.append("no_selected_controls")

    control_segments = []
    for a, b in zip(controls, controls[1:]):
        score_delta = b["score_time"] - a["score_time"]
        audio_delta = b["audio_time"] - a["audio_time"]
        scale = audio_delta / score_delta if score_delta > 0 else None
        status = "plausible" if (
            scale is not None
            and audio_delta > 0
            and cfg.minimum_segment_scale <= scale <= cfg.maximum_segment_scale
        ) else "invalid"
        if status == "invalid":
            warnings.append("control_segment_outside_scale_limits")
        control_segments.append({
            "from": a.get("id"),
            "to": b.get("id"),
            "score_duration_seconds": round(score_delta, 6),
            "audio_duration_seconds": round(audio_delta, 6),
            "implied_scale": None if scale is None else round(scale, 8),
            "status": status,
        })

    xs = np.asarray([x["score_time"] for x in controls], dtype=float)
    ys = np.asarray([x["audio_time"] for x in controls], dtype=float)
    residuals = np.asarray([y - float(baseline(x)) for x, y in zip(xs, ys)], dtype=float)

    def proposal(value):
        t = float(value)
        if not len(xs):
            return float(baseline(t))
        if t < xs[0]:
            return float(baseline(t))
        if t > xs[-1]:
            if cfg.hold_last_residual_after_final_anchor:
                return float(baseline(t) + residuals[-1])
            return float(baseline(t))
        return float(np.interp(t, xs, ys))

    sample_times = {float(x["t_gp"]) for x in beats}
    sample_times.update(float(x) for x in xs)
    sample_times = sorted(sample_times)
    monotonic = all(proposal(b) > proposal(a) for a, b in zip(sample_times, sample_times[1:]))
    if not monotonic:
        warnings.append("proposal_not_monotonic")

    measure_stretches = []
    for a, b in zip(beats, beats[1:]):
        sa, sb = float(a["t_gp"]), float(b["t_gp"])
        base_delta = float(baseline(sb)) - float(baseline(sa))
        proposed_delta = proposal(sb) - proposal(sa)
        if base_delta > 0:
            measure_stretches.append(proposed_delta / base_delta)
    min_stretch = min(measure_stretches or [1.0])
    max_stretch = max(measure_stretches or [1.0])
    if min_stretch < cfg.minimum_measure_stretch or max_stretch > cfg.maximum_measure_stretch:
        warnings.append("proposal_measure_stretch_outside_limits")

    comparison = []
    differences = []
    for row in beats:
        score_time = float(row["t_gp"])
        base_time = float(baseline(score_time))
        proposed_time = proposal(score_time)
        difference = proposed_time - base_time
        differences.append(abs(difference))
        comparison.append({
            "playback_bar": int(row["measure"]),
            "source_bar": int(row.get("source_bar", row["measure"])),
            "occurrence": int(row.get("occurrence", 1)),
            "score_time": round(score_time, 6),
            "baseline_time": round(base_time, 6),
            "proposed_time": round(proposed_time, 6),
            "difference_seconds": round(difference, 6),
        })

    valid = bool(path_valid and controls and monotonic and not warnings)

    # Force mode may override soft validation such as stretch limits, but it
    # must never override monotonicity. Non-monotonic timing can reverse event
    # order and create negative note or beat intervals.
    force_requested = bool(apply and force_invalid)
    force_blocked_non_monotonic = bool(
        force_requested and path_valid and controls and not monotonic
    )
    forced_invalid_applied = bool(
        force_requested
        and path_valid
        and controls
        and monotonic
        and not valid
    )
    timing_changes_applied = bool(apply and (valid or forced_invalid_applied))

    if timing_changes_applied:
        final = proposal
        status = "forced_invalid_diagnostic" if forced_invalid_applied else "applied"
    else:
        final = lambda t: float(baseline(float(t)))
        if force_blocked_non_monotonic:
            status = "baseline_fallback_force_blocked_non_monotonic"
        else:
            status = "diagnostic_only" if valid else "baseline_fallback_invalid_proposal"

    report = {
        "version": 1,
        "status": status,
        "method": "monotonic_continuous_anchor_piecewise_linear",
        "requested_application": bool(apply),
        "timing_changes_applied": timing_changes_applied,
        "forced_invalid_diagnostic": forced_invalid_applied,
        "force_requested": force_requested,
        "force_blocked_non_monotonic": force_blocked_non_monotonic,
        "force_blocked_reason": (
            "proposal_not_monotonic" if force_blocked_non_monotonic else None
        ),
        "config": asdict(cfg),
        "automatic_control_count": sum(x["kind"] == "automatic_continuous_anchor" for x in controls),
        "manual_control_count": sum(x["kind"] == "manual_exact" for x in controls),
        "controls": [
            {
                **x,
                "score_time": round(x["score_time"], 6),
                "audio_time": round(x["audio_time"], 6),
                "baseline_time": round(float(baseline(x["score_time"])), 6),
                "residual_seconds": round(x["audio_time"] - float(baseline(x["score_time"])), 6),
            }
            for x in controls
        ],
        "segments": control_segments,
        "comparison_by_playback_bar": comparison,
        "validation": {
            "preview_path_monotonic": path_valid,
            "proposal_monotonic": monotonic,
            "minimum_measure_stretch_relative_to_baseline": round(min_stretch, 8),
            "maximum_measure_stretch_relative_to_baseline": round(max_stretch, 8),
            "maximum_absolute_difference_seconds": round(max(differences or [0.0]), 6),
            "median_absolute_difference_seconds": round(float(np.median(differences or [0.0])), 6),
            "warnings": sorted(set(warnings)),
            "fallback_used": not timing_changes_applied and bool(apply),
        },
        "ending_behavior": (
            "hold_final_residual_preserve_baseline_slope"
            if cfg.hold_last_residual_after_final_anchor
            else "return_to_baseline"
        ),
        "notes": [
            "Exact manual anchors supersede automatic controls at the same score position.",
            "Before the first control the original baseline is preserved.",
            "After the last control the baseline slope is preserved with the final residual held constant.",
        ],
    }
    final.alignment_diagnostics = report
    return final, report
