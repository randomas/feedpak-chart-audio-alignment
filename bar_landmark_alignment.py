"""Regularized per-bar landmark alignment.

This module is intentionally independent of the existing checkpoint and
continuous-anchor implementations. It reuses their baseline warp, score events,
audio stems, and measure downbeats, but creates a new alignment method rather
than changing existing behaviour.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None


@dataclass(frozen=True)
class BarTrackingConfig:
    search_radius_seconds: float = 0.45
    first_event_fraction: float = 0.40
    minimum_peak_strength: float = 0.20
    smoothing_window_bars: int = 5
    maximum_residual_step_seconds: float = 0.120
    maximum_absolute_correction_seconds: float = 2.0
    minimum_local_stretch: float = 0.90
    maximum_local_stretch: float = 1.10
    minimum_observed_bars: int = 4


class PiecewiseBarWarp:
    def __init__(self, score_times, audio_times):
        self.score=np.asarray(score_times,dtype=float)
        self.audio=np.asarray(audio_times,dtype=float)
        if len(self.score)<2 or np.any(np.diff(self.score)<=0) or np.any(np.diff(self.audio)<=0):
            raise ValueError("bar timing controls must be strictly monotonic")
    def __call__(self,t):
        t=float(t)
        if t<=self.score[0]: return float(self.audio[0]+t-self.score[0])
        if t>=self.score[-1]: return float(self.audio[-1]+t-self.score[-1])
        return float(np.interp(t,self.score,self.audio))


def _event_time(event):
    if isinstance(event,(int,float)): return float(event)
    return float(event.get("t_gp",event.get("time",0.0)))


def _bar_landmarks(score_events, downbeats, fraction):
    bars=[]
    for i,row in enumerate(downbeats):
        start=float(row["t_gp"])
        if i+1<len(downbeats): end=float(downbeats[i+1]["t_gp"])
        else:
            previous=start-float(downbeats[i-1]["t_gp"]) if i else 2.0
            end=start+max(previous,0.25)
        limit=start+fraction*(end-start)
        role_events={}
        for role,events in score_events.items():
            candidates=[_event_time(x) for x in (events or []) if start-1e-6<=_event_time(x)<limit]
            if candidates: role_events[role]=min(candidates)
        if role_events:
            first=min(role_events.values())
            supporting=sorted(role for role,t in role_events.items() if abs(t-first)<=0.060)
            landmark=first
        else:
            supporting=[]; landmark=start
        bars.append({"measure":int(row.get("measure",i+1)),"score_bar_time":start,
                     "score_landmark_time":landmark,"within_bar_seconds":landmark-start,
                     "score_bar_end":end,"score_roles":supporting,
                     "landmark_type":"first_event" if role_events else "bar_boundary"})
    return bars


def _load_onsets(path,sr):
    if librosa is None or not path or not Path(path).is_file(): return None
    y,actual_sr=librosa.load(path,sr=sr,mono=True)
    hop=512
    env=librosa.onset.onset_strength(y=y,sr=actual_sr,hop_length=hop)
    if not len(env): return None
    env=np.asarray(env,dtype=float)
    scale=float(np.percentile(env,95)) or 1.0
    normalized=np.clip(env/scale,0.0,2.0)
    times=librosa.frames_to_time(np.arange(len(env)),sr=actual_sr,hop_length=hop)
    return times,normalized


def _local_candidate(features,predicted,radius,minimum_strength):
    if features is None: return None
    times,strength=features
    mask=(times>=predicted-radius)&(times<=predicted+radius)
    indexes=np.flatnonzero(mask)
    if not len(indexes): return None
    local=indexes[np.argmax(strength[indexes])]
    peak=float(strength[local]); observed=float(times[local])
    if peak<minimum_strength: return None
    distance=abs(observed-predicted)
    confidence=float(np.clip(0.65*min(1.0,peak)+0.35*(1.0-distance/radius),0.0,1.0))
    return observed,confidence,peak


def _weighted_median(values,weights):
    order=np.argsort(values); values=np.asarray(values)[order]; weights=np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights),0.5*np.sum(weights))])


def _smooth_residuals(raw,confidence,config):
    count=len(raw); smoothed=np.zeros(count,dtype=float)
    half=max(1,config.smoothing_window_bars//2)
    for i in range(count):
        lo=max(0,i-half);hi=min(count,i+half+1)
        valid=[j for j in range(lo,hi) if math.isfinite(raw[j]) and confidence[j]>0]
        smoothed[i]=_weighted_median([raw[j] for j in valid],[confidence[j] for j in valid]) if valid else np.nan
    known=np.flatnonzero(np.isfinite(smoothed))
    if not len(known): return np.zeros(count)
    smoothed=np.interp(np.arange(count),known,smoothed[known])
    smoothed=np.clip(smoothed,-config.maximum_absolute_correction_seconds,config.maximum_absolute_correction_seconds)
    # Limit sudden residual movement. Genuine direction changes survive when
    # supported over several bars; isolated jumps are absorbed gradually.
    for i in range(1,count):
        delta=np.clip(smoothed[i]-smoothed[i-1],-config.maximum_residual_step_seconds,config.maximum_residual_step_seconds)
        smoothed[i]=smoothed[i-1]+delta
    for i in range(count-2,-1,-1):
        delta=np.clip(smoothed[i]-smoothed[i+1],-config.maximum_residual_step_seconds,config.maximum_residual_step_seconds)
        smoothed[i]=smoothed[i+1]+delta
    return smoothed


def build_bar_landmark_warp(base_warp,score_events,audio_paths,downbeats,sr=22050,config=None):
    config=config or BarTrackingConfig()
    bars=_bar_landmarks(score_events,downbeats,config.first_event_fraction)
    features={role:_load_onsets(path,sr) for role,path in audio_paths.items() if path}
    raw=[];confidence=[]
    for bar in bars:
        predicted=float(base_warp(bar["score_landmark_time"]))
        observations=[]
        preferred=bar["score_roles"] or [r for r in ("drums","bass","piano","guitar","full") if r in features]
        for role in preferred:
            candidate=_local_candidate(features.get(role),predicted,config.search_radius_seconds,config.minimum_peak_strength)
            if candidate:
                observed,conf,peak=candidate
                # Full mix confirms but never supplies independent multi-source weight.
                weight=0.55*conf if role=="full" else conf
                observations.append({"source":role,"time":observed,"confidence":conf,"peak":peak,"weight":weight})
        if observations:
            observed=_weighted_median([x["time"] for x in observations],[x["weight"] for x in observations])
            residual=observed-predicted
            conf=float(np.clip(sum(x["weight"] for x in observations)/max(1,len(observations)),0,1))
            status="observed"
        else:
            observed=None;residual=float("nan");conf=0.0;status="interpolated"
        raw.append(residual);confidence.append(conf)
        bar.update({"predicted_landmark_time":round(predicted,6),"observed_landmark_time":None if observed is None else round(observed,6),
                    "raw_residual_ms":None if not math.isfinite(residual) else round(1000*residual,3),
                    "confidence":round(conf,4),"status":status,"observations":observations})
    fitted=_smooth_residuals(np.asarray(raw),np.asarray(confidence),config)
    score=np.asarray([b["score_bar_time"] for b in bars],dtype=float)
    baseline=np.asarray([base_warp(t) for t in score],dtype=float)
    corrected=baseline+fitted
    # Enforce local stretch relative to baseline interval rather than rejecting
    # the whole path. Clamp the weaker/local correction and keep the trace.
    for i in range(1,len(corrected)):
        base_delta=baseline[i]-baseline[i-1]
        lo=corrected[i-1]+config.minimum_local_stretch*base_delta
        hi=corrected[i-1]+config.maximum_local_stretch*base_delta
        corrected[i]=float(np.clip(corrected[i],lo,hi))
    for i,b in enumerate(bars):
        b["fitted_bar_time"]=round(float(corrected[i]),6)
        b["fitted_residual_ms"]=round(1000.0*(corrected[i]-baseline[i]),3)
        if b["status"]=="interpolated": b["confidence"]=0.0
    observed=sum(b["status"]=="observed" for b in bars)
    if observed<config.minimum_observed_bars:
        return base_warp,{"schema":"feedpak.bar_landmark_alignment","version":1,"status":"insufficient_observations",
                         "timing_changes_applied":False,"summary":{"bars":len(bars),"observed_bars":observed},"bars":bars}
    warp=PiecewiseBarWarp(score,corrected)
    stretch=np.diff(corrected)/np.diff(baseline)
    report={"schema":"feedpak.bar_landmark_alignment","version":1,"status":"applied",
            "timing_changes_applied":True,"config":config.__dict__,
            "summary":{"bars":len(bars),"observed_bars":observed,"interpolated_bars":len(bars)-observed,
                       "minimum_local_stretch":round(float(np.min(stretch)),6),"maximum_local_stretch":round(float(np.max(stretch)),6),
                       "maximum_absolute_correction_seconds":round(float(np.max(np.abs(corrected-baseline))),6)},"bars":bars}
    return warp,report
