"""Checkpoint 6.2: diagnostic-only anchor-bounded local DTW.

Changes from 6.1:
* trusted Tier-A evidence on an interior segment boundary can activate exactly
  one adjacent segment, selected deterministically from baseline need;
* minimum segment length uses the real GP meter grid rather than assuming 4/4;
* baseline and candidate metrics share fixed audio observations.
No candidate warp is returned to the packaging pipeline.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
try:
    import librosa
except ImportError:
    librosa=None

@dataclass(frozen=True)
class LocalDTWConfig:
    preferred_maximum_measures:int=8
    hard_maximum_measures:int=16
    minimum_segment_beats:float=2.0
    corridor_interior_ms:float=100.0
    corridor_tier_a_boundary_ms:float=60.0
    minimum_baseline_median_ms:float=20.0
    minimum_baseline_p95_ms:float=40.0
    minimum_chroma_residual_ms:float=30.0
    minimum_silence_residual_ms:float=25.0
    minimum_median_improvement_ms:float=8.0
    minimum_p95_improvement_ms:float=15.0
    minimum_relative_improvement:float=.20
    minimum_held_out_relative_improvement:float=.15
    maximum_median_degradation_ms:float=10.0
    maximum_p95_degradation_ms:float=20.0
    maximum_residual_excursion_ms:float=75.0
    maximum_corridor_edge_percent:float=5.0
    maximum_non_diagonal_percent:float=20.0
    maximum_direction_changes_per_measure:float=2.0
    minimum_beat_stretch:float=.96
    maximum_beat_stretch:float=1.04
    minimum_measure_stretch:float=.98
    maximum_measure_stretch:float=1.02
    feature_cost_weight:float=1.0
    baseline_distance_weight:float=.50
    slope_change_weight:float=.35
    repeated_step_weight:float=.25
    hop_length:int=256
    minimum_symbolic_events:int=4

_CACHE={}
def _features(path,sr,hop):
    if not path or librosa is None:return None
    k=(str(path),int(sr),int(hop))
    if k in _CACHE:return _CACHE[k]
    y,ar=librosa.load(path,sr=sr,mono=True)
    env=np.asarray(librosa.onset.onset_strength(y=y,sr=ar,hop_length=hop),float)
    if len(env):env=(env-env.min())/max(float(env.max()-env.min()),1e-9)
    times=librosa.frames_to_time(np.arange(len(env)),sr=ar,hop_length=hop)
    onsets=np.asarray(librosa.onset.onset_detect(onset_envelope=env,sr=ar,
        hop_length=hop,units='time',backtrack=False),float)
    _CACHE[k]=(env,times,onsets,float(ar));return _CACHE[k]

def _metrics(values):
    v=np.asarray(values,float);a=np.abs(v)
    return {'matched_events':int(len(v)),
      'median_residual_ms':round(float(np.median(a)),3) if len(v) else None,
      'p95_residual_ms':round(float(np.percentile(a,95)),3) if len(v) else None,
      'maximum_residual_ms':round(float(a.max()),3) if len(v) else None,
      'signed_median_residual_ms':round(float(np.median(v)),3) if len(v) else None}

def _fixed_observations(events,onsets,baseline,radius=.12):
    obs=[]
    for t in events:
        target=float(baseline(float(t)));lo=np.searchsorted(onsets,target-radius)
        hi=np.searchsorted(onsets,target+radius,side='right')
        if hi>lo:
            hit=float(onsets[lo:hi][np.argmin(np.abs(onsets[lo:hi]-target))])
            obs.append((float(t),hit))
    return obs

def _residuals(observations,warp):
    return np.asarray([(audio-float(warp(t)))*1000.0 for t,audio in observations])

def _boundaries(linear_report,silence_report,beats,cfg):
    by_measure={int(b['measure']):b for b in beats}
    pts=[{'measure':int(beats[0]['measure']),'nominal_time':float(beats[0]['t_gp']),
          'tier':'fixed','evidence':['fixed_start'],'boundary_kind':'observed'}]
    for a in (linear_report or {}).get('anchors',[]):
        pts.append({'measure':int(a['measure']),'nominal_time':float(a['nominal_time']),
          'tier':a.get('tier','B'),'evidence':list(a.get('evidence',[])),'boundary_kind':'observed'})
    for cp in (silence_report or {}).get('checkpoints',[]):
        bm=cp.get('boundary_measurement') or {}
        if cp.get('accepted') and bm.get('accepted'):
            pts.append({'measure':int(cp['measure']),'nominal_time':float(cp['score_end']),
              'tier':'A' if len(bm.get('observations',[]))>=2 else 'B',
              'evidence':['silence'],'boundary_kind':'observed'})
    pts.append({'measure':int(beats[-1]['measure']),'nominal_time':float(beats[-1]['t_gp']),
                'tier':'fixed','evidence':['fixed_end'],'boundary_kind':'observed'})
    pts.sort(key=lambda x:x['nominal_time']);ded=[]
    for p in pts:
        if ded and abs(p['nominal_time']-ded[-1]['nominal_time'])<=.05:
            ded[-1]['evidence']=sorted(set(ded[-1]['evidence']+p['evidence']))
            if p['tier']=='A':ded[-1]['tier']='A'
        else:ded.append(p)
    out=[ded[0]]
    for target in ded[1:]:
        prev=out[-1]
        while target['measure']-prev['measure']>cfg.preferred_maximum_measures:
            m=prev['measure']+cfg.preferred_maximum_measures;b=by_measure.get(m)
            if b is None:break
            prev={'measure':m,'nominal_time':float(b['t_gp']),'tier':'computational',
                  'evidence':['computational_split'],'boundary_kind':'computational'}
            out.append(prev)
        out.append(target)
    return out

def _meter_profile(beats,start,end):
    """Build clipped beat intervals and fractional beat coverage.

    Unlike 6.2, this includes the measure active at ``start`` and the first
    downbeat at/after ``end``. Partial beats contribute fractional beat units.
    """
    if end <= start or len(beats) < 2:
        return {"intervals": [], "beat_units": 0.0,
                "complete_intervals": 0, "partial_start": 0.0,
                "partial_end": 0.0}
    rows=sorted(beats,key=lambda x:float(x['t_gp']))
    down=np.asarray([float(x['t_gp']) for x in rows],float)
    first=max(0,int(np.searchsorted(down,start,side='right'))-1)
    last=min(len(rows)-1,int(np.searchsorted(down,end,side='left')))
    intervals=[];units=0.0;complete=0;partial_start=0.0;partial_end=0.0
    for i in range(first,last):
        m0=float(rows[i]['t_gp']);m1=float(rows[i+1]['t_gp'])
        if m1 <= start or m0 >= end or m1 <= m0:
            continue
        numerator=max(1,int(rows[i].get('ts_num',4)))
        edges=np.linspace(m0,m1,numerator+1)
        for b0,b1 in zip(edges,edges[1:]):
            c0=max(start,float(b0));c1=min(end,float(b1))
            if c1 <= c0:
                continue
            fraction=(c1-c0)/(float(b1)-float(b0))
            intervals.append((c0,c1,float(fraction)))
            units+=fraction
            if fraction >= 1.0-1e-7:
                complete+=1
            else:
                if b0 < start < b1:
                    partial_start=max(partial_start,float(fraction))
                if b0 < end < b1:
                    partial_end=max(partial_end,float(fraction))
    return {"intervals": intervals, "beat_units": float(units),
            "complete_intervals": int(complete),
            "partial_start": float(partial_start),
            "partial_end": float(partial_end)}

def _meter_grid(beats,start,end):
    """Compatibility helper returning clipped beat interval pairs."""
    return [(a,b) for a,b,_ in _meter_profile(beats,start,end)["intervals"]]

def _segment_beat_count(beats,start,end):
    """Return fractional metric beats, including partial boundary beats."""
    return _meter_profile(beats,start,end)["beat_units"]

def _trusted_boundary_evidence(boundary,chroma_report,silence_report,linear_report):
    if boundary.get('tier')!='A':return []
    measure=int(boundary['measure']);items=[]
    admitted={int(a['measure']):set(a.get('evidence',[])) for a in (linear_report or {}).get('anchors',[]) if a.get('tier')=='A'}
    if 'chroma' in admitted.get(measure,set()):
        cp=next((x for x in (chroma_report or {}).get('checkpoints',[]) if int(x.get('measure',-1))==measure and x.get('accepted')),None)
        if cp and cp.get('combined_residual_ms') is not None:
            items.append({'kind':'chroma','measure':measure,'residual_ms':abs(float(cp['combined_residual_ms']))})
    for cp in (silence_report or {}).get('checkpoints',[]):
        bm=cp.get('boundary_measurement') or {}
        if int(cp.get('measure',-1))==measure and cp.get('accepted') and bm.get('accepted') and bm.get('measured_residual_ms') is not None:
            items.append({'kind':'silence','measure':measure,'residual_ms':abs(float(bm['measured_residual_ms']))})
    return items

def _assign_boundary_activation(boundaries,segment_needs,chroma,silence,linear,cfg):
    """Assign each trusted interior boundary to at most one adjacent segment."""
    assigned={i:[] for i in range(len(boundaries)-1)}
    for j in range(1,len(boundaries)-1):
        evidence=_trusted_boundary_evidence(boundaries[j],chroma,silence,linear)
        evidence=[e for e in evidence if (e['kind']=='chroma' and e['residual_ms']>=cfg.minimum_chroma_residual_ms) or
                  (e['kind']=='silence' and e['residual_ms']>=cfg.minimum_silence_residual_ms)]
        if not evidence:continue
        choices=[]
        for idx in (j-1,j):
            need=segment_needs[idx]
            # Prefer the adjacent segment with the larger normalized baseline need.
            choices.append((need, -idx, idx))
        idx=max(choices)[2]
        assigned[idx].extend({**e,'boundary_measure':int(boundaries[j]['measure']),
                              'assigned_side':'left' if idx==j-1 else 'right'} for e in evidence)
    return assigned

def _candidate(events,path,baseline,start,end,start_tier,end_tier,sr,cfg):
    feat=_features(path,sr,cfg.hop_length)
    if feat is None:return None,'missing_drum_audio'
    env,times,_,ar=feat;events=[float(t) for t in events if start<=float(t)<=end]
    if len(events)<cfg.minimum_symbolic_events:return None,'insufficient_drum_events'
    a0=float(baseline(start));a1=float(baseline(end));frame=cfg.hop_length/ar
    n=max(8,int(round((a1-a0)/frame))+1);nominal=np.linspace(start,end,n)
    base_audio=np.asarray([baseline(t) for t in nominal]);synth=np.zeros(n)
    for t in events:synth[int(np.argmin(np.abs(nominal-t)))]=1
    if n>=5:synth=np.convolve(synth,[.15,.5,1,.5,.15],mode='same')
    if synth.max()>0:synth/=synth.max()
    lo=np.searchsorted(times,a0-cfg.corridor_interior_ms/1000)
    hi=np.searchsorted(times,a1+cfg.corridor_interior_ms/1000,side='right')
    observed=env[lo:hi];ot=times[lo:hi]
    if len(observed)<8:return None,'inactive_audio_features'
    expected=np.interp(ot,[a0,a1],[0,n-1]);disp=np.abs(np.arange(n)[:,None]-expected[None,:])*frame*1000
    corridor=np.full((n,1),cfg.corridor_interior_ms);q=max(1,n//8)
    if start_tier=='A':corridor[:q]=cfg.corridor_tier_a_boundary_ms
    if end_tier=='A':corridor[-q:]=cfg.corridor_tier_a_boundary_ms
    cost=np.abs(synth[:,None]-observed[None,:])*cfg.feature_cost_weight+cfg.baseline_distance_weight*disp/corridor
    cost[disp>corridor]=np.inf
    j0=int(np.argmin(np.abs(ot-a0)));j1=int(np.argmin(np.abs(ot-a1)))
    cost[0,:]=np.inf;cost[-1,:]=np.inf;cost[0,j0]=0;cost[-1,j1]=0
    moves=[(1,1),(1,0),(0,1)];D=np.full((n,len(observed),3),np.inf);back=np.full((n,len(observed),3,3),-1,int);D[0,j0,0]=0
    for i in range(n):
      for j in range(j0,j1+1):
       if not np.isfinite(cost[i,j]):continue
       for state,(di,dj) in enumerate(moves):
        pi,pj=i-di,j-dj
        if pi<0 or pj<0:continue
        options=D[pi,pj]+cfg.slope_change_weight*(np.arange(3)!=state)+cfg.repeated_step_weight*(state!=0)
        prev=int(np.argmin(options))
        if np.isfinite(options[prev]) and cost[i,j]+options[prev]<D[i,j,state]:
            D[i,j,state]=cost[i,j]+options[prev];back[i,j,state]=[pi,pj,prev]
    state=int(np.argmin(D[-1,j1]))
    if not np.isfinite(D[-1,j1,state]):return None,'no_path_inside_corridor'
    path_points=[];i=n-1;j=j1
    while True:
        path_points.append((i,j,state))
        if i==0 and j==j0:break
        i,j,state=back[i,j,state]
        if i<0:return None,'broken_backtrace'
    path_points.reverse();buckets={}
    for i,j,_ in path_points:buckets.setdefault(i,[]).append(float(ot[j]))
    raw=np.asarray([np.median(buckets.get(i,[base_audio[i]])) for i in range(n)])
    residual=raw-base_audio
    if len(residual)>=5:residual=np.asarray([np.median(residual[max(0,k-2):min(len(residual),k+3)]) for k in range(len(residual))])
    residual[0]=residual[-1]=0;residual=np.clip(residual,-cfg.maximum_residual_excursion_ms/1000,cfg.maximum_residual_excursion_ms/1000)
    cx=np.linspace(start,end,9);cr=np.interp(cx,nominal,residual);cr[0]=cr[-1]=0
    def warp(t):return float(baseline(float(t))+np.interp(float(t),cx,cr))
    steps=np.diff(np.asarray([(i,j) for i,j,_ in path_points]),axis=0)
    return {'warp':warp,'raw_path_points':len(path_points),'control_points':len(cx),
      'maximum_residual_excursion_ms':float(np.max(np.abs(cr))*1000),
      'non_diagonal_percent':100*float(np.mean(np.any(steps!=1,axis=1))) if len(steps) else 0,
      'corridor_edge_percent':100*float(np.mean(np.abs(residual)*1000>=.9*corridor[:,0])),
      'residual_direction_changes':int(np.sum(np.diff(np.asarray([s for _,_,s in path_points]))!=0)),
      'control_nominal_times':[round(float(x),6) for x in cx],
      'control_residual_ms':[round(float(x)*1000,3) for x in cr]},None

def diagnose_local_dtw(baseline,checkpoint_report,silence_report,chroma_report,linear_report,
                       measure_downbeats,score_events,audio_paths,sr=22050,config=None):
    cfg=config or LocalDTWConfig();bounds=_boundaries(linear_report,silence_report,measure_downbeats,cfg)
    prelim=[]
    for a,b in zip(bounds,bounds[1:]):
        start,end=float(a['nominal_time']),float(b['nominal_time']);events=[t for t in score_events.get('drums',[]) if start<=float(t)<=end]
        feat=_features(audio_paths.get('drums'),sr,cfg.hop_length);obs=_fixed_observations(events,feat[2],baseline) if feat else []
        met=_metrics(_residuals(obs,baseline));need=max((met['median_residual_ms'] or 0)/cfg.minimum_baseline_median_ms,(met['p95_residual_ms'] or 0)/cfg.minimum_baseline_p95_ms)
        prelim.append((events,feat,obs,met,need))
    assigned=_assign_boundary_activation(bounds,[x[4] for x in prelim],chroma_report,silence_report,linear_report,cfg)
    segments=[]
    for idx,(a,b) in enumerate(zip(bounds,bounds[1:])):
        start,end=float(a['nominal_time']),float(b['nominal_time']);events,feat,obs,bm,need=prelim[idx]
        measures=int(b['measure'])-int(a['measure']);meter=_meter_profile(measure_downbeats,start,end);beat_grid=[(x,y) for x,y,_ in meter['intervals']];beat_count=meter['beat_units']
        activation=assigned[idx]
        rec={'segment_id':f"m{a['measure']:03d}-m{b['measure']:03d}",'start_measure':a['measure'],'end_measure':b['measure'],
          'start_nominal_time':round(start,6),'end_nominal_time':round(end,6),'duration_seconds':round(float(baseline(end)-baseline(start)),4),
          'meter_derived_beats':round(beat_count,6),'complete_beat_intervals':meter['complete_intervals'],'partial_start_beat':round(meter['partial_start'],6),'partial_end_beat':round(meter['partial_end'],6),'boundary_sources':{'start':a['evidence'],'end':b['evidence']},
          'boundary_kinds':{'start':a['boundary_kind'],'end':b['boundary_kind']},'assigned_boundary_activation':activation,
          'optimization_sources':['drums'] if audio_paths.get('drums') else [],'held_out_sources':[],'baseline':bm}
        if measures>cfg.hard_maximum_measures:
            rec['decision']={'state':'skipped_insufficient_features','warnings':['hard_maximum_measures_exceeded']};segments.append(rec);continue
        if beat_count<cfg.minimum_segment_beats or len(events)<cfg.minimum_symbolic_events or feat is None:
            reasons=[]
            if beat_count<cfg.minimum_segment_beats:reasons.append('meter_derived_segment_too_short')
            if len(events)<cfg.minimum_symbolic_events:reasons.append('insufficient_drum_events')
            if feat is None:reasons.append('missing_drum_audio')
            rec['decision']={'state':'skipped_insufficient_features','warnings':reasons};segments.append(rec);continue
        baseline_active=(bm['median_residual_ms']>=cfg.minimum_baseline_median_ms or bm['p95_residual_ms']>=cfg.minimum_baseline_p95_ms)
        if not baseline_active and not activation:
            rec['decision']={'state':'skipped_baseline_accurate','warnings':[]};segments.append(rec);continue
        cand,error=_candidate(score_events.get('drums',[]),audio_paths.get('drums'),baseline,start,end,a['tier'],b['tier'],sr,cfg)
        if cand is None:
            rec['decision']={'state':'skipped_insufficient_features','warnings':[error]};segments.append(rec);continue
        warp=cand.pop('warp');cm=_metrics(_residuals(obs,warp));med=bm['median_residual_ms']-cm['median_residual_ms'];p95=bm['p95_residual_ms']-cm['p95_residual_ms'];relative=med/max(bm['median_residual_ms'],1e-9)
        cm.update({'median_improvement_ms':round(med,3),'p95_improvement_ms':round(p95,3),'relative_median_improvement':round(relative,4)});rec['candidate']=cm
        held={};improved=0;degraded=False
        for name in ('bass','guitar','piano'):
            ev=[t for t in score_events.get(name,[]) if start<=float(t)<=end];ff=_features(audio_paths.get(name),sr,cfg.hop_length)
            if len(ev)<cfg.minimum_symbolic_events or ff is None:continue
            fixed=_fixed_observations(ev,ff[2],baseline);B=_metrics(_residuals(fixed,baseline));A=_metrics(_residuals(fixed,warp))
            if not fixed:continue
            mg=B['median_residual_ms']-A['median_residual_ms'];pg=B['p95_residual_ms']-A['p95_residual_ms'];rr=mg/max(B['median_residual_ms'],1e-9);ok=rr>=cfg.minimum_held_out_relative_improvement;bad=mg < -cfg.maximum_median_degradation_ms or pg < -cfg.maximum_p95_degradation_ms
            improved+=int(ok);degraded|=bad;rec['held_out_sources'].append(name+'_onsets')
            held[name+'_onsets']={'baseline':B,'candidate':A,'fixed_observation_count':len(fixed),'coverage_preserved':True,'relative_median_improvement':round(rr,4),'improved':ok,'degraded':bad}
        rec['held_out_evaluation']=held
        measure_st=[];beat_st=[]
        for x,y in zip([float(q['t_gp']) for q in measure_downbeats if start<=float(q['t_gp'])<=end],[float(q['t_gp']) for q in measure_downbeats if start<=float(q['t_gp'])<=end][1:]):
            measure_st.append((warp(y)-warp(x))/(baseline(y)-baseline(x)))
        for x,y in beat_grid:beat_st.append((warp(y)-warp(x))/(baseline(y)-baseline(x)))
        path={**cand,'minimum_measure_stretch':round(min(measure_st or [1]),6),'maximum_measure_stretch':round(max(measure_st or [1]),6),'minimum_beat_stretch':round(min(beat_st or [1]),6),'maximum_beat_stretch':round(max(beat_st or [1]),6),'monotonic':all(warp(y)>warp(x) for x,y in beat_grid),'fixed_start_error_ms':round((warp(start)-baseline(start))*1000,6),'fixed_end_error_ms':round((warp(end)-baseline(end))*1000,6)};rec['path']=path
        state='would_apply';warnings=[]
        if not path['monotonic'] or abs(path['fixed_start_error_ms'])>.001 or abs(path['fixed_end_error_ms'])>.001:state='rejected_non_monotonic';warnings=['endpoint_or_monotonicity']
        elif path['minimum_measure_stretch']<cfg.minimum_measure_stretch or path['maximum_measure_stretch']>cfg.maximum_measure_stretch or path['minimum_beat_stretch']<cfg.minimum_beat_stretch or path['maximum_beat_stretch']>cfg.maximum_beat_stretch:state='rejected_stretch_violation';warnings=['beat_or_measure_stretch']
        elif path['non_diagonal_percent']>cfg.maximum_non_diagonal_percent or path['corridor_edge_percent']>cfg.maximum_corridor_edge_percent or path['residual_direction_changes']>cfg.maximum_direction_changes_per_measure*max(1,measures):state='rejected_path_complexity';warnings=['path_complexity']
        elif degraded:state='rejected_held_out_degradation';warnings=['held_out_degradation']
        elif med<cfg.minimum_median_improvement_ms or p95<cfg.minimum_p95_improvement_ms or relative<cfg.minimum_relative_improvement or improved<1:state='rejected_no_meaningful_improvement';warnings=['improvement_gate']
        rec['simplification']={'raw_path_points':path['raw_path_points'],'control_points':path['control_points'],'simplified_candidate_still_passes':state=='would_apply'};rec['decision']={'state':state,'warnings':warnings};segments.append(rec)
    states=[x['decision']['state'] for x in segments];summary={'generated_segments':len(segments),'evaluated_segments':sum('candidate' in x for x in segments),'eligible_segments':sum(s.startswith('rejected_') or s=='would_apply' for s in states),'boundary_assignments':sum(bool(x.get('assigned_boundary_activation')) for x in segments),'boundary_activated_evaluated':sum(bool(x.get('assigned_boundary_activation')) and 'candidate' in x for x in segments),'boundary_activated_skipped':sum(bool(x.get('assigned_boundary_activation')) and 'candidate' not in x for x in segments),'skipped_baseline_accurate':states.count('skipped_baseline_accurate'),'skipped_insufficient_features':states.count('skipped_insufficient_features'),'would_apply':states.count('would_apply'),'rejected':sum(s.startswith('rejected_') for s in states),'maximum_proposed_change_ms':round(max([x.get('path',{}).get('maximum_residual_excursion_ms',0) for x in segments] or [0]),3),'timing_changes_applied':False}
    return {'version':'1.2.1','status':'diagnostic_only','timing_changes_applied':False,'config':asdict(cfg),'summary':summary,'boundaries':bounds,'segments':segments}

def build_selective_warp(baseline, local_report, measure_downbeats, config=None):
    """Promote only ``would_apply`` local candidates into a production warp.

    Accepted segments are disjoint by construction. Their simplified control
    residuals are added to the checkpoint-linear baseline only inside the
    segment. A global monotonicity/stretch pass is then performed. Any global
    failure rejects the complete local layer and returns the baseline unchanged.
    """
    cfg=config or LocalDTWConfig()
    accepted=[]
    for segment in (local_report or {}).get('segments',[]):
        if segment.get('decision',{}).get('state')!='would_apply':
            continue
        path=segment.get('path') or {}
        xs=np.asarray(path.get('control_nominal_times') or [],dtype=float)
        rs=np.asarray(path.get('control_residual_ms') or [],dtype=float)/1000.0
        start=float(segment['start_nominal_time']);end=float(segment['end_nominal_time'])
        valid=(len(xs)>=2 and len(xs)==len(rs) and np.all(np.diff(xs)>0) and
               abs(xs[0]-start)<=1e-4 and abs(xs[-1]-end)<=1e-4 and
               abs(rs[0])<=1e-6 and abs(rs[-1])<=1e-6)
        if not valid:
            continue
        accepted.append({'segment_id':segment['segment_id'],'start':start,'end':end,
                         'control_nominal_times':xs,'control_residual_seconds':rs,
                         'maximum_change_ms':float(np.max(np.abs(rs))*1000.0)})
    accepted.sort(key=lambda x:x['start'])
    overlap=any(b['start']<a['end']-1e-7 for a,b in zip(accepted,accepted[1:]))
    def proposal(t):
        t=float(t);adjustment=0.0
        for item in accepted:
            if item['start']-1e-9<=t<=item['end']+1e-9:
                adjustment=float(np.interp(t,item['control_nominal_times'],item['control_residual_seconds']))
                break
        return float(baseline(t)+adjustment)
    # Validate on true metric beat edges, measure boundaries and all controls.
    samples={float(x['t_gp']) for x in measure_downbeats}
    if measure_downbeats:
        start=float(measure_downbeats[0]['t_gp']);end=float(measure_downbeats[-1]['t_gp'])
        meter=_meter_profile(measure_downbeats,start,end)
        for a,b,_ in meter['intervals']:samples.add(float(a));samples.add(float(b))
    for item in accepted:samples.update(float(x) for x in item['control_nominal_times'])
    samples=sorted(samples);monotonic=all(proposal(b)>proposal(a) for a,b in zip(samples,samples[1:]))
    beat_stretches=[]
    if measure_downbeats:
        for a,b,_ in _meter_profile(measure_downbeats,float(measure_downbeats[0]['t_gp']),float(measure_downbeats[-1]['t_gp']))['intervals']:
            base_delta=float(baseline(b)-baseline(a))
            if base_delta>0:beat_stretches.append((proposal(b)-proposal(a))/base_delta)
    measure_stretches=[]
    down=[float(x['t_gp']) for x in measure_downbeats]
    for a,b in zip(down,down[1:]):
        base_delta=float(baseline(b)-baseline(a))
        if base_delta>0:measure_stretches.append((proposal(b)-proposal(a))/base_delta)
    beat_min=min(beat_stretches or [1.0]);beat_max=max(beat_stretches or [1.0])
    measure_min=min(measure_stretches or [1.0]);measure_max=max(measure_stretches or [1.0])
    valid=bool(not overlap and monotonic and beat_min>=cfg.minimum_beat_stretch and
               beat_max<=cfg.maximum_beat_stretch and measure_min>=cfg.minimum_measure_stretch and
               measure_max<=cfg.maximum_measure_stretch)
    if not valid:
        final=lambda t:float(baseline(float(t)))
        status='checkpoint_linear_fallback_global_validation'
        applied=[]
    elif not accepted:
        final=lambda t:float(baseline(float(t)))
        status='checkpoint_linear_fallback_no_accepted_segments'
        applied=[]
    else:
        final=proposal;status='applied';applied=accepted
    report={'version':'1.0','status':status,'timing_changes_applied':bool(applied),
      'accepted_segment_count':len(accepted),'applied_segment_count':len(applied),
      'applied_segments':[{'segment_id':x['segment_id'],'start_nominal_time':round(x['start'],6),
        'end_nominal_time':round(x['end'],6),'maximum_change_ms':round(x['maximum_change_ms'],3)} for x in applied],
      'validation':{'overlap_free':not overlap,'monotonic':monotonic,
        'minimum_beat_stretch':round(beat_min,6),'maximum_beat_stretch':round(beat_max,6),
        'minimum_measure_stretch':round(measure_min,6),'maximum_measure_stretch':round(measure_max,6),
        'fallback_used':not bool(applied)}}
    return final,report
