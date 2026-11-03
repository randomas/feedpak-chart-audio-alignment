"""Prepared MuScriptor MIDI evidence for score/audio alignment.

MuScriptor runs outside this codebase.  This module discovers canonical
*.muscriptor.mid files in the material folder, projects their unpadded source
clock into the padded alignment clock, and performs an always-bidirectional,
asymmetric partial-sequence comparison against score events.

The initial policy is diagnostic-only: no transcription candidate changes
production timing.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json, math
import statistics

import feedpak_common as fc

try:
    import mido
except Exception:  # optional until the investigator is enabled
    mido = None

SUPPORTED_METHODS = frozenset({
    "dtw-checkpoint", "checkpoint-linear", "checkpoint-dtw-diagnostic",
    "checkpoint-dtw-selective", "continuous-anchor-linear",
    "continuous-anchor-dtw", "continuous-anchor-structural",
    "bar-landmark-regularized",
})
UNSUPPORTED_METHODS = frozenset({"nominal", "offset", "linear", "dtw"})
ROLE_FILENAMES = {
    "piano": "Piano.muscriptor.mid",
    "guitar": "Guitar.muscriptor.mid",
    "bass": "Bass.muscriptor.mid",
    "drums": "Drums.muscriptor.mid",
}

@dataclass(frozen=True)
class InvestigatorConfig:
    search_radius_seconds: float = 8.0
    window_seconds: float = 10.0
    window_step_seconds: float = 2.0
    minimum_score_events: int = 5
    minimum_audio_events: int = 5
    minimum_matched_events: int = 5
    target_phrase_groups: int = 10
    maximum_phrase_groups: int = 16
    onset_group_tolerance_seconds: float = 0.04
    distinct_location_seconds: float = 0.75
    minimum_similarity: float = 0.58
    minimum_uniqueness_margin: float = 0.05
    bidirectional_agreement_seconds: float = 0.30
    exact_pitch_reward: float = 1.0
    octave_pitch_reward: float = 0.72
    pitch_class_reward: float = 0.45
    missing_chart_penalty: float = 0.20
    extra_transcription_penalty: float = 0.035
    timing_sigma_seconds: float = 0.12
    local_stretch_min: float = 0.9
    local_stretch_max: float = 1.1
    incipit_sequence_groups: int = 6
    incipit_search_radius_seconds: float = 3.0
    minimum_authority_distinctiveness: float = 0.42
    minimum_authority_sequence_support: int = 6
    soft_origin_disagreement_seconds: float = 1.0
    incipit_opening_horizon_seconds: float = 8.0
    incipit_acquisition_phrase_limit: int = 4
    incipit_tracking_radius_seconds: float = 2.5
    incipit_tracking_min_confirmed_phrases: int = 3
    incipit_tracking_max_coast_phrases: int = 2
    incipit_tracking_max_contradictions: int = 1
    incipit_tracking_reacquire_radius_seconds: float = 5.0
    incipit_tracking_once_reacquire: bool = True
    streak_fit_min_landmarks: int = 8
    streak_fit_max_median_residual_seconds: float = 0.12
    streak_fit_max_residual_seconds: float = 0.35
    streak_fit_outlier_mad_multiplier: float = 3.5
    landmark_score_tolerance_seconds: float = 0.035
    landmark_midi_cluster_seconds: float = 0.10
    landmark_chain_stretch_min: float = 0.88
    landmark_chain_stretch_max: float = 1.12
    landmark_chain_start_tolerance_seconds: float = 0.35
    incipit_chain_root_tolerance_seconds: float = 2.0
    wide_monotonic_warp_enabled: bool = False
    version: str = "prepared-muscriptor-sequence-authority-v3"


def validate_method(method: str) -> None:
    if method not in SUPPORTED_METHODS:
        allowed=", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(
            f"--transcription-investigator is not available for alignment mode {method!r}. "
            f"Supported modes: {allowed}"
        )


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_case_insensitive(folder: Path, expected: str):
    target=expected.casefold()
    for item in folder.iterdir() if folder.is_dir() else []:
        if item.is_file() and item.name.casefold()==target:
            return item
    return None


def discover_prepared_files(song_folder):
    folder=Path(song_folder)
    return {role:_discover_case_insensitive(folder,name) for role,name in ROLE_FILENAMES.items()}


def _load_provenance(midi_path: Path):
    candidates=[midi_path.with_suffix(".json"), midi_path.with_name(midi_path.name[:-4]+".json")]
    for path in candidates:
        if path.is_file():
            try:return json.loads(path.read_text(encoding="utf-8")),str(path)
            except Exception as exc:return {"status":"invalid","reason":f"{type(exc).__name__}: {exc}"},str(path)
    return {"status":"unverified","timeline_reference":"source_audio_unpadded"},None


def parse_muscriptor_midi(path, role, padding_added=0.0):
    if mido is None:
        return {"provider":"muscriptor","source":role,"status":"unavailable",
                "reason":"mido_not_installed","events":[]}
    midi_path=Path(path); provenance,provenance_path=_load_provenance(midi_path)
    try:
        midi=mido.MidiFile(str(midi_path))
        current=0.0; active={}; events=[]
        # mido.merge_tracks preserves delta ticks and combines the tempo/event stream.
        for message in mido.merge_tracks(midi.tracks):
            current += mido.tick2second(message.time,midi.ticks_per_beat,500000)
            if message.type=="set_tempo":
                # mido MidiFile.length handles tempo changes, but tick2second needs current tempo.
                # Reparse below using a maintained tempo; this branch is retained for clarity.
                pass
        tempo=500000; current=0.0
        for message in mido.merge_tracks(midi.tracks):
            current += mido.tick2second(message.time,midi.ticks_per_beat,tempo)
            if message.type=="set_tempo":
                tempo=message.tempo; continue
            if message.type=="note_on" and message.velocity>0:
                active.setdefault((getattr(message,"channel",0),message.note),[]).append((current,message.velocity))
            elif message.type in ("note_off","note_on"):
                key=(getattr(message,"channel",0),message.note)
                if active.get(key):
                    start,velocity=active[key].pop(0)
                    event={"provider":"muscriptor","source":role,
                           "event_type":"drum_hit" if role=="drums" else "pitched_note",
                           "source_time":round(start,6),"audio_time":round(start+padding_added,6),
                           "duration":round(max(0.0,current-start),6),"pitch":int(message.note),
                           "pitch_class":int(message.note)%12,"velocity":int(velocity),
                           "raw_midi_note":int(message.note),
                           "canonical_drum_piece":fc.gm_drum_to_piece(int(message.note)) if role=="drums" else None,
                           "padding_added":float(padding_added)}
                    events.append(event)
        # Drums may omit note-offs in some MIDI writers: retain unmatched onsets as onset-only.
        for (_channel,note),starts in active.items():
            for start,velocity in starts:
                events.append({"provider":"muscriptor","source":role,
                    "event_type":"drum_hit" if role=="drums" else "pitched_note",
                    "source_time":round(start,6),"audio_time":round(start+padding_added,6),
                    "duration":0.0,"pitch":int(note),"pitch_class":int(note)%12,
                    "velocity":int(velocity),"raw_midi_note":int(note),
                    "canonical_drum_piece":fc.gm_drum_to_piece(int(note)) if role=="drums" else None,
                    "padding_added":float(padding_added)})
        events.sort(key=lambda x:(x["audio_time"],x["pitch"]))
        return {"provider":"muscriptor","source":role,"status":"available",
                "prepared_file":str(midi_path),"prepared_file_sha256":_sha256(midi_path),
                "provenance":provenance,"provenance_path":provenance_path,
                "timeline":{"midi_reference":"source_audio_unpadded","alignment_reference":"padded_audio",
                            "transform":"audio_time = source_time + padding_added",
                            "padding_added_seconds":float(padding_added)},
                "events":events,"retained_event_count":len(events)}
    except Exception as exc:
        return {"provider":"muscriptor","source":role,"status":"failed_nonfatal",
                "prepared_file":str(midi_path),"reason":f"{type(exc).__name__}: {exc}","events":[]}


def _score_event(row,role):
    if isinstance(row,(int,float)):
        return {"time":float(row),"pitches":[],"pieces":[],"role":role}
    pitches=row.get("pitches") or ([] if row.get("midi") is None else [row.get("midi")])
    piece=row.get("p") or row.get("piece") or row.get("canonical_drum_piece")
    return {"time":float(row.get("t_gp",row.get("time",0.0))),
            "pitches":[int(x) for x in pitches if x is not None],
            "pieces":([str(piece)] if piece else []),"role":role,
            "track_id":row.get("track_id") or row.get("track")}


def _group_onsets(events,time_key,tolerance,role,score_side=False):
    rows=sorted(events,key=lambda x:float(x[time_key])); groups=[]
    for row in rows:
        t=float(row[time_key])
        if not groups or t-groups[-1]["time"]>tolerance:
            groups.append({"time":t,"pitches":set(),"pieces":set(),"members":[],"tracks":set()})
        group=groups[-1]; group["members"].append(row)
        group["pitches"].update(int(x) for x in (row.get("pitches") or ([] if row.get("pitch") is None else [row["pitch"]])))
        if role=="drums":
            pieces=row.get("pieces") or ([] if not row.get("canonical_drum_piece") else [row["canonical_drum_piece"]])
            group["pieces"].update(str(x) for x in pieces)
        if row.get("track_id") is not None: group["tracks"].add(str(row["track_id"]))
    for group in groups:
        group["pitches"]=sorted(group["pitches"]); group["pieces"]=sorted(group["pieces"]); group["tracks"]=sorted(group["tracks"])
    return groups


def _set_similarity(score_group,audio_group,role,cfg):
    if role=="drums":
        expected=set(score_group["pieces"]); observed=set(audio_group["pieces"])
        if not expected or not observed:return 0.0
        inter=expected&observed
        backbone={"kick","snare"}
        reward=sum(1.25 if p in backbone else 1.0 for p in inter)
        denom=sum(1.25 if p in backbone else 1.0 for p in expected)
        return min(1.0,reward/max(denom,1e-9))
    expected=set(score_group["pitches"]); observed=set(audio_group["pitches"])
    if not expected or not observed:return 0.0
    rewards=[]
    for pitch in expected:
        if pitch in observed:rewards.append(cfg.exact_pitch_reward)
        elif any(abs(pitch-x)%12==0 for x in observed):rewards.append(cfg.octave_pitch_reward)
        elif any(pitch%12==x%12 for x in observed):rewards.append(cfg.pitch_class_reward)
        else:rewards.append(0.0)
    # Extras are intentionally not in the denominator.
    return sum(rewards)/len(rewards)


def _align_groups(score,audio,role,cfg,reverse=False):
    s=list(reversed(score)) if reverse else list(score); a=list(reversed(audio)) if reverse else list(audio)
    ns,na=len(s),len(a); neg=-1e9
    dp=[[neg]*(na+1) for _ in range(ns+1)]; cnt=[[0]*(na+1) for _ in range(ns+1)]; back=[[None]*(na+1) for _ in range(ns+1)]
    dp[0][0]=0.0
    for i in range(1,ns+1):dp[i][0]=dp[i-1][0]-cfg.missing_chart_penalty;back[i][0]=(i-1,0,"skip_score")
    for j in range(1,na+1):dp[0][j]=dp[0][j-1]-cfg.extra_transcription_penalty;back[0][j]=(0,j-1,"skip_audio")
    for i in range(1,ns+1):
        for j in range(1,na+1):
            sim=_set_similarity(s[i-1],a[j-1],role,cfg)
            options=[(dp[i-1][j]-cfg.missing_chart_penalty,cnt[i-1][j],i-1,j,"skip_score"),
                     (dp[i][j-1]-cfg.extra_transcription_penalty,cnt[i][j-1],i,j-1,"skip_audio")]
            if sim>0: options.append((dp[i-1][j-1]+sim,cnt[i-1][j-1]+1,i-1,j-1,"match"))
            val,n,pi,pj,op=max(options,key=lambda x:(x[0],x[1]));dp[i][j]=val;cnt[i][j]=n;back[i][j]=(pi,pj,op)
    pairs=[];i,j=ns,na
    while i or j:
        pi,pj,op=back[i][j]
        if op=="match":
            si=i-1;aj=j-1
            if reverse:si=ns-1-si;aj=na-1-aj
            pairs.append((si,aj,_set_similarity(score[si],audio[aj],role,cfg)))
        i,j=pi,pj
    pairs.sort(); matched=len(pairs)
    return {"score":dp[ns][na]/max(1,matched),"matched":matched,"coverage":matched/max(1,ns),
            "direction":"backward" if reverse else "forward","matched_pairs":pairs}


def _robust_affine_fit(score,audio,pairs):
    if not pairs:return {"offset":0.0,"stretch":1.0,"median_residual":None,"pair_count":0,"valid":False}
    st=[score[i]["time"] for i,j,_ in pairs]; at=[audio[j]["time"] for i,j,_ in pairs]
    slopes=[]
    for x in range(len(st)):
        for y in range(x+1,len(st)):
            ds=st[y]-st[x]
            if abs(ds)>1e-4:slopes.append((at[y]-at[x])/ds)
    stretch=statistics.median(slopes) if slopes else 1.0
    offset=statistics.median([a-stretch*s for s,a in zip(st,at)])
    residuals=[abs(a-(offset+stretch*s)) for s,a in zip(st,at)]
    median_residual=statistics.median(residuals)
    valid=all(math.isfinite(x) for x in (offset,stretch,median_residual)) and len(pairs)>=2
    return {"offset":offset if math.isfinite(offset) else 0.0,"stretch":stretch if math.isfinite(stretch) else 1.0,
            "median_residual":median_residual if math.isfinite(median_residual) else None,"pair_count":len(pairs),"valid":valid}


def _phrase_fingerprint(groups,role):
    times=[x["time"] for x in groups]
    spacings=[round((b-a)/max(times[-1]-times[0],1e-6),2) for a,b in zip(times,times[1:])]
    if role=="drums": symbols=[tuple(x["pieces"]) for x in groups]
    else:
        roots=[min(x["pitches"]) if x["pitches"] else None for x in groups]
        symbols=[None if a is None or b is None else b-a for a,b in zip(roots,roots[1:])]
    return {"symbols":symbols,"relative_spacings":spacings,"group_count":len(groups)}


def _phrase_windows(groups,cfg):
    out=[]; n=len(groups); size=min(cfg.target_phrase_groups,cfg.maximum_phrase_groups)
    for start in range(0,n-cfg.minimum_score_events+1,max(1,size//3)):
        end=min(n,start+size)
        if end-start>=cfg.minimum_score_events:out.append(groups[start:end])
    return out


def _candidate_for_phrase(score_phrase,audio_phrase,role,predicted,cfg):
    fw=_align_groups(score_phrase,audio_phrase,role,cfg,False); bw=_align_groups(score_phrase,audio_phrase,role,cfg,True)
    # Use the union of pairs supported in either direction; intersection is explicit confirmation.
    fset={(i,j) for i,j,_ in fw["matched_pairs"]}; bset={(i,j) for i,j,_ in bw["matched_pairs"]}; common=fset&bset
    pairs=fw["matched_pairs"] if len(fw["matched_pairs"])>=len(bw["matched_pairs"]) else bw["matched_pairs"]
    fit=_robust_affine_fit(score_phrase,audio_phrase,pairs)
    centre=(score_phrase[0]["time"]+score_phrase[-1]["time"])/2
    audio_centre=fit["offset"]+fit["stretch"]*centre
    sequence=(fw["coverage"]+bw["coverage"])/2
    timing=(math.exp(-fit["median_residual"]/max(cfg.timing_sigma_seconds,1e-6)) if fit["median_residual"] is not None and math.isfinite(fit["median_residual"]) else 0.0)
    stretch_score=1.0 if cfg.local_stretch_min<=fit["stretch"]<=cfg.local_stretch_max else max(0.0,1.0-abs(math.log(max(fit["stretch"],1e-6)))*1.5)
    prior=math.exp(-abs(audio_centre-predicted)/max(cfg.search_radius_seconds,1e-6))
    combined=.50*sequence+.25*timing+.15*stretch_score+.10*prior
    landmarks=[{"score_time":float(score_phrase[i]["time"]),
                "midi_time":float(audio_phrase[j]["time"]),"similarity":float(sim)}
               for i,j,sim in pairs]
    return {"score":combined,"audio_time":audio_centre,"forward":fw,"backward":bw,"fit":fit,
            "bidirectional_pair_count":len(common),"bidirectional_confirmed":len(common)>=cfg.minimum_matched_events,
            "support_count":min(fw["matched"],bw["matched"]),"timing_similarity":timing,
            "matched_landmarks":landmarks,
            "fingerprint":_phrase_fingerprint(score_phrase,role)}


def _cluster_distinct_locations(hypotheses,cfg):
    clusters=[]
    for row in sorted(hypotheses,key=lambda x:x["audio_time"]):
        if not clusters or row["audio_time"]-clusters[-1][-1]["audio_time"]>cfg.distinct_location_seconds:clusters.append([row])
        else:clusters[-1].append(row)
    representatives=[max(c,key=lambda x:x["score"]) for c in clusters]
    representatives.sort(key=lambda x:x["score"],reverse=True)
    return representatives


def build_phrase_correspondences(score_events,provider_results,baseline,cfg):
    candidates=[]; role_stats={}
    for provider in provider_results:
        role=provider["source"]
        score_rows=[_score_event(x,role) for x in (score_events.get(role) or [])]
        score_groups=_group_onsets(score_rows,"time",cfg.onset_group_tolerance_seconds,role,True)
        audio_rows=[]
        for event in provider.get("events") or []:
            row=dict(event);row["time"]=row["audio_time"];audio_rows.append(row)
        audio_groups=_group_onsets(audio_rows,"time",cfg.onset_group_tolerance_seconds,role,False)
        stats={"score_onset_groups":len(score_groups),"audio_onset_groups":len(audio_groups),"phrase_windows":0,"viable_windows":0,"candidates":0,"strong":0}
        for phrase in _phrase_windows(score_groups,cfg):
            stats["phrase_windows"]+=1; centre=(phrase[0]["time"]+phrase[-1]["time"])/2; predicted=float(baseline(centre))
            local=[x for x in audio_groups if predicted-cfg.search_radius_seconds<=x["time"]<=predicted+cfg.search_radius_seconds]
            if len(local)<cfg.minimum_audio_events:continue
            stats["viable_windows"]+=1; hypotheses=[]
            size=min(cfg.maximum_phrase_groups,max(cfg.minimum_audio_events,len(phrase)+4))
            for start in range(0,max(1,len(local)-cfg.minimum_audio_events+1)):
                aw=local[start:min(len(local),start+size)]
                if len(aw)>=cfg.minimum_audio_events:hypotheses.append(_candidate_for_phrase(phrase,aw,role,predicted,cfg))
            distinct=_cluster_distinct_locations(hypotheses,cfg)
            if not distinct:continue
            best=distinct[0]; runner=distinct[1]["score"] if len(distinct)>1 else 0.0; margin=best["score"]-runner
            strong=(best["score"]>=cfg.minimum_similarity and best["support_count"]>=cfg.minimum_matched_events
                    and best["bidirectional_confirmed"] and best["fit"]["valid"]
                    and cfg.local_stretch_min<=best["fit"]["stretch"]<=cfg.local_stretch_max
                    and best["fit"]["median_residual"] is not None and best["fit"]["median_residual"]<=0.30
                    and margin>=cfg.minimum_uniqueness_margin)
            candidate={"score_time":round(centre,6),"audio_time":round(best["audio_time"],6),"residual_seconds":round(best["audio_time"]-predicted,6),
                "provider":"muscriptor","source":role,"score_role":role,"candidate_type":"drum_pattern" if role=="drums" else "melodic_or_motif_phrase",
                "observation_type":"bidirectional_phrase_sequence","support_count":best["support_count"],"confidence":round(best["score"],4),
                "uniqueness_margin":round(margin,4),"distinct_location_count":len(distinct),"timing_similarity":round(best["timing_similarity"],4),
                "local_stretch":round(best["fit"]["stretch"],5),"median_timing_residual":(round(best["fit"]["median_residual"],5) if best["fit"]["median_residual"] is not None else None),
                "timing_fit_valid":bool(best["fit"]["valid"]),
                "forward":best["forward"],"backward":best["backward"],"bidirectional_pair_count":best["bidirectional_pair_count"],
                "bidirectional_confirmed":best["bidirectional_confirmed"],"phrase_fingerprint":best["fingerprint"],
                "diagnostic_status":"strong_candidate" if strong else "ambiguous_or_weak","production_eligible":False}
            candidates.append(candidate);stats["candidates"]+=1;stats["strong"]+=int(strong)
        role_stats[role]=stats
    return candidates,role_stats

def _global_monotonic_preview(candidates):
    """Independent forward/backward dynamic chains reconciled in the middle."""
    rows=sorted((dict(x) for x in candidates if x["diagnostic_status"]=="strong_candidate"),key=lambda x:(x["score_time"],x["audio_time"]))
    n=len(rows)
    if not n:return {"status":"diagnostic_only","selected_count":0,"selected":[],"unsupported_gaps":[],"monotonic":True}
    fw=[float(x["confidence"]) for x in rows]; fp=[None]*n
    for i in range(n):
        for j in range(i):
            if rows[j]["score_time"]<rows[i]["score_time"] and rows[j]["audio_time"]<rows[i]["audio_time"]:
                gap_penalty=.01*abs((rows[i]["audio_time"]-rows[j]["audio_time"])-(rows[i]["score_time"]-rows[j]["score_time"]))
                value=fw[j]+rows[i]["confidence"]-gap_penalty
                if value>fw[i]:fw[i]=value;fp[i]=j
    bw=[float(x["confidence"]) for x in rows]; bp=[None]*n
    for i in range(n-1,-1,-1):
        for j in range(i+1,n):
            if rows[i]["score_time"]<rows[j]["score_time"] and rows[i]["audio_time"]<rows[j]["audio_time"]:
                gap_penalty=.01*abs((rows[j]["audio_time"]-rows[i]["audio_time"])-(rows[j]["score_time"]-rows[i]["score_time"]))
                value=bw[j]+rows[i]["confidence"]-gap_penalty
                if value>bw[i]:bw[i]=value;bp[i]=j
    meet=max(range(n),key=lambda i:fw[i]+bw[i]-rows[i]["confidence"])
    left=[];i=meet
    while i is not None:left.append(i);i=fp[i]
    left.reverse();right=[];i=bp[meet]
    while i is not None:right.append(i);i=bp[i]
    selected_indices=left+right;selected=[]
    for i in selected_indices:
        row=rows[i];row["chain_support"]={"forward_score":round(fw[i],4),"backward_score":round(bw[i],4),
            "meet_in_middle":i==meet,"supported_from_left":fp[i] is not None or i==selected_indices[0],
            "supported_from_right":bp[i] is not None or i==selected_indices[-1]}
        selected.append(row)
    gaps=[]
    for left_row,right_row in zip(selected,selected[1:]):
        if right_row["score_time"]-left_row["score_time"]>12.0:
            gaps.append({"start_score_time":left_row["score_time"],"end_score_time":right_row["score_time"],
                         "forward_boundary_audio_time":left_row["audio_time"],"backward_boundary_audio_time":right_row["audio_time"],
                         "status":"unsupported","reason":"no_reliable_phrase_correspondence"})
    return {"status":"diagnostic_only","selected_count":len(selected),"selected":selected,"unsupported_gaps":gaps,
            "meet_index":meet,"forward_best_score":round(max(fw),4),"backward_best_score":round(max(bw),4),
            "monotonic":all(b["score_time"]>a["score_time"] and b["audio_time"]>a["audio_time"] for a,b in zip(selected,selected[1:]))}

def investigate(score_events,song_folder,baseline,padding_added=0.0,alignment_mode=None,config=None):
    cfg=config or InvestigatorConfig()
    if alignment_mode is not None:validate_method(alignment_mode)
    discovered=discover_prepared_files(song_folder); providers=[]
    for role,path in discovered.items():
        if path is None:
            providers.append({"provider":"muscriptor","source":role,"status":"not_prepared",
                              "expected_file":ROLE_FILENAMES[role],"events":[]})
        else:
            providers.append(parse_muscriptor_midi(path,role,padding_added))
    candidates,role_stats=build_phrase_correspondences(score_events,providers,baseline,cfg)
    preview=_global_monotonic_preview(candidates)
    strong=[x for x in candidates if x["diagnostic_status"]=="strong_candidate"]
    available=sum(x["status"]=="available" for x in providers)
    return {"version":"2.0","enabled":True,"provider":"prepared_muscriptor_midi",
            "status":"complete" if available==len(ROLE_FILENAMES) else "partial",
            "alignment_mode":alignment_mode,"timing_changes_applied":False,"config":asdict(cfg),
            "prepared_files":providers,"anchor_candidates":candidates,
            "dense_anchor_matrix":{"representation":"sparse_bidirectional_candidates","candidate_count":len(candidates)},
            "global_monotonic_path_preview":preview,"role_diagnostics":role_stats,
            "summary":{"prepared_role_count":available,"expected_role_count":len(ROLE_FILENAMES),
                "normalized_event_count":sum(len(x.get("events") or []) for x in providers),
                "candidate_correspondence_count":len(candidates),"strong_bidirectional_candidates":len(strong),
                "ambiguous_candidates":len(candidates)-len(strong),"selected_monotonic_candidates":preview["selected_count"]},
            "method_consumption":{"mode":"diagnostic_only","candidate_count_received":len(candidates),"candidate_count_applied":0}}

# ---------------------------------------------------------------------------
# v3 nominal-prior, multi-pass, globally constrained phrase alignment
# ---------------------------------------------------------------------------

def _cfg(cfg, name, default):
    return getattr(cfg, name, default)


def make_nominal_opening_prior(opening_score=None, opening_audio=None, global_scale=None):
    """Method-independent coarse chart-to-audio prior.

    Score products already include the chart/count-in offset.  Identity is
    therefore the safest fallback.  An accepted opening pair and scale refine
    it without inheriting any alignment method's later warp.
    """
    if opening_score is None or opening_audio is None:
        return lambda t: float(t), {"kind":"nominal_identity","scale":1.0,"offset":0.0}
    scale=float(global_scale) if global_scale is not None and math.isfinite(float(global_scale)) else 1.0
    score=float(opening_score); audio=float(opening_audio)
    return (lambda t: audio+scale*(float(t)-score)), {
        "kind":"opening_scale","opening_score":score,"opening_audio":audio,
        "scale":scale,"offset":audio-scale*score,
    }


def _phrase_distinctiveness(groups, role):
    fp=_phrase_fingerprint(groups,role); symbols=fp["symbols"]
    usable=[repr(x) for x in symbols if x is not None]
    diversity=len(set(usable))/max(1,len(usable))
    spacing=fp["relative_spacings"]
    spacing_diversity=len(set(spacing))/max(1,len(spacing))
    length=min(1.0,len(groups)/10.0)
    return round(.50*diversity+.25*spacing_diversity+.25*length,4)


def _candidate_search(score_phrase,audio_groups,role,predicted,radius,cfg,pass_name):
    local=[x for x in audio_groups if predicted-radius<=x["time"]<=predicted+radius]
    if len(local)<cfg.minimum_audio_events:return []
    hypotheses=[]; size=min(cfg.maximum_phrase_groups,max(cfg.minimum_audio_events,len(score_phrase)+4))
    for start in range(0,max(1,len(local)-cfg.minimum_audio_events+1)):
        aw=local[start:min(len(local),start+size)]
        if len(aw)>=cfg.minimum_audio_events:
            row=_candidate_for_phrase(score_phrase,aw,role,predicted,cfg)
            row["search_pass"]=pass_name; row["search_radius_seconds"]=radius
            row["search_centre_audio_time"]=predicted
            hypotheses.append(row)
    return _cluster_distinct_locations(hypotheses,cfg)


def _materialize_candidate(phrase,role,predicted,best,runner,distinct,cfg,pass_name,prior_kind):
    centre=(phrase[0]["time"]+phrase[-1]["time"])/2
    margin=best["score"]-(runner["score"] if runner else 0.0)
    fit=best["fit"]; local_valid=(best["score"]>=cfg.minimum_similarity
        and best["support_count"]>=cfg.minimum_matched_events
        and best["bidirectional_confirmed"] and fit["valid"]
        and cfg.local_stretch_min<=fit["stretch"]<=cfg.local_stretch_max
        and fit["median_residual"] is not None and fit["median_residual"]<=0.30)
    repeated=(margin<cfg.minimum_uniqueness_margin and best["support_count"]>=max(7,cfg.minimum_matched_events))
    status=("valid_unique_phrase" if local_valid and margin>=cfg.minimum_uniqueness_margin else
            "valid_repeated_motif_hypothesis" if local_valid and repeated else "ambiguous_or_weak")
    return {"score_time":round(centre,6),"audio_time":round(best["audio_time"],6),
        "residual_seconds":round(best["audio_time"]-predicted,6),"provider":"muscriptor",
        "source":role,"score_role":role,"candidate_type":"drum_pattern" if role=="drums" else "melodic_or_motif_phrase",
        "observation_type":"bidirectional_phrase_sequence","support_count":best["support_count"],
        "local_phrase_confidence":round(best["score"],4),"confidence":round(best["score"],4),
        "global_placement_confidence":0.0,"uniqueness_margin":round(margin,4),
        "distinct_location_count":len(distinct),"timing_similarity":round(best["timing_similarity"],4),
        "local_stretch":round(fit["stretch"],5),
        "median_timing_residual":round(fit["median_residual"],5) if fit["median_residual"] is not None else None,
        "timing_fit_valid":bool(fit["valid"]),
        "sequence_match_valid":bool(best["score"]>=cfg.minimum_similarity
            and best["support_count"]>=cfg.minimum_matched_events
            and best["bidirectional_confirmed"]
            and fit["median_residual"] is not None and fit["median_residual"]<=0.30),
        "matched_landmarks":best.get("matched_landmarks",[]),
        "forward":best["forward"],"backward":best["backward"],
        "bidirectional_pair_count":best["bidirectional_pair_count"],
        "bidirectional_confirmed":best["bidirectional_confirmed"],"phrase_fingerprint":best["fingerprint"],
        "phrase_distinctiveness":_phrase_distinctiveness(phrase,role),"search_pass":pass_name,
        "search_prior_kind":prior_kind,"search_centre_audio_time":round(predicted,6),
        "diagnostic_status":status,"production_eligible":False}


def build_multipass_phrase_correspondences(score_events,provider_results,nominal_prior,cfg,
                                             recovery_prior=None, pass_label="initial",
                                             observed_incipit_prior=None):
    """Chart-first sequence search around expected and observed music origins.

    Discovery is deliberately permissive: expected-origin, offset-incipit and
    recovery centres all get a chance to produce a candidate.  Authority is
    decided later from sequence support, uniqueness and path coherence.
    """
    candidates=[]; role_stats={}
    narrow=_cfg(cfg,"nominal_search_radius_seconds",3.0)
    wide=_cfg(cfg,"wide_search_radius_seconds",8.0)
    recovery_radius=_cfg(cfg,"recovery_search_radius_seconds",4.0)
    for provider in provider_results:
        role=provider["source"]
        score_rows=[_score_event(x,role) for x in (score_events.get(role) or [])]
        sg=_group_onsets(score_rows,"time",cfg.onset_group_tolerance_seconds,role,True)
        ar=[]
        for event in provider.get("events") or []:
            row=dict(event);row["time"]=row["audio_time"];ar.append(row)
        ag=_group_onsets(ar,"time",cfg.onset_group_tolerance_seconds,role,False)
        stats={"score_onset_groups":len(sg),"audio_onset_groups":len(ag),"phrase_windows":0,
               "expected_origin_matches":0,"observed_incipit_matches":0,
               "wide_recoveries":0,"path_recoveries":0,"candidates":0}
        phrases=sorted(_phrase_windows(sg,cfg),key=lambda p:_phrase_distinctiveness(p,role),reverse=True)
        for phrase in phrases:
            stats["phrase_windows"]+=1
            centre=(phrase[0]["time"]+phrase[-1]["time"])/2
            searches=[("expected_origin",nominal_prior,narrow)]
            if observed_incipit_prior is not None:
                searches.append(("offset_incipit",observed_incipit_prior,
                                 _cfg(cfg,"incipit_search_radius_seconds",3.0)))
            distinct=[]; pass_name="none"; prior_kind="none"; predicted=float(nominal_prior(centre))
            for label,prior,radius in searches:
                centre_prediction=float(prior(centre))
                found=_candidate_search(phrase,ag,role,centre_prediction,radius,cfg,label)
                if found:
                    found[0]["origin_hypothesis"]=label
                    distinct.extend(found)
            distinct=_cluster_distinct_locations(distinct,cfg)
            if distinct:
                pass_name=distinct[0].get("search_pass","dual_origin")
                prior_kind=distinct[0].get("origin_hypothesis",pass_name)
                predicted=float(distinct[0].get("search_centre_audio_time",predicted))
            if not distinct or distinct[0]["score"]<cfg.minimum_similarity:
                found=_candidate_search(phrase,ag,role,float(nominal_prior(centre)),wide,cfg,"expected_wide")
                distinct=_cluster_distinct_locations(distinct+found,cfg)
                pass_name="expected_wide"; prior_kind="expected_origin"
            if recovery_prior is not None and (not distinct or distinct[0]["score"]<cfg.minimum_similarity):
                recovery_predicted=float(recovery_prior(centre))
                found=_candidate_search(phrase,ag,role,recovery_predicted,recovery_radius,cfg,"path_recovery")
                distinct=_cluster_distinct_locations(distinct+found,cfg)
                pass_name="path_recovery"; prior_kind="recovery_path"; predicted=recovery_predicted
            if not distinct: continue
            best=distinct[0]; runner=distinct[1] if len(distinct)>1 else None
            row=_materialize_candidate(phrase,role,predicted,best,runner,distinct,cfg,pass_name,prior_kind)
            row["origin_hypothesis"]=best.get("origin_hypothesis",prior_kind)
            row["sequence_group_count"]=len(phrase)
            row["sequence_first_score_time"]=round(phrase[0]["time"],6)
            row["sequence_last_score_time"]=round(phrase[-1]["time"],6)
            # A single onset can be retained as a diagnostic, but can never
            # become a MIDI timing control.
            min_sequence=_cfg(cfg,"minimum_authority_sequence_support",6)
            distinctiveness=float(row.get("phrase_distinctiveness",0.0))
            distinct_floor=(.52 if role=="drums" else .32)
            sequence_ok=(len(phrase)>=min_sequence and row["support_count"]>=min_sequence
                         and row["bidirectional_confirmed"])
            recognizable=sequence_ok and distinctiveness>=distinct_floor
            unique=recognizable and row.get("uniqueness_margin",0.0)>=cfg.minimum_uniqueness_margin
            row["authority_tier"]=("A" if unique else "B" if recognizable else
                                   "C" if sequence_ok else "D")
            row["sequence_authority_eligible"]=recognizable
            row["authority_rationale"]={"sequence_groups":len(phrase),
                "matched_groups":row["support_count"],"distinctiveness":distinctiveness,
                "distinctiveness_floor":distinct_floor,"single_event_authority":False}
            candidates.append(row);stats["candidates"]+=1
            key=row["origin_hypothesis"]
            if key=="expected_origin":stats["expected_origin_matches"]+=1
            elif key=="offset_incipit":stats["observed_incipit_matches"]+=1
            elif key=="recovery_path":stats["path_recoveries"]+=1
            else:stats["wide_recoveries"]+=1
        role_stats[role]=stats
    candidates.sort(key=lambda x:(x["score_time"],-x["local_phrase_confidence"]))
    return candidates,role_stats


def build_role_activity_profiles(score_events, providers):
    """Describe score activity separately from noisy provider labelling."""
    by_provider={x.get("source"):x for x in providers}
    all_first=[]
    role_times={}
    for role in ROLE_FILENAMES:
        times=sorted(_score_event(x,role)["time"] for x in (score_events.get(role) or []))
        role_times[role]=times
        if times: all_first.append(times[0])
    song_first=min(all_first) if all_first else 0.0
    out={}
    for role in ROLE_FILENAMES:
        times=role_times[role]
        provider=by_provider.get(role,{})
        incipit_active=bool(times and len(times)>=3 and times[0]<=song_first+8.0)
        out[role]={"score_event_count":len(times),"first_score_time":times[0] if times else None,
                   "last_score_time":times[-1] if times else None,
                   "prepared":provider.get("status")=="available",
                   "incipit_authority_eligible":incipit_active,
                   "policy":"inactive_role_observations_retained_but_not_authoritative"}
    return out



def classify_opening_role(score_events, cfg):
    """Classify a chart-defined solo/dominant opening role.

    Stem names are deliberately ignored.  The decision uses chart activity in
    the first opening horizon so contaminated MIDI cannot nominate itself.
    """
    rows={}
    first=[]
    for role in ROLE_FILENAMES:
        events=[_score_event(x,role) for x in (score_events.get(role) or [])]
        events.sort(key=lambda x:x["time"]); rows[role]=events
        if events:first.append(events[0]["time"])
    if not first:
        return {"status":"unavailable","classification":"uncertain","role":None,
                "reason":"no_chart_roles"}
    origin=min(first); end=origin+_cfg(cfg,"incipit_opening_horizon_seconds",8.0)
    activity={}
    for role,events in rows.items():
        active=[x for x in events if origin<=x["time"]<=end]
        groups=_group_onsets(active,"time",cfg.onset_group_tolerance_seconds,role,True)
        activity[role]={"first_score_time":events[0]["time"] if events else None,
                        "opening_group_count":len(groups),
                        "opening_pitch_count":sum(len(x.get("pitches") or []) for x in groups),
                        "opening_piece_count":sum(len(x.get("pieces") or []) for x in groups)}
    ranked=sorted(activity.items(),key=lambda kv:(kv[1]["opening_group_count"],kv[1]["opening_pitch_count"]),reverse=True)
    role,lead=ranked[0]; second=ranked[1][1]["opening_group_count"] if len(ranked)>1 else 0
    lead_count=lead["opening_group_count"]
    if lead_count<cfg.minimum_score_events:
        classification="uncertain";reason="insufficient_opening_sequence"
    elif second==0 or lead_count>=max(3,2*second):
        classification="solo";reason="single_role_opening_dominance"
    elif lead_count>=1.35*max(1,second):
        classification="dominant";reason="clear_opening_role_dominance"
    else:
        classification="ensemble";reason="multiple_roles_share_opening"
    return {"status":"available","classification":classification,
            "role":role if classification in {"solo","dominant"} else None,
            "song_opening_score_time":origin,"opening_horizon_end":end,
            "reason":reason,"role_activity":activity}


def _tracking_candidate(phrase,audio_groups,role,predictions,radius,cfg,label):
    found=[]
    for kind,predicted in predictions:
        for row in _candidate_search(phrase,audio_groups,role,float(predicted),radius,cfg,label+"_"+kind):
            item=dict(row);item["tracking_origin_hypothesis"]=kind;found.append(item)
    found=_cluster_distinct_locations(found,cfg)
    return found




def select_monotonic_landmark_chain(landmarks,cfg):
    """Select one sequential chart->MIDI occurrence before affine fitting.

    Overlapping phrase windows may assign one chart onset to several MIDI
    onsets.  This groups those alternatives and uses dynamic programming to
    select one monotonic, locally safe chain rooted at the acquired incipit.
    """
    raw=[]
    for item in landmarks:
        try:x=float(item["score_time"]);y=float(item["midi_time"])
        except (KeyError,TypeError,ValueError):continue
        if math.isfinite(x) and math.isfinite(y):
            raw.append({"score_time":x,"midi_time":y,"similarity":float(item.get("similarity",1.0)),
                        "phrase_index":int(item.get("phrase_index",0)),
                        "acquisition":bool(item.get("acquisition",False))})
    if not raw:return {"status":"no_landmarks","selected":[],"alternatives":[],"raw_count":0}
    raw.sort(key=lambda r:(r["score_time"],r["midi_time"]))
    # Group near-identical chart onsets, then cluster their competing MIDI times.
    score_groups=[];tol=_cfg(cfg,"landmark_score_tolerance_seconds",.035)
    for row in raw:
        if not score_groups or row["score_time"]-score_groups[-1][-1]["score_time"]>tol:score_groups.append([row])
        else:score_groups[-1].append(row)
    nodes=[];alternatives=[];midi_tol=_cfg(cfg,"landmark_midi_cluster_seconds",.10)
    for gi,group in enumerate(score_groups):
        clusters=[]
        for row in sorted(group,key=lambda r:r["midi_time"]):
            if not clusters or row["midi_time"]-clusters[-1][-1]["midi_time"]>midi_tol:clusters.append([row])
            else:clusters[-1].append(row)
        for ci,cluster in enumerate(clusters):
            weights=[max(.05,r["similarity"]) for r in cluster];total=sum(weights)
            x=sum(r["score_time"]*w for r,w in zip(cluster,weights))/total
            y=sum(r["midi_time"]*w for r,w in zip(cluster,weights))/total
            node={"score_time":x,"midi_time":y,"similarity":max(r["similarity"] for r in cluster),
                  "support":len(cluster),"score_group":gi,"midi_cluster":ci,
                  "acquisition_support":sum(r["acquisition"] for r in cluster),
                  "first_phrase_index":min(r["phrase_index"] for r in cluster),
                  "last_phrase_index":max(r["phrase_index"] for r in cluster)}
            nodes.append(node)
            if len(clusters)>1:alternatives.append(node)
    nodes.sort(key=lambda r:(r["score_time"],r["midi_time"]))
    if not nodes:return {"status":"no_clustered_landmarks","selected":[],"alternatives":alternatives,"raw_count":len(raw)}
    first_score=min(r["score_time"] for r in nodes if r["acquisition_support"]>0) if any(r["acquisition_support"]>0 for r in nodes) else nodes[0]["score_time"]
    start_tol=_cfg(cfg,"landmark_chain_start_tolerance_seconds",.35)
    acquisition_nodes=[r for r in nodes if r["acquisition_support"]>0 and r["score_time"]<=first_score+start_tol]
    if not acquisition_nodes:acquisition_nodes=[r for r in nodes if r["score_time"]<=nodes[0]["score_time"]+start_tol]
    # The median acquisition translation is only a soft start preference.
    acquisition_offset=statistics.median([r["midi_time"]-r["score_time"] for r in acquisition_nodes])
    n=len(nodes);neg=-1e18;dp=[neg]*n;prev=[None]*n;length=[0]*n
    min_edge=_cfg(cfg,"landmark_chain_stretch_min",.88);max_edge=_cfg(cfg,"landmark_chain_stretch_max",1.12)
    acquisition_keys={(r["score_group"],r["midi_cluster"]) for r in acquisition_nodes}
    for i,node in enumerate(nodes):
        if (node["score_group"],node["midi_cluster"]) in acquisition_keys:
            start_error=abs((node["midi_time"]-node["score_time"])-acquisition_offset)
            dp[i]=2.0+node["similarity"]+.15*math.log1p(node["support"])-start_error
            length[i]=1
        for j in range(i):
            a=nodes[j]
            if node["score_group"]<=a["score_group"]:continue
            ds=node["score_time"]-a["score_time"];da=node["midi_time"]-a["midi_time"]
            if ds<=0 or da<=0:continue
            edge=da/ds
            if not min_edge<=edge<=max_edge:continue
            phrase_backtrack=max(0,a["first_phrase_index"]-node["last_phrase_index"])
            value=dp[j]+1.0+node["similarity"]+.12*math.log1p(node["support"])-.8*abs(edge-1.0)-.2*phrase_backtrack
            if value>dp[i]:dp[i]=value;prev[i]=j;length[i]=length[j]+1
    valid=[i for i in range(n) if dp[i]>neg/2]
    if not valid:return {"status":"no_monotonic_safe_chain","selected":[],"alternatives":alternatives,
                         "raw_count":len(raw),"node_count":len(nodes)}
    end=max(valid,key=lambda i:(length[i],dp[i],nodes[i]["score_time"]-first_score))
    idxs=[];cur=end
    while cur is not None:idxs.append(cur);cur=prev[cur]
    idxs.reverse();selected=[nodes[i] for i in idxs]
    selected_keys={(r["score_group"],r["midi_cluster"]) for r in selected}
    rejected=[r for r in nodes if (r["score_group"],r["midi_cluster"]) not in selected_keys]
    return {"status":"available","selected":selected,"alternatives":rejected,
            "raw_count":len(raw),"node_count":len(nodes),"selected_count":len(selected),
            "competing_node_count":len(rejected),"acquisition_offset_seconds":round(acquisition_offset,6),
            "edge_stretch_range":[min_edge,max_edge]}

def fit_chart_to_midi_streak(landmarks,cfg):
    """Robustly fit the chart clock onto fixed performance-MIDI timestamps."""
    chain=select_monotonic_landmark_chain(landmarks,cfg)
    rows=[{"score_time":float(r["score_time"]),"midi_time":float(r["midi_time"]),
           "similarity":float(r.get("similarity",1.0)),"support":int(r.get("support",1))}
          for r in chain.get("selected",[])]
    if len(rows)<2:return {"valid":False,"reason":"unsafe_or_incoherent_chart_to_midi_fit",
                           "landmark_count":len(landmarks),"chain_summary":{k:v for k,v in chain.items() if k not in ('selected','alternatives')}}
    active=rows[:]
    for _ in range(3):
        fit=_robust_affine_fit([{"time":r["score_time"]} for r in active],
                               [{"time":r["midi_time"]} for r in active],
                               [(i,i,1.0) for i in range(len(active))])
        residuals=[abs(r["midi_time"]-(fit["offset"]+fit["stretch"]*r["score_time"])) for r in active]
        med=statistics.median(residuals) if residuals else math.inf
        mad=statistics.median([abs(x-med) for x in residuals]) if residuals else math.inf
        threshold=max(_cfg(cfg,"streak_fit_max_median_residual_seconds",.12),
                      med+_cfg(cfg,"streak_fit_outlier_mad_multiplier",3.5)*max(mad,.01))
        kept=[r for r,res in zip(active,residuals) if res<=threshold]
        if len(kept)==len(active) or len(kept)<2:break
        active=kept
    fit=_robust_affine_fit([{"time":r["score_time"]} for r in active],
                           [{"time":r["midi_time"]} for r in active],
                           [(i,i,1.0) for i in range(len(active))])
    residuals=[abs(r["midi_time"]-(fit["offset"]+fit["stretch"]*r["score_time"])) for r in active]
    median=statistics.median(residuals) if residuals else math.inf; maximum=max(residuals) if residuals else math.inf
    span=active[-1]["score_time"]-active[0]["score_time"] if active else 0.0
    valid=(len(active)>=_cfg(cfg,"streak_fit_min_landmarks",8) and span>=2.0 and
           cfg.local_stretch_min<=fit["stretch"]<=cfg.local_stretch_max and
           median<=_cfg(cfg,"streak_fit_max_median_residual_seconds",.12) and
           maximum<=_cfg(cfg,"streak_fit_max_residual_seconds",.35))
    timing_classification=("normal_timing_fit" if cfg.local_stretch_min>=.9 and cfg.local_stretch_max<=1.1 else
        "probable_chart_timing_discrepancy" if .75<=fit["stretch"]<=1.25 else
        "extreme_chart_timing_discrepancy")
    return {"valid":valid,"reason":None if valid else "unsafe_or_incoherent_chart_to_midi_fit",
            "timing_classification":timing_classification,
            "wide_warp_enabled":bool(_cfg(cfg,"wide_monotonic_warp_enabled",False)),
            "timeline_model":"chart_to_performance_midi","fixed_timeline":"prepared_midi_audio_clock",
            "transformed_timeline":"chart_clock","offset":round(fit["offset"],6),
            "chart_stretch":round(fit["stretch"],8),"landmark_count":len(landmarks),
            "chain_landmark_count":len(rows),"inlier_count":len(active),
            "rejected_landmark_count":len(landmarks)-len(active),
            "chain_summary":{k:v for k,v in chain.items() if k not in ("selected","alternatives")},
            "chart_start":active[0]["score_time"] if active else None,
            "chart_end":active[-1]["score_time"] if active else None,
            "midi_start":active[0]["midi_time"] if active else None,
            "midi_end":active[-1]["midi_time"] if active else None,
            "median_residual_seconds":round(median,6) if math.isfinite(median) else None,
            "maximum_residual_seconds":round(maximum,6) if math.isfinite(maximum) else None,
            "inliers":active}

def track_incipit_role(score_events, providers, nominal_prior, observed_incipit_prior, cfg):
    """Acquire the first recognizable opening-role phrase, then follow it.

    Missing transcription causes COASTING and creates no controls.  Strongly
    contradictory observations or an exhausted coast end the streak with an
    explicit reason.  A single cautious reacquisition is permitted.
    """
    opening=classify_opening_role(score_events,cfg)
    role=opening.get("role")
    report={"enabled":True,"opening_role":opening,"state":"RELEASED",
            "events":[],"streaks":[],"representative_controls":[],
            "timing_changes_applied":False}
    if role is None:
        report.update({"status":"not_applicable","release_reason":opening.get("reason")});return report
    provider=next((x for x in providers if x.get("source")==role and x.get("status")=="available"),None)
    if provider is None:
        report.update({"status":"acquisition_failed","release_reason":"opening_role_midi_unavailable"});return report
    score_rows=[_score_event(x,role) for x in (score_events.get(role) or [])]
    sg=_group_onsets(score_rows,"time",cfg.onset_group_tolerance_seconds,role,True)
    midi_rows=[]
    for event in provider.get("events") or []:
        row=dict(event);row["time"]=row["audio_time"];midi_rows.append(row)
    ag=_group_onsets(midi_rows,"time",cfg.onset_group_tolerance_seconds,role,False)
    phrases=sorted(_phrase_windows(sg,cfg),key=lambda p:p[0]["time"])
    first_score=sg[0]["time"] if sg else 0.0
    acquisition=[p for p in phrases if p[0]["time"]<=first_score+_cfg(cfg,"incipit_opening_horizon_seconds",8.0)]
    acquisition=acquisition[:_cfg(cfg,"incipit_acquisition_phrase_limit",4)]
    acquired=None
    for phrase in acquisition:
        centre=(phrase[0]["time"]+phrase[-1]["time"])/2
        predictions=[("expected_origin",nominal_prior(centre))]
        if observed_incipit_prior is not None:predictions.append(("offset_incipit",observed_incipit_prior(centre)))
        candidates=_tracking_candidate(phrase,ag,role,predictions,
            _cfg(cfg,"incipit_tracking_radius_seconds",2.5),cfg,"acquire")
        if not candidates:continue
        best=candidates[0];runner=candidates[1] if len(candidates)>1 else None
        row=_materialize_candidate(phrase,role,predictions[0][1],best,runner,candidates,cfg,
                                   "incipit_acquisition",best.get("tracking_origin_hypothesis","expected_origin"))
        if row["diagnostic_status"]!="ambiguous_or_weak" and row["support_count"]>=cfg.minimum_authority_sequence_support:
            acquired=(phrase,row);break
    if acquired is None:
        report.update({"status":"acquisition_failed","release_reason":"no_recognizable_opening_sequence"})
        report["events"].append({"event":"tracking_acquisition_failed","role":role,
                                 "reason":"no_recognizable_opening_sequence","action":"retain_baseline"})
        return report
    phrase,row=acquired
    accepted=[];streak=[];misses=0;contradictions=0;reacquired=False
    current_offset=float(row["audio_time"])-float(row["score_time"])
    current_stretch=float(row.get("local_stretch") or 1.0)
    report["state"]="LOCKED"
    report["events"].append({"event":"tracking_lock_acquired","role":role,
        "score_time":row["score_time"],"audio_time":row["audio_time"],
        "origin_hypothesis":row.get("search_prior_kind"),"support_count":row["support_count"]})
    start_index=phrases.index(phrase)
    for idx in range(start_index,len(phrases)):
        p=phrases[idx]
        if idx==start_index:candidate=row
        else:
            centre=(p[0]["time"]+p[-1]["time"])/2
            predicted=current_offset+current_stretch*centre
            radius=(_cfg(cfg,"incipit_tracking_reacquire_radius_seconds",5.0) if report["state"]=="REACQUIRING"
                    else _cfg(cfg,"incipit_tracking_radius_seconds",2.5))
            found=_tracking_candidate(p,ag,role,[("tracked_projection",predicted)],radius,cfg,"continue")
            candidate=None
            if found:
                best=found[0];runner=found[1] if len(found)>1 else None
                candidate=_materialize_candidate(p,role,predicted,best,runner,found,cfg,
                    "incipit_tracking","tracked_projection")
        observable=bool(candidate and candidate.get("support_count",0)>=cfg.minimum_authority_sequence_support)
        # Local short-phrase slope is diagnostic only.  Continuation is judged
        # by sequence evidence; production stretch is fitted robustly over the
        # accumulated chart-to-MIDI landmarks.
        valid=bool(observable and candidate.get("sequence_match_valid"))
        if valid:
            candidate=dict(candidate);candidate["authority_tier"]="opening_role_streak"
            candidate["sequence_authority_eligible"]=True;accepted.append(candidate);streak.append(candidate)
            deltas=[float(x["midi_time"])-float(x["score_time"]) for x in (candidate.get("matched_landmarks") or [])]
            if deltas:current_offset=statistics.median(deltas)
            current_stretch=1.0
            misses=0;contradictions=0
            if report["state"]=="REACQUIRING":
                reacquired=True;report["events"].append({"event":"tracking_lock_reacquired","role":role,
                    "score_time":candidate["score_time"],"audio_time":candidate["audio_time"]})
            report["state"]="LOCKED";continue
        if not observable:
            misses+=1;report["state"]="COASTING"
            report["events"].append({"event":"tracking_phrase_unobserved","role":role,
                "score_time":round((p[0]["time"]+p[-1]["time"])/2,6),
                "consecutive_unobserved_phrases":misses,"action":"coast_without_control"})
            if misses<=_cfg(cfg,"incipit_tracking_max_coast_phrases",2):continue
            reason="insufficient_midi_observation"
        else:
            contradictions+=1;reason="contradictory_or_ambiguous_sequence"
            if contradictions<=_cfg(cfg,"incipit_tracking_max_contradictions",1):continue
        if _cfg(cfg,"incipit_tracking_once_reacquire",True) and not reacquired:
            report["state"]="REACQUIRING";misses=0;contradictions=0;reacquired=True
            report["events"].append({"event":"tracking_lock_lost","role":role,"reason":reason,
                "last_confirmed_score_time":accepted[-1]["score_time"] if accepted else None,
                "action":"attempt_single_reacquisition"});continue
        report["state"]="LOST"
        report["events"].append({"event":"tracking_lock_lost","role":role,"reason":reason,
            "last_confirmed_score_time":accepted[-1]["score_time"] if accepted else None,
            "first_unresolved_score_time":round((p[0]["time"]+p[-1]["time"])/2,6),
            "action":"release_to_baseline","timing_controls_created_in_gap":False});break
    minimum=_cfg(cfg,"incipit_tracking_min_confirmed_phrases",3)
    landmarks=[]
    for phrase_index,item in enumerate(accepted):
        for raw in item.get("matched_landmarks") or []:
            landmark=dict(raw);landmark["phrase_index"]=phrase_index;landmark["acquisition"]=(phrase_index==0)
            landmark["phrase_confidence"]=item.get("local_phrase_confidence")
            landmarks.append(landmark)
    streak_fit=fit_chart_to_midi_streak(landmarks,cfg)
    geometry_valid=len(accepted)>=minimum and bool(streak_fit.get("valid"))
    opening_start=float(opening.get("song_opening_score_time",first_score))
    chain_start=streak_fit.get("chart_start")
    root_distance=None if chain_start is None else float(chain_start)-opening_start
    root_tolerance=_cfg(cfg,"incipit_chain_root_tolerance_seconds",2.0)
    root_valid=bool(geometry_valid and root_distance is not None and -0.05<=root_distance<=root_tolerance)
    classification=("incipit_role_streak" if root_valid else "regional_role_streak" if geometry_valid else "rejected_geometry")
    controls=[];diagnostic_controls=[];incipit_control=None
    if geometry_valid:
        offset=float(streak_fit["offset"]);stretch=float(streak_fit["chart_stretch"])
        first_chart=float(streak_fit["chart_start"]);last_chart=float(streak_fit["chart_end"])
        sample_times=[first_chart]
        for item in accepted:
            t=float(item["score_time"])
            if first_chart<t<last_chart and t-sample_times[-1]>=4.0:sample_times.append(t)
        if last_chart>sample_times[-1]:sample_times.append(last_chart)
        for t in sample_times:
            diagnostic_controls.append({"score_time":t,"audio_time":offset+stretch*t,
                "kind":"incipit_acquisition_onset" if root_valid and t==first_chart else "regional_monotonic_chain_fit",
                "authority":classification,"source":role,"chart_stretch":stretch,
                "streak_median_residual_seconds":streak_fit["median_residual_seconds"]})
        if root_valid:
            controls=list(diagnostic_controls);incipit_control=dict(controls[0])
    confirmed=root_valid
    streak_fit["root_validation"]={"opening_score_time":round(opening_start,6),
        "selected_chain_start":None if chain_start is None else round(float(chain_start),6),
        "root_distance_seconds":None if root_distance is None else round(root_distance,6),
        "maximum_root_distance_seconds":root_tolerance,"passed":root_valid,"classification":classification}
    if accepted:
        report["streaks"].append({"streak_id":role+"-opening-1","role":role,"status":"confirmed" if confirmed else classification,
            "chart_start":streak_fit.get("chart_start"),"chart_end":streak_fit.get("chart_end"),
            "midi_start":streak_fit.get("midi_start"),"midi_end":streak_fit.get("midi_end"),
            "accepted_phrases":len(accepted),"representative_control_count":len(controls),
            "diagnostic_control_count":len(diagnostic_controls),"chart_to_midi_fit":streak_fit})
    release_reason=(None if confirmed else "selected_chain_not_rooted_at_incipit" if geometry_valid else
                    streak_fit.get("reason") or "fewer_than_minimum_confirmed_phrases")
    report.update({"status":"confirmed" if confirmed else classification,"state":report["state"],
        "accepted_phrase_count":len(accepted),"timeline_model":"chart_to_performance_midi",
        "chain_classification":classification,"streak_fit":streak_fit,"incipit_control":incipit_control,
        "representative_controls":controls,"monotonic_chain_diagnostic_controls":diagnostic_controls,
        "release_reason":release_reason})
    return report

def build_incipit_consensus(candidates, nominal_prior, observed_incipit_prior,
                            role_profiles, cfg):
    """Rank sequence-based opening placements without requiring perfect consensus."""
    opening=[]
    first_times=[x["first_score_time"] for x in role_profiles.values() if x["first_score_time"] is not None]
    opening_limit=(min(first_times)+20.0) if first_times else 20.0
    for raw in candidates:
        if raw.get("sequence_first_score_time",raw["score_time"])>opening_limit: continue
        row=dict(raw); profile=role_profiles.get(row.get("source"),{})
        score=float(row["score_time"]);audio=float(row["audio_time"])
        expected=float(nominal_prior(score)); observed=(float(observed_incipit_prior(score))
            if observed_incipit_prior is not None else expected)
        expected_delta=abs(audio-expected); observed_delta=abs(audio-observed)
        active=bool(profile.get("incipit_authority_eligible"))
        sequence=float(row.get("local_phrase_confidence",0.0))
        distinct=float(row.get("phrase_distinctiveness",0.0))
        uniqueness=max(0.0,min(1.0,float(row.get("uniqueness_margin",0.0))/.15))
        origin_support=max(0.0,1.0-min(expected_delta,observed_delta)/max(1.0,cfg.search_radius_seconds))
        score_value=.45*sequence+.22*distinct+.18*uniqueness+.15*origin_support
        if not active: score_value-=.12
        row.update({"expected_origin_delta_seconds":round(expected_delta,6),
                    "offset_incipit_delta_seconds":round(observed_delta,6),
                    "role_incipit_authority_eligible":active,
                    "incipit_consensus_score":round(score_value,6)})
        opening.append(row)
    opening.sort(key=lambda x:x["incipit_consensus_score"],reverse=True)
    best=opening[0] if opening else None
    return {"status":"available" if best else "insufficient_sequence_evidence",
            "selected":best,"ranked_hypotheses":opening[:12],
            "policy":"offset_incipit_and_expected_origin_are_competing_soft_hypotheses",
            "single_note_and_silence_events_authoritative":False}


def build_role_paths(candidates, nominal_prior, cfg, boundary_controls=None):
    paths={}
    for role in ROLE_FILENAMES:
        rows=[x for x in candidates if x.get("source")==role and x.get("sequence_authority_eligible")]
        paths[role]=select_globally_constrained_path(rows,nominal_prior,cfg,boundary_controls)
    fused=[]
    for role,path in paths.items():
        for row in path.get("selected",[]):
            r=dict(row);r["role_path"]=role;fused.append(r)
    # The final selector still enforces monotonicity, now over candidates that
    # have already survived a coherent same-role sequence path.
    return paths,select_globally_constrained_path(fused,nominal_prior,cfg,boundary_controls)


def _extract_boundary_controls(boundary_controls):
    out=[]
    for row in boundary_controls or []:
        s=row.get("score_time",row.get("t_gp"));a=row.get("audio_time",row.get("time"))
        if s is not None and a is not None and math.isfinite(float(s)) and math.isfinite(float(a)):
            out.append({"score_time":float(s),"audio_time":float(a),"kind":row.get("kind","boundary"),"immutable":True})
    return sorted(out,key=lambda x:x["score_time"])


def select_globally_constrained_path(candidates,nominal_prior,cfg,boundary_controls=None):
    """Forward/backward DP with hard edge stretch and immutable boundaries."""
    controls=_extract_boundary_controls(boundary_controls)
    rows=[]
    for row in candidates:
        if row["diagnostic_status"]=="ambiguous_or_weak":continue
        r=dict(row); nominal_error=abs(float(r["audio_time"])-float(nominal_prior(r["score_time"])))
        placement=math.exp(-nominal_error/max(_cfg(cfg,"wide_search_radius_seconds",8.0),1e-6))
        r["global_placement_confidence"]=round(placement,4);r["nominal_prior_error_seconds"]=round(nominal_error,6)
        rows.append(r)
    rows.sort(key=lambda x:(x["score_time"],x["audio_time"]))
    min_edge=_cfg(cfg,"global_edge_stretch_min",0.9);max_edge=_cfg(cfg,"global_edge_stretch_max",1.1)
    def edge(a,b):
        ds=b["score_time"]-a["score_time"];da=b["audio_time"]-a["audio_time"]
        if ds<=0 or da<=0:return None
        stretch=da/ds
        if not min_edge<=stretch<=max_edge:return None
        return stretch
    # Candidate must also be compatible with enclosing immutable controls.
    filtered=[]
    for r in rows:
        ok=True
        for left,right in zip(controls,controls[1:]):
            if left["score_time"]<r["score_time"]<right["score_time"]:
                ok=edge(left,r) is not None and edge(r,right) is not None;break
        if ok:filtered.append(r)
    rows=filtered;n=len(rows)
    if not n:return {"status":"no_globally_valid_path","selected_count":0,"selected":[],"boundary_controls":controls,"unsupported_gaps":[]}
    node=[.7*r["local_phrase_confidence"]+.3*r["global_placement_confidence"] for r in rows]
    fw=node[:];fp=[None]*n
    for i in range(n):
        for j in range(i):
            stretch=edge(rows[j],rows[i])
            if stretch is None:continue
            value=fw[j]+node[i]-.25*abs(stretch-1.0)
            if value>fw[i]:fw[i]=value;fp[i]=j
    bw=node[:];bp=[None]*n
    for i in range(n-1,-1,-1):
        for j in range(i+1,n):
            stretch=edge(rows[i],rows[j])
            if stretch is None:continue
            value=bw[j]+node[i]-.25*abs(stretch-1.0)
            if value>bw[i]:bw[i]=value;bp[i]=j
    meet=max(range(n),key=lambda i:fw[i]+bw[i]-node[i]);idx=[];i=meet
    while i is not None:idx.append(i);i=fp[i]
    idx.reverse();i=bp[meet]
    while i is not None:idx.append(i);i=bp[i]
    selected=[]
    for i in idx:
        r=dict(rows[i]);r["diagnostic_status"]=("valid_chain_supported_motif" if r["diagnostic_status"]=="valid_repeated_motif_hypothesis" else r["diagnostic_status"])
        r["chain_support"]={"forward_score":round(fw[i],4),"backward_score":round(bw[i],4),"meet_in_middle":i==meet}
        selected.append(r)
    gaps=[]
    for a,b in zip(selected,selected[1:]):
        if b["score_time"]-a["score_time"]>12:gaps.append({"start_score_time":a["score_time"],"end_score_time":b["score_time"],"status":"unsupported"})
    return {"status":"available","selected_count":len(selected),"selected":selected,"boundary_controls":controls,
            "edge_stretch_range":[min_edge,max_edge],"meet_index":meet,"unsupported_gaps":gaps,
            "monotonic":all(edge(a,b) is not None for a,b in zip(selected,selected[1:]))}


def build_transcription_prior(report,fallback):
    selected=((report or {}).get("global_monotonic_path_preview") or {}).get("selected") or []
    if len(selected)<2:return fallback,{"status":"insufficient_transcription_controls","control_count":len(selected)}
    xs=[float(x["score_time"]) for x in selected];ys=[float(x["audio_time"]) for x in selected]
    def prior(t):
        t=float(t)
        if t<=xs[0]:return float(fallback(t))+(ys[0]-float(fallback(xs[0])))
        if t>=xs[-1]:return float(fallback(t))+(ys[-1]-float(fallback(xs[-1])))
        for i in range(1,len(xs)):
            if t<=xs[i]:
                f=(t-xs[i-1])/(xs[i]-xs[i-1]);return ys[i-1]+f*(ys[i]-ys[i-1])
        return float(fallback(t))
    return prior,{"status":"available","control_count":len(selected),"controls":[{"score_time":x,"audio_time":y} for x,y in zip(xs,ys)]}


def investigate(score_events,song_folder,baseline,padding_added=0.0,alignment_mode=None,config=None,
                nominal_prior=None,nominal_prior_report=None,boundary_controls=None,recovery_prior=None,
                pass_label="initial",observed_incipit_prior=None,observed_incipit_report=None):
    cfg=config or InvestigatorConfig()
    if alignment_mode is not None:validate_method(alignment_mode)
    nominal_prior=nominal_prior or (lambda t:float(t))
    discovered=discover_prepared_files(song_folder);providers=[]
    for role,path in discovered.items():
        providers.append({"provider":"muscriptor","source":role,"status":"not_prepared","expected_file":ROLE_FILENAMES[role],"events":[]}
                         if path is None else parse_muscriptor_midi(path,role,padding_added))
    profiles=build_role_activity_profiles(score_events,providers)
    incipit_tracking=track_incipit_role(score_events,providers,nominal_prior,observed_incipit_prior,cfg)
    candidates,stats=build_multipass_phrase_correspondences(
        score_events,providers,nominal_prior,cfg,recovery_prior,pass_label,
        observed_incipit_prior=observed_incipit_prior)
    role_paths,preview=build_role_paths(candidates,nominal_prior,cfg,boundary_controls)
    incipit=build_incipit_consensus(candidates,nominal_prior,observed_incipit_prior,profiles,cfg)
    valid=[x for x in candidates if x["diagnostic_status"]!="ambiguous_or_weak"]
    return {"version":"4.0","enabled":True,"provider":"prepared_muscriptor_midi","status":"complete" if all(x["status"]=="available" for x in providers) else "partial",
        "alignment_mode":alignment_mode,"pass_label":pass_label,"timing_changes_applied":False,"config":asdict(cfg),
        "nominal_prior":nominal_prior_report or {"kind":"expected_identity"},
        "observed_incipit_prior":observed_incipit_report or {"kind":"unavailable"},
        "incipit_consensus":incipit,"incipit_role_tracking":incipit_tracking,
        "role_activity_profiles":profiles,"prepared_files":providers,
        "anchor_candidates":candidates,"role_monotonic_paths":role_paths,
        "global_monotonic_path_preview":preview,"role_diagnostics":stats,
        "summary":{"prepared_role_count":sum(x["status"]=="available" for x in providers),"expected_role_count":len(ROLE_FILENAMES),
            "normalized_event_count":sum(len(x.get("events") or []) for x in providers),"candidate_correspondence_count":len(candidates),
            "locally_valid_candidates":len(valid),"selected_monotonic_candidates":preview["selected_count"],
            "sequence_authority_candidates":sum(bool(x.get("sequence_authority_eligible")) for x in candidates)},
        "method_consumption":{"mode":"diagnostic_then_graded_authority","candidate_count_received":len(candidates),"candidate_count_applied":0,
                              "single_note_controls_allowed":False}}

# ---------------------------------------------------------------------------
# v4 MIDI-authority segments and audio-anchor compatibility
# ---------------------------------------------------------------------------

def build_trusted_midi_segments(report, maximum_phrase_gap=3.0, minimum_phrases=3,
                                minimum_span=1.0, representative_spacing=4.0):
    """Collapse overlapping phrase observations into robust MIDI timing segments."""
    selected=sorted((((report or {}).get("final_global_monotonic_path_preview") or
                      (report or {}).get("global_monotonic_path_preview") or {}).get("selected") or []),
                    key=lambda x:float(x["score_time"]))
    clusters=[]
    for row in selected:
        if not clusters or float(row["score_time"])-float(clusters[-1][-1]["score_time"])>maximum_phrase_gap:
            clusters.append([row])
        else:clusters[-1].append(row)
    segments=[];supporting=[];controls=[]
    for index,cluster in enumerate(clusters):
        if len(cluster)<minimum_phrases:
            supporting.extend(cluster);continue
        start=float(cluster[0]["score_time"]);end=float(cluster[-1]["score_time"])
        if end-start<minimum_span:
            supporting.extend(cluster);continue
        xs=[float(x["score_time"]) for x in cluster];ys=[float(x["audio_time"]) for x in cluster]
        slopes=[]
        for i in range(len(xs)):
            for j in range(i+1,len(xs)):
                if xs[j]-xs[i]>=.25:slopes.append((ys[j]-ys[i])/(xs[j]-xs[i]))
        stretch=statistics.median(slopes) if slopes else 1.0
        offset=statistics.median([y-stretch*x for x,y in zip(xs,ys)])
        residuals=[abs(y-(offset+stretch*x)) for x,y in zip(xs,ys)]
        roles=sorted({x.get("source") for x in cluster if x.get("source")})
        locally_safe=(.9<=stretch<=1.1 and statistics.median(residuals)<=.12 and max(residuals)<=.35)
        segment={"id":f"midi-segment-{index}","authority":"trusted_midi_segment" if locally_safe else "supporting_midi_segment",
                 "start_score_time":round(start,6),"end_score_time":round(end,6),
                 "start_audio_time":round(offset+stretch*start,6),"end_audio_time":round(offset+stretch*end,6),
                 "fitted_stretch":round(stretch,6),"fitted_offset":round(offset,6),
                 "supporting_phrase_count":len(cluster),"roles":roles,"role_count":len(roles),
                 "median_residual_seconds":round(statistics.median(residuals),6),
                 "maximum_residual_seconds":round(max(residuals),6),"trusted":bool(locally_safe)}
        segments.append(segment)
        if locally_safe:
            chosen=[cluster[0]];last=start
            for row in cluster[1:-1]:
                if float(row["score_time"])-last>=representative_spacing:
                    chosen.append(row);last=float(row["score_time"])
            if cluster[-1] is not chosen[-1]:chosen.append(cluster[-1])
            for row in chosen:
                controls.append({"score_time":float(row["score_time"]),"audio_time":float(row["audio_time"]),
                                 "kind":"trusted_midi","authority":"trusted_midi",
                                 "segment_id":segment["id"],"immutable":False,
                                 "local_phrase_confidence":row.get("local_phrase_confidence"),
                                 "global_placement_confidence":row.get("global_placement_confidence")})
        else:supporting.extend(cluster)
    return {"segments":segments,"trusted_segments":[x for x in segments if x["trusted"]],
            "supporting_segments":[x for x in segments if not x["trusted"]],
            "representative_controls":controls,"supporting_observation_count":len(supporting),
            "trusted_control_count":len(controls)}


def assess_midi_authority(report, minimum_controls=4, minimum_coverage=.35,
                          maximum_gap=30.0, minimum_roles=1):
    tracker=(report or {}).get("incipit_role_tracking") or {}
    tracker_controls=tracker.get("representative_controls") or []
    hierarchy=build_trusted_midi_segments(report)
    if tracker.get("status")=="confirmed" and len(tracker_controls)>=2:
        hierarchy=dict(hierarchy);hierarchy["representative_controls"]=tracker_controls
        hierarchy["trusted_control_count"]=len(tracker_controls)
        hierarchy["authority_source"]="incipit_role_streak"
    controls=hierarchy["representative_controls"]
    selected=sorted((((report or {}).get("final_global_monotonic_path_preview") or
                      (report or {}).get("global_monotonic_path_preview") or {}).get("selected") or []),
                    key=lambda x:float(x["score_time"]))
    if selected:
        span=float(selected[-1]["score_time"])-float(selected[0]["score_time"])
        maximum_score=max(float(x["score_time"]) for x in selected)
        coverage=span/max(maximum_score,1e-6)
        gaps=[float(b["score_time"])-float(a["score_time"]) for a,b in zip(selected,selected[1:])]
        largest_gap=max(gaps) if gaps else 0.0
        roles=sorted({x.get("source") for x in selected if x.get("source")})
    else:coverage=0.0;largest_gap=math.inf;roles=[]
    tracker_confirmed=(tracker.get("status")=="confirmed" and len(tracker_controls)>=2)
    if tracker_confirmed:
        first=float(tracker_controls[0]["score_time"]);last=float(tracker_controls[-1]["score_time"])
        span=max(0.0,last-first);coverage=1.0 if span>=4.0 else span/4.0
        largest_gap=max([float(b["score_time"])-float(a["score_time"]) for a,b in zip(tracker_controls,tracker_controls[1:])] or [0.0])
        roles=sorted({x.get("source") for x in tracker_controls if x.get("source")})
    predicates={"enough_trusted_controls":len(controls)>=(2 if tracker_confirmed else minimum_controls),
                "sufficient_path_coverage":coverage>=(0.5 if tracker_confirmed else minimum_coverage),
                "bounded_largest_gap":largest_gap<=maximum_gap,
                "sufficient_role_diversity":len(roles)>=minimum_roles}
    available=all(predicates.values())
    return available,{"status":"available" if available else "insufficient_midi_authority",
        "authority":("incipit_role_streak" if available and tracker_confirmed else
                     "trusted_midi_primary" if available else "diagnostic_only"),
        "predicates":predicates,"path_coverage":round(coverage,4),
        "largest_score_gap_seconds":None if not math.isfinite(largest_gap) else round(largest_gap,6),
        "roles":roles,"role_count":len(roles),**hierarchy}


def build_midi_authority_prior(report,fallback,force=False,force_monotonic_chain=False):
    available,authority=assess_midi_authority(report)
    tracker=(report or {}).get("incipit_role_tracking") or {}
    controls=authority["representative_controls"]
    if force_monotonic_chain:
        raw=tracker.get("monotonic_chain_diagnostic_controls") or []
        finite=sorted([x for x in raw if math.isfinite(float(x.get("score_time",math.nan))) and math.isfinite(float(x.get("audio_time",math.nan)))],key=lambda x:float(x["score_time"]))
        monotonic=len(finite)>=2 and all(float(b["score_time"])>float(a["score_time"]) and float(b["audio_time"])>float(a["audio_time"]) for a,b in zip(finite,finite[1:]))
        if not monotonic:
            authority.update({"status":"force_blocked","authority":"force_monotonic_chain_blocked",
                "force_monotonic_chain_requested":True,"force_blocked_reason":"no_finite_monotonic_chain_controls"})
            return fallback,authority
        controls=finite;available=True
        authority.update({"status":"available","authority":"forced_monotonic_chain","force_requested":False,
            "force_monotonic_chain_requested":True,"normal_gate_passed":False,
            "chain_classification":tracker.get("chain_classification"),"diagnostic_unsafe_timing":True,
            "representative_controls":controls,"trusted_control_count":len(controls),"force_blocked_reason":None})
    elif force and not available:
        selected=sorted((((report or {}).get("final_global_monotonic_path_preview") or
                          (report or {}).get("global_monotonic_path_preview") or {}).get("selected") or []),
                        key=lambda x:float(x["score_time"]))
        finite=[x for x in selected if math.isfinite(float(x["score_time"])) and math.isfinite(float(x["audio_time"]))]
        monotonic=len(finite)>=2 and all(float(b["score_time"])>float(a["score_time"]) and float(b["audio_time"])>float(a["audio_time"]) for a,b in zip(finite,finite[1:]))
        if not monotonic:
            authority.update({"status":"force_blocked","authority":"force_blocked","force_requested":True,
                              "force_blocked_reason":"fewer_than_two_monotonic_finite_midi_controls"})
            return fallback,authority
        controls=[]; last=None
        for row in finite:
            if last is None or float(row["score_time"])-last>=4.0 or row is finite[-1]:
                controls.append({"score_time":float(row["score_time"]),"audio_time":float(row["audio_time"]),
                                 "kind":"forced_midi","authority":"forced_midi"}); last=float(row["score_time"])
        authority.update({"status":"available","authority":"forced_midi_primary","force_requested":True,
                          "normal_gate_passed":False,"representative_controls":controls,
                          "trusted_control_count":len(controls),"force_blocked_reason":None})
        available=True
    else:
        authority["force_requested"]=bool(force); authority["normal_gate_passed"]=bool(available)
    if not available:return fallback,authority
    controls=sorted(controls,key=lambda x:x["score_time"]);xs=[x["score_time"] for x in controls];ys=[x["audio_time"] for x in controls]
    def prior(t):
        t=float(t)
        tracker_limited=authority.get("authority") in {"incipit_role_streak","forced_monotonic_chain"}
        if t<xs[0]:return float(fallback(t)) if tracker_limited else float(fallback(t))+(ys[0]-float(fallback(xs[0])))
        if t>xs[-1]:return float(fallback(t)) if tracker_limited else float(fallback(t))+(ys[-1]-float(fallback(xs[-1])))
        for i in range(1,len(xs)):
            if t<=xs[i]:
                f=(t-xs[i-1])/(xs[i]-xs[i-1]);return ys[i-1]+f*(ys[i]-ys[i-1])
        return float(fallback(t))
    return prior,authority


def classify_audio_anchor_compatibility(continuous_report,midi_prior,strong_seconds=.10,
                                        compatible_seconds=.25,uncertain_seconds=.50):
    """Attach audio evidence to MIDI; never move MIDI toward incompatible audio."""
    counts={"strong_corroboration":0,"compatible_support":0,"uncertain":0,"conflict":0}
    report=dict(continuous_report or {});matches=[]
    for match in report.get("matches",[]) or []:
        m=dict(match);score=float(m.get("score_time",0.0));target=float(midi_prior(score));candidates=[]
        for raw in m.get("candidates",[]) or []:
            c=dict(raw);time=float(c.get("time",c.get("audio_time",target)));delta=time-target;absolute=abs(delta)
            cls=("strong_corroboration" if absolute<=strong_seconds else
                 "compatible_support" if absolute<=compatible_seconds else
                 "uncertain" if absolute<=uncertain_seconds else "conflict")
            c["midi_compatibility"]=cls;c["midi_delta_seconds"]=round(delta,6)
            c["production_eligible_against_midi"]=cls in {"strong_corroboration","compatible_support"}
            counts[cls]+=1;candidates.append(c)
        m["candidates"]=candidates;matches.append(m)
    report["matches"]=matches
    report["midi_compatibility_summary"]={**counts,"policy":"audio_corroborates_or_fills_midi_voids",
                                           "strong_seconds":strong_seconds,"compatible_seconds":compatible_seconds,
                                           "uncertain_seconds":uncertain_seconds}
    return report
