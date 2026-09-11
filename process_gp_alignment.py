"""
process_gp_alignment.py

Script 2 of the feedpak pipeline. Parses a .gp5 file, builds a tick->second
map from the file's own tempo data, extracts fretted/drum/keyboard tracks,
then time-warps every nominal (GP-clock) timestamp onto the real audio's
clock via frequency-filtered DTW against the drum stem.

v1 scope note: technique fields (bends, slides, hammer-on/pull-off,
fingering) are intentionally NOT extracted yet — timing/sync correctness
is the current priority (per project decision). Fretted-note output is
limited to {t, s, f, sus}. This keeps Script 2 simpler and easier to
validate against real audio before technique notation is layered on.

Usage:
    python process_gp_alignment.py song.gp5 drums.ogg --out intermediate_arrangements.json
"""
import argparse
import json
import math

import numpy as np

import feedpak_common as fc
import checkpoint_dtw as cdtw
import project_config as pc
import alphatab_score as ats

try:
    import guitarpro
except ImportError:
    guitarpro = None

try:
    import librosa
    from scipy.interpolate import interp1d
except ImportError:
    librosa = None
    interp1d = None


SUSTAIN_MIN_SECONDS = 0.05


# --------------------------------------------------------------------------
# Tempo map extraction
# --------------------------------------------------------------------------

def build_tempo_events(song):
    """
    Walk every beat in every track looking for a mixTableChange.tempo
    (GP stores tempo changes on beats, not measure headers). Returns a
    sorted, deduplicated list of (tick, bpm) starting with (0, song.tempo).

    Only one track's beats need scanning in principle (tempo changes are
    song-wide), but different tracks can have different beat grids, so we
    scan the first track that has measures and trust GP authoring tools
    to keep tempo meta-events consistent across tracks.
    """
    events = [(0, song.tempo)]
    track = next((t for t in song.tracks if t.measures), None)
    if track is None:
        return events

    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                mtc = getattr(beat.effect, "mixTableChange", None)
                if mtc is not None and mtc.tempo is not None:
                    tick = beat.start if beat.start is not None else measure.header.start
                    events.append((tick, mtc.tempo.value))

    events.sort(key=lambda e: e[0])
    deduped = [events[0]]
    for tick, bpm in events[1:]:
        if tick == deduped[-1][0]:
            deduped[-1] = (tick, bpm)
        else:
            deduped.append((tick, bpm))
    return deduped


def rebase_tempo_events(tempo_events, first_tick):
    first_tick=int(first_tick or 0)
    active=float(tempo_events[0][1]) if tempo_events else 120.0
    for tick,bpm in sorted(tempo_events):
        if int(tick)<=first_tick: active=float(bpm)
        else: break
    out=[(0,active)]
    for tick,bpm in sorted(tempo_events):
        rel=int(tick)-first_tick
        if rel<0: continue
        item=(rel,float(bpm))
        if item[0]==out[-1][0]: out[-1]=item
        elif item!=out[-1]: out.append(item)
    return out

def first_configured_instrument_time(song,role_by_name,tempo_events,first_tick):
    active={"guitar","bass","drums","piano_left","piano_right","piano_combined"}; times=[]
    for track in song.tracks:
        if role_by_name.get(track.name) not in active: continue
        for measure in track.measures:
            for voice in measure.voices:
                for beat in voice.beats:
                    if beat.start is not None and beat.notes:
                        times.append(fc.tick_to_seconds(beat.start-first_tick,tempo_events))
    return min(times) if times else 0.0

def choose_chart_offset(audio_content_start,score_content_time):
    return float(audio_content_start)-float(score_content_time)

def trustworthy_checkpoint_count(chroma_report,silence_report):
    return int((chroma_report or {}).get("summary",{}).get("strong",0))+int((silence_report or {}).get("measured_boundaries",0))

def adaptive_checkpoint_diagnostic(pitched_events,audio_paths,base_warp,measure_downbeats,onset_report,silence_report,every_measures,initial_radius,sr=22050,maximum_radius=4.0):
    attempts=[]; radius=float(initial_radius); current=None
    while radius<=float(maximum_radius)+1e-9:
        current=diagnose_chroma_checkpoints(pitched_events,audio_paths,base_warp,measure_downbeats,onset_report,every_measures,radius,sr)
        trusted=trustworthy_checkpoint_count(current,silence_report)
        attempts.append({"radius_seconds":radius,"chroma_strong":current["summary"]["strong"],"measured_silence_boundaries":int((silence_report or {}).get("measured_boundaries",0)),"trustworthy":trusted})
        if trusted>0: break
        radius*=2.0
    return {"version":1,"status":"diagnostic_only","attempts":attempts,"selected_radius_seconds":attempts[-1]["radius_seconds"],"expanded":len(attempts)>1,"trustworthy":attempts[-1]["trustworthy"],"report":current}

def build_key_signature_events(song, tempo_events, first_tick):
    """
    Deterministic keys.json events, straight from GP measure headers'
    keySignature (accidentals, is_minor) — no audio analysis needed.
    Only emits an event when the key actually changes.
    """
    events = []
    last_key = None
    for header in song.measureHeaders:
        ks = header.keySignature
        accidentals, is_minor = ks.value[0], bool(ks.value[1])
        name = fc.key_signature_to_name(accidentals, is_minor)
        if name is None or name == last_key:
            continue
        last_key = name
        t_gp = fc.tick_to_seconds(header.start - first_tick, tempo_events)
        events.append({"t_gp": t_gp, "key": name})
    return events


def build_song_timeline(song, tempo_events, first_tick):
    """
    Bar-boundary data for song_timeline.json. Each beat entry also
    carries the active time-signature numerator (ts_num), because the
    ACTUAL bar-subdivision signal a working reference feedpak relies on
    isn't a sparse authored-tempo list or a bare beats[] array — it's a
    DENSE per-measure tempo curve: one tempos[] entry per measure, with
    the BPM back-computed from how long that measure actually took in
    real (post-DTW) audio. That's built in process() after warping,
    once we know each measure's real duration; this function only
    supplies the nominal (pre-warp) measure boundaries + numerators it
    needs to do that.
    """
    time_signatures = []
    beats = []
    last_ts = None
    active_num = 4
    for header in song.measureHeaders:
        t_gp = fc.tick_to_seconds(header.start - first_tick, tempo_events)
        if header.timeSignature:
            ts = (header.timeSignature.numerator, header.timeSignature.denominator.value)
            if ts != last_ts:
                time_signatures.append({"t_gp": t_gp, "ts": list(ts)})
                last_ts = ts
            active_num = ts[0]
        beats.append({"t_gp": t_gp, "measure": header.number, "ts_num": active_num})

    return {"time_signatures": time_signatures, "beats": beats}


def compute_dense_measure_tempos(warped_beats, fallback_bpm):
    """
    One BPM per measure, back-computed from the real (warped) duration
    between consecutive measure downbeats: bpm = ts_num * 60 / duration.
    This is what a working reference feedpak's song_timeline.json
    actually contains (168 entries for a ~168-measure song, each
    reflecting that measure's real, DTW-observed tempo) — not a sparse
    list of only the tempo events the GP file happens to declare. The
    last measure has no "next" downbeat to diff against, so it just
    repeats the previous measure's effective BPM.
    """
    dense = []
    for i, b in enumerate(warped_beats):
        if i + 1 < len(warped_beats):
            duration = warped_beats[i + 1]["time"] - b["time"]
            bpm = round(b["ts_num"] * 60.0 / duration, 3) if duration > 0 else \
                (dense[-1]["bpm"] if dense else fallback_bpm)
        else:
            bpm = dense[-1]["bpm"] if dense else fallback_bpm
        dense.append({"time": b["time"], "bpm": bpm})
    return dense



def _event_times(entries, key="t_gp"):
    return sorted({float(e[key]) for e in entries if key in e})


def _keyboard_onsets(notation):
    times=[]
    for measure in notation.get("measures", []):
        for stave in measure.get("staves", {}).values():
            for voice in stave.get("voices", []):
                times.extend(b["t_gp"] for b in voice.get("beats", []) if "t_gp" in b)
    return sorted(set(times))


def _source_band(source, sr):
    nyquist=sr/2.0
    if source == "bass": return 30.0, min(1200.0, nyquist)
    if source == "piano": return 45.0, min(6000.0, nyquist)
    return 30.0, nyquist


def _source_click_frequency(source):
    return {"bass": 90.0, "piano": 880.0, "drums": 180.0}.get(source, 180.0)



def load_alignment_anchors(path, nominal_downbeats, padding_added=0.0):
    """Load manual anchors. Missing time_reference defaults to original source audio."""
    if not path:
        return [], "source"
    with open(path, "r", encoding="utf-8") as f:
        payload=json.load(f)
    reference=payload.get("time_reference", "source").lower()
    if reference not in ("source", "padded"):
        raise ValueError("Anchor time_reference must be 'source' or 'padded'")
    by_measure={int(b["measure"]):b for b in nominal_downbeats}
    anchors=[]
    for raw in payload.get("anchors", []):
        measure=int(raw["measure"]); beat=float(raw.get("beat",1))
        if measure not in by_measure:
            raise ValueError(f"Anchor references unknown measure {measure}")
        current=by_measure[measure]; nominal=float(current["t_gp"])
        if beat != 1:
            nxt=by_measure.get(measure+1)
            if not nxt: raise ValueError("Non-downbeat anchor cannot target final measure")
            nominal += ((beat-1)/max(1,int(current.get("ts_num",4))))*(nxt["t_gp"]-current["t_gp"])
        source_time=float(raw["audio_time"])
        padded_time=source_time+padding_added if reference == "source" else source_time
        anchors.append({"measure":measure,"beat":beat,"nominal_time":nominal,
                        "source_audio_time":source_time-padding_added if reference=="padded" else source_time,
                        "padded_audio_time":padded_time,"label":raw.get("label")})
    anchors.sort(key=lambda a:a["nominal_time"])
    for a,b in zip(anchors,anchors[1:]):
        if b["nominal_time"]<=a["nominal_time"] or b["padded_audio_time"]<=a["padded_audio_time"]:
            raise ValueError("Anchors must increase in score and audio time")
    return anchors, reference


def apply_hard_anchors(base_warp, anchors, nominal_downbeats, max_chunk_measures=16):
    """Create a continuous piecewise warp through manual and heuristic boundaries."""
    points=[]
    for i,b in enumerate(nominal_downbeats):
        if i==0 or i==len(nominal_downbeats)-1 or i % max(2,max_chunk_measures)==0:
            points.append({"nominal_time":float(b["t_gp"]),"padded_audio_time":float(base_warp(b["t_gp"])),"kind":"heuristic"})
    # Manual anchors replace heuristic points at the same score position.
    for a in anchors:
        points=[p for p in points if abs(p["nominal_time"]-a["nominal_time"])>1e-6]
        points.append({**a,"kind":"manual"})
    points.sort(key=lambda p:p["nominal_time"])
    clean=[]
    for point in points:
        if clean and point["padded_audio_time"]<=clean[-1]["padded_audio_time"]:
            if point.get("kind")=="manual":
                while clean and clean[-1].get("kind")!="manual" and point["padded_audio_time"]<=clean[-1]["padded_audio_time"]:
                    clean.pop()
                if clean and point["padded_audio_time"]<=clean[-1]["padded_audio_time"]:
                    raise ValueError("Manual anchor conflicts with an earlier manual anchor")
            else:
                continue
        clean.append(point)
    x=np.asarray([p["nominal_time"] for p in clean]); y=np.asarray([p["padded_audio_time"] for p in clean])
    if len(x)<2: return base_warp, clean
    fn=interp1d(x,y,kind="linear",bounds_error=False,fill_value="extrapolate")
    return (lambda t:float(fn(float(t)))), clean


def estimate_last_measure_end(measure_downbeats):
    """Estimate the final bar boundary from the authored measure grid."""
    beats=sorted(measure_downbeats,key=lambda x:float(x["t_gp"]))
    if not beats: return 0.0
    if len(beats)==1: return float(beats[0]["t_gp"])
    durations=[float(b["t_gp"])-float(a["t_gp"]) for a,b in zip(beats,beats[1:]) if float(b["t_gp"])>float(a["t_gp"])]
    final_duration=durations[-1] if durations else 0.0
    return float(beats[-1]["t_gp"])+final_duration

def detect_audio_content_end(audio_path, known_start, sr=22050):
    """Return the last sustained full-mix activity after ``known_start``."""
    if librosa is None or not audio_path: return None,{"available":False,"reason":"missing_full_mix_or_librosa"}
    y,actual_sr=librosa.load(audio_path,sr=sr,mono=True); hop=512
    rms=librosa.feature.rms(y=y,hop_length=hop)[0]
    times=librosa.frames_to_time(np.arange(len(rms)),sr=actual_sr,hop_length=hop)
    if not len(rms): return None,{"available":False,"reason":"empty_full_mix"}
    # A full mix can be dense for more than 80% of its duration, so its
    # 20th-percentile RMS is not a noise floor. Multiplying that value by
    # three produced impossible thresholds (0.789 RMS on Spellbound) and
    # therefore no endpoint. Use a small fraction of the active programme
    # level instead, then require sustained activity to reject isolated
    # codec clicks and low-level tail noise.
    active=float(np.percentile(rms,70))
    threshold=max(active*0.03,1e-5)
    mask=(rms>=threshold)&(times>=float(known_start))
    width=max(1,int(round(.30*actual_sr/hop)))
    sustained=np.convolve(mask.astype(int),np.ones(width,dtype=int),mode="same")>=max(1,int(np.ceil(width*.60)))
    idx=np.where(sustained)[0]
    if not len(idx): return None,{"available":False,"reason":"no_sustained_content","threshold_rms":threshold}
    end=min(len(y)/actual_sr,float(times[idx[-1]]+hop/actual_sr))
    return end,{"available":True,"content_end":round(end,6),"file_duration":round(len(y)/actual_sr,6),"threshold_rms":round(threshold,8),"sustain_window_seconds":.30}

def build_instrument_scale_candidates(specs, score_start, score_end, sr=22050,
                                      minimum_events=12, minimum_span_seconds=45.0,
                                      minimum_coverage=0.55, minimum_scale=0.95,
                                      maximum_scale=1.05):
    """Build start-independent scale candidates from instrument onset spans.

    ``compute_simple_warp(..., "linear")`` remains unchanged. This helper
    calls it only to measure each source's first-to-last scale, discards its
    offset, and records enough evidence for deterministic selection.
    """
    score_span=max(0.0,float(score_end)-float(score_start)); candidates=[]
    for priority,(source,event_times,audio_path) in enumerate(specs):
        times=sorted(float(t) for t in event_times)
        symbolic_span=(times[-1]-times[0]) if len(times)>=2 else 0.0
        coverage=(symbolic_span/score_span) if score_span>0 else 0.0
        rec={"source":source,"priority":priority,"event_count":len(times),
             "symbolic_first":times[0] if times else None,
             "symbolic_last":times[-1] if times else None,
             "symbolic_span_seconds":round(symbolic_span,6),
             "score_span_coverage":round(coverage,6),"accepted":False,"warnings":[]}
        if not audio_path: rec["warnings"].append("missing_audio")
        if len(times)<minimum_events: rec["warnings"].append("insufficient_events")
        if symbolic_span<minimum_span_seconds: rec["warnings"].append("insufficient_symbolic_span")
        if coverage<minimum_coverage: rec["warnings"].append("insufficient_score_coverage")
        if rec["warnings"]:
            candidates.append(rec); continue
        raw=compute_simple_warp(times,audio_path,"linear",sr=sr)
        diag=dict(getattr(raw,"alignment_diagnostics",{}) or {})
        scale=diag.get("scale")
        rec.update({"raw_linear_diagnostics":diag,
                    "measured_audio_first":diag.get("audio_start"),
                    "measured_audio_last":diag.get("audio_end"),
                    "measured_scale":scale})
        if scale is None: rec["warnings"].append("missing_measured_scale")
        elif not minimum_scale<=float(scale)<=maximum_scale:
            rec["warnings"].append("scale_outside_5_percent_gate")
        rec["accepted"]=not rec["warnings"]
        candidates.append(rec)
    return candidates

def choose_instrument_scale(candidates, agreement_tolerance=0.005):
    """Select a scale without granting the source authority over song start."""
    accepted=[x for x in candidates if x.get("accepted")]
    if not accepted:
        return 1.0,{"status":"fallback_no_accepted_candidate",
                   "selected_source":None,"agreement_tolerance":agreement_tolerance,
                   "candidates":candidates}
    # Coverage is the primary requirement. Event count breaks near ties, then
    # the stable specs order (drums, bass, piano) is the deterministic fallback.
    best=max(accepted,key=lambda x:(float(x["score_span_coverage"]),
                                    int(x["event_count"]),-int(x["priority"])))
    agreeing=[x for x in accepted
              if abs(float(x["measured_scale"])-float(best["measured_scale"]))<=agreement_tolerance]
    if len(agreeing)>=2:
        weights=[max(1.0,float(x["score_span_coverage"])*float(x["event_count"])) for x in agreeing]
        scale=sum(float(x["measured_scale"])*w for x,w in zip(agreeing,weights))/sum(weights)
        status="fused_agreeing_instrument_scales"; selected=[x["source"] for x in agreeing]
    else:
        scale=float(best["measured_scale"]); status="selected_best_instrument_scale"; selected=best["source"]
    return scale,{"status":status,"selected_source":selected,
                  "agreement_tolerance":agreement_tolerance,"candidates":candidates,
                  "applied_scale":round(float(scale),9)}

def build_opening_anchor_candidates(specs, score_start, audio_content_start, sr=22050,
                                    maximum_score_delay=15.0,
                                    maximum_anchor_shift=4.0,
                                    minimum_events=4):
    """Find a reliable early instrument anchor without changing simple linear.

    Sources qualify only when their first symbolic event is near the score
    beginning. ``compute_simple_warp(..., "linear")`` supplies the measured
    first audio onset, but its scale/offset are not used here.
    """
    out=[]
    for priority,(source,event_times,audio_path) in enumerate(specs):
        times=sorted(float(t) for t in event_times)
        first=times[0] if times else None
        delay=None if first is None else first-float(score_start)
        item={'source':source,'priority':priority,'event_count':len(times),
              'symbolic_first':first,'score_start_delay_seconds':delay,
              'accepted':False,'warnings':[]}
        if not audio_path:item['warnings'].append('missing_audio')
        if len(times)<minimum_events:item['warnings'].append('insufficient_events')
        if delay is None or delay<-.025 or delay>maximum_score_delay:
            item['warnings'].append('not_an_early_score_entrance')
        if item['warnings']:
            out.append(item);continue
        raw=compute_simple_warp(times,audio_path,'linear',sr=sr)
        diag=dict(getattr(raw,'alignment_diagnostics',{}) or {})
        audio_first=diag.get('audio_start')
        shift=None if audio_first is None else float(audio_first)-float(audio_content_start)
        item.update({'measured_audio_first':audio_first,
                     'anchor_shift_from_content_start_seconds':shift,
                     'raw_linear_diagnostics':diag})
        if audio_first is None:item['warnings'].append('missing_audio_first_onset')
        elif abs(shift)>maximum_anchor_shift:item['warnings'].append('opening_anchor_shift_too_large')
        item['accepted']=not item['warnings']
        out.append(item)
    return out

def choose_opening_anchor(candidates, score_start, audio_content_start,
                          simultaneous_score_tolerance=.050,
                          audio_agreement_tolerance=.250):
    """Choose the earliest accepted score entrance, fusing corroborating audio."""
    accepted=[x for x in candidates if x.get('accepted')]
    if not accepted:
        return float(score_start),float(audio_content_start),{
            'status':'fallback_content_start','selected_sources':[],
            'score_anchor':float(score_start),'audio_anchor':float(audio_content_start),
            'candidates':candidates}
    earliest=min(float(x['symbolic_first']) for x in accepted)
    cohort=[x for x in accepted if abs(float(x['symbolic_first'])-earliest)<=simultaneous_score_tolerance]
    # Reject mutually inconsistent audio anchors. In that case select the
    # deterministic candidate closest to the independently observed content start.
    audio_values=[float(x['measured_audio_first']) for x in cohort]
    spread=max(audio_values)-min(audio_values) if len(audio_values)>1 else 0.0
    if len(cohort)>1 and spread<=audio_agreement_tolerance:
        audio_anchor=float(np.median(audio_values)); score_anchor=float(np.median([x['symbolic_first'] for x in cohort]))
        status='fused_early_instrument_anchors'; sources=[x['source'] for x in cohort]
    else:
        best=min(cohort,key=lambda x:(abs(float(x['measured_audio_first'])-float(audio_content_start)),x['priority']))
        audio_anchor=float(best['measured_audio_first']);score_anchor=float(best['symbolic_first'])
        status='selected_early_instrument_anchor';sources=best['source']
    return score_anchor,audio_anchor,{
        'status':status,'selected_sources':sources,'score_anchor':round(score_anchor,9),
        'audio_anchor':round(audio_anchor,9),'cohort_audio_spread_seconds':round(spread,6),
        'simultaneous_score_tolerance_seconds':simultaneous_score_tolerance,
        'audio_agreement_tolerance_seconds':audio_agreement_tolerance,
        'candidates':candidates}

def reanchor_scale_to_song_start(scale, score_start, audio_start):
    """Apply only measured scale while preserving the independent start."""
    return lambda t:float(audio_start)+float(scale)*(float(t)-float(score_start))

def build_guarded_global_span_warp(measure_downbeats, full_audio_path, known_audio_start,
                                   sr=22050, minimum_scale=.95, maximum_scale=1.05):
    """Fit a start-fixed global linear baseline when full-mix span is plausible."""
    nominal_start=float(measure_downbeats[0]["t_gp"]) if measure_downbeats else float(known_audio_start)
    nominal_end=estimate_last_measure_end(measure_downbeats)
    audio_end,evidence=detect_audio_content_end(full_audio_path,known_audio_start,sr)
    nominal_span=nominal_end-nominal_start
    proposed=None if audio_end is None or nominal_span<=0 else (float(audio_end)-float(known_audio_start))/nominal_span
    accepted=bool(proposed is not None and minimum_scale<=proposed<=maximum_scale)
    scale=float(proposed) if accepted else 1.0
    def warp(t): return float(known_audio_start)+(float(t)-nominal_start)*scale
    report={"version":1,"method":"known_start_plus_guarded_full_mix_end_linear",
      "nominal_start":round(nominal_start,6),"nominal_end_estimate":round(nominal_end,6),
      "audio_start":round(float(known_audio_start),6),"audio_end":None if audio_end is None else round(float(audio_end),6),
      "nominal_span_seconds":round(nominal_span,6),
      "audio_span_seconds":None if audio_end is None else round(float(audio_end)-float(known_audio_start),6),
      "proposed_scale":None if proposed is None else round(float(proposed),8),"applied_scale":round(scale,8),
      "minimum_scale":minimum_scale,"maximum_scale":maximum_scale,"accepted":accepted,
      "fallback_used":not accepted,"evidence":evidence,
      "warnings":[] if accepted else ["global_span_rejected_missing_or_outside_5_percent_gate"]}
    # Never attach the same report object that will later contain the
    # alignment decision. Doing so creates report["global_span"] = report
    # and json.dump correctly rejects the circular reference.
    warp.alignment_diagnostics=dict(report)
    return warp,report

def compute_full_mix_frame_warp(nominal_start, nominal_end, full_audio_path,
                                known_audio_start, sr=22050):
    """Anchor the global song frame to the full mix.

    The builder already knows the post-padding musical start. The full mix is
    used to locate the sustained-content end, so a late instrument entrance
    can never move measure 1.
    """
    diagnostics={"mode":"full-mix-frame-v1","selected_source":"full",
                 "candidates":[],"agreement":[],"method":"known_start_full_mix_end_linear"}
    nominal_start=float(nominal_start); nominal_end=float(nominal_end)
    audio_start=float(known_audio_start)
    if librosa is None or not full_audio_path or nominal_end <= nominal_start:
        scale=1.0
        fn=lambda t:audio_start+(float(t)-nominal_start)*scale
        diagnostics.update({"reason":"missing full mix, librosa, or score span",
            "nominal_start":nominal_start,"nominal_end":nominal_end,
            "audio_start":audio_start,"audio_end":audio_start+(nominal_end-nominal_start),
            "offset_seconds":audio_start-nominal_start,"scale":scale})
        fn.alignment_diagnostics=diagnostics; return fn
    y,actual_sr=librosa.load(full_audio_path,sr=sr,mono=True)
    hop=512
    rms=librosa.feature.rms(y=y,hop_length=hop)[0]
    times=librosa.frames_to_time(np.arange(len(rms)),sr=actual_sr,hop_length=hop)
    floor=float(np.percentile(rms,20)) if len(rms) else 0.0
    active=float(np.percentile(rms,70)) if len(rms) else 0.0
    threshold=max(floor*3.0, active*0.08, 1e-5)
    mask=rms>=threshold
    # Require roughly 150 ms of nearby activity to reject isolated codec clicks.
    width=max(1,int(round(0.15*actual_sr/hop)))
    sustained=np.convolve(mask.astype(int),np.ones(width,dtype=int),mode="same")>=max(1,width//2)
    indices=np.where(sustained & (times>=audio_start))[0]
    audio_end=float(times[indices[-1]]) if len(indices) else len(y)/actual_sr
    if audio_end <= audio_start:
        audio_end=len(y)/actual_sr
    scale=(audio_end-audio_start)/(nominal_end-nominal_start)
    fn=lambda t,a=audio_start,n=nominal_start,k=scale:a+(float(t)-n)*k
    diagnostics.update({"audio":str(full_audio_path),"nominal_start":nominal_start,
        "nominal_end":nominal_end,"audio_start":audio_start,"audio_end":audio_end,
        "offset_seconds":audio_start-nominal_start,"scale":scale,
        "content_end_threshold_rms":threshold,"start_anchor_error_ms":0.0})
    fn.alignment_diagnostics=diagnostics; return fn

def _activity_span_from_features(rms,times,onset,onset_times,minimum_sustained_seconds=0.20):
    """Pure helper for robust stem-coverage diagnostics."""
    rms=np.asarray(rms,dtype=float); times=np.asarray(times,dtype=float)
    onset=np.asarray(onset,dtype=float); onset_times=np.asarray(onset_times,dtype=float)
    if not len(rms) or not len(times):
        return {"available":False,"reason":"empty_audio_features"}
    floor=float(np.percentile(rms,20)); active=float(np.percentile(rms,75))
    threshold=max(floor*3.0,active*0.12,1e-6)
    hop=float(np.median(np.diff(times))) if len(times)>1 else minimum_sustained_seconds
    width=max(1,int(round(minimum_sustained_seconds/max(hop,1e-6))))
    mask=rms>=threshold
    sustained=np.convolve(mask.astype(int),np.ones(width,dtype=int),mode="same")>=max(1,int(np.ceil(width*0.60)))
    idx=np.where(sustained)[0]
    if not len(idx):
        return {"available":False,"reason":"no_sustained_activity","rms_threshold":round(threshold,8)}
    onset_threshold=max(float(np.percentile(onset,75)) if len(onset) else 0.0,1e-9)
    onset_idx=np.where(onset>=onset_threshold)[0] if len(onset) else np.asarray([],dtype=int)
    return {"available":True,"first_sustained_activity":round(float(times[idx[0]]),6),
            "last_sustained_activity":round(float(times[idx[-1]]),6),
            "first_strong_onset":round(float(onset_times[onset_idx[0]]),6) if len(onset_idx) else None,
            "rms_threshold":round(threshold,8),"minimum_sustained_seconds":minimum_sustained_seconds}

def diagnose_track_audio_coverage(score_times,audio_path,warp_fn,sr=22050,source="guitar"):
    score_times=sorted(float(t) for t in score_times)
    gp={"unique_onset_count":len(score_times),"first_note_time":round(float(warp_fn(score_times[0])),6) if score_times else None,
        "last_note_time":round(float(warp_fn(score_times[-1])),6) if score_times else None}
    if librosa is None or not audio_path:
        return {"version":1,"status":"diagnostic_only","source":source,"gp":gp,
                "audio":{"available":False,"reason":"missing_audio_or_librosa"}}
    y,actual_sr=librosa.load(audio_path,sr=sr,mono=True); hop=512
    rms=librosa.feature.rms(y=y,hop_length=hop)[0]
    times=librosa.frames_to_time(np.arange(len(rms)),sr=actual_sr,hop_length=hop)
    onset=librosa.onset.onset_strength(y=y,sr=actual_sr,hop_length=hop)
    onset_times=librosa.frames_to_time(np.arange(len(onset)),sr=actual_sr,hop_length=hop)
    audio=_activity_span_from_features(rms,times,onset,onset_times)
    comparison={}
    if gp["first_note_time"] is not None and audio.get("first_sustained_activity") is not None:
        lead=gp["first_note_time"]-float(audio["first_sustained_activity"])
        comparison={"audio_leads_gp_seconds":round(lead,6),
                    "classification":("stem_contains_substantial_unmapped_or_cross_instrument_audio" if lead>1.0 else
                                      "gp_precedes_audio" if lead<-1.0 else "entrances_nearby")}
    return {"version":1,"status":"diagnostic_only","source":source,"gp":gp,"audio":audio,"comparison":comparison}

def compute_simple_warp(event_times, audio_path, mode, sr=22050):
    diagnostics={"mode":mode,"selected_source":None,"candidates":[],"agreement":[]}
    if mode == "nominal":
        fn=lambda t:float(t)
        diagnostics.update({"method":"identity_gp_clock","offset_seconds":0.0,"scale":1.0})
        fn.alignment_diagnostics=diagnostics
        return fn
    if librosa is None or not audio_path or not event_times:
        fn=lambda t:float(t)
        diagnostics.update({"method":"identity_fallback","reason":"missing audio, events, or librosa",
                            "offset_seconds":0.0,"scale":1.0})
        fn.alignment_diagnostics=diagnostics
        return fn
    audio,actual_sr=librosa.load(audio_path,sr=sr,mono=True)
    envelope=librosa.onset.onset_strength(y=audio,sr=actual_sr)
    detected=librosa.onset.onset_detect(onset_envelope=envelope,sr=actual_sr,units="time",backtrack=False)
    if len(detected)==0:
        fn=lambda t:float(t)
        diagnostics.update({"method":"identity_fallback","reason":"no audio onsets detected",
                            "offset_seconds":0.0,"scale":1.0})
        fn.alignment_diagnostics=diagnostics
        return fn
    nominal_start=float(event_times[0]); audio_start=float(detected[0])
    if mode == "offset" or len(event_times)<2 or len(detected)<2:
        offset=audio_start-nominal_start
        fn=lambda t,o=offset:float(t)+o
        diagnostics.update({"method":"first_onset_offset","audio":str(audio_path),
                            "nominal_start":nominal_start,"audio_start":audio_start,
                            "offset_seconds":offset,"scale":1.0})
    else:
        nominal_end=float(event_times[-1]); audio_end=float(detected[-1])
        span=nominal_end-nominal_start
        scale=(audio_end-audio_start)/span if span>0 else 1.0
        fn=lambda t,a=audio_start,n=nominal_start,k=scale:a+(float(t)-n)*k
        diagnostics.update({"method":"first_last_onset_linear","audio":str(audio_path),
                            "nominal_start":nominal_start,"nominal_end":nominal_end,
                            "audio_start":audio_start,"audio_end":audio_end,
                            "offset_seconds":audio_start-nominal_start,"scale":scale})
    fn.alignment_diagnostics=diagnostics
    return fn

def compute_warp_candidate(event_times, audio_path, source, sr=22050, band_rad=0.25):
    """Build one instrument-specific DTW candidate without altering chart timing."""
    if librosa is None or interp1d is None or not event_times or not audio_path:
        return None
    y, actual_sr=librosa.load(audio_path, sr=sr, mono=True)
    duration=len(y)/actual_sr
    synth=synthesize_click_track(event_times, actual_sr, duration,
                                 freq=_source_click_frequency(source))
    fmin,fmax=_source_band(source,actual_sr)
    # Symmetric representation: real and synthetic use the same Mel band.
    real_mel=librosa.feature.melspectrogram(y=y,sr=actual_sr,fmin=fmin,fmax=fmax)
    synth_mel=librosa.feature.melspectrogram(y=synth,sr=actual_sr,fmin=fmin,fmax=fmax)
    real_db=librosa.power_to_db(real_mel,ref=np.max)
    synth_db=librosa.power_to_db(synth_mel,ref=np.max)
    onset_real=librosa.onset.onset_strength(S=real_db,sr=actual_sr)
    onset_synth=librosa.onset.onset_strength(S=synth_db,sr=actual_sr)
    _,wp=librosa.sequence.dtw(X=onset_synth[np.newaxis,:],Y=onset_real[np.newaxis,:],
                              global_constraints=True,band_rad=band_rad)
    fs=wp[::-1,0]; fr=wp[::-1,1]
    ts=librosa.frames_to_time(fs,sr=actual_sr)
    tr=librosa.frames_to_time(fr,sr=actual_sr)
    x,yv=fc.dedupe_monotonic_path(list(ts),list(tr))
    if len(x)<2: return None
    interp=interp1d(x,yv,kind="linear",bounds_error=False,fill_value=(yv[0],yv[-1]))
    def warp(t): return float(interp(t))
    real_onsets=librosa.onset.onset_detect(onset_envelope=onset_real,sr=actual_sr,
                                           units="time",backtrack=False)
    residuals=[]
    if len(real_onsets):
        real_onsets=np.asarray(real_onsets,dtype=float)
        for t in event_times:
            m=warp(t); pos=int(np.searchsorted(real_onsets,m)); choices=[]
            if pos<len(real_onsets): choices.append(abs(real_onsets[pos]-m))
            if pos>0: choices.append(abs(real_onsets[pos-1]-m))
            if choices: residuals.append(min(choices)*1000.0)
    median=float(np.median(residuals)) if residuals else 1000.0
    p95=float(np.percentile(residuals,95)) if residuals else 1500.0
    max_frames=max(len(onset_synth),len(onset_real),1)
    limit=max(1.0,band_rad*max_frames)
    dist=np.abs(wp[:,0].astype(float)-wp[:,1].astype(float))
    edge=float(np.mean(dist>=0.9*limit)*100.0) if len(dist) else 100.0
    event_span=max(event_times[-1]-event_times[0],1e-6)
    coverage=max(0.0,min(1.0,(float(x[-1])-float(x[0]))/event_span))
    score=1.5*coverage-median/180.0-p95/600.0-edge/35.0+0.12*math.log1p(len(event_times))
    diag={"source":source,"audio":str(audio_path),"event_count":len(event_times),
          "frequency_band_hz":[fmin,fmax],"median_residual_ms":round(median,3),
          "p95_residual_ms":round(p95,3),"coverage":round(coverage,4),
          "band_edge_percent":round(edge,3),"quality_score":round(score,6),
          "nominal_anchor_start":round(float(x[0]),6),
          "nominal_anchor_end":round(float(x[-1]),6)}
    return {"source":source,"warp":warp,"score":score,"diagnostics":diag}


def _candidate_agreement(candidates, sample_times):
    """Pairwise disagreement at shared measure boundaries, in milliseconds."""
    pairs=[]
    for i in range(len(candidates)):
        for j in range(i+1,len(candidates)):
            diffs=np.asarray([abs(candidates[i]["warp"](t)-candidates[j]["warp"](t))*1000.0
                              for t in sample_times],dtype=float)
            if len(diffs):
                pairs.append({"sources":[candidates[i]["source"],candidates[j]["source"]],
                              "median_ms":round(float(np.median(diffs)),3),
                              "p95_ms":round(float(np.percentile(diffs,95)),3),
                              "max_ms":round(float(np.max(diffs)),3)})
    return pairs


def choose_probabilistic_warp(candidates, sample_times, selection_threshold=0.55,
                              agreement_median_ms=75.0, agreement_p95_ms=200.0,
                              agreement_max_ms=400.0):
    valid=[c for c in candidates if c]
    if not valid:
        fn=lambda t:t
        fn.alignment_diagnostics={"mode":"identity","reason":"no usable alignment reference",
                                  "candidates":[],"agreement":[]}
        return fn
    scores=np.asarray([c["score"] for c in valid],dtype=float)
    weights=np.exp(scores-np.max(scores)); weights/=np.sum(weights)
    for c,w in zip(valid,weights): c["diagnostics"]["probability"]=round(float(w),6)
    agreement=_candidate_agreement(valid,sample_times)
    agree=all(p["median_ms"]<=agreement_median_ms and p["p95_ms"]<=agreement_p95_ms
              and p["max_ms"]<=agreement_max_ms for p in agreement)
    best=int(np.argmax(weights))
    if len(valid)==1 or weights[best]>=selection_threshold or not agree:
        chosen=valid[best]; fn=chosen["warp"]
        mode="selected" if agree or len(valid)==1 else "selected_due_to_disagreement"
        fn.alignment_diagnostics={"mode":mode,"selected_source":chosen["source"],
                                  "candidates":[c["diagnostics"] for c in valid],
                                  "agreement":agreement}
        return fn
    def fused(t): return float(sum(w*c["warp"](t) for c,w in zip(valid,weights)))
    fused.alignment_diagnostics={"mode":"probability_fusion","selected_source":None,
                                 "candidates":[c["diagnostics"] for c in valid],
                                 "agreement":agreement}
    return fused


def sanitize_warped_downbeats(nominal, warped, min_ratio=0.30, max_ratio=2.00,
                              min_seconds=0.20, max_bpm=400.0):
    """Repair bounded local DTW failures before any tempo or beat output is built."""
    n=min(len(nominal),len(warped)); clean=[dict(x) for x in warped[:n]]
    bad=set(); reasons={}
    for i in range(n-1):
        nom=nominal[i+1]["t_gp"]-nominal[i]["t_gp"]
        dur=clean[i+1]["time"]-clean[i]["time"]
        ratio=dur/nom if nom>0 else 1.0
        bpm=clean[i].get("ts_num",4)*60.0/dur if dur>0 else float("inf")
        why=[]
        if dur<=0: why.append("non_monotonic")
        if dur<min_seconds: why.append("too_short")
        if nom>0 and (ratio<min_ratio or ratio>max_ratio): why.append("stretch_ratio")
        if bpm>max_bpm: why.append("extreme_bpm")
        if why: bad.add(i); bad.add(i+1); reasons[i]=why
    repairs=[]
    if bad:
        groups=[]
        for idx in sorted(bad):
            if not groups or idx>groups[-1][-1]+1: groups.append([idx])
            else: groups[-1].append(idx)
        for group in groups:
            left=group[0]-1; right=group[-1]+1
            if left<0 or right>=n:
                continue
            t0=clean[left]["time"]; t1=clean[right]["time"]
            if t1<=t0: continue
            nominal_span=nominal[right]["t_gp"]-nominal[left]["t_gp"]
            if nominal_span<=0: continue
            for k in range(left+1,right):
                frac=(nominal[k]["t_gp"]-nominal[left]["t_gp"])/nominal_span
                clean[k]["time"]=round(t0+frac*(t1-t0),4)
            repairs.append({"from_measure":clean[group[0]]["measure"],
                            "to_measure":clean[group[-1]]["measure"],
                            "method":"bounded_linear_interpolation"})
    # Final strict validation. No duplicate or backward boundaries may ship.
    severe=[]
    for i in range(n-1):
        dur=clean[i+1]["time"]-clean[i]["time"]
        if dur<=0 or dur<min_seconds:
            severe.append({"measure":clean[i]["measure"],"duration":round(dur,6)})
    return clean,{"detected_bad_indices":sorted(bad),"repairs":repairs,
                  "unresolved_severe":severe,"original_reasons":reasons}


def build_explicit_feedpak_beats(downbeats):
    out=[]; previous=None
    for i,b in enumerate(downbeats):
        start=float(b["time"]); count=max(1,int(b.get("ts_num",4)))
        duration=(float(downbeats[i+1]["time"])-start) if i+1<len(downbeats) else previous
        if duration and duration>0: previous=duration
        if out and start<=out[-1]["time"]: continue
        out.append({"time":round(start,4),"measure":int(b["measure"])})
        if not duration or duration<=0: continue
        for j in range(1,count):
            t=round(start+j*duration/count,4)
            if t>out[-1]["time"]: out.append({"time":t,"measure":-1})
    return out


def analyze_alignment_quality(nominal,warped,warp_fn,sanitization):
    warnings=[]; ratios=[]
    for i in range(min(len(nominal),len(warped))-1):
        nd=nominal[i+1]["t_gp"]-nominal[i]["t_gp"]
        wd=warped[i+1]["time"]-warped[i]["time"]
        if nd>0 and wd>0:
            r=wd/nd; ratios.append(r)
            if abs(r-1)>0.15: warnings.append({"level":"warning","kind":"measure_stretch",
                                               "measure":warped[i]["measure"],"ratio":round(r,4)})
    if sanitization["repairs"]:
        warnings.append({"level":"warning","kind":"repaired_timeline",
                         "repairs":sanitization["repairs"]})
    if sanitization["unresolved_severe"]:
        warnings.append({"level":"severe","kind":"unresolved_timeline",
                         "items":sanitization["unresolved_severe"]})
    levels={w["level"] for w in warnings}
    severity="severe" if "severe" in levels else "warning" if "warning" in levels else "ok"
    return {"version":1,"severity":severity,"decision":getattr(warp_fn,"alignment_diagnostics",{}),
            "sanitization":sanitization,"min_stretch":round(min(ratios or [1]),4),
            "max_stretch":round(max(ratios or [1]),4),"warnings":warnings}


def shift_nominal_times(entries, offset, time_key="t_gp"):
    """Adds a constant offset to every entry's nominal-time field, in place-safe copy form."""
    if offset == 0.0:
        return entries
    out = []
    for e in entries:
        e2 = dict(e)
        e2[time_key] = e2[time_key] + offset
        out.append(e2)
    return out


# --------------------------------------------------------------------------
# Fretted-track parsing (guitar / bass)
# --------------------------------------------------------------------------

def parse_fretted_track(track, tempo_events, first_tick):
    string_count = len(track.strings)
    notes = []

    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                if beat.start is None:
                    continue
                beat_t = fc.tick_to_seconds(beat.start - first_tick, tempo_events)
                dur_ticks = beat.duration.time
                beat_end_t = fc.tick_to_seconds(
                    beat.start - first_tick + dur_ticks, tempo_events
                )
                sustain = beat_end_t - beat_t

                for note in beat.notes:
                    s = fc.remap_string_index(note.string, string_count)
                    entry = {"t_gp": beat_t, "s": s, "f": note.value}
                    fc.omit_if_negligible(entry, "sus", sustain, SUSTAIN_MIN_SECONDS)
                    notes.append(entry)

    notes.sort(key=lambda n: (n["t_gp"], n["s"]))

    absolute_midi = sorted(
        (gs.number, gs.value) for gs in track.strings
    )
    # track.strings numbers are 1-based highest-first, same convention as
    # notes; sort ascending by string number then reverse to get low->high
    absolute_midi_low_to_high = [v for _, v in sorted(absolute_midi, key=lambda x: -x[0])]

    anchors = compute_anchors(notes)

    return {
        "name": track.name or ("Bass" if track.channel and track.channel.instrument in (33, 34, 35, 36, 37, 38, 39) else "Guitar"),
        "string_count": string_count,
        "absolute_tuning_midi": absolute_midi_low_to_high,
        "capo": getattr(track, "offset", 0) or 0,
        "notes": notes,
        "anchors": anchors,
    }


def compute_anchors(notes, window_seconds=2.0):
    """
    Hand-position anchors, computed dynamically from played notes since
    there's no native GP marker for this (ported from a prior converter's
    tested heuristic — see architecture doc §2.2a).
    """
    from collections import Counter

    fretted = [n for n in notes if n["f"] > 0]
    if not fretted:
        return [{"time": 0.0, "fret": 1, "width": 4}]

    grouped = {}
    for n in fretted:
        grouped.setdefault(n["t_gp"], []).append(n)

    timestamps = sorted(grouped.keys())
    anchors = []
    last_fret = None
    last_width = None
    start_idx = 0

    while start_idx < len(timestamps):
        window_start = timestamps[start_idx]
        window_end = window_start + window_seconds

        window_frets = []
        chord_found = False
        chord_lowest = chord_highest = None

        end_idx = start_idx
        while end_idx < len(timestamps) and timestamps[end_idx] <= window_end:
            simultaneous = grouped[timestamps[end_idx]]
            if len(simultaneous) > 1:
                chord_found = True
                chord_lowest = min(n["f"] for n in simultaneous)
                chord_highest = max(n["f"] for n in simultaneous)
                break
            window_frets.append(simultaneous[0]["f"])
            end_idx += 1

        if chord_found:
            target_fret = chord_lowest
            target_width = max(4, chord_highest - chord_lowest + 1)
            start_idx = end_idx + 1
        else:
            if not window_frets:
                start_idx = end_idx + 1
                continue
            counts = Counter(window_frets)
            top_count = counts.most_common(1)[0][1]
            target_fret = min(f for f, c in counts.items() if c == top_count)
            close_frets = [f for f in window_frets if abs(f - target_fret) <= 4]
            target_width = max(4, max(close_frets) - min(close_frets) + 1) if close_frets else 4
            start_idx = end_idx

        if target_fret != last_fret or target_width != last_width:
            anchors.append({"time": window_start, "fret": target_fret, "width": target_width})
            last_fret, last_width = target_fret, target_width

    return anchors if anchors else [{"time": 0.0, "fret": 1, "width": 4}]


# --------------------------------------------------------------------------
# Drum-track parsing
# --------------------------------------------------------------------------

def parse_drum_track(track, tempo_events, first_tick):
    hits = []
    for measure in track.measures:
        for voice in measure.voices:
            for beat in voice.beats:
                if beat.start is None:
                    continue
                beat_t = fc.tick_to_seconds(beat.start - first_tick, tempo_events)
                for note in beat.notes:
                    piece = fc.gm_drum_to_piece(note.value)
                    hits.append({
                        "t_gp": beat_t,
                        "p": piece,
                        "v": max(1, min(127, note.velocity)),
                    })
    hits.sort(key=lambda h: h["t_gp"])
    return hits


# --------------------------------------------------------------------------
# Keyboard-track -> notation.json
# --------------------------------------------------------------------------

MIDDLE_C = 60


def gp_note_to_midi(track, note):
    """Return the sounding MIDI pitch for a fretted GP note.

    pyguitarpro Note.value is a fret/position, not an absolute MIDI pitch.
    Track strings carry the open-string MIDI values and are keyed by their
    Guitar Pro string number.
    """
    string_number = int(getattr(note, "string", 0) or 0)
    open_pitch = next((int(gs.value) for gs in track.strings
                       if int(getattr(gs, "number", 0)) == string_number), None)
    if open_pitch is None:
        raise ValueError(f"note references unknown GP string {string_number}")
    midi = open_pitch + int(note.value)
    if not 0 <= midi <= 127:
        raise ValueError(f"computed MIDI pitch {midi} is outside 0..127")
    return midi

def _duration_value(duration):
    index = int(getattr(duration, "index", 0) or 0)
    return min(32, max(1, 2 ** index if index else 1))


def parse_keyboard_track(track, tempo_events, first_tick, hand="combined"):
    """Preserve an explicitly configured piano track and all nonempty GP voices.

    Separate LH/RH tracks are never pitch-split. A genuinely combined piano
    track retains the legacy middle-C stave split, but still preserves voices.
    Note end times are kept until warp application so durations follow the same
    accepted timing map as onsets.
    """
    measures_out=[]; last_ts=None
    stave_defs = ([{"id":"lh","clef":"F4","label":"Left Hand"}] if hand=="left" else
                  [{"id":"rh","clef":"G2","label":"Right Hand"}] if hand=="right" else
                  [{"id":"rh","clef":"G2","label":"Right Hand"},{"id":"lh","clef":"F4","label":"Left Hand"}])
    for idx,measure in enumerate(track.measures,start=1):
        header=measure.header
        mout={"idx":idx,"t_gp":fc.tick_to_seconds(header.start-first_tick,tempo_events)}
        ts=(header.timeSignature.numerator,header.timeSignature.denominator.value) if header.timeSignature else None
        if ts and ts!=last_ts: mout["ts"]=list(ts); last_ts=ts
        staves={}
        for voice_index,voice in enumerate(measure.voices,start=1):
            by_stave={}
            for beat in voice.beats:
                if beat.start is None or not beat.notes: continue
                start=fc.tick_to_seconds(beat.start-first_tick,tempo_events)
                end=fc.tick_to_seconds(beat.start-first_tick+beat.duration.time,tempo_events)
                groups={"lh":[],"rh":[]}
                for note in beat.notes:
                    try:
                        midi = gp_note_to_midi(track, note)
                    except ValueError:
                        continue
                    target=("lh" if hand=="left" else "rh" if hand=="right" else
                            "rh" if midi>=MIDDLE_C else "lh")
                    groups[target].append({"midi":midi,"end_gp":end})
                for stave_id,notes in groups.items():
                    if notes:
                        by_stave.setdefault(stave_id,[]).append({"t_gp":start,"dur":_duration_value(beat.duration),"notes":notes})
            for stave_id,beats in by_stave.items():
                staves.setdefault(stave_id,{"voices":[]})["voices"].append({"v":voice_index,"beats":beats})
        if staves: mout["staves"]=staves
        measures_out.append(mout)
    return {"version":1,"instrument":"piano","source_track":track.name,
            "configured_hand":hand,"staves":stave_defs,"measures":measures_out}


def _parse_gp_lyric_items(text):
    """Expand GP lyric text into syllable and continuation items."""
    import re
    items=[]
    for raw in re.findall(r"_|-|[^\s|]+",str(text or "")):
        if raw=="_": items.append({"kind":"continuation","value":"+"}); continue
        if raw=="-":
            if items and items[-1]["kind"]=="text" and not items[-1]["value"].endswith("-"):
                items[-1]["value"]+="-"
            continue
        parts=[x for x in raw.split("-") if x]
        if len(parts)>1:
            items.extend({"kind":"text","value":x+("-" if i<len(parts)-1 else "")}
                         for i,x in enumerate(parts))
        else: items.append({"kind":"text","value":raw})
    return items

def _lyric_tokens(text):
    return [x["value"] for x in _parse_gp_lyric_items(text)]

def _collect_monophonic_vocal_slots(track,tempo_events,first_tick):
    by_time={}; rejected=[]
    for measure_index,measure in enumerate(track.measures,start=1):
        for voice in measure.voices:
            for beat in voice.beats:
                if beat.start is None or not beat.notes: continue
                start=fc.tick_to_seconds(beat.start-first_tick,tempo_events)
                end=fc.tick_to_seconds(beat.start-first_tick+beat.duration.time,tempo_events)
                pitches=[]
                for note in beat.notes:
                    try: pitches.append(gp_note_to_midi(track,note))
                    except ValueError: pass
                if not pitches: continue
                rec=by_time.setdefault(round(start,9),{"t_gp":start,"end_gp":end,"measure":measure_index,"midis":set()})
                rec["end_gp"]=max(rec["end_gp"],end); rec["midis"].update(pitches)
    slots=[]
    for rec in by_time.values():
        midis=sorted(rec.pop("midis"))
        if len(midis)!=1:
            rejected.append({"measure":rec["measure"],"time":round(rec["t_gp"],6),"reason":"polyphonic_vocal_onset","pitches":midis}); continue
        rec["midi"]=midis[0]; slots.append(rec)
    return sorted(slots,key=lambda x:x["t_gp"]),rejected

def _spread_measure_items(items,slots):
    n=len(slots); m=len(items)
    if not n or not m:return [],"empty"
    if m<=n:
        indices=[round(i*(n-1)/max(1,m-1)) for i in range(m)] if m>1 else [0]
        labels=["+"]*n
        for item,index in zip(items,indices): labels[index]=item["value"]
        mode="exact" if m==n else "melisma_linear"
    else:
        groups=[[] for _ in range(n)]
        for i,item in enumerate(items): groups[round(i*(n-1)/max(1,m-1))].append(item["value"])
        labels=[" ".join(x) if x else "+" for x in groups]; mode="grouped_linear"
    return [{"t_gp":slot["t_gp"],"end_gp":slot["end_gp"],"w":label,"measure":slot["measure"],"assignment":"measure_bounded_"+mode}
            for slot,label in zip(slots,labels)],mode

def allocate_measure_lyric_line(slots,text,starting_measure):
    """Consume the lyric stream by each measure's real note-slot budget."""
    items=_parse_gp_lyric_items(text); by_measure={}
    for slot in slots: by_measure.setdefault(slot["measure"],[]).append(slot)
    words=[]; pitches=[]; reports=[]; cursor=0
    for measure in sorted(m for m in by_measure if m>=int(starting_measure or 1)):
        note_slots=sorted(by_measure[measure],key=lambda x:x["t_gp"])
        local=items[cursor:cursor+len(note_slots)]; cursor+=len(local)
        if not local:
            reports.append({"measure":measure,"status":"no_lyric_items","notes":len(note_slots)}); continue
        assigned,mode=_spread_measure_items(local,note_slots); words.extend(assigned)
        pitches.extend({"t_gp":x["t_gp"],"end_gp":x["end_gp"],"midi":x["midi"],"measure":measure} for x in note_slots)
        reports.append({"measure":measure,"status":"ok","mode":mode,"items":len(local),"notes":len(note_slots)})
    return words,pitches,{"measures":reports,"total_items":len(items),"consumed_items":cursor,"unresolved_items":max(0,len(items)-cursor)}

def allocate_timed_gp_lyrics(slots,anchors,tolerance_seconds=0.025,maximum_anchor_lag_seconds=1.0):
    """Compatibility allocator for explicit timed beat text."""
    slots=sorted((dict(x) for x in slots),key=lambda x:x["t_gp"])
    anchors=sorted((dict(x) for x in anchors),key=lambda x:x["t_gp"])
    words=[]; pitches=[]; used=set(); assignments=[]; missing=[]
    for ai,anchor in enumerate(anchors):
        limit=anchors[ai+1]["t_gp"] if ai+1<len(anchors) else float("inf")
        choices=[]
        for si,slot in enumerate(slots):
            if si in used or slot["t_gp"]>=limit-tolerance_seconds: continue
            delta=slot["t_gp"]-anchor["t_gp"]
            if abs(delta)<=tolerance_seconds or 0<=delta<=maximum_anchor_lag_seconds:
                choices.append((abs(delta),si))
        if not choices:
            missing.append({**anchor,"reason":"no_vocal_note_near_anchor"}); continue
        first=min(choices)[1]; owned=[]
        for si in range(first,len(slots)):
            if slots[si]["t_gp"]>=limit-tolerance_seconds: break
            if si not in used: owned.append(si)
        for pos,si in enumerate(owned):
            slot=slots[si]; used.add(si); label=anchor["w"] if pos==0 else "+"
            words.append({"t_gp":slot["t_gp"],"end_gp":slot["end_gp"],"w":label,"measure":slot["measure"]})
            pitches.append({"t_gp":slot["t_gp"],"end_gp":slot["end_gp"],"midi":slot["midi"],"measure":slot["measure"]})
        assignments.append({"anchor_time":round(anchor["t_gp"],6),"anchor_text":anchor["w"],"first_note_time":round(slots[first]["t_gp"],6),"note_count":len(owned)})
    return words,pitches,{"assignments":assignments,"anchors_without_notes":missing,"unresolved_vocal_notes":len(slots)-len(used)}

def parse_gp_vocals(song,track,tempo_events,first_tick):
    slots,polyphonic=_collect_monophonic_vocal_slots(track,tempo_events,first_tick)
    words=[]; pitches=[]; reports=[]; lines=[]
    for index,line in enumerate(getattr(getattr(song,"lyrics",None),"lines",[]) or [],start=1):
        text=getattr(line,"lyrics",None) if line else None
        if not text: continue
        starting=int(getattr(line,"startingMeasure",1) or 1)
        lw,lp,report=allocate_measure_lyric_line(slots,text,starting)
        words.extend(lw); pitches.extend(lp); reports.append({"line":index,**report})
        lines.append({"line":index,"starting_measure":starting,"raw_lyrics":text,"parsed_items":report["total_items"]})
    if not words:
        anchors=[]
        for measure_index,measure in enumerate(track.measures,start=1):
            for voice in measure.voices:
                for beat in voice.beats:
                    text=getattr(beat,"text",None)
                    if beat.start is None or not text: continue
                    labels=_lyric_tokens(text)
                    if labels:
                        anchors.append({"t_gp":fc.tick_to_seconds(beat.start-first_tick,tempo_events),
                                        "w":" ".join(x for x in labels if x!="+") or "+",
                                        "measure":measure_index})
        words,pitches,timed=allocate_timed_gp_lyrics(slots,anchors)
        reports.append({"source":"timed_beat_text_compatibility","assigned_anchors":len(timed["assignments"])})
    used={x["measure"] for x in words}; note_measures={x["measure"] for x in slots}; rejected={x["measure"] for x in polyphonic}
    return {"version":4,"source":"guitar_pro_measure_lyrics","track":track.name,
        "lyrics":sorted(words,key=lambda x:x["t_gp"]),"pitch_notes":sorted(pitches,key=lambda x:x["t_gp"]),
        "diagnostics":{"allocator":"measure_bounded_syllable_line_v2","lyric_lines":lines,"measure_fallback":reports,
        "explicit_text_notes":sum(x["w"]!="+" for x in words),"continuation_notes":sum(x["w"]=="+" for x in words),
        "usable_vocal_notes":len(slots),"unresolved_vocal_notes":max(0,len(slots)-len(pitches)),
        "polyphonic_onsets_rejected":polyphonic,"bars_written":len(used),"bars_skipped":len((note_measures|rejected)-used)}}

def midi_to_virtual_position(midi):
    """Encode piano pitch exactly like the proven RB3 keys converter.

    Playable keys use 24-semitone virtual blocks:
        s = midi // 24
        f = midi % 24
    Only the standard 88-key piano range is emitted.
    """
    midi = int(midi)
    if not 21 <= midi <= 108:
        raise ValueError(f"Piano MIDI pitch {midi} is outside 21..108")
    return midi // 24, midi % 24


def playable_keyboard_from_notation(notation):
    """Flatten one configured GP piano track into an RB3-compatible keys lane."""
    notes = []
    skipped_out_of_range = 0
    for measure in notation.get("measures", []):
        for stave in measure.get("staves", {}).values():
            for voice in stave.get("voices", []):
                for beat in voice.get("beats", []):
                    start = float(beat["t"])
                    for raw in beat.get("notes", []):
                        try:
                            string, fret = midi_to_virtual_position(raw["midi"])
                        except ValueError:
                            skipped_out_of_range += 1
                            continue
                        note = {"t": round(start, 4), "s": string, "f": fret}
                        fc.omit_if_negligible(
                            note, "d", float(raw.get("d", 0.0)), SUSTAIN_MIN_SECONDS
                        )
                        notes.append(note)
    notes.sort(key=lambda n: (n["t"], n["s"], n["f"]))
    if skipped_out_of_range:
        fc.log(
            f"'{notation.get('source_track') or 'Piano'}': skipped "
            f"{skipped_out_of_range} pitch(es) outside MIDI 21..108",
            indent=2,
        )
    return {
        "name": notation.get("source_track") or "Piano",
        "type": "piano",
        "tuning": [0, 0, 0, 0, 0, 0],
        "capo": 0,
        "notes": notes,
        "chords": [],
        # Keys do not use the guitar hand-position heuristic. Match the
        # known-working RB3 converter's fixed piano anchor exactly.
        "anchors": [{"time": 0.0, "fret": 1, "width": 4}],
        "handshapes": [],
        "templates": [],
    }

def shift_and_warp_notation(notation, offset, warp_fn):
    out_measures=[]
    for m in notation["measures"]:
        m2=dict(m); m2["t"]=round(warp_fn(m2.pop("t_gp")+offset),4)
        if "staves" in m2:
            staves2={}
            for sid,stave in m2["staves"].items():
                voices2=[]
                for voice in stave["voices"]:
                    beats2=[]
                    for beat in voice["beats"]:
                        b2=dict(beat); nominal=b2.pop("t_gp")+offset; real=warp_fn(nominal); b2["t"]=round(real,4)
                        notes=[]
                        for note in b2.get("notes",[]):
                            n=dict(note); end_nominal=n.pop("end_gp")+offset
                            n["d"]=round(max(0.0,warp_fn(end_nominal)-real),4); notes.append(n)
                        b2["notes"]=notes; beats2.append(b2)
                    voices2.append({**voice,"beats":beats2})
                staves2[sid]={**stave,"voices":voices2}
            m2["staves"]=staves2
        out_measures.append(m2)
    return {**notation,"measures":out_measures}


def warp_gp_vocals(vocals, offset, warp_fn):
    if not vocals: return None
    out=dict(vocals)
    out["lyrics"]=[]; out["pitch_notes"]=[]
    for raw in vocals.get("lyrics",[]):
        n=dict(raw); start=warp_fn(n.pop("t_gp")+offset); end=warp_fn(n.pop("end_gp")+offset)
        n["t"]=round(start,4); n["d"]=round(max(0.0,end-start),4); out["lyrics"].append(n)
    for raw in vocals.get("pitch_notes",[]):
        n=dict(raw); start=warp_fn(n.pop("t_gp")+offset); end=warp_fn(n.pop("end_gp")+offset)
        n["t"]=round(start,4); n["d"]=round(max(0.0,end-start),4); out["pitch_notes"].append(n)
    return out

# --------------------------------------------------------------------------
# DTW alignment
# --------------------------------------------------------------------------

def synthesize_click_track(hit_times, sr, duration_s, freq=180.0, decay_s=0.04):
    """
    Short exponentially-decaying clicks, not sustained tones — a sine
    tone has no sharp transient and onset_strength localizes poorly
    against it (see architecture doc §2.2 step 4).
    """
    n_samples = max(1, int(duration_s * sr))
    audio = np.zeros(n_samples, dtype=np.float32)
    click_n = max(1, int(decay_s * sr))
    t = np.arange(click_n) / sr
    click = np.sin(2 * np.pi * freq * t) * np.exp(-t / (decay_s / 5.0))
    for ht in hit_times:
        start = int(ht * sr)
        if start >= n_samples:
            continue
        end = min(n_samples, start + click_n)
        audio[start:end] += click[: end - start]
    return audio


def compute_warp_function(gp_drum_hits, drums_audio_path, sr=22050, band_rad=0.25):
    """
    Returns a function t_gp -> t_real built from DTW between a synthetic
    click track (from GP's nominal drum timing) and onset-strength of the
    real drum stem. Falls back to identity if librosa/scipy aren't
    available or there are no drum hits to align against.
    """
    if librosa is None or interp1d is None or not gp_drum_hits:
        return lambda t: t

    with fc.timed_step("Loading drum stem audio", indent=2):
        y, actual_sr = librosa.load(drums_audio_path, sr=sr, mono=True)
    duration_s = len(y) / actual_sr
    fc.log(f"{duration_s:.1f}s of audio at {actual_sr}Hz", indent=2)

    hit_times = [h["t_gp"] for h in gp_drum_hits]
    synth = synthesize_click_track(hit_times, actual_sr, duration_s)

    with fc.timed_step("Computing onset-strength envelopes", indent=2):
        onset_real = librosa.onset.onset_strength(y=y, sr=actual_sr)
        onset_synth = librosa.onset.onset_strength(y=synth, sr=actual_sr)

    with fc.timed_step("Running DTW", indent=2):
        _, wp = librosa.sequence.dtw(
            X=onset_synth[np.newaxis, :],
            Y=onset_real[np.newaxis, :],
            global_constraints=True,
            band_rad=band_rad,
        )
    # wp is ordered from the end backward; frame indices for synth (col 0)
    # and real (col 1)
    frames_synth = wp[::-1, 0]
    frames_real = wp[::-1, 1]
    times_synth = librosa.frames_to_time(frames_synth, sr=actual_sr)
    times_real = librosa.frames_to_time(frames_real, sr=actual_sr)

    x, y_vals = fc.dedupe_monotonic_path(list(times_synth), list(times_real))
    fc.log(f"Warp path: {len(wp)} DTW points -> {len(x)} unique time anchors", indent=2)
    if len(x) < 2:
        fc.log("Not enough unique warp points — falling back to identity mapping", indent=2)
        return lambda t: t

    warp = interp1d(x, y_vals, kind="linear", bounds_error=False,
                     fill_value=(y_vals[0], y_vals[-1]))

    def warp_fn(t_gp):
        return float(warp(t_gp))

    return warp_fn




def _cluster_event_times(times,tolerance=0.025):
    out=[]
    for value in sorted(float(t) for t in times):
        if not out or value-out[-1]>tolerance:
            out.append(value)
    return out


def _score_gaps(times,start_time,end_time):
    points=[float(start_time)]+_cluster_event_times(times)+[float(end_time)]
    return [(a,b) for a,b in zip(points,points[1:]) if b>a]


_SILENCE_FEATURE_CACHE={}


def _audio_inactivity(audio_path,start,end,sr=22050,hop=512):
    if not audio_path or end<=start: return None
    key=(str(audio_path),int(sr),int(hop))
    cached=_SILENCE_FEATURE_CACHE.get(key)
    if cached is None:
        y,actual_sr=librosa.load(audio_path,sr=sr,mono=True)
        rms=librosa.feature.rms(y=y,hop_length=hop)[0]
        times=librosa.frames_to_time(np.arange(len(rms)),sr=actual_sr,hop_length=hop)
        onset=librosa.onset.onset_strength(y=y,sr=actual_sr,hop_length=hop)
        onset_times=librosa.frames_to_time(np.arange(len(onset)),sr=actual_sr,hop_length=hop)
        cached=(rms,times,onset,onset_times)
        _SILENCE_FEATURE_CACHE[key]=cached
    rms,times,onset,onset_times=cached
    mask=(times>=start)&(times<end)
    if not np.any(mask): return None
    low=float(np.percentile(rms,30)); high=float(np.percentile(rms,75)); mean=float(np.mean(rms[mask]))
    normalized=float(np.clip(1.0-(mean-low)/max(high-low,1e-9),0.0,1.0))
    omask=(onset_times>=start)&(onset_times<end)
    onset_density=float(np.mean(onset[omask])) if np.any(omask) else 0.0
    onset_ref=float(np.percentile(onset,60)) if len(onset) else 1.0
    onset_quiet=float(np.clip(1.0-onset_density/max(onset_ref,1e-9),0.0,1.0))
    confidence=0.65*normalized+0.35*onset_quiet
    return {"confidence":round(confidence,4),"mean_rms":round(mean,7),
            "rms_quiet":round(normalized,4),"onset_quiet":round(onset_quiet,4),
            "confirmed":bool(confidence>=0.60)}


def diagnose_silence_checkpoints(score_events,audio_paths,base_warp,measure_downbeats,
                                  tempo_bpm,search_radius=1.0,sr=22050):
    """Find internal silence landmarks without modifying timing.

    Drum-only gaps must exceed one second. Gaps shared by at least two
    non-vocal instrument stems must last at least one local beat. The full
    mix confirms audibility but never counts as an independent stem.
    """
    beat_seconds=60.0/max(float(tempo_bpm),1e-6)
    start=float(measure_downbeats[0]["t_gp"]); end=float(measure_downbeats[-1]["t_gp"])
    gaps={name:_score_gaps(times,start,end) for name,times in score_events.items() if times}
    raw=[]
    # Drum-only long gaps.
    for a,b in gaps.get("drums",[]):
        if b-a>1.0:
            raw.append({"score_start":a,"score_end":b,"supporting_stems":["drums"],
                        "rule":"drum_only_over_1_second"})
    # Intersections shared by at least two independent instrument stems.
    names=sorted(gaps)
    for i,name_a in enumerate(names):
        for name_b in names[i+1:]:
            for a0,a1 in gaps[name_a]:
                for b0,b1 in gaps[name_b]:
                    lo=max(a0,b0); hi=min(a1,b1)
                    if hi-lo>beat_seconds:
                        raw.append({"score_start":lo,"score_end":hi,
                                    "supporting_stems":[name_a,name_b],
                                    "rule":"two_plus_stems_one_beat"})
    # Merge the same musical boundary within 500ms, unioning support.
    merged=[]
    for cand in sorted(raw,key=lambda x:x["score_end"]):
        if merged and abs(cand["score_end"]-merged[-1]["score_end"])<=0.5:
            old=merged[-1]
            old["score_start"]=max(old["score_start"],cand["score_start"])
            old["supporting_stems"]=sorted(set(old["supporting_stems"]+cand["supporting_stems"]))
            if len(old["supporting_stems"])>=2: old["rule"]="two_plus_stems_one_beat"
        else: merged.append(dict(cand))
    output=[]
    for cand in merged:
        boundary=cand["score_end"]; support=cand["supporting_stems"]
        duration=boundary-cand["score_start"]
        # Recognizable post-silence boundary: three clusters in 750ms, or
        # entries from at least two supporting sources within 120ms.
        per_source={name:[t for t in score_events.get(name,[]) if boundary<=t<=boundary+.75]
                    for name in support}
        post_clusters=_cluster_event_times([t for values in per_source.values() for t in values])
        entering=[name for name,values in per_source.items() if values and values[0]-boundary<=.12]
        boundary_ok=len(post_clusters)>=3 or len(entering)>=2
        audio_start=float(base_warp(cand["score_start"])); audio_end=float(base_warp(boundary))
        confirmations={}
        for name in support:
            evidence=_audio_inactivity(audio_paths.get(name),audio_start,audio_end,sr=sr)
            if evidence is not None: confirmations[name]=evidence
        confirmed=[name for name,e in confirmations.items() if e["confirmed"]]
        required_audio=1 if support==["drums"] else 2
        full_evidence=_audio_inactivity(audio_paths.get("full"),audio_start,audio_end,sr=sr)
        accepted=(duration>1.0 and len(confirmed)>=1 if support==["drums"] else
                  duration>beat_seconds and len(confirmed)>=2)
        accepted=bool(accepted and boundary_ok)
        nearest=min(measure_downbeats,key=lambda b:abs(float(b["t_gp"])-boundary))
        output.append({"measure":int(nearest["measure"]),"pattern_type":"silence_transition",
            "score_start":round(cand["score_start"],4),"score_end":round(boundary,4),
            "predicted_audio_start":round(audio_start,4),"predicted_audio_end":round(audio_end,4),
            "duration_seconds":round(duration,4),"duration_beats":round(duration/beat_seconds,4),
            "rule":cand["rule"],"supporting_stems":support,"supporting_stem_count":len(support),
            "audio_confirmed_stems":confirmed,"audio_confirmation":confirmations,
            "full_mix_confirmation":full_evidence,"post_silence_clusters_750ms":len(post_clusters),
            "boundary_entering_stems_120ms":entering,"boundary_recognizable":bool(boundary_ok),
            "accepted":accepted,
            "warnings":([] if accepted else
                (["rejected_boundary_not_recognizable"] if not boundary_ok else [])+
                (["rejected_insufficient_audio_stem_confirmation"] if len(confirmed)<required_audio else []))})
    strong=sum(1 for x in output if x["accepted"])
    return {"version":1,"status":"diagnostic_only","vocals_ignored":True,
            "drum_only_minimum_seconds":1.0,"multi_stem_minimum_beats":1.0,
            "beat_seconds":round(beat_seconds,6),"generated":len(output),"strong":strong,
            "rejected":len(output)-strong,"checkpoints":output}

_CHROMA_FEATURE_CACHE={}


def _load_audio_chroma(audio_path,sr=22050,hop=512):
    if not audio_path: return None
    key=(str(audio_path),int(sr),int(hop))
    if key in _CHROMA_FEATURE_CACHE: return _CHROMA_FEATURE_CACHE[key]
    y,actual_sr=librosa.load(audio_path,sr=sr,mono=True)
    harmonic=librosa.effects.harmonic(y,margin=3.0)
    chroma=librosa.feature.chroma_cqt(y=harmonic,sr=actual_sr,hop_length=hop)
    times=librosa.frames_to_time(np.arange(chroma.shape[1]),sr=actual_sr,hop_length=hop)
    norms=np.linalg.norm(chroma,axis=0,keepdims=True)
    chroma=chroma/np.maximum(norms,1e-9)
    result=(chroma,times,float(actual_sr),int(hop))
    _CHROMA_FEATURE_CACHE[key]=result
    return result


def _score_chroma_template(pitch_events,start,end,frame_times):
    template=np.zeros((12,len(frame_times)),dtype=float)
    if not pitch_events or end<=start: return template
    for event in pitch_events:
        t=float(event["t_gp"])
        if start<=t<=end:
            j=int(np.argmin(np.abs(frame_times-t)))
            template[int(event["midi"])%12,j]=1.0
            if j+1<len(frame_times): template[int(event["midi"])%12,j+1]=max(template[int(event["midi"])%12,j+1],0.5)
    # Mild temporal smoothing, preserving absolute pitch class.
    if template.shape[1]>=3:
        template=(np.roll(template,1,axis=1)+2*template+np.roll(template,-1,axis=1))/4.0
        template[:,0]*=4/3; template[:,-1]*=4/3
    norms=np.linalg.norm(template,axis=0,keepdims=True)
    return template/np.maximum(norms,1e-9)


def _chroma_candidate_for_source(pitch_events,audio_path,nominal_start,nominal_end,
                                 linear_prediction,base_warp,search_radius,sr=22050,hop=512):
    loaded=_load_audio_chroma(audio_path,sr,hop)
    if loaded is None: return {"available":False,"reason":"missing_audio"}
    audio_chroma,audio_times,actual_sr,hop=loaded
    frame_seconds=hop/actual_sr
    n=max(16,int(round((float(base_warp(nominal_end))-float(base_warp(nominal_start)))/frame_seconds)))
    score_times=np.linspace(nominal_start,nominal_end,n,endpoint=False)
    template=_score_chroma_template(pitch_events,nominal_start,nominal_end,score_times)
    active=np.where(np.sum(template,axis=1)>0)[0]
    usable=int(np.sum(np.linalg.norm(template,axis=0)>0))
    if len(active)<3 or usable<16:
        return {"available":False,"reason":"insufficient_pitched_content",
                "active_pitch_classes":int(len(active)),"usable_chroma_frames":usable}
    center=float(linear_prediction)
    offsets=np.arange(-search_radius,search_radius+frame_seconds/2,frame_seconds)
    scored=[]
    template_active=np.linalg.norm(template,axis=0)>0
    for residual in offsets:
        candidate_start=float(base_warp(nominal_start))+float(residual)
        query=candidate_start+np.arange(n)*frame_seconds
        indices=np.searchsorted(audio_times,query)
        indices=np.clip(indices,0,audio_chroma.shape[1]-1)
        observed=audio_chroma[:,indices]
        sims=np.sum(template[:,template_active]*observed[:,template_active],axis=0)
        score=float(np.mean(sims)) if len(sims) else 0.0
        scored.append((score,float(residual),candidate_start))
    scored.sort(reverse=True,key=lambda x:x[0])
    # Non-maximum suppression makes second-best a genuinely distinct location.
    best=scored[0]; distinct=[x for x in scored[1:] if abs(x[1]-best[1])>=0.10]
    second=distinct[0] if distinct else scored[-1]
    null=min(scored,key=lambda x:abs(x[1]))
    margin=best[0]-second[0]; gain=best[0]-null[0]
    # Pattern uniqueness combines local ambiguity with chroma diversity.
    entropy=[]
    for j in range(template.shape[1]):
        col=template[:,j]
        if np.sum(col)>0:
            q=col/np.sum(col); entropy.append(-float(np.sum(q[q>0]*np.log(q[q>0])))/math.log(12))
    diversity=float(np.mean(entropy)) if entropy else 0.0
    uniqueness=float(np.clip(0.7*margin/0.10+0.3*diversity,0,1))
    accepted=bool(best[0]>=0.55 and margin>=0.10 and gain>=0.025 and abs(best[1])<=0.250)
    warnings=[]
    if best[0]<0.55: warnings.append("rejected_low_chroma_similarity")
    if margin<0.10: warnings.append("rejected_insufficient_best_second_margin")
    if gain<0.025: warnings.append("rejected_insufficient_gain_over_linear")
    if abs(best[1])>0.250: warnings.append("rejected_excessive_residual")
    return {"available":True,"active_pitch_classes":int(len(active)),
            "usable_chroma_frames":usable,"pattern_uniqueness":round(uniqueness,4),
            "best_time":round(center+best[1],6),"residual_ms":round(best[1]*1000,3),
            "best_score":round(best[0],4),"second_score":round(second[0],4),
            "margin":round(margin,4),"linear_null_score":round(null[0],4),
            "gain_over_linear":round(gain,4),"accepted":accepted,"warnings":warnings}


def diagnose_chroma_checkpoints(pitched_events,audio_paths,base_warp,measure_downbeats,
                                onset_report,every_measures=4,search_radius=1.0,sr=22050):
    config={"checkpoint_measures":int(every_measures),"search_radius_seconds":float(search_radius),
            "context_measures_before":1,"context_measures_after":1,
            "minimum_active_pitch_classes":3,"minimum_chroma_frames":16,
            "minimum_similarity":0.55,"minimum_best_second_margin":0.10,
            "minimum_gain_over_linear":0.025,"maximum_cross_stem_disagreement_ms":50.0,
            "maximum_chroma_onset_disagreement_ms":50.0,"maximum_residual_ms":250.0}
    periodic={int(x["measure"]):x for x in (onset_report or {}).get("checkpoints",[])}
    checkpoints=[]; source_counts={"bass":0,"guitar":0,"piano":0,"full":0}
    step=max(1,int(every_measures)); bounds=measure_downbeats
    for i,b in enumerate(bounds):
        if i%step: continue
        lo=max(0,i-1); hi=min(len(bounds)-1,i+1)
        nominal_start=float(bounds[lo]["t_gp"]); nominal_end=float(bounds[hi]["t_gp"])
        nominal=float(b["t_gp"]); linear=float(base_warp(nominal))
        sources={}
        for name in ("bass","guitar","piano"):
            result=_chroma_candidate_for_source(pitched_events.get(name,[]),audio_paths.get(name),
                nominal_start,nominal_end,linear,base_warp,search_radius,sr)
            sources[name]=result
            if result.get("available"): source_counts[name]+=1
        # Full mix corroborates the combined pitched score, never independently accepts.
        combined=[]
        for name in ("bass","guitar","piano"): combined.extend(pitched_events.get(name,[]))
        full=_chroma_candidate_for_source(combined,audio_paths.get("full"),nominal_start,nominal_end,
                                          linear,base_warp,search_radius,sr)
        if full.get("available"): source_counts["full"]+=1
        full["corroborating_only"]=True; sources["full"]=full
        strong=[(name,x) for name,x in sources.items() if name!="full" and x.get("accepted")]
        residuals=[x["residual_ms"] for _,x in strong]
        disagreement=(max(residuals)-min(residuals)) if len(residuals)>=2 else None
        onset=periodic.get(int(b["measure"])); onset_ms=None
        if onset and onset.get("accepted") and onset.get("median_correction_seconds") is not None:
            onset_ms=1000*float(onset["median_correction_seconds"])
        combined_ms=float(np.median(residuals)) if residuals else None
        chroma_onset=abs(combined_ms-onset_ms) if combined_ms is not None and onset_ms is not None else None
        warnings=[]
        if not strong: warnings.append("rejected_no_strong_isolated_chroma_source")
        if disagreement is not None and disagreement>50: warnings.append("rejected_cross_stem_disagreement")
        if chroma_onset is not None and chroma_onset>50: warnings.append("rejected_chroma_onset_disagreement")
        accepted=bool(strong and not warnings)
        checkpoints.append({"measure":int(b["measure"]),"nominal_time":round(nominal,4),
            "linear_prediction":round(linear,4),"window_nominal_start":round(nominal_start,4),
            "window_nominal_end":round(nominal_end,4),"sources":sources,
            "strong_isolated_sources":[name for name,_ in strong],
            "cross_stem_disagreement_ms":None if disagreement is None else round(disagreement,3),
            "onset_residual_ms":None if onset_ms is None else round(onset_ms,3),
            "chroma_onset_disagreement_ms":None if chroma_onset is None else round(chroma_onset,3),
            "combined_candidate_time":None if combined_ms is None else round(linear+combined_ms/1000,6),
            "combined_residual_ms":None if combined_ms is None else round(combined_ms,3),
            "accepted":accepted,"warnings":warnings})
    strong=sum(1 for x in checkpoints if x["accepted"])
    return {"version":1,"status":"diagnostic_only","config":config,
            "summary":{"generated":len(checkpoints),"strong":strong,
                       "rejected":len(checkpoints)-strong,"by_source":source_counts,
                       "timing_changes_applied":False},"checkpoints":checkpoints}


def diagnose_checkpoints(event_times,audio_path,base_warp,measure_downbeats,
                         every_measures=4,search_radius=1.0,sr=22050):
    """Measure local onset agreement without changing the linear warp."""
    if librosa is None or not audio_path or not event_times:
        return {"version":1,"status":"unavailable","checkpoints":[],
                "reason":"missing librosa, audio, or symbolic events"}
    y,actual_sr=librosa.load(audio_path,sr=sr,mono=True)
    env=librosa.onset.onset_strength(y=y,sr=actual_sr)
    detected=np.asarray(librosa.onset.onset_detect(onset_envelope=env,sr=actual_sr,
                                                  units="time",backtrack=False),dtype=float)
    checkpoints=[]; strong=0; rejected=0
    step=max(1,int(every_measures))
    events=np.asarray(sorted(float(t) for t in event_times),dtype=float)
    for index,b in enumerate(measure_downbeats):
        if index % step: continue
        nominal=float(b["t_gp"]); predicted=float(base_warp(nominal))
        # Pick symbolic events close to this score boundary, then compare each
        # predicted event with the nearest detected onset inside the radius.
        half_window=max(0.35,min(1.5,float(search_radius)))
        local=events[np.abs(events-nominal)<=half_window]
        residuals=[]
        for event in local:
            target=float(base_warp(event))
            lo=np.searchsorted(detected,target-search_radius)
            hi=np.searchsorted(detected,target+search_radius,side="right")
            if hi>lo:
                nearest=detected[lo:hi][np.argmin(np.abs(detected[lo:hi]-target))]
                residuals.append(float(nearest-target))
        if len(residuals)>=3:
            correction=float(np.median(residuals)); spread=float(np.median(np.abs(np.asarray(residuals)-correction)))
            accepted=abs(correction)<=search_radius and spread<=0.12
        else:
            correction=None; spread=None; accepted=False
        strong+=int(accepted); rejected+=int(not accepted)
        checkpoints.append({"measure":int(b["measure"]),"nominal_time":round(nominal,4),
                            "predicted_time":round(predicted,4),"matched_events":len(residuals),
                            "median_correction_seconds":round(correction,4) if correction is not None else None,
                            "mad_seconds":round(spread,4) if spread is not None else None,
                            "accepted":accepted})
    return {"version":1,"status":"diagnostic_only","checkpoint_measures":step,
            "search_radius_seconds":float(search_radius),"strong":strong,
            "rejected":rejected,"checkpoints":checkpoints}


_CHECKPOINT_ONSET_CACHE = {}

def _detected_onset_times(audio_path, sr=22050):
    """Return cached onset times for checkpoint boundary refinement."""
    if not audio_path or librosa is None:
        return np.asarray([], dtype=float)
    key=(str(audio_path),int(sr))
    if key not in _CHECKPOINT_ONSET_CACHE:
        y,actual_sr=librosa.load(audio_path,sr=sr,mono=True)
        envelope=librosa.onset.onset_strength(y=y,sr=actual_sr)
        _CHECKPOINT_ONSET_CACHE[key]=np.asarray(
            librosa.onset.onset_detect(onset_envelope=envelope,sr=actual_sr,
                                       units="time",backtrack=False),dtype=float)
    return _CHECKPOINT_ONSET_CACHE[key]

def measure_silence_boundary_residuals(silence_report,audio_paths,search_radius=1.0,sr=22050):
    """Measure accepted silence ends using post-silence attacks in supporting stems.

    The silence detector establishes that a boundary is structurally useful. This
    pass gives the boundary its own residual rather than borrowing a periodic
    checkpoint. Drum-only landmarks require one measured source; multi-stem
    landmarks require two. Cross-source spread above 50 ms is rejected.
    """
    measured=[]
    for cp in (silence_report or {}).get("checkpoints",[]):
        item=dict(cp)
        item["boundary_measurement"]={"available":False}
        if not cp.get("accepted"):
            measured.append(item); continue
        predicted=float(cp["predicted_audio_end"])
        observations=[]
        for source in cp.get("audio_confirmed_stems",[]):
            onsets=_detected_onset_times(audio_paths.get(source),sr=sr)
            lo=np.searchsorted(onsets,predicted-search_radius)
            hi=np.searchsorted(onsets,predicted+search_radius,side="right")
            if hi<=lo: continue
            local=onsets[lo:hi]
            hit=float(local[np.argmin(np.abs(local-predicted))])
            observations.append({"source":source,"time":round(hit,6),
                                 "residual_ms":round((hit-predicted)*1000.0,3)})
        required=1 if cp.get("rule")=="drum_only_over_1_second" else 2
        residuals=[x["residual_ms"] for x in observations]
        spread=(max(residuals)-min(residuals)) if len(residuals)>=2 else 0.0
        accepted=bool(len(observations)>=required and spread<=50.0)
        median=float(np.median(residuals)) if observations else None
        item["boundary_measurement"]={
            "available":bool(observations),"required_sources":required,
            "observations":observations,"cross_source_spread_ms":round(spread,3),
            "measured_residual_ms":None if median is None else round(median,3),
            "accepted":accepted,
            "warnings":([] if accepted else
                (["rejected_insufficient_boundary_sources"] if len(observations)<required else [])+
                (["rejected_boundary_source_disagreement"] if spread>50.0 else []))}
        measured.append(item)
    out=dict(silence_report or {})
    out["checkpoints"]=measured
    out["measured_boundaries"]=sum(1 for x in measured if x.get("boundary_measurement",{}).get("accepted"))
    return out

def _checkpoint_region_candidates(onset_report,silence_report,chroma_report):
    candidates=[]
    for cp in (onset_report or {}).get("checkpoints",[]):
        if not cp.get("accepted") or cp.get("median_correction_seconds") is None: continue
        if int(cp.get("matched_events",0))<5 or float(cp.get("mad_seconds") or 9)>0.020: continue
        candidates.append({"measure":int(cp["measure"]),"nominal_time":float(cp["nominal_time"]),
            "kind":"onset","residual_ms":1000.0*float(cp["median_correction_seconds"]),
            "quality":min(1.0,0.55+0.025*int(cp.get("matched_events",0)))*
                      max(0.25,1.0-float(cp.get("mad_seconds") or 0)/0.020),
            "detail":{"matched_events":int(cp.get("matched_events",0)),
                      "mad_ms":round(1000.0*float(cp.get("mad_seconds") or 0),3)}})
    for cp in (chroma_report or {}).get("checkpoints",[]):
        if not cp.get("accepted") or cp.get("combined_residual_ms") is None: continue
        candidates.append({"measure":int(cp["measure"]),"nominal_time":float(cp["nominal_time"]),
            "kind":"chroma","residual_ms":float(cp["combined_residual_ms"]),
            "quality":0.90,"detail":{"sources":cp.get("strong_isolated_sources",[]),
                "onset_disagreement_ms":cp.get("chroma_onset_disagreement_ms")}})
    for cp in (silence_report or {}).get("checkpoints",[]):
        bm=cp.get("boundary_measurement") or {}
        if not cp.get("accepted") or not bm.get("accepted") or bm.get("measured_residual_ms") is None: continue
        candidates.append({"measure":int(cp["measure"]),"nominal_time":float(cp["score_end"]),
            "kind":"silence","residual_ms":float(bm["measured_residual_ms"]),
            "quality":0.95 if len(bm.get("observations",[]))>=2 else 0.82,
            "detail":{"supporting_stems":cp.get("supporting_stems",[]),
                "measured_sources":[x["source"] for x in bm.get("observations",[])],
                "spread_ms":bm.get("cross_source_spread_ms")}})
    return candidates

def build_checkpoint_linear_warp(base_warp,onset_report,silence_report,chroma_report,
                                 measure_downbeats,minimum_actionable_ms=10.0,
                                 shrinkage=0.50,maximum_applied_ms=50.0,
                                 minimum_interior_anchors=3):
    """Build a conservative residual curve over the validated linear map."""
    candidates=_checkpoint_region_candidates(onset_report,silence_report,chroma_report)
    # Consolidate observations within 500 ms in nominal score time.
    groups=[]
    for c in sorted(candidates,key=lambda x:x["nominal_time"]):
        if groups and c["nominal_time"]-groups[-1][-1]["nominal_time"]<=0.5:
            groups[-1].append(c)
        else: groups.append([c])
    selected=[]; rejected=[]
    first=float(measure_downbeats[0]["t_gp"]); last=float(measure_downbeats[-1]["t_gp"])
    for group in groups:
        kinds=sorted({x["kind"] for x in group})
        residuals=[x["residual_ms"] for x in group]
        spread=max(residuals)-min(residuals) if len(residuals)>1 else 0.0
        weighted=sum(x["residual_ms"]*x["quality"] for x in group)/sum(x["quality"] for x in group)
        onset=next((x for x in group if x["kind"]=="onset"),None)
        strong_onset=bool(onset and onset["detail"]["matched_events"]>=10 and onset["detail"]["mad_ms"]<=10.0)
        tier="A" if len(kinds)>=2 and spread<=50.0 else "B" if ("silence" in kinds or strong_onset) else "C"
        reasons=[]
        if spread>50.0: reasons.append("rejected_evidence_disagreement")
        if abs(weighted)<minimum_actionable_ms: reasons.append("rejected_negligible_residual")
        nominal=float(np.median([x["nominal_time"] for x in group]))
        if nominal<=first+1e-6 or nominal>=last-1e-6: reasons.append("rejected_exterior_candidate")
        if tier=="C": reasons.append("rejected_diagnostic_only_tier")
        record={"measure":int(round(np.median([x["measure"] for x in group]))),
            "nominal_time":round(nominal,6),"evidence":kinds,"tier":tier,
            "observations":group,"evidence_spread_ms":round(spread,3),
            "measured_residual_ms":round(weighted,3)}
        if reasons:
            record.update({"accepted":False,"warnings":reasons}); rejected.append(record); continue
        applied=float(np.clip(weighted*shrinkage,-maximum_applied_ms,maximum_applied_ms))
        record.update({"accepted":True,"applied_residual_ms":round(applied,3),"warnings":[]})
        selected.append(record)
    status="applied"
    if len(selected)<minimum_interior_anchors:
        status="linear_fallback_insufficient_anchors"
        selected=[]
    # Validate residual stretch, dropping the lower-confidence anchor in a bad interval.
    dropped=[]
    def points():
        return [{"nominal_time":first,"applied_residual_ms":0.0,"tier":"fixed"}]+selected+[
               {"nominal_time":last,"applied_residual_ms":0.0,"tier":"fixed"}]
    changed=True
    while status=="applied" and changed:
        changed=False; pts=points()
        for a,b in zip(pts,pts[1:]):
            base_delta=float(base_warp(b["nominal_time"]))-float(base_warp(a["nominal_time"]))
            corrected_delta=base_delta+(b["applied_residual_ms"]-a["applied_residual_ms"])/1000.0
            stretch=corrected_delta/base_delta if base_delta>0 else 0.0
            if not (0.98<=stretch<=1.02):
                options=[x for x in (a,b) if x.get("tier")!="fixed"]
                if not options:
                    status="linear_fallback_stretch_validation"; selected=[]; changed=False; break
                victim=min(options,key=lambda x:(0 if x["tier"]=="B" else 1,abs(x["applied_residual_ms"])))
                selected.remove(victim); dropped.append({**victim,"reason":"dropped_stretch_violation"})
                changed=True; break
        if len(selected)<minimum_interior_anchors and status=="applied":
            status="linear_fallback_insufficient_anchors_after_validation"; selected=[]; changed=False
    if status=="applied":
        pts=points(); xs=np.asarray([x["nominal_time"] for x in pts],dtype=float)
        residual=np.asarray([x["applied_residual_ms"]/1000.0 for x in pts],dtype=float)
        def warp(t): return float(base_warp(float(t))+np.interp(float(t),xs,residual))
    else:
        pts=points()
        def warp(t): return float(base_warp(float(t)))
    # Final validation metrics sampled at every measure boundary.
    samples=np.asarray([float(x["t_gp"]) for x in measure_downbeats],dtype=float)
    diffs=np.asarray([(warp(t)-float(base_warp(t)))*1000.0 for t in samples])
    stretches=[]
    for a,b in zip(samples,samples[1:]):
        bd=float(base_warp(b))-float(base_warp(a)); cd=warp(b)-warp(a)
        if bd>0: stretches.append(cd/bd)
    report={"version":1,"status":status,
        "config":{"minimum_actionable_residual_ms":minimum_actionable_ms,"shrinkage":shrinkage,
            "maximum_applied_correction_ms":maximum_applied_ms,"minimum_residual_stretch":0.98,
            "maximum_residual_stretch":1.02,"minimum_interior_anchors":minimum_interior_anchors},
        "selection":{"raw_candidates":len(candidates),"consolidated_regions":len(groups),
            "actionable_anchors":len(selected),"rejected_regions":len(rejected),"dropped_anchors":len(dropped)},
        "anchors":selected,"rejected":rejected,"dropped":dropped,
        "validation":{"monotonic":bool(all(warp(b)>warp(a) for a,b in zip(samples,samples[1:]))),
            "minimum_observed_residual_stretch":round(min(stretches or [1.0]),6),
            "maximum_observed_residual_stretch":round(max(stretches or [1.0]),6),
            "maximum_timing_difference_ms":round(float(np.max(np.abs(diffs))) if len(diffs) else 0.0,3),
            "median_timing_difference_ms":round(float(np.median(np.abs(diffs))) if len(diffs) else 0.0,3),
            "timing_changes_applied":bool(status=="applied")}}
    return warp,report

def apply_warp(entries, warp_fn, time_key="t_gp", out_key="t"):
    out = []
    for e in entries:
        e2 = dict(e)
        e2[out_key] = round(warp_fn(e2.pop(time_key)), 4)
        out.append(e2)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def get_initial_tempo(gp_path):
    """
    Lightweight helper for build_feedpak.py to call directly (as a module
    import, not a subprocess) when it needs the song's starting BPM to
    size a count-in pad, before deciding whether to run the full pipeline.
    """
    if guitarpro is None:
        raise RuntimeError("pyguitarpro is required: pip install pyguitarpro")
    song = guitarpro.parse(gp_path)
    return float(song.tempo)


def process(gp_path, drums_audio_path=None, bass_audio_path=None, piano_audio_path=None,
            guitar_audio_path=None, full_audio_path=None,
            sr=22050, count_in_offset=0.0, timeline_mode="both",
            allow_severe_alignment=False, anchors_path=None, padding_added=0.0,
            auto_chunk_measures=16, alignment_mode="dtw",
            checkpoint_measures=4, checkpoint_search_radius=1.0, project_config_path=None,
            gp_parser="gp5", score_json_path=None, alphatab_extractor=None, node_executable="node"):
    if gp_parser not in ("gp5", "alphatab"):
        raise ValueError("gp_parser must be 'gp5' or 'alphatab'")
    if gp_parser == "alphatab":
        fc.log(f"Processing modern GP through alphaTab: {gp_path}")
        json_path = score_json_path or ats.ensure_json(gp_path, extractor=alphatab_extractor, node=node_executable)
        score_data=ats.load(json_path); ap=ats.products(score_data, project_config_path=project_config_path)
        tempo_events=ap["tempo_events"]; role_by_name=ap["role_by_name"]; fretted_tracks=ap["fretted_tracks"]
        drum_hits_gp=ap["drum_hits_gp"]; bass_alignment_events=ap["bass_alignment_events"]; piano_alignment_events=ap["piano_alignment_events"]
        guitar_alignment_events=ap["guitar_alignment_events"]; bass_pitch_events=ap["bass_pitch_events"]; guitar_pitch_events=ap["guitar_pitch_events"]
        piano_pitch_events=ap["piano_pitch_events"]; notation_tracks=ap["notation_tracks"]; gp_vocals=ap["gp_vocals"]
        key_events_gp=ap["key_events_gp"]; song_timeline_gp=ap["song_timeline_gp"]
        audio_content_start=float(count_in_offset); score_content_time=ap["score_content_time"]
        chart_offset=choose_chart_offset(audio_content_start,score_content_time)
        fc.log(f"alphaTab: {ap['track_count']} track(s), {ap['measure_count']} playback measure(s), {len(tempo_events)} tempo segment(s)",indent=1)
    else:
        if guitarpro is None:
            raise RuntimeError("pyguitarpro is required for --gp-parser gp5: pip install pyguitarpro")
        fc.log(f"Processing GP file: {gp_path}")
        audio_content_start = float(count_in_offset)
        if audio_content_start:
            fc.log(f"Post-padding audio content start: {audio_content_start:.3f}s "
                   f"(chart offset will be derived after GP parsing)")
        fc.log_step(1, 5, "Parsing .gp5 structure")
        with fc.timed_step("guitarpro.parse()", indent=1):
            song = guitarpro.parse(gp_path)
        first_tick = song.measureHeaders[0].start if song.measureHeaders else 0
        tempo_events = rebase_tempo_events(build_tempo_events(song), first_tick)
        fc.log(f"{len(song.tracks)} track(s), {len(song.measureHeaders)} measure(s), "
               f"{len(tempo_events)} tempo segment(s) (starting {tempo_events[0][1]:.0f} BPM)",
               indent=1)

        fc.log_step(2, 5, "Routing tracks (fretted / drum / keyboard)")
        fretted_tracks = {}
        drum_hits_gp = []
        bass_alignment_events = []
        piano_alignment_events = []
        guitar_alignment_events = []
        bass_pitch_events = []
        guitar_pitch_events = []
        piano_pitch_events = []
        notation_tracks = {}
        gp_vocals = None
        inventory=pc.inventory_song(song)
        if project_config_path:
            with open(project_config_path,"r",encoding="utf-8") as f: project_data=json.load(f)
            validation=pc.validate_song_config(project_data,inventory)
            if not validation["valid"]: raise ValueError("Invalid project track configuration: "+json.dumps(validation["errors"],ensure_ascii=False))
            role_by_name=pc.resolved_roles(project_data,inventory)
        else: role_by_name={x["name"]:x["automatic_role"] for x in inventory}

        score_content_time=first_configured_instrument_time(
            song,role_by_name,tempo_events,first_tick)
        chart_offset=choose_chart_offset(audio_content_start,score_content_time)
        fc.log(f"Timing origin: audio content {audio_content_start:.3f}s; "
               f"first configured instrumental score onset {score_content_time:.3f}s; "
               f"chart offset {chart_offset:+.3f}s",indent=1)

        for track in song.tracks:
            if not track.measures: continue
            role=role_by_name.get(track.name,"unsupported")
            if role == "drums":
                fc.log(f"'{track.name}' -> main drums", indent=1); drum_hits_gp=parse_drum_track(track,tempo_events,first_tick); fc.log(f"{len(drum_hits_gp)} drum hits parsed",indent=2)
            elif role in ("piano_left","piano_right","piano_combined"):
                fc.log(f"'{track.name}' -> {role}",indent=1)
                hand={"piano_left":"left","piano_right":"right","piano_combined":"combined"}[role]
                notation=parse_keyboard_track(track,tempo_events,first_tick,hand=hand); notation_tracks[_safe_track_id(track)]=notation
                piano_alignment_events.extend({"t_gp":t} for t in _keyboard_onsets(notation))
                for measure in notation.get("measures",[]):
                    for stave in measure.get("staves",{}).values():
                        for voice in stave.get("voices",[]):
                            for beat in voice.get("beats",[]):
                                for note in beat.get("notes",[]): piano_pitch_events.append({"t_gp":beat["t_gp"],"midi":int(note["midi"])})
            elif role == "lead_vocal":
                capability=pc.classify_vocal_capability(song,inventory,role_by_name)
                if capability.get("direct_gp_lyrics_supported"):
                    fc.log(f"'{track.name}' -> GP lead vocal ({capability['classification']})",indent=1)
                    gp_vocals=parse_gp_vocals(song,track,tempo_events,first_tick)
                else:
                    fc.log(f"'{track.name}' -> GP vocal lyrics rejected: {capability['classification']}",indent=1)
                    fc.log(capability["reason"],indent=2)
            elif role in ("guitar","bass"):
                fc.log(f"'{track.name}' -> {role} fretted track",indent=1); data=parse_fretted_track(track,tempo_events,first_tick); fretted_tracks[_safe_track_id(track)]=data
                if role=="bass":
                    bass_alignment_events.extend({"t_gp":n["t_gp"]} for n in data["notes"]); bass_pitch_events.extend({"t_gp":n["t_gp"],"midi":int(data["absolute_tuning_midi"][n["s"]]+n["f"])} for n in data["notes"])
                else:
                    guitar_alignment_events.extend({"t_gp":n["t_gp"]} for n in data["notes"]); guitar_pitch_events.extend({"t_gp":n["t_gp"],"midi":int(data["absolute_tuning_midi"][n["s"]]+n["f"])} for n in data["notes"])
                fc.log(f"{len(data['notes'])} notes, {len(data['anchors'])} anchors",indent=2)
            else: fc.log(f"'{track.name}' -> {role}; skipped",indent=1)

        fc.log_step(3, 5, "Reading key signatures + building song timeline")
        key_events_gp = build_key_signature_events(song, tempo_events, first_tick)
        song_timeline_gp = build_song_timeline(song, tempo_events, first_tick)
        fc.log(f"{len(key_events_gp)} key change event(s), "
               f"{len(song_timeline_gp['beats'])} measure boundaries", indent=1)

    # Apply the count-in offset to every nominal (GP-clock) time before
    # DTW, so the synthetic click track (built from these same nominal
    # drum-hit times) lines up with the now-equivalently-padded real
    # audio — DTW doesn't need special-casing for this, it just sees a
    # consistently shifted reference.
    for data in fretted_tracks.values():
        data["notes"] = shift_nominal_times(data["notes"], chart_offset)
        data["anchors"] = shift_nominal_times(data["anchors"], chart_offset, time_key="time")
    drum_hits_gp = shift_nominal_times(drum_hits_gp, chart_offset)
    bass_alignment_events = shift_nominal_times(bass_alignment_events, chart_offset)
    piano_alignment_events = shift_nominal_times(piano_alignment_events, chart_offset)
    guitar_alignment_events = shift_nominal_times(guitar_alignment_events, chart_offset)
    bass_pitch_events = shift_nominal_times(bass_pitch_events, chart_offset)
    guitar_pitch_events = shift_nominal_times(guitar_pitch_events, chart_offset)
    piano_pitch_events = shift_nominal_times(piano_pitch_events, chart_offset)
    key_events_gp = shift_nominal_times(key_events_gp, chart_offset)
    song_timeline_gp["time_signatures"] = shift_nominal_times(
        song_timeline_gp["time_signatures"], chart_offset)
    song_timeline_gp["beats"] = shift_nominal_times(song_timeline_gp["beats"], chart_offset)

    manual_anchors, anchor_reference = load_alignment_anchors(
        anchors_path, song_timeline_gp["beats"], padding_added=padding_added)
    if manual_anchors:
        fc.log(f"Loaded {len(manual_anchors)} manual anchor(s); input clock={anchor_reference}, padding conversion=+{padding_added:.3f}s", indent=1)

    fc.log_step(4, 5, "DTW-aligning nominal GP timing to real audio")
    specs=[("drums",_event_times(drum_hits_gp),drums_audio_path),
           ("bass",_event_times(bass_alignment_events),bass_audio_path),
           ("piano",_event_times(piano_alignment_events),piano_audio_path)]
    if alignment_mode == "dtw":
        candidates=[]
        for source,event_times,audio_path in specs:
            if event_times and audio_path:
                fc.log(f"Building {source} alignment candidate from {len(event_times)} onsets",indent=1)
                candidates.append(compute_warp_candidate(event_times,audio_path,source,sr=sr))
        sample_times=[b["t_gp"] for b in song_timeline_gp["beats"]]
        warp_fn=choose_probabilistic_warp(candidates,sample_times)
        decision=warp_fn.alignment_diagnostics
        warp_fn,chunk_points=apply_hard_anchors(warp_fn,manual_anchors,song_timeline_gp["beats"],
                                                max_chunk_measures=auto_chunk_measures)
        decision["manual_anchors"]=manual_anchors
        decision["chunk_points"]=chunk_points
        decision["anchor_time_reference"]=anchor_reference
        decision["padding_added"]=padding_added
    else:
        reference=next(((source,times,path) for source,times,path in specs if times and path),None)
        source,event_times,audio_path=reference if reference else ("none",[],None)
        checkpoint_mode=alignment_mode in ("dtw-checkpoint","checkpoint-linear","checkpoint-dtw-diagnostic","checkpoint-dtw-selective")
        simple_mode="linear" if checkpoint_mode else alignment_mode
        if checkpoint_mode:
            # KISS baseline: times were already shifted by the known count-in.
            # Preserve the native GP tempo map and never derive a global scale
            # from the full mix's detected content end. Local/checkpoint layers
            # may still add small, confidence-gated corrections below.
            # Checkpoint production baseline: estimate only clock scale from
            # broad instrument spans, then force the independently established
            # score/audio start. The simple `linear` mode branch below/above is
            # intentionally untouched and remains the comparison/fallback mode.
            score_start=float(song_timeline_gp["beats"][0]["t_gp"])
            score_end=estimate_last_measure_end(song_timeline_gp["beats"])
            scale_candidates=build_instrument_scale_candidates(
                specs,score_start,score_end,sr=sr,minimum_events=12,
                minimum_span_seconds=45.0,minimum_coverage=0.55,
                minimum_scale=.95,maximum_scale=1.05)
            scale,scale_report=choose_instrument_scale(scale_candidates,agreement_tolerance=.005)
            # Offset authority is independent from scale authority. Include
            # guitar for the opening only, because guitar may be the earliest
            # reliable source even though it is intentionally excluded from
            # broad global-scale estimation.
            opening_specs=specs+[("guitar",_event_times(guitar_alignment_events),guitar_audio_path)]
            opening_candidates=build_opening_anchor_candidates(
                opening_specs,score_start,audio_content_start,sr=sr,
                maximum_score_delay=15.0,maximum_anchor_shift=4.0,minimum_events=4)
            score_anchor,audio_anchor,opening_report=choose_opening_anchor(
                opening_candidates,score_start,audio_content_start,
                simultaneous_score_tolerance=.050,audio_agreement_tolerance=.250)
            warp_fn=reanchor_scale_to_song_start(scale,score_anchor,audio_anchor)
            source=scale_report.get("selected_source") or "gp_tempo_map"
            event_times=[]; audio_path=None
            span_report={"status":"diagnostic_only","role":"full_mix_corroboration_deferred"}
            warp_fn.alignment_diagnostics={
                "mode":"instrument-scale-reanchored-v1" if scale_report["status"]!="fallback_no_accepted_candidate" else "gp-tempo-map-baseline-v3",
                "method":"instrument_first_last_scale_plus_independent_song_start",
                "selected_source":source,
                "offset_source":"early_instrument_onset",
                "opening_anchor":opening_report,
                "scale_source":source,
                "scale":round(float(scale),9),
                "global_end_scaling_applied":bool(abs(float(scale)-1.0)>1e-9),
                "instrument_scale":scale_report,
                "full_mix_role":"diagnostic_only",
                "global_span":span_report,
            }
            fc.log(f"Checkpoint global scale: {scale:.6f} from {source}",indent=1)
            fc.log(f"Opening anchor: score {score_anchor:.3f}s -> audio {audio_anchor:.3f}s from {opening_report['selected_sources']}",indent=1)
            for item in opening_candidates:
                fc.log(f"Opening candidate {item['source']}: score_first={item.get('symbolic_first')}, "
                       f"audio_first={item.get('measured_audio_first')}, accepted={item['accepted']}, "
                       f"warnings={item['warnings']}",indent=2)
            for item in scale_candidates:
                fc.log(f"Scale candidate {item['source']}: coverage={item['score_span_coverage']:.3f}, "
                       f"events={item['event_count']}, scale={item.get('measured_scale')}, "
                       f"accepted={item['accepted']}, warnings={item['warnings']}",indent=2)
        else:
            fc.log(f"Using simple alignment mode '{simple_mode}' with reference '{source}'",indent=1)
            warp_fn=compute_simple_warp(event_times,audio_path,simple_mode,sr=sr)
        decision=warp_fn.alignment_diagnostics
        if not checkpoint_mode:
            decision["selected_source"]=source if source != "none" else None
        decision["manual_anchors_ignored"]=bool(manual_anchors)
        decision["padding_added"]=padding_added
        decision["timing_origin"]={"audio_content_start":round(audio_content_start,6),
            "first_configured_instrument_score_onset":round(score_content_time,6),
            "chart_offset":round(chart_offset,6),"tempo_events_rebased_to_first_tick":True}
        if alignment_mode in ("dtw-checkpoint","checkpoint-linear","checkpoint-dtw-diagnostic","checkpoint-dtw-selective"):
            diagnostic_reference=next(((name,times,path) for name,times,path in specs if times and path), ("none",[],None))
            diagnostic_source,diagnostic_times,diagnostic_path=diagnostic_reference
            fc.log(f"Detecting diagnostic {diagnostic_source} checkpoints every {checkpoint_measures} measures",indent=1)
            checkpoint_report=diagnose_checkpoints(diagnostic_times,diagnostic_path,warp_fn,
                song_timeline_gp["beats"],checkpoint_measures,checkpoint_search_radius,sr)
            decision["mode"]="dtw-checkpoint-diagnostic-v3"
            decision["checkpoint_diagnostics"]=checkpoint_report
            score_events={"drums":_event_times(drum_hits_gp),"bass":_event_times(bass_alignment_events),
                          "guitar":_event_times(guitar_alignment_events),"piano":_event_times(piano_alignment_events)}
            audio_paths={"drums":drums_audio_path,"bass":bass_audio_path,"guitar":guitar_audio_path,
                         "piano":piano_audio_path,"full":full_audio_path}
            silence_report=diagnose_silence_checkpoints(score_events,audio_paths,warp_fn,
                song_timeline_gp["beats"],float(tempo_events[0][1]),checkpoint_search_radius,sr)
            decision["silence_checkpoint_diagnostics"]=silence_report
            pitched_events={"bass":bass_pitch_events,"guitar":guitar_pitch_events,"piano":piano_pitch_events}
            chroma_report=diagnose_chroma_checkpoints(pitched_events,audio_paths,warp_fn,
                song_timeline_gp["beats"],checkpoint_report,checkpoint_measures,
                checkpoint_search_radius,sr)
            decision["chroma_checkpoint_diagnostics"]=chroma_report
            decision["mode"]="dtw-checkpoint-diagnostic-v4"
            if alignment_mode in ("checkpoint-linear","checkpoint-dtw-diagnostic","checkpoint-dtw-selective"):
                silence_report=measure_silence_boundary_residuals(
                    silence_report,audio_paths,checkpoint_search_radius,sr)
                decision["silence_checkpoint_diagnostics"]=silence_report
                adaptive=adaptive_checkpoint_diagnostic(
                    pitched_events,audio_paths,warp_fn,song_timeline_gp["beats"],checkpoint_report,
                    silence_report,checkpoint_measures,checkpoint_search_radius,sr)
                decision["adaptive_checkpoint_search"]=adaptive
                if adaptive["expanded"]:
                    fc.log(f"No corroborated checkpoints at initial radius; whole-song search expanded through {adaptive['selected_radius_seconds']:.1f}s (diagnostic only)",indent=2)
                if adaptive["trustworthy"] > 0:
                    chroma_report=adaptive["report"]
                    decision["chroma_checkpoint_diagnostics"]=chroma_report
                corrected_warp,correction_report=build_checkpoint_linear_warp(
                    warp_fn,checkpoint_report,silence_report,chroma_report,
                    song_timeline_gp["beats"])
                corrected_warp.alignment_diagnostics=decision
                warp_fn=corrected_warp
                decision["mode"]="checkpoint-linear-v1"
                decision["checkpoint_linear"]=correction_report
                if alignment_mode in ("checkpoint-dtw-diagnostic","checkpoint-dtw-selective"):
                    local_report=cdtw.diagnose_segment_source_dtw(
                        warp_fn,song_timeline_gp["beats"],score_events,audio_paths,
                        segment_measures=checkpoint_measures,sr=sr)
                    decision["local_dtw"]=local_report
                    if alignment_mode=="checkpoint-dtw-selective":
                        production_warp,production_report=cdtw.build_selective_warp(
                            warp_fn,local_report,song_timeline_gp["beats"])
                        production_warp.alignment_diagnostics=decision
                        warp_fn=production_warp
                        decision["mode"]="checkpoint-dtw-selective-v1"
                        decision["timing_output"]=("checkpoint-dtw-selective-v1" if production_report["timing_changes_applied"] else "checkpoint-linear-v1")
                        decision["local_dtw_production"]=production_report
                    else:
                        decision["mode"]="checkpoint-dtw-diagnostic-v1.2.1"
                        decision["timing_output"]="checkpoint-linear-v1"
                    q=local_report["summary"]
                    fc.log(f"Local DTW diagnostic: {q['generated_segments']} generated, {q['evaluated_segments']} evaluated",indent=2)
                    fc.log(f"Local DTW diagnostic: {q['boundary_assignments']} boundary assignments, {q['boundary_activated_evaluated']} evaluated, {q['boundary_activated_skipped']} skipped",indent=2)
                    fc.log(f"Local DTW diagnostic: {q['would_apply']} would apply, {q['rejected']} rejected",indent=2)
                    fc.log(f"Local DTW diagnostic: {q['skipped_baseline_accurate']} baseline-accurate, "
                           f"{q['skipped_insufficient_features']} insufficient features",indent=2)
                    if alignment_mode=="checkpoint-dtw-selective":
                        pr=decision["local_dtw_production"]
                        fc.log(f"Selective local DTW: {pr['applied_segment_count']} segment(s) applied; status={pr['status']}",indent=2)
                        fc.log(f"Timing output: {decision['timing_output']}",indent=2)
                    else:
                        fc.log("Timing output: checkpoint-linear; local DTW not applied",indent=2)
            fc.log(f"Chroma checkpoints: {chroma_report['summary']['strong']} strong, "
                   f"{chroma_report['summary']['rejected']} rejected from "
                   f"{chroma_report['summary']['generated']} periodic candidate(s)",indent=2)
            fc.log(f"Silence checkpoints: {silence_report.get('strong',0)} strong, "
                   f"{silence_report.get('rejected',0)} rejected from {silence_report.get('generated',0)} consolidated candidate(s)",indent=2)
            fc.log(f"Checkpoints: {checkpoint_report.get('strong',0)} strong, "
                   f"{checkpoint_report.get('rejected',0)} rejected; "
                   f"timing {'selective local DTW production' if alignment_mode=='checkpoint-dtw-selective' else ('diagnostic local DTW; packaged checkpoint-linear' if alignment_mode=='checkpoint-dtw-diagnostic' else ('guardedly corrected' if alignment_mode=='checkpoint-linear' else 'unchanged'))}",indent=2)
    guitar_coverage=diagnose_track_audio_coverage(
        _event_times(guitar_alignment_events),guitar_audio_path,warp_fn,sr=sr,source="guitar")
    decision["guitar_coverage_diagnostic"]=guitar_coverage
    gc=guitar_coverage
    if gc.get("gp",{}).get("first_note_time") is not None:
        fc.log(f"Guitar coverage: GP begins {gc['gp']['first_note_time']:.3f}s; "
               f"audio sustained activity begins {gc.get('audio',{}).get('first_sustained_activity')}",indent=1)
        if gc.get("comparison",{}).get("classification")=="stem_contains_substantial_unmapped_or_cross_instrument_audio":
            fc.log(f"Guitar coverage inconclusive: mapped stem activity leads configured GP track by "
                   f"{gc['comparison']['audio_leads_gp_seconds']:.3f}s",indent=2)
    fc.log(f"Alignment decision: {decision.get('mode')} {decision.get('selected_source') or ''}",indent=1)
    for c in decision.get("candidates",[]):
        fc.log(f"{c['source']}: p={c.get('probability',0):.3f}, median={c.get('median_residual_ms',0):.1f}ms, coverage={c.get('coverage',0):.2f}",indent=2)
    for pair in decision.get("agreement",[]):
        fc.log(f"Agreement {pair['sources']}: median={pair['median_ms']:.1f}ms, p95={pair['p95_ms']:.1f}ms",indent=2)

    drum_hits = apply_warp(drum_hits_gp, warp_fn)
    key_events = apply_warp(key_events_gp, warp_fn)

    warped_beats = [
        {"time": round(warp_fn(e["t_gp"]), 4), "measure": e["measure"], "ts_num": e["ts_num"]}
        for e in song_timeline_gp["beats"]
    ]
    warped_beats,sanitization=sanitize_warped_downbeats(song_timeline_gp["beats"],warped_beats)
    dense_tempos=compute_dense_measure_tempos(warped_beats,fallback_bpm=float(tempo_events[0][1]))
    explicit_beats=build_explicit_feedpak_beats(warped_beats)
    alignment_report=analyze_alignment_quality(song_timeline_gp["beats"],warped_beats,warp_fn,sanitization)
    fc.log(f"ALIGNMENT QUALITY: {alignment_report['severity'].upper()}",indent=1)
    for repair in sanitization["repairs"]:
        fc.log(f"REPAIRED measures {repair['from_measure']}..{repair['to_measure']} by bounded interpolation",indent=2)
    for severe in sanitization["unresolved_severe"]:
        fc.log(f"SEVERE measure {severe['measure']}: duration={severe['duration']:.4f}s",indent=2)
    if alignment_report["severity"]=="severe" and not allow_severe_alignment:
        raise RuntimeError("Severe alignment remains after repair; refusing to package. Review the report or pass --allow-severe-alignment.")
    song_timeline={
        "tempos":dense_tempos,
        "time_signatures":[{"time":round(warp_fn(e["t_gp"]),4),"ts":e["ts"]}
                           for e in song_timeline_gp["time_signatures"]],
    }
    if timeline_mode in ("both","beats"):
        song_timeline["beats"]=explicit_beats
    if timeline_mode=="beats":
        song_timeline.pop("tempos",None)

    fc.log_step(5, 5, "Applying warp to all tracks")
    arrangements_out = {}
    for track_id, data in fretted_tracks.items():
        warped_notes = apply_warp(data["notes"], warp_fn)
        warped_anchors = [
            {"time": round(warp_fn(a["time"]), 4), "fret": a["fret"], "width": a["width"]}
            for a in data["anchors"]
        ]
        tuning_offsets, nonstandard = fc.tuning_offsets_from_absolute(
            data["absolute_tuning_midi"], is_bass="bass" in data["name"].lower()
        )
        if nonstandard:
            fc.log(f"'{data['name']}': nonstandard string count, tuning offsets "
                   f"are best-effort", indent=1)
        arrangements_out[track_id] = {
            "name": data["name"],
            "tuning": tuning_offsets,
            "tuning_nonstandard_string_count": nonstandard,
            "capo": max(0, data["capo"]),
            "notes": warped_notes,
            "chords": [],
            "anchors": warped_anchors,
            "handshapes": [],
            "templates": [],
        }
        fc.log(f"'{data['name']}' warped ({len(warped_notes)} notes)", indent=1)

    warped_notation = {}
    for track_id, notation in notation_tracks.items():
        warped_notation[track_id] = shift_and_warp_notation(notation, chart_offset, warp_fn)
        arrangements_out[track_id] = playable_keyboard_from_notation(warped_notation[track_id])
        fc.log(f"'{track_id}' notation and playable piano lane warped ({len(arrangements_out[track_id]['notes'])} notes)", indent=1)
    result = {
        "arrangements": arrangements_out,
        "drum_tab": {"version": 1, "hits": drum_hits} if drum_hits else None,
        "notation": warped_notation if warped_notation else None,
        "keys": {"version": 1, "events": [{"t": e["t"], "key": e["key"]} for e in key_events]}
                 if key_events else None,
        "song_timeline": song_timeline,
        "alignment_report": alignment_report,
        "gp_vocals": warp_gp_vocals(gp_vocals,chart_offset,warp_fn) if gp_vocals else None,
    }
    fc.log(f"Done: {len(arrangements_out)} fretted arrangement(s), "
           f"{len(drum_hits)} drum hits, {len(warped_notation)} notation track(s)")
    return result


def _looks_like_keyboard(track):
    name = (track.name or "").upper()
    return "KEY" in name or "PIANO" in name


def _safe_track_id(track):
    name = (track.name or f"track{track.number}").strip().lower()
    return "".join(c if c.isalnum() else "_" for c in name).strip("_") or f"track{track.number}"


def main():
    parser = argparse.ArgumentParser(description="Parse a GP score and align it to real audio via DTW.")
    parser.add_argument("gp_file", help="Path to .gp5, modern .gp, or schema-v4 JSON")
    parser.add_argument("--gp-parser", choices=("gp5","alphatab"), default="gp5")
    parser.add_argument("--score-json", default=None)
    parser.add_argument("--alphatab-extractor", default=None)
    parser.add_argument("--node-executable", default="node")
    parser.add_argument("drums_audio", nargs="?", default=None, help="Optional isolated drums stem")
    parser.add_argument("--bass-audio", default=None)
    parser.add_argument("--piano-audio", default=None)
    parser.add_argument("--guitar-audio", default=None)
    parser.add_argument("--full-audio", default=None)
    parser.add_argument("--alignment-mode",choices=("nominal","offset","linear","dtw","dtw-checkpoint","checkpoint-linear","checkpoint-dtw-diagnostic","checkpoint-dtw-selective"),default="dtw")
    parser.add_argument("--timeline-mode", choices=("tempos","beats","both"), default="both",
                        help="A/B test output: dense tempos only, explicit beats only, or both")
    parser.add_argument("--allow-severe-alignment", action="store_true",
                        help="Package even if severe structural timing errors remain after repair")
    parser.add_argument("--anchors", default=None)
    parser.add_argument("--padding-added", type=float, default=0.0)
    parser.add_argument("--auto-chunk-measures", type=int, default=16)
    parser.add_argument("--checkpoint-measures",type=int,default=4)
    parser.add_argument("--checkpoint-search-radius",type=float,default=1.0)
    parser.add_argument("--project-config",default=None)
    parser.add_argument("--out", default="intermediate_arrangements.json")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--count-in-offset", type=float, default=0.0,
                         help="Seconds to add to every nominal GP-clock time before DTW "
                              "(matches a count-in silence pad prepended to the audio stems "
                              "by build_feedpak.py; 0.0 if the audio already had none).")
    args = parser.parse_args()

    result=process(args.gp_file,args.drums_audio,bass_audio_path=args.bass_audio,
                   piano_audio_path=args.piano_audio,guitar_audio_path=args.guitar_audio,
                   full_audio_path=args.full_audio,sr=args.sr,
                   count_in_offset=args.count_in_offset,timeline_mode=args.timeline_mode,
                   allow_severe_alignment=args.allow_severe_alignment,
                   anchors_path=args.anchors, padding_added=args.padding_added,
                   auto_chunk_measures=args.auto_chunk_measures,alignment_mode=args.alignment_mode,
                   checkpoint_measures=args.checkpoint_measures,
                   checkpoint_search_radius=args.checkpoint_search_radius, project_config_path=args.project_config,
                   gp_parser=args.gp_parser, score_json_path=args.score_json, alphatab_extractor=args.alphatab_extractor, node_executable=args.node_executable)
    fc.write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()





















