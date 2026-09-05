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


def parse_keyboard_track(track, tempo_events, first_tick):
    """
    Splits notes into right-hand (>= middle C) / left-hand (< middle C)
    staves by pitch threshold, matching the convention used in the
    project's example notation_keys.json. This is a simplification —
    a real player's hand split doesn't always follow a fixed pitch
    threshold — flagged as a v2 improvement, not a blocker for v1.

    Emits nominal "t_gp" fields (not final "t") at both the measure and
    beat level — this track still needs its timing warped to real audio
    and, before that, shifted by any count-in offset — same as every
    other track. See warp_notation() for the nested-structure warp.
    """
    measures_out = []
    last_ts = None

    for idx, measure in enumerate(track.measures, start=1):
        header = measure.header
        m_t = fc.tick_to_seconds(header.start - first_tick, tempo_events)

        rh_beats, lh_beats = [], []
        for voice in measure.voices:
            for beat in voice.beats:
                if beat.start is None or not beat.notes:
                    continue
                beat_t = fc.tick_to_seconds(beat.start - first_tick, tempo_events)
                dur_index = beat.duration.index
                dur_value = 2 ** dur_index if dur_index else 1
                dur_value = min(32, max(1, dur_value))
                midis = [n.value for n in beat.notes]
                beat_entry = {"t_gp": beat_t, "dur": dur_value,
                              "notes": [{"midi": m} for m in midis]}
                if all(m >= MIDDLE_C for m in midis):
                    rh_beats.append(beat_entry)
                elif all(m < MIDDLE_C for m in midis):
                    lh_beats.append(beat_entry)
                else:
                    # mixed chord spanning the split point: put it on rh,
                    # simplest correct-but-imperfect choice for v1
                    rh_beats.append(beat_entry)

        m_out = {"idx": idx, "t_gp": m_t}
        ts = (header.timeSignature.numerator, header.timeSignature.denominator.value) \
            if header.timeSignature else None
        if ts and ts != last_ts:
            m_out["ts"] = list(ts)
            last_ts = ts
        staves = {}
        if rh_beats:
            staves["rh"] = {"voices": [{"v": 1, "beats": rh_beats}]}
        if lh_beats:
            staves["lh"] = {"voices": [{"v": 1, "beats": lh_beats}]}
        if staves:
            m_out["staves"] = staves
        measures_out.append(m_out)

    return {
        "version": 1,
        "instrument": "piano",
        "staves": [
            {"id": "rh", "clef": "G2", "label": "Right Hand"},
            {"id": "lh", "clef": "F4", "label": "Left Hand"},
        ],
        "measures": measures_out,
    }


def shift_and_warp_notation(notation, offset, warp_fn):
    """
    Notation's timing lives in a nested measures -> staves -> voices ->
    beats structure, not a flat list, so it needs its own walk rather
    than the flat apply_warp() used for notes/hits/keys. Applies the
    count-in offset first, then the audio warp, at both the measure and
    beat level, and renames t_gp -> t as it goes (matches every other
    track's convention: t_gp is nominal/GP-clock only, t is real/warped).
    """
    out_measures = []
    for m in notation["measures"]:
        m2 = dict(m)
        m2["t"] = round(warp_fn(m2.pop("t_gp") + offset), 4)
        if "staves" in m2:
            staves2 = {}
            for stave_id, stave in m2["staves"].items():
                voices2 = []
                for voice in stave["voices"]:
                    beats2 = []
                    for beat in voice["beats"]:
                        b2 = dict(beat)
                        b2["t"] = round(warp_fn(b2.pop("t_gp") + offset), 4)
                        beats2.append(b2)
                    voices2.append({**voice, "beats": beats2})
                staves2[stave_id] = {**stave, "voices": voices2}
            m2["staves"] = staves2
        out_measures.append(m2)
    return {**notation, "measures": out_measures}


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
            checkpoint_measures=4, checkpoint_search_radius=1.0):
    if guitarpro is None:
        raise RuntimeError("pyguitarpro is required: pip install pyguitarpro")

    fc.log(f"Processing GP file: {gp_path}")
    if count_in_offset:
        fc.log(f"Count-in offset: {count_in_offset:.3f}s "
               f"(applied to all nominal times before DTW)")
    fc.log_step(1, 5, "Parsing .gp5 structure")
    with fc.timed_step("guitarpro.parse()", indent=1):
        song = guitarpro.parse(gp_path)
    tempo_events = build_tempo_events(song)
    first_tick = song.measureHeaders[0].start if song.measureHeaders else 0
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

    for track in song.tracks:
        if not track.measures:
            continue
        if track.isPercussionTrack:
            fc.log(f"'{track.name}' -> drum track", indent=1)
            drum_hits_gp = parse_drum_track(track, tempo_events, first_tick)
            fc.log(f"{len(drum_hits_gp)} drum hits parsed", indent=2)
        elif _looks_like_keyboard(track):
            fc.log(f"'{track.name}' -> keyboard/notation track", indent=1)
            notation = parse_keyboard_track(track, tempo_events, first_tick)
            notation_tracks[_safe_track_id(track)] = notation
            piano_alignment_events.extend({"t_gp": t} for t in _keyboard_onsets(notation))
            for measure in notation.get("measures",[]):
                for stave in measure.get("staves",{}).values():
                    for voice in stave.get("voices",[]):
                        for beat in voice.get("beats",[]):
                            for note in beat.get("notes",[]):
                                piano_pitch_events.append({"t_gp":beat["t_gp"],"midi":int(note["midi"])})
        else:
            fc.log(f"'{track.name}' -> fretted track", indent=1)
            data = parse_fretted_track(track, tempo_events, first_tick)
            fretted_tracks[_safe_track_id(track)] = data
            if "bass" in (track.name or "").lower() or "bass" in data["name"].lower():
                bass_alignment_events.extend({"t_gp": n["t_gp"]} for n in data["notes"])
                bass_pitch_events.extend({"t_gp":n["t_gp"],"midi":int(data["absolute_tuning_midi"][n["s"]]+n["f"])} for n in data["notes"])
            else:
                guitar_alignment_events.extend({"t_gp": n["t_gp"]} for n in data["notes"])
                guitar_pitch_events.extend({"t_gp":n["t_gp"],"midi":int(data["absolute_tuning_midi"][n["s"]]+n["f"])} for n in data["notes"])
            fc.log(f"{len(data['notes'])} notes, {len(data['anchors'])} anchors", indent=2)

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
        data["notes"] = shift_nominal_times(data["notes"], count_in_offset)
        data["anchors"] = shift_nominal_times(data["anchors"], count_in_offset, time_key="time")
    drum_hits_gp = shift_nominal_times(drum_hits_gp, count_in_offset)
    bass_alignment_events = shift_nominal_times(bass_alignment_events, count_in_offset)
    piano_alignment_events = shift_nominal_times(piano_alignment_events, count_in_offset)
    guitar_alignment_events = shift_nominal_times(guitar_alignment_events, count_in_offset)
    bass_pitch_events = shift_nominal_times(bass_pitch_events, count_in_offset)
    guitar_pitch_events = shift_nominal_times(guitar_pitch_events, count_in_offset)
    piano_pitch_events = shift_nominal_times(piano_pitch_events, count_in_offset)
    key_events_gp = shift_nominal_times(key_events_gp, count_in_offset)
    song_timeline_gp["time_signatures"] = shift_nominal_times(
        song_timeline_gp["time_signatures"], count_in_offset)
    song_timeline_gp["beats"] = shift_nominal_times(song_timeline_gp["beats"], count_in_offset)

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
        simple_mode="linear" if alignment_mode=="dtw-checkpoint" else alignment_mode
        fc.log(f"Using simple alignment mode '{simple_mode}' with reference '{source}'",indent=1)
        warp_fn=compute_simple_warp(event_times,audio_path,simple_mode,sr=sr)
        decision=warp_fn.alignment_diagnostics
        decision["selected_source"]=source if source != "none" else None
        decision["manual_anchors_ignored"]=bool(manual_anchors)
        decision["padding_added"]=padding_added
        if alignment_mode=="dtw-checkpoint":
            fc.log(f"Detecting diagnostic {source} checkpoints every {checkpoint_measures} measures",indent=1)
            checkpoint_report=diagnose_checkpoints(event_times,audio_path,warp_fn,
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
            fc.log(f"Chroma checkpoints: {chroma_report['summary']['strong']} strong, "
                   f"{chroma_report['summary']['rejected']} rejected from "
                   f"{chroma_report['summary']['generated']} periodic candidate(s)",indent=2)
            fc.log(f"Silence checkpoints: {silence_report.get('strong',0)} strong, "
                   f"{silence_report.get('rejected',0)} rejected from {silence_report.get('generated',0)} consolidated candidate(s)",indent=2)
            fc.log(f"Checkpoints: {checkpoint_report.get('strong',0)} strong, "
                   f"{checkpoint_report.get('rejected',0)} rejected; timing unchanged",indent=2)
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
        warped_notation[track_id] = shift_and_warp_notation(notation, count_in_offset, warp_fn)
        fc.log(f"'{track_id}' notation warped", indent=1)

    result = {
        "arrangements": arrangements_out,
        "drum_tab": {"version": 1, "hits": drum_hits} if drum_hits else None,
        "notation": warped_notation if warped_notation else None,
        "keys": {"version": 1, "events": [{"t": e["t"], "key": e["key"]} for e in key_events]}
                 if key_events else None,
        "song_timeline": song_timeline,
        "alignment_report": alignment_report,
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
    parser = argparse.ArgumentParser(description="Parse a .gp5 file and align it to real audio via DTW.")
    parser.add_argument("gp_file", help="Path to the .gp5 file")
    parser.add_argument("drums_audio", nargs="?", default=None, help="Optional isolated drums stem")
    parser.add_argument("--bass-audio", default=None)
    parser.add_argument("--piano-audio", default=None)
    parser.add_argument("--guitar-audio", default=None)
    parser.add_argument("--full-audio", default=None)
    parser.add_argument("--alignment-mode",choices=("nominal","offset","linear","dtw","dtw-checkpoint"),default="dtw")
    parser.add_argument("--timeline-mode", choices=("tempos","beats","both"), default="both",
                        help="A/B test output: dense tempos only, explicit beats only, or both")
    parser.add_argument("--allow-severe-alignment", action="store_true",
                        help="Package even if severe structural timing errors remain after repair")
    parser.add_argument("--anchors", default=None)
    parser.add_argument("--padding-added", type=float, default=0.0)
    parser.add_argument("--auto-chunk-measures", type=int, default=16)
    parser.add_argument("--checkpoint-measures",type=int,default=4)
    parser.add_argument("--checkpoint-search-radius",type=float,default=1.0)
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
                   checkpoint_search_radius=args.checkpoint_search_radius)
    fc.write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()









