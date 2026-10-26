"""Canonical playback-expanded beatmap and Feedpak timeline projection."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import math

SCHEMA_VERSION=1
DEFAULT_SAFE_STRETCH=(0.9,1.1)
DEFAULT_EQUIVALENCE_TOLERANCE_MS=0.11

def _duration(xs,i,key):
    if i+1<len(xs):
        d=float(xs[i+1][key])-float(xs[i][key]); return d if d>0 else None
    if i:
        d=float(xs[i][key])-float(xs[i-1][key]); return d if d>0 else None
    return None

def _meter(t,events,fallback):
    active=[max(1,int(fallback)),4]
    for event in events:
        if float(event.get("t_gp",0))>t+1e-9: break
        if len(event.get("ts") or [])>=2: active=[max(1,int(event["ts"][0])),max(1,int(event["ts"][1]))]
    return active

def _repaired(report):
    out=set()
    for r in ((report.get("sanitization") or {}).get("repairs") or []):
        a=int(r.get("from_measure",r.get("measure",-1))); b=int(r.get("to_measure",a))
        if a>=0: out.update(range(a,b+1))
    return out

def validate_beatmap(beats,safe_stretch=DEFAULT_SAFE_STRETCH):
    errors=[]; warnings=[]; ids=set(); previous={}; low,high=safe_stretch
    for p in beats:
        identity=p["identity"]; timing=p["timing"]; bid=identity["beat_id"]; segment=identity["playback_segment"]
        if bid in ids: errors.append({"kind":"duplicate_beat_id","beat_id":bid})
        ids.add(bid)
        if not all(math.isfinite(float(timing[k])) for k in ("score_time","audio_time")):
            errors.append({"kind":"non_finite_time","beat_id":bid})
        if segment in previous and timing["audio_time"]<=previous[segment]["timing"]["audio_time"]:
            errors.append({"kind":"non_monotonic_audio_time","beat_id":bid,"previous_beat_id":previous[segment]["identity"]["beat_id"]})
        previous[segment]=p
        stretch=timing.get("local_stretch")
        if stretch is not None and not low<=stretch<=high:
            warnings.append({"kind":"stretch_outside_safe_range","beat_id":bid,"ratio":round(stretch,6)})
    return {"status":"error" if errors else "warning" if warnings else "ok","safe_stretch_range":[low,high],
            "error_count":len(errors),"warning_count":len(warnings),"errors":errors,"warnings":warnings}

def build_beatmap(nominal_timeline,warped_downbeats,alignment_report=None,metadata=None,safe_stretch=DEFAULT_SAFE_STRETCH):
    report=alignment_report or {}; nominal=list(nominal_timeline.get("beats") or []); warped=list(warped_downbeats)
    signatures=list(nominal_timeline.get("time_signatures") or [])
    if len(nominal)!=len(warped): raise ValueError("Beatmap nominal/warped downbeat counts differ")
    repaired=_repaired(report); beats=[]
    for i,(source,mapped) in enumerate(zip(nominal,warped)):
        measure=int(source.get("measure",i+1)); src=int(source.get("source_bar",measure)); occ=int(source.get("occurrence",1)); seg=int(source.get("playback_segment",0))
        meter=_meter(float(source["t_gp"]),signatures,source.get("ts_num",4)); count=meter[0]
        nd=_duration(nominal,i,"t_gp"); wd=_duration(warped,i,"time"); stretch=wd/nd if nd and wd else None
        nbpm=count*60/nd if nd else None; mbpm=count*60/wd if wd else None
        for b in range(count):
            f=b/count; flags=[]
            if stretch is not None and not safe_stretch[0]<=stretch<=safe_stretch[1]: flags.append("stretch_outside_safe_range")
            if measure in repaired: flags.append("repaired_timeline_interval")
            beats.append({"identity":{"beat_id":f"seg{seg:03d}.m{measure:04d}.o{occ:02d}.b{b:02d}","playback_segment":seg,
                                      "playback_measure":measure,"source_measure":src,"source_occurrence":occ,"beat_index":b},
                          "timing":{"score_time":round(float(source["t_gp"])+(nd or 0)*f,6),"audio_time":round(float(mapped["time"])+(wd or 0)*f,6),
                                    "nominal_bpm":None if nbpm is None else round(nbpm,6),"mapped_bpm":None if mbpm is None else round(mbpm,6),
                                    "local_stretch":None if stretch is None else round(stretch,6)},
                          "meter":meter,"structural_role":"downbeat" if b==0 else "beat",
                          "provenance":{"status":"inferred","method":"accepted_measure_warp_even_subdivision","left_measure":measure,
                                        "right_measure":int(nominal[i+1].get("measure",measure+1)) if i+1<len(nominal) else None,
                                        "repaired_interval":measure in repaired,"confidence":None,"flags":flags}})
    validation=validate_beatmap(beats,safe_stretch); downs=[b for b in beats if b["structural_role"]=="downbeat"]
    bpms=[b["timing"]["mapped_bpm"] for b in downs if b["timing"]["mapped_bpm"] is not None]; stretches=[b["timing"]["local_stretch"] for b in downs if b["timing"]["local_stretch"] is not None]
    decision=report.get("decision") or {}; method=decision.get("mode") or decision.get("method") or "accepted_warp"
    summary={"playback_measure_count":len(nominal),"beat_count":len(beats),"playback_segment_count":len({b["identity"]["playback_segment"] for b in beats}),
             "source_measure_count":len({b["identity"]["source_measure"] for b in beats}),"first_audio_time":beats[0]["timing"]["audio_time"] if beats else None,
             "last_audio_time":beats[-1]["timing"]["audio_time"] if beats else None,"min_mapped_bpm":round(min(bpms),6) if bpms else None,
             "max_mapped_bpm":round(max(bpms),6) if bpms else None,"min_local_stretch":round(min(stretches),6) if stretches else None,
             "max_local_stretch":round(max(stretches),6) if stretches else None,"repaired_measure_count":len(repaired),"method":method}
    return {"schema":"feedpak.beatmap","schema_version":SCHEMA_VERSION,"metadata":dict(metadata or {}),"summary":summary,"validation":validation,"beats":beats}

def beatmap_to_explicit_beats(beatmap):
    return [{"time":round(float(p["timing"]["audio_time"]),4),"measure":int(p["identity"]["playback_measure"]) if p["structural_role"]=="downbeat" else -1} for p in beatmap.get("beats",[])]

def beatmap_to_dense_tempos(beatmap,fallback_bpm):
    down=[p for p in beatmap.get("beats",[]) if p["structural_role"]=="downbeat"]; out=[]
    for i,p in enumerate(down):
        t=float(p["timing"]["audio_time"])
        if i+1<len(down):
            d=float(down[i+1]["timing"]["audio_time"])-t; bpm=round(max(1,int(p["meter"][0]))*60/d,3) if d>0 else (out[-1]["bpm"] if out else fallback_bpm)
        else: bpm=out[-1]["bpm"] if out else fallback_bpm
        out.append({"time":round(t,4),"bpm":bpm})
    return out

def compare_timeline_series(legacy,projected,value_keys,tolerance_ms=DEFAULT_EQUIVALENCE_TOLERANCE_MS):
    first=None; mismatches=0; max_delta=0.0
    for i,(a,b) in enumerate(zip(legacy,projected)):
        delta=abs(float(a.get("time",0))-float(b.get("time",0)))*1000; max_delta=max(max_delta,delta); differences={k:{"legacy":a.get(k),"beatmap":b.get(k)} for k in value_keys if a.get(k)!=b.get(k)}
        if delta>tolerance_ms or differences:
            mismatches+=1
            if first is None: first={"index":i,"time_delta_ms":round(delta,6),"differences":differences,"legacy":dict(a),"beatmap":dict(b)}
    mismatches+=abs(len(legacy)-len(projected)); status="equivalent" if len(legacy)==len(projected) and not mismatches else "mismatch"
    return {"status":status,"legacy_count":len(legacy),"beatmap_count":len(projected),"count_match":len(legacy)==len(projected),"tolerance_ms":tolerance_ms,
            "maximum_delta_ms":round(max_delta,6),"mismatch_count":mismatches,"first_mismatch":first}

def compare_timeline_outputs(lb,pb,lt,pt,tolerance_ms=DEFAULT_EQUIVALENCE_TOLERANCE_MS):
    b=compare_timeline_series(lb,pb,("measure",),tolerance_ms); t=compare_timeline_series(lt,pt,("bpm",),tolerance_ms)
    return {"status":"equivalent" if b["status"]==t["status"]=="equivalent" else "mismatch","beats":b,"tempos":t}

def require_timeline_equivalence(c):
    if c.get("status")!="equivalent": raise RuntimeError("Beatmap timeline projection differs from legacy timing output: "+repr(dict(c)))

def build_beatmap_report(beatmap,timeline_equivalence=None):
    flagged=[]
    for p in beatmap.get("beats",[]):
        flags=(p.get("provenance") or {}).get("flags") or []
        if flags: flagged.append({"beat_id":p["identity"]["beat_id"],"playback_measure":p["identity"]["playback_measure"],"beat_index":p["identity"]["beat_index"],"audio_time":p["timing"]["audio_time"],"local_stretch":p["timing"].get("local_stretch"),"flags":flags})
    result={"schema":"feedpak.beatmap.report","schema_version":SCHEMA_VERSION,"summary":dict(beatmap.get("summary") or {}),"validation":dict(beatmap.get("validation") or {}),"flagged_beats":flagged}
    if timeline_equivalence is not None: result["timeline_equivalence"]=dict(timeline_equivalence)
    return result
