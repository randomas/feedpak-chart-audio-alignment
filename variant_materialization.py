"""Materialize A/B timing variants from one completed analysis pass.

This module never parses a score or analyses audio. It consumes the safe build
work directory plus the complete alignment report, derives a monotonic
baseline-audio -> proposal-audio map, rewrites timing-bearing JSON documents,
and packages a separately labelled diagnostic Feedpak.
"""
from __future__ import annotations
import copy, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
import yaml
import feedpak_common as fc


class VariantMaterializationUnavailable(RuntimeError):
    """Expected, non-production failure with structured diagnostics."""
    def __init__(self, diagnostics):
        self.diagnostics=diagnostics
        super().__init__(diagnostics.get("materialization_state") or "variant_materialization_unavailable")


def _rows_to_controls(rows):
    controls=[]
    for row in rows or []:
        if not isinstance(row,dict):
            continue
        x=row.get("baseline_time",row.get("source_audio_time",row.get("baseline_audio_time")))
        y=row.get("proposed_time",row.get("target_audio_time",row.get("audio_time")))
        if x is None or y is None:
            continue
        controls.append((float(x),float(y)))
    return sorted(dict(controls).items())


def continuous_comparison_controls(report, return_diagnostics=False):
    """Find variant controls in current and historical report schemas.

    Expected absence is reported as a materialization state rather than thrown;
    malformed non-monotonic controls remain unusable but are also diagnostic.
    """
    decision=((report or {}).get("decision") or {})
    candidates=[]
    linear=decision.get("continuous_anchor_linear") or {}
    candidates.append(("decision.continuous_anchor_linear.comparison_by_playback_bar",
                       linear.get("comparison_by_playback_bar"),linear))
    for key in ("continuous_comparison_controls","comparison_controls"):
        candidates.append((f"decision.{key}",decision.get(key),linear))
    candidates.append(("decision.continuous_anchor_linear.controls",linear.get("controls"),linear))
    for key in ("selected_path_controls","selected_anchor_path_controls","controls"):
        candidates.append((f"decision.{key}",decision.get(key),linear))
    legacy=(report or {}).get("continuous_anchor_linear") or {}
    candidates.append(("continuous_anchor_linear.comparison_by_playback_bar",
                       legacy.get("comparison_by_playback_bar"),legacy))

    diagnostics={"materialization_state":"insufficient_continuous_comparison_controls",
                 "control_source":None,"control_count":0,"reason":"fewer_than_two_controls"}
    for source,rows,owner in candidates:
        controls=_rows_to_controls(rows)
        if len(controls)<2:
            continue
        if any(bx<=ax or by<=ay for (ax,ay),(bx,by) in zip(controls,controls[1:])):
            diagnostics.update(materialization_state="invalid_continuous_comparison_controls",
                               control_source=source,control_count=len(controls),reason="not_strictly_monotonic")
            continue
        diagnostics.update(materialization_state="materialized",control_source=source,
                           control_count=len(controls),reason=None)
        return (controls,owner,diagnostics) if return_diagnostics else (controls,owner)
    return ([],linear,diagnostics) if return_diagnostics else ([],linear)


class AudioTimeMap:
    def __init__(self, controls):
        self.x=np.asarray([x for x,_ in controls],float)
        self.y=np.asarray([y for _,y in controls],float)
    def __call__(self,value):
        t=float(value)
        if t<=self.x[0]: return float(t+(self.y[0]-self.x[0]))
        if t>=self.x[-1]: return float(t+(self.y[-1]-self.x[-1]))
        return float(np.interp(t,self.x,self.y))
    def interval(self,start,duration):
        a=self(start); b=self(float(start)+float(duration or 0.0))
        return a,max(0.0,b-a)


def _warp_timed_rows(rows,warp,duration_key="d"):
    out=[]
    for raw in rows or []:
        row=copy.deepcopy(raw)
        if "t" in row:
            old=float(row["t"]); row["t"]=round(warp(old),4)
            if duration_key in row:
                _,duration=warp.interval(old,row[duration_key]);row[duration_key]=round(duration,4)
        out.append(row)
    return out


def _rewrite_arrangement(data,warp):
    data=copy.deepcopy(data)
    for note in data.get("notes") or []:
        old=float(note.get("t",0)); new,duration=warp.interval(old,note.get("sus",0))
        note["t"]=round(new,4)
        if "sus" in note: note["sus"]=round(duration,4)
        if note.get("bnv") and float(note.get("sus",0))>0:
            old_duration=max(float(note.get("sus",0)),1e-9)
            scale=duration/old_duration
            for point in note["bnv"]: point["t"]=round(float(point.get("t",0))*scale,4)
    for anchor in data.get("anchors") or []:
        if "time" in anchor: anchor["time"]=round(warp(anchor["time"]),4)
    for key in ("beats","sections"):
        for row in data.get(key) or []:
            if "time" in row: row["time"]=round(warp(row["time"]),4)
    return data


def _rewrite_notation(data,warp):
    data=copy.deepcopy(data)
    for measure in data.get("measures") or []:
        if "t" in measure: measure["t"]=round(warp(measure["t"]),4)
        for stave in (measure.get("staves") or {}).values():
            for voice in stave.get("voices") or []:
                for beat in voice.get("beats") or []:
                    old=float(beat.get("t",0));beat["t"]=round(warp(old),4)
                    for note in beat.get("notes") or []:
                        if "d" in note:
                            _,duration=warp.interval(old,note["d"]);note["d"]=round(duration,4)
    return data


def _rewrite_timeline(data,warp):
    data=copy.deepcopy(data)
    for key in ("beats","time_signatures","sections"):
        for row in data.get(key) or []:
            if "time" in row: row["time"]=round(warp(row["time"]),4)
    tempos=data.get("tempos") or []
    for row in tempos:
        if "time" in row: row["time"]=round(warp(row["time"]),4)
    # Recompute dense BPM from transformed downbeats where possible.
    down=[x for x in data.get("beats") or [] if int(x.get("measure",-1))>=0]
    if tempos and len(down)>=2:
        for i,row in enumerate(tempos):
            if i+1<len(down):
                duration=float(down[i+1]["time"])-float(down[i]["time"])
                if duration>0:
                    old_duration=None
                    if i+1<len(tempos) and float(tempos[i+1]["time"])>float(row["time"]):
                        old_duration=float(tempos[i+1]["time"])-float(row["time"])
                    # Numerator is not carried by tempo rows. Preserve beat count by scaling old BPM.
                    if old_duration: row["bpm"]=round(float(row["bpm"])*old_duration/duration,3)
            elif i and "bpm" in tempos[i-1]: row["bpm"]=tempos[i-1]["bpm"]
    return data


def rewrite_json_document(relative_path,data,warp):
    name=relative_path.replace("\\","/")
    if name.startswith("arrangements/"):
        return _rewrite_notation(data,warp) if "notation" in Path(name).name else _rewrite_arrangement(data,warp)
    if name=="song_timeline.json": return _rewrite_timeline(data,warp)
    if name=="drum_tab.json":
        out=copy.deepcopy(data);out["hits"]=_warp_timed_rows(out.get("hits"),warp,duration_key="d");return out
    if name=="vocal_pitch.json":
        out=copy.deepcopy(data);out["notes"]=_warp_timed_rows(out.get("notes"),warp);return out
    if name=="vocal_pitch_contour.json":
        out=copy.deepcopy(data);out["samples"]=_warp_timed_rows(out.get("samples"),warp,duration_key="d");return out
    if name=="keys.json":
        out=copy.deepcopy(data);out["events"]=_warp_timed_rows(out.get("events"),warp,duration_key="d");return out
    if Path(name).name.startswith("lyrics"):
        return _warp_timed_rows(data,warp)
    return data


def _copytree_hardlink(source,target):
    def copier(src,dst):
        try: os.link(src,dst); return dst
        except OSError: return shutil.copy2(src,dst)
    shutil.copytree(source,target,copy_function=copier)


def materialize_continuous_variant(work_dir,safe_output_path,alignment_report,package_feedpak,label="UNSAFE-CONTINUOUS"):
    controls,linear,control_diagnostics=continuous_comparison_controls(alignment_report,return_diagnostics=True)
    if len(controls)<2:
        raise VariantMaterializationUnavailable(control_diagnostics)
    warp=AudioTimeMap(controls)
    output_dir=os.path.dirname(safe_output_path);stem=Path(safe_output_path).stem
    base_stem=stem[:-len(".feedpak")] if stem.endswith(".feedpak") else stem
    variant_name=base_stem+"."+label+".feedpak"
    output_path=os.path.join(output_dir,variant_name)
    temp=Path(tempfile.mkdtemp(prefix="feedpak_variant_",dir=output_dir))
    try:
        _copytree_hardlink(work_dir,temp/"work")
        variant_dir=temp/"work"
        for path in variant_dir.rglob("*.json"):
            rel=path.relative_to(variant_dir).as_posix()
            data=json.loads(path.read_text(encoding="utf-8"))
            rewritten=rewrite_json_document(rel,data,warp)
            if rewritten is not data: fc.write_json(path,rewritten)
        manifest_path=variant_dir/"manifest.yaml"
        manifest=yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        manifest["diagnostic_unsafe_timing"]=True
        manifest["diagnostic_label"]=label
        manifest["diagnostic_materialization"]="single_analysis_pass"
        package_feedpak(str(variant_dir),manifest,output_path)
        variant_report=copy.deepcopy(alignment_report)
        decision=variant_report.setdefault("decision",{})
        decision["diagnostic_unsafe_timing"]=True
        decision["variant_materialization"]={"status":"applied","label":label,"analysis_reused":True,
            "score_parsing_repeated":False,"audio_analysis_repeated":False,"control_count":len(controls),
            "source":control_diagnostics["control_source"],
            "control_discovery":control_diagnostics,
            "continuous_linear_status":linear.get("status")}
        fc.write_json(output_path+".alignment_report.json",variant_report)
        return output_path
    finally:
        shutil.rmtree(temp,ignore_errors=True)
