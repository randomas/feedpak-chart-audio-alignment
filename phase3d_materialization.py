#!/usr/bin/env python3
"""Phase 3D: analyze once, materialize many complete feedpak variants.

Consumes one completed Stage 3C.4.3 run. It never reruns GP parsing, stem
analysis, feature extraction, DTW, structural search, or proposal scoring.
Baseline packages are copied; candidate packages are derived from the same
baseline package and frozen proposal reports.
"""
from __future__ import annotations
import argparse, copy, csv, json, math, re, shutil, tempfile, zipfile
from pathlib import Path
import numpy as np

SAFE_METHODS={"regional_constant_offset","regional_linear_scale","source_fused_piecewise_linear"}
STRUCTURAL_METHOD="experimental_persistent_measure_displacement"
ABSOLUTE_KEYS={"time","starttime","endtime","start_time","end_time","audio_time","audiotime","t_audio","onset","offset","timestamp"}
START_KEYS=("time","startTime","start_time","audio_time","t_audio","onset","timestamp")
DURATION_KEYS=("duration","durationSeconds","duration_seconds","length")


def load(path):
    with path.open("r",encoding="utf-8") as f:return json.load(f)

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")

def sanitize(s):return re.sub(r'[^A-Za-z0-9._-]+','_',s).strip('_')

class Warp:
    def __init__(self,x,y):
        order=np.argsort(x);self.x=np.asarray(x,dtype=float)[order];self.y=np.asarray(y,dtype=float)[order]
        if len(self.x)<2 or np.any(np.diff(self.x)<=0) or np.any(np.diff(self.y)<=0):raise ValueError("warp controls must be strictly monotonic")
    def __call__(self,t):
        t=float(t)
        if t<=self.x[0]:return float(self.y[0]+t-self.x[0])
        if t>=self.x[-1]:return float(self.y[-1]+t-self.x[-1])
        return float(np.interp(t,self.x,self.y))
    def max_delta(self):return float(np.max(np.abs(self.y-self.x)))
    def stretch(self):return np.diff(self.y)/np.diff(self.x)


def proposal_warp(proposal):
    rows=proposal.get("mapped_downbeats") or []
    x=[r["baseline_audio_time"] for r in rows];y=[r["proposed_audio_time"] for r in rows]
    return Warp(x,y)

def structural_warp(cluster,shift_override=None):
    shift=float(cluster.get("median_audio_shift_seconds",0))
    if shift_override is not None:
        rep=float(cluster.get("representative_measure_shift") or 1);shift*=float(shift_override)/rep
    boundary=(cluster.get("sequence_validation") or {}).get("best_boundary") or {}
    boundary_measure=boundary.get("boundary_measure",cluster.get("source_start_measure"))
    members=cluster.get("members") or []
    points=[]
    for m in members:
        gap=m.get("gap") or {}
        for ev in m.get("source_evaluations") or []:
            for r in ev.get("measure_results") or []:
                if r.get("source_measure")==boundary_measure:points.append(float(r.get("target_audio_shift_seconds",shift)))
    # Report audio shift is the most reliable saved duration. Boundary time is
    # reconstructed from mapped-downbeat evidence in member results when present.
    boundary_time=None
    for m in members:
        for ev in m.get("source_evaluations") or []:
            for r in ev.get("measure_results") or []:
                if r.get("source_measure")==boundary_measure and r.get("target_audio_shift_seconds") is not None:
                    # No absolute source time is stored here; caller replaces this using beatmap.
                    pass
    return boundary_measure,shift


def retime_node(node,warp):
    if isinstance(node,list):
        for item in node:retime_node(item,warp)
    elif isinstance(node,dict):
        originals={k:v for k,v in node.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
        # Preserve durations by mapping their endpoint from the original start.
        start_key=next((k for k in START_KEYS if k in originals),None)
        if start_key:
            old_start=float(originals[start_key]);new_start=warp(old_start);node[start_key]=round(new_start,9)
            for dk in DURATION_KEYS:
                if dk in originals and originals[dk]>=0:
                    node[dk]=round(max(0,warp(old_start+float(originals[dk]))-new_start),9)
        for k,v in originals.items():
            if k==start_key or k in DURATION_KEYS:continue
            if k.lower() in ABSOLUTE_KEYS:node[k]=round(warp(v),9)
        for value in node.values():retime_node(value,warp)


def semantic_times(node,out=None,path=""):
    out=[] if out is None else out
    if isinstance(node,list):
        for i,v in enumerate(node):semantic_times(v,out,f"{path}[{i}]")
    elif isinstance(node,dict):
        for k,v in node.items():
            if isinstance(v,(int,float)) and not isinstance(v,bool) and k.lower() in ABSOLUTE_KEYS:out.append((f"{path}.{k}",float(v)))
            else:semantic_times(v,out,f"{path}.{k}")
    return out


def retime_package(src_feedpak,dst_feedpak,warp,manifest):
    with tempfile.TemporaryDirectory(prefix="phase3d_") as td:
        root=Path(td)
        with zipfile.ZipFile(src_feedpak,"r") as z:z.extractall(root)
        changed=0;time_count=0
        for path in root.rglob("*.json"):
            try:data=load(path)
            except Exception:continue
            before=semantic_times(data)
            retime_node(data,warp)
            after=semantic_times(data)
            if before!=after:dump(path,data);changed+=1;time_count+=len(after)
        # Visible label where a YAML manifest exists, without requiring PyYAML.
        for path in list(root.rglob("manifest.yaml"))+list(root.rglob("manifest.yml")):
            text=path.read_text(encoding="utf-8")
            suffix=manifest["display_suffix"]
            text=re.sub(r'(?m)^(title\s*:\s*)(.+)$',lambda m:m.group(1)+m.group(2).rstrip()+f" {suffix}",text,count=1)
            path.write_text(text,encoding="utf-8")
        dump(root/"phase3d_variant_manifest.json",manifest|{"retimed_json_files":changed,"retimed_time_value_count":time_count})
        with zipfile.ZipFile(dst_feedpak,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(root.rglob("*")):
                if path.is_file():z.write(path,path.relative_to(root))
    return changed,time_count


def beatmap_controls(beatmap_path):
    beatmap=load(beatmap_path);pairs=[]
    for b in beatmap.get("beats") or []:
        ident=b.get("identity") or {};timing=b.get("timing") or {}
        if b.get("structural_role")=="downbeat" and timing.get("audio_time") is not None:
            pairs.append((int(ident.get("playback_measure",0)),float(timing["audio_time"])))
    return dict(pairs)

def persistent_structural_warp(beatmap_path,cluster,measure_shift):
    measures=beatmap_controls(beatmap_path)
    boundary=((cluster.get("sequence_validation") or {}).get("best_boundary") or {}).get("boundary_measure",cluster.get("source_start_measure"))
    if boundary not in measures:raise ValueError(f"boundary measure {boundary} absent from beatmap")
    base=measures[boundary];rep=float(cluster.get("representative_measure_shift") or 1)
    delta=float(cluster.get("median_audio_shift_seconds",0))*float(measure_shift)/rep
    # Abrupt structural hypotheses are listening-only. A tiny epsilon keeps the
    # map strict while expressing a persistent post-boundary displacement.
    eps=.001
    x=[0,max(0,base-eps),base,max(measures.values())+60]
    y=[0,max(0,base-eps),base+delta,max(measures.values())+60+delta]
    return Warp(x,y),boundary,delta


def select_safe(proposals,limit=2):
    byid={p.get("proposal_id"):p for p in proposals.get("proposals") or []}
    results=[]
    for rank in proposals.get("timing_proposal_ranking") or proposals.get("ranking") or []:
        p=byid.get(rank.get("proposal_id"));v=(p or {}).get("validation") or {}
        if not p or p.get("method") not in SAFE_METHODS or v.get("status")!="ok" or v.get("error_count",0):continue
        if not p.get("mapped_downbeats"):continue
        try:w=proposal_warp(p)
        except Exception:continue
        stretch=w.stretch()
        if np.min(stretch)<.9 or np.max(stretch)>1.1:continue
        if w.max_delta()<.02:continue
        results.append((p,w))
        if len(results)>=limit:break
    return results

def select_structural(report,limit=2):
    clusters=list(report.get("clusters") or [])
    order={"persistent_displacement_candidate":4,"persistent_correspondence_without_boundary":3,"ambiguous_repeated_material":2,"isolated_correspondence":1}
    clusters.sort(key=lambda c:(order.get((c.get("sequence_validation") or {}).get("classification"),0),float(((c.get("sequence_validation") or {}).get("fuzzy_assessment") or {}).get("structural_validation_priority",0))),reverse=True)
    return clusters[:limit]




def materialize_from_work_dir(work_dir, manifest, output_path, proposals_report, structural_report,
                              package_fn, safe_limit=2, structural_limit=1):
    """Create comparison feedpaks before cleanup, reusing the current build's analysis."""
    work_dir=Path(work_dir); output_path=Path(output_path)
    variants=[]
    candidates=[]
    for p, warp in select_safe(proposals_report or {}, safe_limit):
        candidates.append((f"SAFE-{len(candidates)+1:02d}", p.get("method","proposal"), warp, p, False))
    structural_refusals=[]
    for c in select_structural(structural_report or {}, structural_limit):
        structural_refusals.append({"cluster_id":c.get("cluster_id"),"status":"not_materialized",
            "reason":"structural correspondence has no validated executable timing map"})
    import tempfile, shutil
    for variant_id, method, warp, source, experimental in candidates:
        with tempfile.TemporaryDirectory(prefix="feedpak-phase3d-") as td:
            variant_work=Path(td)/"work"; shutil.copytree(work_dir,variant_work)
            changed=count=0
            for path in variant_work.rglob("*.json"):
                try: obj=json.loads(path.read_text(encoding="utf-8"))
                except Exception: continue
                before=json.dumps(obj,sort_keys=True)
                retime_node(obj,warp)
                after=json.dumps(obj,sort_keys=True)
                if after!=before:
                    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8"); changed+=1
            vm=dict(manifest)
            vm["phase3d_variant"]={"id":variant_id,"method":method,"experimental":experimental,
                                    "analysis_reused":True,"source":source}
            suffix=f".{variant_id}.feedpak"
            dst=output_path.with_name(output_path.name[:-8]+suffix)
            package_fn(str(variant_work),vm,str(dst))
            variants.append({"variant_id":variant_id,"method":method,"experimental":experimental,
                             "file":str(dst),"retimed_json_files":changed})
    report={"phase":"3D","analysis_passes":1,"materialization":"in_process_pre_cleanup",
            "baseline":str(output_path),"variants":variants,"structural_refusals":structural_refusals}
    report_path=output_path.with_suffix(output_path.suffix+".phase3d_variants.json")
    dump(report_path,report)
    return report
