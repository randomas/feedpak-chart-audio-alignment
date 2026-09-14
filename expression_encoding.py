"""Parser-neutral Tier 1 expression encoding for Feedpak 1.19 notes."""
from __future__ import annotations
from collections import Counter

def enum_name(v):
    if isinstance(v,dict): v=v.get("name",v.get("value"))
    if v is None:return ""
    return str(getattr(v,"name",v)).replace("_","").replace("-","").replace(" ","").lower()

def enum_number(v,default=None):
    if isinstance(v,dict):v=v.get("value")
    v=getattr(v,"value",v)
    try:return int(v)
    except (TypeError,ValueError):return default

def finger(v):
    n=enum_name(v); names={"thumb":0,"indexfinger":1,"index":1,"middlefinger":2,"middle":2,"annularfinger":3,"ringfinger":3,"ring":3,"littlefinger":4,"pinky":4,"little":4}
    x=names.get(n,enum_number(v,-1));return x if x in range(5) else -1

def pick_direction(v):
    n=enum_name(v)
    return 0 if "down" in n else 1 if "up" in n else -1

def bend_intent(v):
    n=enum_name(v)
    if "prebend" in n and "release" in n:return 3
    if "round" in n:return 4
    if "prebend" in n:return 2
    if "release" in n:return 1
    return 0

def bend_curve(points,duration,position_max,value_divisor=2.0):
    out=[]
    for p in points or []:
        pos=p.get("offset",p.get("position")) if isinstance(p,dict) else getattr(p,"position",getattr(p,"offset",None)); val=p.get("value") if isinstance(p,dict) else getattr(p,"value",None)
        try:r={"t":round(max(0,min(float(duration),float(pos)/position_max*float(duration))),4),"v":round(float(val)/value_divisor,3)}
        except (TypeError,ValueError):continue
        if not out or r!=out[-1]:out.append(r)
    return sorted(out,key=lambda x:x["t"])

def encode_alphatab(note,event,destination_fret=None,beat=None):
    t=(note or {}).get("techniques") or {}; l=(note or {}).get("links") or {}; beat=beat or {}; state=beat.get("state") or {}; rhythm=beat.get("rhythm") or {}; out={}
    if l.get("hammer_pull_destination_id") is not None:out["po" if destination_fret is not None and destination_fret<int(event.get("fret",0)) else "ho"]=True
    h=enum_name(t.get("harmonic_type"))
    if "natural" in h:out["hm"]=True
    elif "pinch" in h:out["hp"]=True
    if t.get("palm_mute") or state.get("palm_mute"):out["pm"]=True
    if t.get("dead"):out.update(mt=True,fhm=True)
    if enum_name(t.get("vibrato")) not in ("","none","0"):out["vb"]=True
    if enum_name(t.get("accentuation")) not in ("","none","0"):out["ac"]=True
    if t.get("tapping"):out["tp"]=True
    if t.get("slap"):out["slp"]=True
    if t.get("pop"):out["plk"]=True
    x=pick_direction(rhythm.get("pick_stroke"));
    if x>=0:out["pkd"]=x
    x=finger(t.get("left_hand_finger"));
    if x>=0:out["fg"]=x
    slide=enum_name(t.get("slide_out_type"))
    if slide not in ("","none","0") and destination_fret is not None:out["slu" if "out" in slide or "pickslide" in slide else "sl"]=int(destination_fret)
    pts=t.get("bend_points") or []; bname=enum_name(t.get("bend_type"))
    if pts or bname not in ("","none","0"):
        curve=bend_curve(pts,max(0,float(event.get("sus",0))),60.0); peak=max((abs(p["v"]) for p in curve),default=0)
        if peak>0:
            out["bn"]=round(peak,3); intent=bend_intent(t.get("bend_type"))
            if intent:out["bt"]=intent
            if len(curve)>1:out["bnv"]=curve
    return out

def encode_gp5(note,beat,duration,destination_fret=None):
    ne=getattr(note,"effect",None);be=getattr(beat,"effect",None);out={}
    if ne is None:return out
    if getattr(ne,"hammer",False):out["po" if destination_fret is not None and destination_fret<int(getattr(note,"value",0)) else "ho"]=True
    harmonic=getattr(ne,"harmonic",None);h=enum_name(harmonic.__class__.__name__ if harmonic else None)
    if "natural" in h:out["hm"]=True
    elif "pinch" in h:out["hp"]=True
    if getattr(ne,"palmMute",False) or getattr(be,"palmMute",False):out["pm"]=True
    if getattr(ne,"ghostNote",False) or enum_name(getattr(note,"type",None))=="dead":out["mt"]=True
    if getattr(ne,"vibrato",False):out["vb"]=True
    if getattr(ne,"accentuatedNote",False) or getattr(ne,"heavyAccentuatedNote",False):out["ac"]=True
    slap=enum_name(getattr(be,"slapEffect",None))
    if "slap" in slap:out["slp"]=True
    elif "pop" in slap:out["plk"]=True
    if "tap" in slap:out["tp"]=True
    x=pick_direction(getattr(be,"pickStroke",None));
    if x>=0:out["pkd"]=x
    x=finger(getattr(ne,"leftHandFinger",None));
    if x>=0:out["fg"]=x
    slides=list(getattr(ne,"slides",None) or [])
    if slides and destination_fret is not None:
        names=" ".join(enum_name(v) for v in slides);out["slu" if "out" in names or "pickslide" in names else "sl"]=int(destination_fret)
    bend=getattr(ne,"bend",None)
    if bend:
        curve=bend_curve(list(getattr(bend,"points",None) or []),duration,12.0);peak=max((abs(p["v"]) for p in curve),default=0)
        if not peak:
            try:peak=abs(float(getattr(bend,"value",0)))/2
            except:peak=0
        if peak>0:
            out["bn"]=round(peak,3);intent=bend_intent(getattr(bend,"type",None))
            if intent:out["bt"]=intent
            if len(curve)>1:out["bnv"]=curve
    return out

def count_fields(notes):
    keys=("sl","slu","bn","bt","bnv","ho","po","hm","hp","pm","mt","vb","ac","tp","fhm","plk","slp","rh","pkd","fg")
    return dict(Counter(k for n in notes for k in keys if k in n))
