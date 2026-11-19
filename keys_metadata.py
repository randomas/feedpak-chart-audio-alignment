"""Backward-compatible keys and synth metadata normalization (v16.7K1).

K1 is metadata-only.  It normalizes legacy piano metadata and optional new
keys arrangements/evidence streams, but does not alter chart parsing, stem
routing, MIDI discovery, alignment methods, arrangements, or timing.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence

NORMALIZED_VERSION = 2
VALID_HANDS = {None, "left", "right", "combined"}
VALID_STAFF_MODES = {"single", "grand", "combined"}


def _project(data: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return ((data or {}).get("feedpak_project") or {})


def _tracks(data: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _project(data).get("tracks") or {}


def _arrangement(*, arrangement_id: str, display_name: str, chart_tracks: list[dict[str, Any]],
                 staff_mode: str, evidence_stream: str, source: str) -> dict[str, Any]:
    return {"arrangement_id": arrangement_id, "display_name": display_name,
            "family": "keys", "chart_tracks": chart_tracks,
            "staff_mode": staff_mode, "evidence_stream": evidence_stream,
            "metadata_source": source}


def _legacy_piano(tracks: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    piano = tracks.get("piano") or {}
    chart_tracks=[]
    if piano.get("left"):
        chart_tracks.append({"name": piano["left"], "hand": "left"})
    if piano.get("right"):
        chart_tracks.append({"name": piano["right"], "hand": "right"})
    if piano.get("combined"):
        chart_tracks.append({"name": piano["combined"], "hand": "combined"})
    if not chart_tracks:
        return [], None
    staff_mode = "grand" if any(x["hand"] in ("left", "right") for x in chart_tracks) else "combined"
    arrangement=_arrangement(arrangement_id="piano",display_name="Piano",chart_tracks=chart_tracks,
                             staff_mode=staff_mode,evidence_stream="piano",source="legacy_tracks.piano")
    stream={"stream_id":"piano","family":"keys","chart_tracks":[x["name"] for x in chart_tracks],
            "performance_midi":None,"candidate_audio_source":"piano",
            "deduplicate_symbolic_events":True,"metadata_source":"legacy_tracks.piano"}
    return [arrangement],stream


def normalize_keys_configuration(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a new normalized keys model without modifying input metadata."""
    tracks=_tracks(data); arrangements,legacy_stream=_legacy_piano(tracks)
    streams={"piano":legacy_stream} if legacy_stream else {}
    keys=tracks.get("keys") or {}
    configured_arrangements=keys.get("arrangements") or {}
    if isinstance(configured_arrangements, list):
        configured_arrangements={str(x.get("arrangement_id")):x for x in configured_arrangements if x.get("arrangement_id")}
    for arrangement_id,spec in configured_arrangements.items():
        spec=spec or {}; raw_tracks=spec.get("tracks")
        if raw_tracks is None and spec.get("track") is not None: raw_tracks=[spec.get("track")]
        raw_tracks=raw_tracks or []
        chart_tracks=[]
        for item in raw_tracks:
            if isinstance(item,str): chart_tracks.append({"name":item,"hand":None})
            elif isinstance(item,Mapping): chart_tracks.append({"name":item.get("name"),"hand":item.get("hand")})
        arrangements.append(_arrangement(
            arrangement_id=str(arrangement_id),display_name=str(spec.get("display_name") or arrangement_id),
            chart_tracks=chart_tracks,staff_mode=str(spec.get("staff_mode") or "single"),
            evidence_stream=str(spec.get("evidence_stream") or arrangement_id),source="tracks.keys.arrangements"))
    configured_streams=keys.get("evidence_streams") or {}
    for stream_id,spec in configured_streams.items():
        spec=spec or {}; stream_tracks=list(spec.get("tracks") or [])
        streams[str(stream_id)]={"stream_id":str(stream_id),"family":"keys","chart_tracks":stream_tracks,
            "performance_midi":spec.get("performance_midi"),"candidate_audio_source":spec.get("audio_source"),
            "deduplicate_symbolic_events":bool(spec.get("deduplicate",True)),
            "metadata_source":"tracks.keys.evidence_streams"}
    input_version=int(_project(data).get("version",1) or 1)
    return {"input_schema_version":input_version,"normalized_schema_version":NORMALIZED_VERSION,
        "legacy_compatibility_applied":bool(legacy_stream),"family":"keys",
        "arrangements":arrangements,"evidence_streams":streams}


def validate_keys_configuration(data: Mapping[str, Any] | None,
                                available_tracks: Sequence[str] | None = None,
                                available_stems: Sequence[str] | None = None) -> dict[str, Any]:
    normalized=normalize_keys_configuration(data); errors=[];warnings=[]
    available=set(available_tracks or []); stems=set(x.casefold() for x in (available_stems or []))
    arrangement_ids=set(); track_arrangements={}; track_streams={}
    for arrangement in normalized["arrangements"]:
        aid=arrangement["arrangement_id"]
        if aid in arrangement_ids: errors.append({"code":"duplicate_keys_arrangement_id","arrangement_id":aid})
        arrangement_ids.add(aid)
        if arrangement["staff_mode"] not in VALID_STAFF_MODES:
            errors.append({"code":"invalid_keys_staff_mode","arrangement_id":aid,"staff_mode":arrangement["staff_mode"]})
        stream=arrangement["evidence_stream"]
        for track in arrangement["chart_tracks"]:
            name=track.get("name");hand=track.get("hand")
            if not name: errors.append({"code":"missing_keys_track_name","arrangement_id":aid});continue
            if hand not in VALID_HANDS: errors.append({"code":"invalid_keys_hand","track":name,"hand":hand})
            if available and name not in available: errors.append({"code":"configured_track_not_found","track":name,"arrangement_id":aid})
            track_arrangements.setdefault(name,set()).add(aid);track_streams.setdefault(name,set()).add(stream)
    for name,ids in track_arrangements.items():
        if len(ids)>1: errors.append({"code":"keys_track_in_multiple_arrangements","track":name,"arrangements":sorted(ids)})
    for stream_id,stream in normalized["evidence_streams"].items():
        for name in stream["chart_tracks"]:
            if available and name not in available: errors.append({"code":"configured_track_not_found","track":name,"evidence_stream":stream_id})
            track_streams.setdefault(name,set()).add(stream_id)
        source=stream.get("candidate_audio_source")
        if source and stems and str(source).casefold() not in stems:
            warnings.append({"code":"configured_keys_audio_source_unavailable","evidence_stream":stream_id,"audio_source":source})
        if not stream.get("performance_midi") and stream.get("metadata_source") != "legacy_tracks.piano":
            warnings.append({"code":"keys_performance_midi_not_configured","evidence_stream":stream_id})
    for name,ids in track_streams.items():
        if len(ids)>1: errors.append({"code":"keys_track_in_conflicting_evidence_streams","track":name,"evidence_streams":sorted(ids)})
    for arrangement in normalized["arrangements"]:
        if arrangement["evidence_stream"] not in normalized["evidence_streams"]:
            warnings.append({"code":"keys_evidence_stream_not_configured","arrangement_id":arrangement["arrangement_id"],
                             "evidence_stream":arrangement["evidence_stream"]})
    return {"valid":not errors,"errors":errors,"warnings":warnings,"normalized":normalized}


def legacy_roles_unchanged(data: Mapping[str, Any] | None) -> dict[str, str]:
    """Compatibility helper mirroring the historical piano role assignment."""
    piano=_tracks(data).get("piano") or {};out={}
    for key,role in (("left","piano_left"),("right","piano_right"),("combined","piano_combined")):
        if piano.get(key):out[piano[key]]=role
    return out


def track_descriptors(data: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Map configured Keys chart-track names to parser-neutral descriptors."""
    normalized=normalize_keys_configuration(data);out={}
    for arrangement in normalized["arrangements"]:
        for track in arrangement["chart_tracks"]:
            out[track["name"]]={"family":"keys","arrangement_id":arrangement["arrangement_id"],
                "display_name":arrangement["display_name"],"hand":track.get("hand"),
                "staff_mode":arrangement["staff_mode"],"evidence_stream":arrangement["evidence_stream"]}
    return out


def deduplicate_symbolic_events(events: Sequence[Mapping[str, Any]], time_tolerance: float = 1e-6) -> list[dict[str, Any]]:
    """Union duplicate pitched Keys events while preserving contributing tracks."""
    grouped=[]
    for event in sorted(events,key=lambda x:(float(x["t_gp"]),int(x.get("midi",-1)),str(x.get("source_track","")))):
        t=float(event["t_gp"]);midi=event.get("midi");match=None
        for row in reversed(grouped):
            if t-float(row["t_gp"])>time_tolerance:break
            if midi is not None and row.get("midi")==int(midi) and abs(t-float(row["t_gp"]))<=time_tolerance:
                match=row;break
        if match is None:
            match={k:v for k,v in event.items() if k not in ("source_track","source_tracks","duplicate_count")}
            match["source_tracks"]=[];match["duplicate_count"]=0;grouped.append(match)
        tracks=list(event.get("source_tracks") or [])
        if event.get("source_track"):tracks.append(event["source_track"])
        for name in tracks:
            if name not in match["source_tracks"]:match["source_tracks"].append(name)
        match["duplicate_count"]+=max(1,int(event.get("duplicate_count",1)))
        if event.get("end_gp") is not None:
            match["end_gp"]=max(float(match.get("end_gp",match["t_gp"])),float(event["end_gp"]))
    return grouped


def resolve_evidence_sources(data: Mapping[str, Any] | None, padded_paths: Mapping[str, Any] | None = None, song_folder=None) -> dict[str, dict[str, Any]]:
    """Resolve configured physical audio and MIDI sources without making them authoritative."""
    from pathlib import Path as _Path
    normalized=normalize_keys_configuration(data); paths={str(k).casefold():v for k,v in dict(padded_paths or {}).items()}
    result={}
    for stream_id,stream in normalized["evidence_streams"].items():
        audio_id=stream.get("candidate_audio_source"); audio_path=paths.get(str(audio_id).casefold()) if audio_id else None
        midi_name=stream.get("performance_midi"); midi_path=None
        if midi_name and song_folder:
            folder=_Path(song_folder);target=str(midi_name).casefold()
            midi_path=next((x for x in folder.iterdir() if x.is_file() and x.name.casefold()==target),None) if folder.is_dir() else None
        result[stream_id]={"stream_id":stream_id,"logical_role":stream_id,"family":"keys",
            "physical_audio_source":audio_id,"audio_path":str(audio_path) if audio_path else None,
            "performance_midi":midi_name,"midi_path":str(midi_path) if midi_path else None,
            "audio_available":bool(audio_path),"midi_available":bool(midi_path),
            "nominal_identity_verified":False,"diagnostic_only":True}
    return result
