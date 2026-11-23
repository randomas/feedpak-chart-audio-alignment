"""Exact cache helpers for reusable FeedPak preparation artifacts."""
from __future__ import annotations
import hashlib, json, os, tempfile, time
from pathlib import Path

CACHE_SCHEMA = 1

def json_safe(value):
    if isinstance(value,dict): return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_safe(x) for x in value]
    if isinstance(value,set): return sorted(json_safe(x) for x in value)
    if hasattr(value,"item"):
        try: return value.item()
        except Exception: pass
    return value

def canonical_hash(payload):
    raw=json.dumps(json_safe(payload),sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def file_identity(path, content_hash=True):
    path=os.path.abspath(os.fspath(path)); st=os.stat(path)
    row={"name":os.path.basename(path),"size":st.st_size,"mtime_ns":st.st_mtime_ns}
    if content_hash:
        h=hashlib.sha256()
        with open(path,"rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
        row["sha256"]=h.hexdigest()
    return row

def load_json(path):
    with open(path,"r",encoding="utf-8") as f:return json.load(f)

def atomic_write_json(path,data):
    path=os.path.abspath(os.fspath(path));os.makedirs(os.path.dirname(path),exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".cache-",suffix=".tmp",dir=os.path.dirname(path))
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="\n") as f:
            json.dump(data,f,indent=2,allow_nan=False,sort_keys=True)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.remove(tmp)

def cache_lookup(path,fingerprint,required_files=()):
    if not os.path.isfile(path):return None,"cache_file_missing"
    try:data=load_json(path)
    except Exception:return None,"cache_load_error"
    if data.get("schema_version")!=CACHE_SCHEMA:return None,"schema_mismatch"
    if data.get("fingerprint")!=fingerprint:return None,"fingerprint_mismatch"
    if any(not os.path.isfile(x) for x in required_files):return None,"cached_artifact_missing"
    return data,"exact_cache_hit"

def timer_record(performance,name,started,**extra):
    row={"seconds":time.perf_counter()-started};row.update(extra);performance[name]=row;return row

MATERIAL_COMPATIBILITY_SCHEMA = 1

def optional_file_identity(path, role=None, content_hash=True):
    row={'role':role,'path':None if not path else os.path.abspath(os.fspath(path)),
         'status':'missing' if not path or not os.path.isfile(path) else 'present'}
    if row['status']=='present': row.update(file_identity(path,content_hash=content_hash))
    return row

def build_material_compatibility_report(*, chart_path, audio_paths=None,
                                        prepared_midi_files=None, anchors_path=None,
                                        padding_added=0.0, provider='muscriptor'):
    """Build a compact, content-addressed material ledger for a timing run."""
    audio_paths=dict(audio_paths or {}); prepared_midi_files=dict(prepared_midi_files or {})
    chart=optional_file_identity(chart_path,'chart')
    anchors=optional_file_identity(anchors_path,'manual_anchors') if anchors_path else {
        'role':'manual_anchors','path':None,'status':'not_configured'}
    audio={role:optional_file_identity(path,role) for role,path in sorted(audio_paths.items())}
    midi={}
    for role,path in sorted(prepared_midi_files.items()):
        row=optional_file_identity(path,role)
        row.update({'provider':provider,'timeline_basis':'source_audio_unpadded',
                    'provenance_status':'legacy_unverified'})
        sidecar=(str(path)+'.provenance.json') if path else None
        if sidecar and os.path.isfile(sidecar):
            try:
                provenance=load_json(sidecar);row['provenance']=provenance
                expected=(audio.get(role) or {}).get('sha256')
                declared=provenance.get('source_stem_sha256')
                if expected and declared==expected:
                    row['provenance_status']='verified'
                elif expected and declared:
                    row['provenance_status']='mismatch'
                    row['provenance_error']='source_stem_sha256_mismatch'
                else:
                    row['provenance_status']='incomplete'
            except Exception as exc:
                row['provenance_status']='invalid'
                row['provenance_error']=type(exc).__name__
        midi[role]=row
    payload={'schema_version':MATERIAL_COMPATIBILITY_SCHEMA,'chart':chart,'anchors':anchors,
             'audio':audio,'prepared_midi':midi,'padding_added':float(padding_added)}
    payload['material_set_sha256']=canonical_hash(payload)
    provenance_states={x.get('provenance_status') for x in midi.values() if x.get('status')=='present'}
    if 'mismatch' in provenance_states: payload['status']='mismatch'
    elif provenance_states and provenance_states <= {'verified'}: payload['status']='verified'
    elif provenance_states: payload['status']='legacy_unverified'
    else: payload['status']='no_prepared_midi'
    payload['warnings']=[]
    if payload['status']=='legacy_unverified': payload['warnings'].append('prepared_midi_source_stem_provenance_unverified')
    if payload['status']=='mismatch': payload['warnings'].append('prepared_midi_source_stem_mismatch')
    return payload

def classify_cache_dependency(layer):
    """Declare whether a cached layer remains valid when anchors/baseline change."""
    independent={'prepared_stems','parsed_chart','audio_features','prepared_midi','midi_section_role_candidates'}
    dependent={'baseline_residuals','authority_decision','occurrence_selection','cross_role_fusion','section_materialization','final_warp','final_verification'}
    if layer in independent:return 'anchor_independent'
    if layer in dependent:return 'anchor_dependent'
    return 'unknown'

def build_cache_compatibility(cache_events=None,anchor_fingerprint=None,baseline_fingerprint=None):
    rows=[]
    for event in cache_events or []:
        row=dict(event);layer=str(row.get('layer') or row.get('cache_kind') or 'unknown')
        dependency=classify_cache_dependency(layer);row['dependency']=dependency
        if dependency=='anchor_dependent':
            row['anchor_fingerprint']=anchor_fingerprint;row['baseline_fingerprint']=baseline_fingerprint
        state=row.get('status') or row.get('reason') or 'legacy_unverified'
        row['compatibility_state']=('exact_hit' if state in {'exact_cache_hit','cache_hit','hit'} else
                                    'miss' if 'missing' in state or state=='miss' else
                                    'mismatch' if 'mismatch' in state else 'legacy_unverified')
        rows.append(row)
    return {'schema_version':1,'anchor_fingerprint':anchor_fingerprint,
            'baseline_fingerprint':baseline_fingerprint,'layers':rows,
            'status':'mismatch' if any(x['compatibility_state']=='mismatch' for x in rows) else
                     ('legacy_unverified' if any(x['compatibility_state']=='legacy_unverified' for x in rows) else 'compatible')}
