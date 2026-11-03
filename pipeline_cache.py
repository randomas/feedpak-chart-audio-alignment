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
