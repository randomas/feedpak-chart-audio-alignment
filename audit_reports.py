"""Compact, truthful audit products for FeedPak builds and comparison runs."""
from __future__ import annotations
import csv,json
from pathlib import Path

EXCLUDED_PARTS={'Diagnostics','Diagnostics_light','audit_bundle','skeleton','logs','generated','cache','_feedpak_build'}

def _read(path):
    if not path:return None
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except Exception:return None

def _find(folder,suffix):
    rows=sorted(Path(folder).glob('*'+suffix));return rows[0] if rows else None

def _excluded(path,root):
    try:parts=set(path.relative_to(root).parts)
    except ValueError:parts=set(path.parts)
    return bool(parts & EXCLUDED_PARTS)

def build_runtime_contract(*,requested,resolved,delivered,failures=None):
    failures=list(failures or [])
    return {'schema':'feedpak.runtime.contract','schema_version':1,'requested':requested,
            'resolved':resolved,'delivered':delivered,
            'status':'failed' if failures else 'satisfied','failures':failures}

def require_runtime_contract(contract):
    if contract.get('status')!='satisfied':
        raise RuntimeError('RUNTIME CONTRACT FAILED: '+repr(contract.get('failures') or []))

def _production(alignment):
    return (alignment or {}).get('production_timing') or {}

def scorecard_row(folder,material=None,requested_method=None,status=None,exit_code=None):
    folder=Path(folder)
    env=_read(_find(folder,'.method_result_envelope.json')) or {}
    beat=_read(_find(folder,'.beatmap_report.json')) or {}
    mat=_read(_find(folder,'.material_compatibility.json')) or {}
    run=_read(_find(folder,'.runtime_contract.json')) or {}
    alignment=_read(_find(folder,'.alignment_report.json')) or {}
    av=beat.get('manual_anchor_verification') or {}
    requested=run.get('requested') or {};delivered=run.get('delivered') or {}
    production=_production(alignment)
    feedpaks=sorted(folder.glob('*.feedpak'))
    final_output=production.get('timing_output') or env.get('method_version')
    fallback=production.get('fallback_used')
    if fallback is None and isinstance(final_output,str):fallback='fallback' in final_output
    return {'material':material or folder.name,
        'requested_method':requested_method or requested.get('alignment_mode') or production.get('requested_method'),
        'executed_method':delivered.get('method') or env.get('method_id') or production.get('selected_production_method'),
        'status':status or env.get('status') or ('complete' if feedpaks else 'failed'),
        'exit_code':exit_code,'feedpak_produced':bool(feedpaks),
        'feedpak_name':feedpaks[0].name if feedpaks else None,
        'final_timing_output':final_output,'fallback_used':fallback,
        'anchors_requested':requested.get('anchors') or (run.get('resolved') or {}).get('anchors'),
        'anchors_finally_satisfied':av.get('exported_manual_anchor_status'),
        'maximum_anchor_error_ms':av.get('maximum_exported_manual_anchor_error_ms'),
        'vocals_requested':requested.get('vocals_mode'),'vocals_delivered':delivered.get('vocals'),
        'runtime_contract_status':run.get('status'),'material_status':mat.get('status'),
        'material_set_sha256':mat.get('material_set_sha256'),'warning_count':env.get('warning_count'),
        'review_point_count':len(env.get('listening_review_points') or [])}

def _summary_rows(root):
    data=_read(root/'test_material_summary.json') or []
    if isinstance(data,dict):data=[data]
    return data

def _actual_output_dirs(root):
    summary=_summary_rows(root);out=[]
    for row in summary:
        folder=Path(row.get('Output') or '')
        if folder.is_dir():out.append((folder,row))
    if out:return out
    for p in root.iterdir():
        if p.is_dir() and not _excluded(p,root) and (list(p.glob('*.feedpak')) or list(p.glob('*.method_result_envelope.json'))):
            out.append((p,{}))
    return out

def write_comparison_audit(run_root,results=None):
    root=Path(run_root).resolve();out=root/'audit_bundle';out.mkdir(exist_ok=True)
    output_dirs=_actual_output_dirs(root);rows=[]
    for folder,summary in output_dirs:
        rows.append(scorecard_row(folder,material=summary.get('Material'),
            status=summary.get('Status'),exit_code=summary.get('ExitCode')))
    fields=list(rows[0]) if rows else list(scorecard_row(root).keys())
    with (out/'method_scorecard.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    anchors=[];materials=[];routes=[];targets=[]
    for folder,_ in output_dirs:
        beatp=_find(folder,'.beatmap_report.json');matp=_find(folder,'.material_compatibility.json')
        alignp=_find(folder,'.alignment_report.json');envp=_find(folder,'.method_result_envelope.json')
        if beatp:
            d=_read(beatp) or {};anchors.append({'file':str(beatp.relative_to(root)),**(d.get('manual_anchor_verification') or {})})
        if matp:materials.append({'file':str(matp.relative_to(root)),'report':_read(matp)})
        if alignp:
            d=_read(alignp) or {};production=_production(d);decision=d.get('decision') or {}
            routes.append({'file':str(alignp.relative_to(root)),
                'requested_method':production.get('requested_method'),
                'selected_production_method':production.get('selected_production_method'),
                'timing_output':production.get('timing_output'),
                'fallback_used':production.get('fallback_used'),
                'fallback_reason':production.get('fallback_reason'),
                'final_anchor_application':decision.get('final_manual_anchor_application'),
                'anchor_verification':decision.get('manual_anchor_verification'),
                'exported_anchor_verification':decision.get('exported_manual_anchor_verification')})
        if envp:
            d=_read(envp) or {}
            for x in d.get('listening_review_points') or []:targets.append({'file':str(envp.relative_to(root)),**x})
    for name,data in [('final_anchor_verification.json',anchors),('material_compatibility.json',materials),
                      ('timing_route_trace.json',routes),('listening_targets.json',targets)]:
        (out/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    return out
