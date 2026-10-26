#!/usr/bin/env python3
"""Collect compact, method-specific diagnostics without duplicating large artifacts."""
import argparse, hashlib, json, re, shutil, zipfile
from pathlib import Path

SOURCE_PATTERNS=('*.log','*.txt','*.json','*.csv')

def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+','_',value).strip('._') or 'unnamed'

def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def copy_file(source: Path, destination: Path, inventory: list[dict], category: str, seen_hashes=None):
    digest=digest_file(source); size=source.stat().st_size
    record={'category':category,'source':str(source),'relative_path':str(destination),'bytes':size,'sha256':digest}
    if seen_hashes is not None and digest in seen_hashes:
        record.update(status='deduplicated',duplicate_of=seen_hashes[digest]); inventory.append(record); return False
    destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,destination)
    record['status']='copied'; inventory.append(record)
    if seen_hashes is not None: seen_hashes[digest]=str(destination)
    return True

ASCII_ARCHIVE_SUFFIXES={'.json','.yaml','.yml','.txt','.csv','.md','.xml'}

def extract_feedpak_ascii(feedpak: Path, destination: Path, inventory: list[dict]):
    """Extract only human-readable package members, never audio stems."""
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(feedpak) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            member=Path(info.filename)
            if member.suffix.lower() not in ASCII_ARCHIVE_SUFFIXES:
                inventory.append({'category':'feedpak_ascii','source':str(feedpak),
                    'archive_member':info.filename,'status':'omitted_non_ascii_archive_member'})
                continue
            data=z.read(info)
            try: data.decode('utf-8-sig')
            except UnicodeDecodeError:
                inventory.append({'category':'feedpak_ascii','source':str(feedpak),
                    'archive_member':info.filename,'status':'omitted_non_text_archive_member'})
                continue
            out=destination/info.filename; out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(data)
            inventory.append({'category':'feedpak_ascii','source':str(feedpak),'relative_path':str(out),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'status':'extracted'})

def schema_summary(path):
    summary={'source':str(path),'sha256':digest_file(path),'bytes':path.stat().st_size}
    try:
        data=json.loads(path.read_text(encoding='utf-8'))
        summary.update(schema_version=data.get('schemaVersion',data.get('schema_version',data.get('version'))),
            track_count=len(data.get('tracks') or []),
            playback_measure_count=len(data.get('playbackMeasures') or data.get('measures') or []),
            tempo_segment_count=len(data.get('tempoSegments') or data.get('tempos') or []),
            role_mapping=data.get('roleMapping') or data.get('role_mapping'))
    except Exception as exc: summary['parse_error']=f'{type(exc).__name__}: {exc}'
    return summary


def is_full_score_schema(path):
    name=path.name.lower()
    return name.endswith('.alphatab.json') or 'schema4' in name or name in {'active_score.json','active_score_schema.json'} or name.startswith('active_score.')


def _pick(obj, keys):
    return {key:obj.get(key) for key in keys if key in obj}


def skeleton_transcription(data):
    summary=data.get('summary') or {}; inc=data.get('incipit_consensus') or {}
    tracking=data.get('incipit_role_tracking') or {}; authority=data.get('midi_authority') or {}
    path=data.get('global_monotonic_path_preview') or {}
    roles={}
    for role,rp in (data.get('role_monotonic_paths') or {}).items():
        selected=rp.get('selected') or []
        roles[role]={'selected_count':rp.get('selected_count',len(selected)),
                     'unsupported_gap_count':len(rp.get('unsupported_gaps') or []),
                     'first_score_time':selected[0].get('score_time') if selected else None,
                     'last_score_time':selected[-1].get('score_time') if selected else None}
    controls=authority.get('representative_controls') or tracking.get('representative_controls') or []
    compact_controls=[_pick(x,('score_time','audio_time','source','authority','kind','local_phrase_confidence','global_placement_confidence')) for x in controls]
    return {'diagnostic_profile':'light','version':data.get('version'),'alignment_mode':data.get('alignment_mode'),
        'pass_label':data.get('pass_label'),'summary':summary,
        'expected_origin':data.get('nominal_prior'),'observed_incipit_origin':data.get('observed_incipit_prior'),
        'incipit_consensus':{'status':inc.get('status'),'selected':_pick(inc.get('selected') or {},
            ('source','score_time','audio_time','origin_hypothesis','authority_tier','sequence_group_count','support_count','incipit_consensus_score','expected_origin_delta_seconds','offset_incipit_delta_seconds'))},
        'incipit_role_tracking':{'status':tracking.get('status'),'state':tracking.get('state'),
            'opening_role':tracking.get('opening_role'),'release_reason':tracking.get('release_reason'),
            'accepted_phrase_count':tracking.get('accepted_phrase_count'),'events':tracking.get('events') or [],
            'timeline_model':tracking.get('timeline_model'),'incipit_control':tracking.get('incipit_control'),
            'streak_fit':_pick(tracking.get('streak_fit') or {},('valid','reason','timeline_model','fixed_timeline',
                'transformed_timeline','offset','chart_stretch','landmark_count','inlier_count','rejected_landmark_count',
                'chart_start','chart_end','midi_start','midi_end','median_residual_seconds','maximum_residual_seconds')),
            'streaks':tracking.get('streaks') or []},
        'global_path':{'status':path.get('status'),'selected_count':path.get('selected_count'),
            'edge_stretch_range':path.get('edge_stretch_range'),'unsupported_gaps':path.get('unsupported_gaps') or []},
        'role_paths':roles,'midi_authority':_pick(authority,('status','authority','predicates','path_coverage',
            'largest_score_gap_seconds','roles','role_count','trusted_control_count','force_requested','normal_gate_passed','force_blocked_reason')),
        'representative_controls':compact_controls}


def skeleton_alignment(data):
    decision=data.get('decision') or data
    keep=('mode','timing_output','production_timing','timing_origin','opening_anchor','scale','midi_timing_decision',
          'continuous_structural_second_pass_prior','continuous_structural_second_pass','manual_anchor_quality',
          'manual_anchor_status','guitar_coverage_diagnostic','alignment_quality')
    return {'diagnostic_profile':'light','decision':_pick(decision,keep)}


def write_light_json(source, destination):
    try:data=json.loads(source.read_text(encoding='utf-8-sig'))
    except Exception as exc:
        destination.write_text(json.dumps({'diagnostic_profile':'light','source':str(source),
            'parse_error':f'{type(exc).__name__}: {exc}'},indent=2),encoding='utf-8');return
    name=source.name.lower()
    if 'transcription_investigator' in name:out=skeleton_transcription(data)
    elif 'alignment_report' in name:out=skeleton_alignment(data)
    else:out={'diagnostic_profile':'light','source':source.name,'keys':sorted(data) if isinstance(data,dict) else [],
              'summary':data.get('summary') if isinstance(data,dict) else None}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--song-dir',required=True); ap.add_argument('--output-dir',required=True)
    ap.add_argument('--diagnostics-root','--diagnostics-dir',dest='diagnostics_root',required=True); ap.add_argument('--log',action='append',default=[]); ap.add_argument('--material-name',required=True); ap.add_argument('--method',required=True)
    ap.add_argument('--include-full-score-schema',action='store_true',help='Copy the complete active score schema instead of summary-only diagnostics')
    ap.add_argument('--light',action='store_true',help='Write analysis-focused skeleton diagnostics only')
    args=ap.parse_args(); song_dir=Path(args.song_dir); output_dir=Path(args.output_dir)
    song_root=Path(args.diagnostics_root)/safe_name(args.material_name)/safe_name(args.method); song_root.mkdir(parents=True,exist_ok=True)
    inventory=[]; seen_hashes={}
    # Preserve the main run log even when the build failed before producing reports.
    for log_name in args.log:
        log=Path(log_name)
        if log.is_file(): copy_file(log,song_root/'logs'/log.name,inventory,'run_log',seen_hashes)
    if args.light:
        wanted=[]
        for root in (output_dir,song_dir):
            for pattern in ('*transcription_investigator.json','*alignment_report.json'):
                wanted.extend(root.rglob(pattern))
        for source in sorted({x.resolve() for x in wanted if x.is_file()}):
            destination=song_root/'skeleton'/(source.name.replace('.json','.light.json'))
            write_light_json(source,destination)
            inventory.append({'category':'light_skeleton','source':str(source),'relative_path':str(destination),
                              'bytes':destination.stat().st_size,'sha256':digest_file(destination),'status':'generated'})
        manifest={'version':'4.0','profile':'light','material':args.material_name,'method':args.method,
                  'song_source':str(song_dir),'normal_output':str(output_dir),'diagnostics_root':str(song_root),'files':inventory}
        (song_root/'diagnostics_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f'Light diagnostics collected: {song_root} ({len(inventory)} file entries)')
        return
    for feedpak in sorted(output_dir.rglob('*.feedpak')): extract_feedpak_ascii(feedpak,song_root/'feedpak_ascii'/safe_name(feedpak.stem),inventory)
    # Collect every readable run artefact from the output tree. active_score JSON is
    # deliberately omitted; feedpaks are expanded above instead of copied as binaries.
    output_files=[p for p in output_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.json','.log','.txt','.csv','.yaml','.yml'}]
    dated_yaml=[p for p in output_files if p.suffix.lower() in {'.yaml','.yml'} and re.search(r'20\d{6}(?:[_-]?\d{6})?',p.name)]
    newest_dated_yaml=max(dated_yaml,key=lambda p:p.stat().st_mtime,default=None)
    for source in sorted(output_files):
        if source.name.lower() in {'active_score.json','active_score_schema.json'} or source.name.lower().startswith('active_score.'): continue
        if source in dated_yaml and source != newest_dated_yaml: continue
        copy_file(source,song_root/'output_ascii'/source.relative_to(output_dir),inventory,'output_ascii',seen_hashes)
    seen_paths=set()
    for pattern in SOURCE_PATTERNS:
        for source in sorted(song_dir.glob(pattern)):
            if not source.is_file() or source.resolve() in seen_paths: continue
            if is_full_score_schema(source) and not args.include_full_score_schema:
                inventory.append({'category':'source_ascii','source':str(source),'status':'omitted_full_score_schema'})
                continue
            seen_paths.add(source.resolve())
            copy_file(source,song_root/'source_ascii'/source.name,inventory,'source_ascii',seen_hashes)
    for name in ('intermediate_arrangements.json','intermediate_vocals.json','metadata.json'):
        source=song_dir/name
        if source.is_file(): copy_file(source,song_root/'generated'/name,inventory,'method_snapshot',seen_hashes)
    score_jsons=sorted(song_dir.glob('*.alphatab.json'))+sorted(song_dir.glob('*schema4*.json'))
    if score_jsons:
        active=score_jsons[0]; summary=schema_summary(active); summary_path=song_root/'generated'/'active_score_schema_summary.json'
        summary_path.parent.mkdir(parents=True,exist_ok=True); summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
        inventory.append({'category':'schema_summary','source':str(active),'relative_path':str(summary_path),'bytes':summary_path.stat().st_size,'sha256':digest_file(summary_path),'status':'generated'})
        if args.include_full_score_schema:
            copy_file(active,song_root/'generated'/'active_score_schema.json',inventory,'full_score_schema',seen_hashes)
    projects=sorted(list(song_dir.glob('feedpak-project*.yaml'))+list(song_dir.glob('feedpak-project*.yml')),key=lambda x:x.stat().st_mtime,reverse=True)
    if projects: copy_file(projects[0],song_root/'generated'/('active_project_config'+projects[0].suffix),inventory,'method_snapshot',seen_hashes)
    manifest={'version':'3.0','material':args.material_name,'method':args.method,'song_source':str(song_dir),'normal_output':str(output_dir),'diagnostics_root':str(song_root),'full_score_schema_included':args.include_full_score_schema,'files':inventory}
    (song_root/'diagnostics_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Diagnostics collected: {song_root} ({len(inventory)} file entries)')
if __name__=='__main__': main()
