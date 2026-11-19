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
            'timeline_model':tracking.get('timeline_model'),'chain_classification':tracking.get('chain_classification'),
            'incipit_control':tracking.get('incipit_control'),
            'streak_fit':_pick(tracking.get('streak_fit') or {},('valid','reason','timeline_model','fixed_timeline',
                'transformed_timeline','offset','chart_stretch','landmark_count','chain_landmark_count','inlier_count',
                'rejected_landmark_count','chart_start','chart_end','midi_start','midi_end',
                'median_residual_seconds','maximum_residual_seconds','chain_summary','root_validation')),
            'streaks':[_pick(x,('streak_id','role','status','chart_start','chart_end','midi_start','midi_end',
                'accepted_phrases','representative_control_count')) for x in (tracking.get('streaks') or [])]},
        'global_path':{'status':path.get('status'),'selected_count':path.get('selected_count'),
            'edge_stretch_range':path.get('edge_stretch_range'),'unsupported_gaps':path.get('unsupported_gaps') or []},
        'role_paths':roles,'midi_authority':_pick(authority,('status','authority','predicates','path_coverage',
            'largest_score_gap_seconds','roles','role_count','trusted_control_count','force_requested','force_monotonic_chain_requested',
            'normal_gate_passed','force_blocked_reason','chain_classification','diagnostic_unsafe_timing')),
        'representative_controls':compact_controls}



def skeleton_midi_section_fit(data):
    paths=data.get("path_report") or {}; fusion=data.get("fusion_report") or {}
    roles={}
    for role,row in (paths.get("roles") or {}).items():
        roles[role]={"status":row.get("status"),"streak_count":row.get("streak_count"),
            "selected_candidate_count":row.get("selected_candidate_count"),
            "unsupported_window_count":len(row.get("unsupported_windows") or []),
            "monotonicity_validated":row.get("monotonicity_validated"),
            "streaks":[_pick(x,("streak_id","chart_start","chart_end","midi_start","midi_end",
                "score","start_window_index","end_window_index","largest_overlap_disagreement_seconds","monotonic"))
                for x in (row.get("streaks") or [])]}
    return {"diagnostic_profile":"light","version":data.get("version"),"method":data.get("method"),
        "diagnostic_only":data.get("diagnostic_only"),"timing_changes_applied":data.get("timing_changes_applied"),
        "integration":data.get("integration"),
        "performance":data.get("performance"),
        "candidate_summary":((data.get("candidate_report") or {}).get("summary") or {}),
        "candidate_phase_diagnostics":((data.get("candidate_report") or {}).get("phase_diagnostics_summary") or {}),
        "path_summary":paths.get("summary") or {},"selected_phase_diagnostics":paths.get("phase_diagnostics") or {},
        "roles":roles,
        "fusion_summary":fusion.get("summary") or {},
        "solo_role_authority":fusion.get("solo_role_authority"),
        "three_clock_incipit":fusion.get("three_clock_incipit"),
        "production_phase_diagnostics":fusion.get("production_phase_diagnostics"),
        "structural_repeat_diagnostics":fusion.get("structural_repeat_diagnostics") or data.get("structural_repeat_report"),
        "section_evidence_quality":data.get("section_evidence_quality") or fusion.get("section_evidence_quality"),
        "rooted_phrase_ordinal_support":data.get("rooted_phrase_ordinal_support") or fusion.get("rooted_phrase_ordinal_support"),
        "phase_migration_qualification":data.get("phase_migration_qualification") or fusion.get("phase_migration_qualification"),
        "normal_production_limits":fusion.get("normal_production_limits"),
        "monotonicity_certificate":fusion.get("monotonicity_certificate"),
        "regions":[_pick(x,("region_id","chart_start","chart_end","midi_start","midi_end","control_count",
            "minimum_edge_stretch","maximum_edge_stretch","strong_control_count","catch_up_events","monotonic"))
            for x in (fusion.get("regions") or [])],
        "role_conflict_count":len(fusion.get("role_conflicts") or []),
        "rejected_blocks":fusion.get("rejected_blocks") or [],
        "application_report":data.get("application_report")}

def skeleton_audio_evidence(data):
    rows=data.get("evidence") or []
    compact=[]
    for row in rows:
        compact.append({
            "evidence_id":row.get("evidence_id"),"evidence_type":row.get("evidence_type"),
            "detector":row.get("detector"),"source":row.get("source"),
            "nominal_role":row.get("nominal_role"),"observed_signal_type":row.get("observed_signal_type"),
            "clocks":row.get("clocks") or {},"residual_seconds":row.get("residual_seconds"),
            "confidence":row.get("confidence"),"limitations":row.get("limitations") or [],
            "authority":row.get("authority"),"adapter_status":row.get("adapter_status"),
            "native_quality":row.get("native_quality") or {},"provenance":row.get("provenance") or {}})
    return {"diagnostic_profile":"light","schema_version":data.get("schema_version"),
        "diagnostic_only":data.get("diagnostic_only"),"timing_changes_applied":data.get("timing_changes_applied"),
        "midi_authority_changed":data.get("midi_authority_changed"),
        "detectors_executed":data.get("detectors_executed"),"summary":data.get("summary") or {},
        "evidence":compact,"notes":data.get("notes") or []}

def skeleton_alignment(data):
    decision=data.get('decision') or {}
    decision_keep=('mode','timing_output','timing_origin','opening_anchor','scale','midi_timing_decision',
          'continuous_structural_second_pass_prior','continuous_structural_second_pass','manual_anchor_quality',
          'manual_anchor_status','guitar_coverage_diagnostic','alignment_quality')
    root_keep=('version','severity','production_timing','summary','maximum_timing_change_seconds',
               'minimum_measure_stretch','maximum_measure_stretch','warnings','errors')
    return {'diagnostic_profile':'light',**_pick(data,root_keep),'decision':_pick(decision,decision_keep)}



def skeleton_strict_queries(data):
    rows=[]
    for query in data.get("queries") or []:
        answer=query.get("answer") or {}
        rows.append({"query_id":query.get("query_id"),"query_type":query.get("query_type"),
            "chart_start":query.get("chart_start"),"chart_end":query.get("chart_end"),"reason":query.get("reason"),
            "bounded_query_status":query.get("bounded_query_status"),"status":answer.get("status"),
            "shared_evidence_usage":query.get("shared_evidence_usage") or {},
            "shared_evidence_query_count":query.get("shared_evidence_query_count",0),"independent_physical_source_count":answer.get("independent_physical_source_count"),
            "alternative_count":answer.get("alternative_count"),
            "selected_evidence_ids":[x.get("evidence_id") for x in answer.get("selected_support") or []]})
    return {"diagnostic_profile":"light","summary":data.get("summary"),"queries":rows,
        "safety":_pick(data,("diagnostic_only","detectors_executed","anchors_created","timing_changes_applied","midi_authority_changed"))}


def skeleton_query_evaluation(data):
    rows=[]
    for item in data.get("evaluations") or []:
        geometry=item.get("bridge_geometry") or {};point=item.get("representative_audio_support") or {}
        rows.append({"query_id":item.get("query_id"),"query_type":item.get("query_type"),"chart_start":item.get("chart_start"),
            "chart_end":item.get("chart_end"),"reason":item.get("reason"),
            "bounded_query_status":item.get("bounded_query_status"),"q1_status":item.get("q1_status"),
            "shared_evidence_usage":item.get("shared_evidence_usage") or {},
            "shared_evidence_query_count":item.get("shared_evidence_query_count",0),
            "classification":item.get("classification"),"independent_physical_source_count":item.get("independent_physical_source_count"),
            "evidence_id":point.get("evidence_id"),"physical_source_id":point.get("physical_source_id"),
            "bridge_status":geometry.get("status"),"bridge_reason":geometry.get("reason"),
            "minimum_distinct_chart_span_seconds":geometry.get("minimum_distinct_chart_span_seconds"),
            "direct_boundary_corroboration":geometry.get("direct_boundary_corroboration"),
            "coincident_control":geometry.get("coincident_control"),"bridge_edges":geometry.get("edges") or [],
            "future_audio_only_anchor_candidate":item.get("future_audio_only_anchor_candidate")})
    return {"diagnostic_profile":"light","summary":data.get("summary"),"evaluations":rows,
        "safety":_pick(data,("diagnostic_only","detectors_executed","anchors_created","timing_changes_applied","midi_authority_changed"))}


def skeleton_contribution_ledger(data):
    rows=[]
    for event in data.get("events") or []:
        rows.append({"ledger_event_id":event.get("ledger_event_id"),"chart_time":event.get("chart_time"),
            "chart_interval":event.get("chart_interval"),"status":event.get("status"),
            "production_effect":event.get("production_effect"),"audio_sources":event.get("audio_sources") or [],
            "midi_roles":event.get("midi_roles") or [],
            "audio_contributed_to_structural_analysis":event.get("audio_contributed_to_structural_analysis"),
            "audio_contributed_to_production_warp":event.get("audio_contributed_to_production_warp"),
            "midi_contributed_to_region":event.get("midi_contributed_to_region"),
            "midi_survived_materialization":event.get("midi_survived_materialization"),
            "materialization_blocked_elsewhere":event.get("materialization_blocked_elsewhere"),
            "audio_clocks":event.get("audio_clocks") or [],"midi_clocks":event.get("midi_clocks") or [],
            "audio_sources":event.get("audio_sources") or [],
            "reasons":event.get("reasons") or []})
    return {"diagnostic_profile":"light","summary":data.get("summary"),"events":rows,
        "safety":_pick(data,("diagnostic_only","detectors_executed","anchors_created","timing_changes_applied","midi_authority_changed"))}

def skeleton_method_result(data):
    return {"diagnostic_profile":"light",**{k:data.get(k) for k in ("schema_version","orchestration_version",
        "method_id","method_version","status","primary_evidence","prepared_midi_mode","continuous_midi_status",
        "midi_section_status","manual_anchor_count","manual_anchor_status","maximum_manual_anchor_error_ms",
        "manual_anchor_conflicts","opening_root_status","materialization_status","occurrence_warnings","phase_warnings",
        "manual_anchor_recommended","listening_review_points","alignment_warnings","measure_stretch_warning_ranges",
        "warning_count","feedpak_path","midi_method_comparison")}}

def skeleton_comparative_boundary(data):
    inventory=[{k:x.get(k) for k in ("evidence_id","audio_time","confidence","limitations","hard_exclusions",
        "physical_source_id","source_group","detector_family","nominal_role","observed_signal_type","evidence_type",
        "strict_eligible","evidence_level","strict_failure_reasons","comparative_eligible","timing_authority","sigma_seconds")}
        for x in data.get("evidence_inventory") or []]
    return {"diagnostic_profile":"light","version":data.get("version"),"summary":data.get("summary"),
        "policy":data.get("policy"),"evidence_inventory":inventory,
        "boundary_evaluations":data.get("boundary_evaluations") or [],"pair_evaluations":data.get("pair_evaluations") or [],
        "triage":data.get("triage") or {},
        "safety":_pick(data,("diagnostic_only","detectors_executed","audio_decoded","candidate_scores_changed",
            "path_reselected","controls_created","anchors_created","midi_authority_changed","timing_changes_applied"))}

def skeleton_targeted_boundary(data):
    return {"diagnostic_profile":"light","version":data.get("version"),"summary":data.get("summary"),
        "policy":data.get("policy"),"requests":data.get("requests") or [],
        "merged_requests":data.get("merged_requests") or [],"evaluations":data.get("evaluations") or [],
        "pair_evaluations":data.get("pair_evaluations") or [],
        "safety":_pick(data,("diagnostic_only","detectors_executed","whole_song_recomputation",
            "candidate_scores_changed","path_reselected","controls_created","anchors_created",
            "midi_authority_changed","timing_changes_applied"))}

def skeleton_occurrence_resolution(data):
    rows=[]
    for row in data.get("qualifications") or []:
        rows.append({k:row.get(k) for k in ("qualification_id","role","window_id","chart_start","chart_end",
            "phrase_identity_support","occurrence_identity_support","production_status","selected_occurrence_rank",
            "selected_midi_start","favored_alternative_rank","favored_alternative_midi_start","candidate_occurrences",
            "timing_changes_applied")})
    return {"diagnostic_profile":"light","version":data.get("version"),"summary":data.get("summary"),
        "policy":data.get("policy"),"qualifications":rows,
        "regional_consolidation":data.get("regional_consolidation") or [],
        "targeted_boundary_queries":data.get("targeted_boundary_queries") or [],
        "boundary_query_evaluations":data.get("boundary_query_evaluations") or [],
        "boundary_pair_evaluations":data.get("boundary_pair_evaluations") or [],
        "safety":_pick(data,("diagnostic_only","timing_changes_applied","candidate_scores_changed",
            "path_reselected","anchors_created","midi_authority_changed"))}

def skeleton_hybrid_control(data):
    proposals=[]
    for row in data.get("proposals") or []:
        proposals.append({
            "proposal_id":row.get("proposal_id"),"region_id":row.get("region_id"),
            "chart_start":row.get("chart_start"),"chart_end":row.get("chart_end"),
            "classification":row.get("classification"),"decision":row.get("decision"),
            "midi_survived_materialization":row.get("midi_survived_materialization"),
            "midi_control_count":row.get("midi_control_count"),
            "supporting_event_ids":row.get("supporting_event_ids") or [],
            "proposed_control":row.get("proposed_control"),
            "downstream_warp_authorized":row.get("downstream_warp_authorized"),
            "reasons":row.get("reasons") or []})
    return {"diagnostic_profile":"light","version":data.get("version"),
        "summary":data.get("summary"),"policy":data.get("policy"),"proposals":proposals,
        "safety":_pick(data,("diagnostic_only","proposal_only","timing_changes_applied",
            "anchors_created","midi_authority_changed"))}

def write_light_json(source, destination):
    try:data=json.loads(source.read_text(encoding='utf-8-sig'))
    except Exception as exc:
        destination.write_text(json.dumps({'diagnostic_profile':'light','source':str(source),
            'parse_error':f'{type(exc).__name__}: {exc}'},indent=2),encoding='utf-8');return
    name=source.name.lower()
    if 'method_result_envelope' in name:out=skeleton_method_result(data)
    elif 'comparative_boundary_evidence_triage' in name:out=skeleton_comparative_boundary(data)
    elif 'targeted_boundary_evidence_acquisition' in name:out=skeleton_targeted_boundary(data)
    elif 'audio_qualified_occurrence_resolution' in name:out=skeleton_occurrence_resolution(data)
    elif 'hybrid_midi_audio_control' in name:out=skeleton_hybrid_control(data)
    elif 'evidence_contribution_ledger' in name:out=skeleton_contribution_ledger(data)
    elif 'query_result_evaluation' in name:out=skeleton_query_evaluation(data)
    elif 'strict_audio_queries' in name:out=skeleton_strict_queries(data)
    elif 'audio_evidence_bundle' in name:out=skeleton_audio_evidence(data)
    elif 'midi_section_fit' in name:out=skeleton_midi_section_fit(data)
    elif 'transcription_investigator' in name:out=skeleton_transcription(data)
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
            for pattern in ('*method_result_envelope.json','*transcription_investigator.json','*midi_section_fit.json','*audio_evidence_bundle.json','*audio_evidence_qualification.json','*strict_audio_queries.json','*query_result_evaluation.json','*evidence_contribution_ledger.json','*audio_qualified_occurrence_resolution.json','*targeted_boundary_evidence_acquisition.json','*comparative_boundary_evidence_triage.json','*hybrid_midi_audio_control.json','*hybrid_midi_audio_control_proposal.json','*alignment_report.json'):
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
