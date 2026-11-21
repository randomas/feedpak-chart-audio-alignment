import query_result_evaluation as q


def midi_controls():
    return {'fusion_report':{'regions':[{'region_id':'r1','controls':[
        {'score_time':0.0,'audio_time':1.0,'roles':['drums']},
        {'score_time':10.0,'audio_time':11.0,'roles':['drums']},
        {'score_time':20.0,'audio_time':21.0,'roles':['drums']}]}]}}


def query(qtype='materialization_gap_boundary',chart=5.0,audio=6.0,status='strict_audio_support_available',physical='stem-a'):
    support=[] if status!='strict_audio_support_available' else [{'evidence_id':'e1','source':'guitar','physical_source_id':physical,
        'confidence':.9,'residual_seconds':.05,'clocks':{'chart_time':chart,'padded_audio_time':audio,'mapped_audio_time':chart}}]
    return {'query_id':'q1','query_type':qtype,'chart_start':chart-1,'chart_end':chart+1,'reason':'gap',
        'minimum_independent_physical_sources':1,'answer':{'status':status,'selected_support':support,
        'independent_physical_source_count':1 if support else 0,'alternative_count':len(support)}}


def test_safe_audio_supported_void_becomes_future_candidate_but_not_anchor():
    report=q.build_report(strict_query_report={'queries':[query()]},midi_section_fit_report=midi_controls())
    row=report['evaluations'][0]
    assert row['classification']=='audio_supported_midi_void'
    assert row['bridge_geometry']['status'] in {'safe','safe_two_sided_gap_bridge'}
    assert row['future_audio_only_anchor_candidate'] is True
    assert report['anchors_created'] is False and report['timing_changes_applied'] is False


def test_nonmonotonic_audio_is_conflict_and_never_candidate():
    row=q.build_report(strict_query_report={'queries':[query(audio=30)]},midi_section_fit_report=midi_controls())['evaluations'][0]
    assert row['classification']=='audio_midi_conflict'
    assert row['bridge_geometry']['status']=='conflict'
    assert not row['future_audio_only_anchor_candidate']


def test_out_of_policy_bridge_is_too_isolated():
    row=q.build_report(strict_query_report={'queries':[query(audio=8)]},midi_section_fit_report=midi_controls())['evaluations'][0]
    assert row['classification']=='evidence_too_isolated_to_use'
    assert row['bridge_geometry']['status']=='out_of_policy'


def test_phase_support_is_boundary_corroboration():
    row=q.build_report(strict_query_report={'queries':[query('persistent_phase_transition')]},midi_section_fit_report=midi_controls())['evaluations'][0]
    assert row['classification']=='existing_midi_boundary_corroboration'
    assert not row['future_audio_only_anchor_candidate']


def test_insufficient_remains_neutral():
    report=q.build_report(strict_query_report={'queries':[query(status='insufficient_strict_audio_evidence')]},midi_section_fit_report=midi_controls())
    assert report['evaluations'][0]['classification']=='insufficient_strict_audio_evidence'
    assert report['midi_authority_changed'] is False


def test_runtime_and_compact_diagnostics_wiring():
    process=open('process_gp_alignment.py',encoding='utf-8').read();builder=open('build_feedpak.py',encoding='utf-8').read();collector=open('collect_run_diagnostics.py',encoding='utf-8').read()
    assert 'qre.build_report' in process and 'query_result_evaluation_summary' in process
    assert '.query_result_evaluation.json' in builder
    assert 'skeleton_query_evaluation' in collector and 'skeleton_strict_queries' in collector
