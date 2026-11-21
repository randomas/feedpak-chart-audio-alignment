import strict_audio_queries as q
import audio_evidence_qualification as aeq


def evidence(eid,t,physical='stem-a',confidence=.9,residual=.05):
    return {'evidence_id':eid,'evidence_type':'retained_local_audio_hypothesis','source':'guitar',
        'residual_seconds':residual,'confidence':confidence,'adapter_status':'complete','limitations':[],
        'positive_factors':['retained'],'clocks':{'chart_time':t,'padded_audio_time':t+residual,'mapped_audio_time':t},
        'provenance':{'physical_audio_source':physical}}


def qualification(*rows):return aeq.qualify_bundle({'schema_version':'r3.1','evidence':list(rows)})


def test_generates_opening_gap_rejected_phase_and_terminal_queries():
    midi={'fusion_report':{'three_clock_incipit':{'status':'unrooted_opening','opening_horizon_seconds':8},
            'unsupported_points':[{'chart_time':20,'reason':'no_role_support'}],
            'production_phase_diagnostics':{'review_points':[{'kind':'gradual_cumulative_drift','chart_time':30}],
                'terminal':{'unsupported_terminal_tail_seconds':4,'terminal_chart_time':100}}},
        'path_report':{'roles':{'guitar':{'unsupported_windows':[{'chart_start':40,'chart_end':50,'reason':'no_valid_candidates'}]}}},
        'application_report':{'rejected_regions':[{'chart_start':60,'chart_end':70,'reason':'unsafe_transition'}]}}
    rows=q.generate_queries(midi_section_fit_report=midi,chart_start=0,chart_end=100)
    kinds={x['query_type'] for x in rows}
    assert {'opening_root_uncertain','materialization_gap_boundary','rejected_region_boundary',
            'persistent_phase_transition','terminal_mismatch'} <= kinds


def test_query_answer_filters_by_chart_region():
    report=qualification(evidence('inside',5),evidence('outside',25))
    answer=aeq.answer_strict_query(report,'opening_root_uncertain',0,8)
    assert [x['evidence_id'] for x in answer['selected_support']]==['inside']


def test_shared_physical_source_is_one_independent_vote():
    report=qualification(evidence('solo',5,'keys.ogg'),evidence('pad',5.1,'keys.ogg'))
    query={'query_id':'q','query_type':'opening_root_uncertain','chart_start':0,'chart_end':8,'bounded_query_status':'bounded'}
    answered=q.answer_queries(report,[query])[0]['answer']
    assert answered['alternative_count']==2
    assert answered['independent_physical_source_count']==1


def test_clear_audio_only_incipit_is_reported_but_never_materialized():
    midi={'fusion_report':{'three_clock_incipit':{'status':'unvalidated','opening_horizon_seconds':8}}}
    result=q.build_report(qualification_report=qualification(evidence('incipit',3.2)),midi_section_fit_report=midi,chart_start=0,chart_end=100)
    assert result['summary']['strict_support_query_count']==1
    assert result['anchors_created'] is False
    assert result['timing_changes_applied'] is False
    assert result['midi_authority_changed'] is False


def test_insufficient_query_is_explicit_and_neutral():
    midi={'fusion_report':{'unsupported_points':[{'chart_time':50,'reason':'no_role_support'}]}}
    result=q.build_report(qualification_report=qualification(evidence('elsewhere',5)),midi_section_fit_report=midi)
    assert result['summary']['insufficient_query_count']==1
    assert result['queries'][0]['answer']['status']=='insufficient_strict_audio_evidence'


def test_runtime_and_light_diagnostics_wiring():
    process=open('process_gp_alignment.py',encoding='utf-8').read()
    builder=open('build_feedpak.py',encoding='utf-8').read()
    collector=open('collect_run_diagnostics.py',encoding='utf-8').read()
    assert 'saq.build_report' in process and 'strict_audio_query_summary' in process
    assert '.strict_audio_queries.json' in builder
    assert '*strict_audio_queries.json' in collector
    assert '*audio_evidence_qualification.json' in collector
