import phase_migration_qualification as pmq


def measures(n=20,d=2.0):
    return [{'start':i*d,'end':(i+1)*d,'beat_count':4} for i in range(n)]


def fusion(offsets,rooted=True):
    controls=[{'score_time':i*2.0,'audio_time':i*2.0+off,'roles':['piano'],'strong':True} for i,off in enumerate(offsets)]
    return {'three_clock_incipit':{'status':'validated' if rooted else 'no_opening_candidate'},
      'solo_role_authority':{},'regions':[{'region_id':'r1','controls':controls}],
      'production_phase_diagnostics':{'opening':{'status':'rooted' if rooted else 'unrooted_opening','first_fused_control_chart_time':0.0}}}


def test_persistent_monotonic_departure_is_classified_as_migration():
    result=pmq.build(fusion([0,.1,.2,.35,.5,.65,.7]),measures())
    assert result['diagnostic_only'] is True
    assert result['timing_changes_applied'] is False
    assert result['selection_behavior_changed'] is False
    assert any(x['classification']=='persistent_migration' for x in result['episodes'])
    assert result['high_risk_episode_count']>=1


def test_correction_that_returns_is_not_persistent_migration():
    result=pmq.build(fusion([0,.15,.3,.15,0,.02,.01]),measures())
    assert any(x['classification'] in {'correction_and_catch_up','temporary_correction'} for x in result['episodes'])


def test_unrooted_large_movement_is_structurally_unrooted():
    result=pmq.build(fusion([.3,.45,.6,.75,.9],rooted=False),measures())
    assert result['root_validated'] is False
    assert any(x['classification']=='structurally_unrooted' for x in result['episodes'])


def test_section_summary_receives_separate_phase_dimension():
    summary={'sections':[{'section_index':1,'section_start':0.0,'section_end':12.0,
      'quality':'strong','semantic_type':'validated_evidence'}]}
    result=pmq.build(fusion([0,.1,.2,.35,.5,.65,.7]),measures(),summary)
    state=result['section_phase_states'][0]
    assert state['evidence_support']=='strong'
    assert state['phase_stability']=='persistent_migration'
    assert state['production_status']=='validated_evidence'


def test_no_alignment_object_is_modified():
    import copy
    source=fusion([0,.1,.2,.3]);before=copy.deepcopy(source)
    pmq.build(source,measures())
    assert source==before


def test_smooth_multi_role_evolution_is_not_called_migration():
    source=fusion([0,-.10,-.20,-.30,-.40,-.50,-.60],rooted=True)
    for control in source['regions'][0]['controls']:
        control['roles']=['bass','piano']
    result=pmq.build(source,measures())
    assert any(x['classification']=='supported_tempo_evolution' for x in result['episodes'])
    assert not any(x['classification']=='persistent_migration' for x in result['episodes'])


def test_smooth_single_role_large_departure_remains_migration_risk():
    result=pmq.build(fusion([0,-.10,-.20,-.30,-.40,-.50,-.60],rooted=True),measures())
    assert any(x['classification']=='persistent_migration' for x in result['episodes'])


def test_unrooted_episode_near_gap_can_be_occurrence_switch():
    source=fusion([0,.0,.0,.2,.4,.6,.8],rooted=False)
    source['production_phase_diagnostics']['unsupported_gap_phase_accounting']=[{'chart_start':4.0,'chart_end':6.0}]
    result=pmq.build(source,measures())
    assert any(x['classification']=='probable_occurrence_switch' for x in result['episodes'])


def test_metrical_plateau_is_reported():
    result=pmq.build(fusion([0,.25,.5,.5,.5,.5]),measures())
    assert any(x['metrical_plateau'] for x in result['episodes'])


def test_multi_role_support_alone_does_not_make_persistent_migration():
    source=fusion([0,.05,.10,.15,.20,.25],rooted=True)
    for control in source['regions'][0]['controls']:
        control['roles']=['bass','drums']
    result=pmq.build(source,measures())
    assert not any(x['classification']=='persistent_migration' for x in result['episodes'])


def test_movement_and_phase_error_confidence_are_separate():
    source=fusion([0,.05,.10,.15,.20,.25],rooted=True)
    for control in source['regions'][0]['controls']:
        control['roles']=['bass','drums']
    episode=pmq.build(source,measures())['episodes'][0]
    assert episode['movement_confidence']>episode['phase_error_confidence']


def test_materialization_gap_context_can_identify_occurrence_switch():
    source=fusion([0,0,0,.2,.4,.6,.8],rooted=False)
    application={'applied_regions':[{'chart_start':0.0,'chart_end':5.0},{'chart_start':12.0,'chart_end':20.0}],
                 'transitions':[],'rejected_regions':[]}
    result=pmq.build(source,measures(),application_report=application)
    switches=[x for x in result['episodes'] if x['classification']=='probable_occurrence_switch']
    assert switches
    assert switches[0]['near_gap'] is True
    assert switches[0]['nearest_gap_kind']=='between_applied_regions'
    assert switches[0]['nearest_gap_span']==[5.0,12.0]


def test_persistent_migration_requires_positive_error_qualifier():
    source=fusion([0,.05,.10,.15,.20,.25],rooted=True)
    episode=pmq.build(source,measures())['episodes'][0]
    if episode['classification']=='persistent_migration':
        assert episode['positive_error_qualifier'] is True
