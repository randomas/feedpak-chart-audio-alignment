from pathlib import Path
import pytest
import midi_section_fit as msf


def control(t,a,classification='single_role_strong'):
    return {'score_time':float(t),'audio_time':float(a),'classification':classification,
            'roles':['piano'],'role_count':1,'provenance':[]}


def region(name,start,end,offset_start,offset_end,strong=True,catchups=None):
    classification='single_role_strong' if strong else 'single_role_provisional'
    mid=(start+end)/2
    controls=[control(start,start+offset_start,classification),
              control(mid,mid+(offset_start+offset_end)/2,classification),
              control(end,end+offset_end,classification)]
    return {'region_id':name,'chart_start':start,'chart_end':end,
            'midi_start':controls[0]['audio_time'],'midi_end':controls[-1]['audio_time'],
            'control_count':3,'strong_control_count':3 if strong else 0,
            'controls':controls,'catch_up_events':catchups or [],'monotonic':True}


def test_raw_endpoint_delta_does_not_reject_long_valid_region():
    fusion={'regions':[region('r1',0,120,0,.6)]}
    warp,report=msf.build_section_materialized_warp(fusion,lambda t:t,'permissive')
    assert report['timing_changes_applied'] is True
    assert report['raw_endpoint_delta_used_as_rejection'] is False
    assert warp(120)==pytest.approx(120.6)
    assert warp(150)==pytest.approx(150.6)


def test_gap_uses_baseline_shape_with_interpolated_offset():
    fusion={'regions':[region('r1',0,20,0,.2),region('r2',30,50,.4,.5)]}
    warp,report=msf.build_section_materialized_warp(fusion,lambda t:t,'conservative')
    assert len(report['applied_regions'])==2
    assert report['transitions'][0]['midi_validated'] is False
    assert warp(25)==pytest.approx(25.3)
    assert report['unsupported_gaps_midi_validated'] is False


def test_modes_enable_increasing_evidence_tiers():
    tier_a=region('A',0,20,0,.1,strong=True)
    tier_b=region('B',30,50,.1,.2,strong=True,catchups=[{'kind':'correction_and_catch_up'}])
    tier_c=region('C',60,80,.2,.3,strong=False)
    fusion={'regions':[tier_a,tier_b,tier_c]}
    _,con=msf.build_section_materialized_warp(fusion,lambda t:t,'conservative')
    _,per=msf.build_section_materialized_warp(fusion,lambda t:t,'permissive')
    _,unsafe=msf.build_section_materialized_warp(fusion,lambda t:t,'diagnostic-unsafe')
    assert [x['region_id'] for x in con['applied_regions']]==['A']
    assert [x['region_id'] for x in per['applied_regions']]==['A','B']
    assert [x['region_id'] for x in unsafe['applied_regions']]==['A','B','C']


def test_transition_policy_not_endpoint_delta_determines_acceptance():
    # A 600 ms change over 10 s requires only 1.06 stretch and is conservative-safe.
    fusion={'regions':[region('r1',0,20,0,0),region('r2',30,50,.6,.6)]}
    _,report=msf.build_section_materialized_warp(fusion,lambda t:t,'conservative')
    assert len(report['applied_regions'])==2
    assert report['transitions'][0]['maximum_stretch']==pytest.approx(1.06)


def test_catastrophic_reversal_is_never_forced():
    bad=region('bad',0,20,0,.1)
    bad['controls'][1]['audio_time']=-5
    _,report=msf.build_section_materialized_warp({'regions':[bad]},lambda t:t,'diagnostic-unsafe')
    assert not report['applied_regions']
    assert report['rejected_regions'][0]['severity']=='catastrophic'


def test_cli_mode_wiring():
    root=Path(__file__).resolve().parents[1]
    process=(root/'process_gp_alignment.py').read_text()
    builder=(root/'build_feedpak.py').read_text()
    runner=(root/'run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    for mode in ('conservative','permissive','diagnostic-unsafe'):
        assert mode in process and mode in builder and mode in runner
    assert '--midi-section-fit-materialization' in process
    assert '--midi-section-fit-materialization' in builder
    assert 'MidiSectionFitMaterialization' in runner


def test_short_delta_exception_accepts_high_stretch_short_edge():
    policy=msf.MATERIALIZATION_MODES['permissive']
    result=msf._geometry_acceptance(1.0,1.20,policy)
    assert result['accepted'] is True
    assert result['stretch']==pytest.approx(1.20)
    assert result['absolute_correction_seconds']==pytest.approx(.20)
    assert result['acceptance_basis']=='short_region_delta_exception'


def test_short_delta_exception_does_not_accept_large_absolute_error():
    result=msf._geometry_acceptance(1.0,1.40,msf.MATERIALIZATION_MODES['permissive'])
    assert result['accepted'] is False
    assert result['acceptance_basis']=='geometry_out_of_policy'


def test_long_small_drift_passes_on_stretch_not_endpoint_delta():
    result=msf._geometry_acceptance(120.0,120.6,msf.MATERIALIZATION_MODES['conservative'])
    assert result['accepted'] is True
    assert result['acceptance_basis']=='normal_stretch_range'
    assert result['absolute_correction_seconds']==pytest.approx(.6)


def test_short_exception_can_classify_local_chart_defect_catchup():
    policy=msf.MATERIALIZATION_MODES['permissive']
    first=msf._geometry_acceptance(1.0,1.2,policy)
    second=msf._geometry_acceptance(1.0,.8,policy)
    assert first['acceptance_basis']=='short_region_delta_exception'
    assert second['acceptance_basis']=='short_region_delta_exception'
    assert msf._classify_catch_up(first,second)=='possible_local_chart_timing_defect'


def test_fusion_retains_short_bounded_high_stretch_edge_for_materializer():
    a=msf.FusedControlCandidate(0.,0.,['piano'],[],1,0.,1.,'single_role_strong')
    b=msf.FusedControlCandidate(1.,1.2,['piano'],[],1,0.,1.,'single_role_strong')
    valid,reason,stretch,_=msf._fused_transition(a,b,msf.SectionFusionConfig())
    assert valid is True
    assert reason is None
    assert stretch==pytest.approx(1.2)


def test_length_adjusted_episode_delta_exception_allows_dsmn_edge():
    controls=[]
    for i in range(49):
        controls.append({'score_time':float(i)*1.5384615384615472,
                         'audio_time':float(i)*1.5384615384615472,
                         'classification':'single_role_strong' if i<37 else 'single_role_provisional',
                         'roles':['piano']})
    # One local episode around edge 28, with the central edge matching DSMN's 1.25 / 384.6 ms correction.
    for i in range(29,49): controls[i]['audio_time'] += 0.3846153846153868
    region={'region_id':'r','chart_start':controls[0]['score_time'],'chart_end':controls[-1]['score_time'],
            'controls':controls,'strong_control_count':37,
            'catch_up_events':[{'first_edge_index':27,'second_edge_index':28},
                               {'first_edge_index':28,'second_edge_index':29}],
            'supported_playback_measure_count':50,'monotonic':True}
    _,report=msf.build_section_materialized_warp({'regions':[region]},lambda t:t,'permissive')
    assert len(report['applied_regions'])==1
    exceptional=[x for x in report['applied_regions'][0]['edge_assessments']
                 if x['acceptance_basis']=='length_adjusted_episode_delta_exception']
    assert exceptional
    assert exceptional[0]['episode_delta_allowance_seconds']==0.42000000000000004


def _v165_control(score_time,audio_time):
    return {'score_time':float(score_time),'audio_time':float(audio_time),
            'classification':'single_role_strong','roles':['piano']}


def test_v165_splits_mode_unsafe_edge_and_recovers_later_suffix():
    controls=[_v165_control(0,0),_v165_control(5,5),_v165_control(10,10),
              _v165_control(15,14),_v165_control(20,19),_v165_control(25,24),
              _v165_control(30,29)]
    region={'region_id':'source','chart_start':0.0,'chart_end':30.0,'controls':controls,
            'strong_control_count':len(controls),'supported_playback_measure_count':24,
            'catch_up_events':[],'monotonic':True}
    warp,report=msf.build_section_materialized_warp({'regions':[region]},lambda t:t,'conservative')
    assert report['status']=='available_recovered'
    assert report['materialization_recovery_applied'] is True
    assert len(report['materialization_split_edges'])==1
    assert len(report['applied_regions'])==2
    assert report['applied_regions'][0]['source_region_id']=='source'
    assert report['applied_regions'][1]['source_region_id']=='source'
    assert report['applied_regions'][1]['trimmed_leading_control_count']==1
    assert report['unsupported_gaps_midi_validated'] is False
    assert report['global_monotonicity_validated'] is True
    assert warp(10.0)==pytest.approx(10.0)
    assert warp(20.0)==pytest.approx(19.0)


def test_v165_keeps_safe_prefix_when_suffix_cannot_reenter():
    controls=[_v165_control(0,0),_v165_control(5,5),_v165_control(10,10),
              _v165_control(15,12),_v165_control(20,17),_v165_control(25,22)]
    region={'region_id':'source','chart_start':0.0,'chart_end':25.0,'controls':controls,
            'strong_control_count':len(controls),'supported_playback_measure_count':24,
            'catch_up_events':[],'monotonic':True}
    _,report=msf.build_section_materialized_warp({'regions':[region]},lambda t:t,'conservative')
    assert len(report['applied_regions'])==1
    assert report['applied_regions'][0]['chart_end']==10.0
    assert any(x.get('reason')=='transition_stretch_out_of_policy' for x in report['rejected_regions'])
    assert report['materialization_split_edges']


def test_v165_does_not_split_edge_accepted_by_episode_exception():
    controls=[_v165_control(0,0),_v165_control(5,5),_v165_control(10,10),
              _v165_control(11,11.2),_v165_control(16,16.2)]
    region={'region_id':'episode','chart_start':0.0,'chart_end':16.0,'controls':controls,
            'strong_control_count':len(controls),'supported_playback_measure_count':50,
            'catch_up_events':[{'first_edge_index':1,'second_edge_index':2}],
            'monotonic':True}
    _,report=msf.build_section_materialized_warp({'regions':[region]},lambda t:t,'permissive')
    assert report['materialization_split_edges']==[]
    assert len(report['applied_regions'])==1
    assert report['applied_regions'][0]['region_id']=='episode'
