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
