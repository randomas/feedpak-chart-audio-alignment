from pathlib import Path
import midi_section_fit as msf
import process_gp_alignment as pga
import pytest


def candidate(window_id,cs,ce,ms,me,role='piano',score=1.0):
    stretch=(me-ms)/(ce-cs)
    return {'window_id':window_id,'role':role,'chart_start':cs,'chart_end':ce,
        'midi_start':ms,'midi_end':me,'chart_stretch':stretch,
        'translation_seconds':ms-stretch*cs,'matched_chart_groups':12,'chart_group_count':12,
        'midi_group_count':14,'chart_coverage':.9,'median_residual_seconds':.03,
        'maximum_residual_seconds':.08,'sequence_score':score,'context_score':score,
        'combined_score':score,'matched_landmarks':[],'status':'valid','rejection_reasons':[],
        'search_hypothesis':'prepared_chart_clock'}


def path_report(candidates,role='piano'):
    return {'roles':{role:{'streaks':[{'streak_id':role+'-s1','candidates':candidates}],
        'status':'available','monotonicity_validated':True}}}


def test_fusion_keeps_valid_prefix_and_recovers_later_region():
    # First pair is coherent, middle occurrence is incompatible, final pair is coherent.
    rows=[candidate('a',0,10,0,10),candidate('b',7.5,17.5,7.5,17.5),
          candidate('bad',15,25,40,50),
          candidate('c',30,40,30,40),candidate('d',37.5,47.5,37.5,47.5)]
    report=msf.fuse_monotonic_role_paths(path_report(rows))
    assert report['summary']['region_count']>=2
    assert all(x['monotonic'] for x in report['regions'])
    assert report['rejected_blocks']


def test_single_role_multiwindow_is_strong_and_single_window_is_provisional():
    strong=msf.fuse_monotonic_role_paths(path_report([
        candidate('a',0,10,0,10),candidate('b',7.5,17.5,7.5,17.5)]))
    classes={c['classification'] for r in strong['regions'] for c in r['controls']}
    assert 'single_role_strong' in classes
    provisional=msf.fuse_monotonic_role_paths(path_report([candidate('a',0,10,0,10)]))
    classes={c['classification'] for r in provisional['regions'] for c in r['controls']}
    assert classes=={'single_role_provisional'}


def test_common_gm_drum_notes_and_chart_names_share_families():
    expected={35:1000,36:1000,38:1001,40:1001,42:1002,46:1002,
              41:1004,50:1004,49:1003,57:1003}
    for note,family in expected.items():
        assert msf._pitches({'source':'drums','event_type':'drum_hit','pitch':note})=={family}
    assert msf._pitches({'p':'kick'})=={1000}
    assert msf._pitches({'p':'snare'})=={1001}
    assert msf._pitches({'p':'ride'})=={1003}


def test_synthetic_drum_section_path_is_found():
    measures=[{'playback_index':i,'playback_measure_number':i+1,'written_measure_number':i+1,
               'start_time':i*2,'end_time':(i+1)*2} for i in range(8)]
    names=['kick','hihat','snare','hihat']
    notes=[36,42,38,42]
    chart=[];midi=[]
    for i in range(32):
        chart.append({'time':i*.5,'p':names[i%4]})
        midi.append({'audio_time':.2+i*.5,'source':'drums','event_type':'drum_hit','pitch':notes[i%4]})
    report=msf.build_section_path_report(measures,{'drums':chart},{'drums':midi})
    assert report['path_report']['roles']['drums']['selected_candidate_count']>0
    assert report['path_report']['roles']['drums']['monotonicity_validated'] is True


def test_forced_warp_applies_strong_region_and_leaves_gap_on_baseline():
    fusion={'regions':[{'region_id':'r1','monotonic':True,'strong_control_count':3,'catch_up_events':[],'controls':[
        {'score_time':0.,'audio_time':.1,'classification':'single_role_strong'},
        {'score_time':5.,'audio_time':5.2,'classification':'single_role_strong'},
        {'score_time':10.,'audio_time':10.1,'classification':'single_role_strong'}]}]}
    warp,report=msf.build_forced_section_warp(fusion,lambda t:t)
    assert report['timing_changes_applied'] is True
    assert warp(5)>5
    assert warp(20)==pytest.approx(20.1)
    assert report['unsupported_gaps_midi_validated'] is False


def test_gp5_compute_anchors_restored_and_ordered():
    notes=[{'t_gp':0.,'f':1},{'t_gp':1.,'f':3},{'t_gp':2.,'f':7},{'t_gp':3.,'f':5}]
    anchors=pga.compute_anchors(notes)
    assert anchors
    assert all(set(x)=={'time','fret','width'} for x in anchors)
    assert all(b['time']>a['time'] for a,b in zip(anchors,anchors[1:]))
    assert all(x['width']==4 for x in anchors)


def test_force_switch_is_wired_end_to_end():
    root=Path(__file__).resolve().parents[1]
    assert '--force-midi-section-fit' in (root/'process_gp_alignment.py').read_text()
    assert '--force-midi-section-fit' in (root/'build_feedpak.py').read_text()
    runner=(root/'run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    assert '[switch]$ForceMidiSectionFit' in runner
    assert 'midi-section-fit-materialization' in (root/'process_gp_alignment.py').read_text()
