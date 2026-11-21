import copy
import midi_section_fit as msf


def minimal_path_report():
    candidate=msf.SectionCandidate(
        window_id='w1',role='guitar',chart_start=0.0,chart_end=4.0,
        midi_start=1.0,midi_end=5.0,chart_stretch=1.0,translation_seconds=1.0,
        matched_chart_groups=8,chart_group_count=8,midi_group_count=8,chart_coverage=1.0,
        median_residual_seconds=0.0,maximum_residual_seconds=0.0,
        sequence_score=1.0,context_score=1.0,combined_score=1.0,
        matched_landmarks=[],status='valid',rejection_reasons=[],search_hypothesis='prepared_chart_clock')
    return {'roles':{'guitar':{'streaks':[{'streak_id':'s1','selected_candidates':[candidate.__dict__]}]}},
            'summary':{'selected_candidate_count':1,'streak_count':1}}


def strip_progress(value):
    value=copy.deepcopy(value);value.pop('progress_diagnostics',None);return value


def test_fusion_progress_is_selection_equivalent(capsys):
    report=minimal_path_report()
    before=msf.fuse_monotonic_role_paths(report,progress_interval_seconds=9999)
    after=msf.fuse_monotonic_role_paths(report,progress_interval_seconds=0)
    assert strip_progress(before)==strip_progress(after)
    progress=after['progress_diagnostics']
    assert progress['layer_count']>=0 and progress['transition_checks']>=0


def test_stage_progress_markers_are_present():
    text=open(msf.__file__,encoding='utf-8').read()
    for marker in ('MIDI candidate cache:','MIDI role paths: starting','MIDI role paths complete:',
                   'MIDI fusion: starting','MIDI fusion: layer','MIDI fusion complete:'):
        assert marker in text

def test_inner_fusion_callback_is_observational_only():
    def control(t):
        return msf.FusedControlCandidate(t,t+1.0,['guitar'],[],1,0.0,0.9,'single_role_strong')
    layers=[{'chart_time':float(i),'candidates':[control(float(i))]} for i in range(4)]
    events=[]
    without=msf._select_fused_monotonic_controls(layers,msf.SectionFusionConfig())
    with_callback=msf._select_fused_monotonic_controls(layers,msf.SectionFusionConfig(),events.append)
    assert without==with_callback
    assert len(events)==4
    assert {'dp_layer_index','dp_layer_count','previous_state_count','candidate_count',
            'next_state_count','transition_checks','transition_rejections'} <= set(events[-1])
