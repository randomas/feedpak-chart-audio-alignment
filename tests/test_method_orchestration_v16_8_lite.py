import method_orchestration as mo


def report(*anchors):
    return {"quality":{"anchors":[{"id":f"a{i}","mode":"exact","nominal_time":c,"padded_audio_time":a} for i,(c,a) in enumerate(anchors,1)]}}


def test_prepared_midi_auto_only_for_compatible_methods():
    assert mo.validate_prepared_midi_mode('auto','continuous-anchor-dtw')==( 'auto',True,None)
    mode,compatible,reason=mo.validate_prepared_midi_mode('auto','nominal')
    assert mode=='auto' and compatible is False and reason


def test_manual_anchor_verification_is_exact_with_one_ms_tolerance():
    out=mo.verify_manual_anchors(lambda t:t+2,report((10,12),(20,22)),'continuous-anchor-dtw')
    assert out['manual_anchor_status']=='satisfied'
    assert out['maximum_manual_anchor_error_ms']==0


def test_manual_anchor_failure_is_visible():
    out=mo.verify_manual_anchors(lambda t:t+2.01,report((10,12)),'continuous-anchor-dtw')
    assert out['manual_anchor_status']=='materialization_failed'
    assert out['manual_anchor_conflicts']


def test_nominal_and_offset_are_incompatible_when_anchor_not_satisfied():
    nominal=mo.check_simple_method_compatibility(lambda t:t,report((10,12)),'nominal')
    offset=mo.check_simple_method_compatibility(lambda t:t+2,report((10,12),(20,23)),'offset')
    assert nominal['status']=='incompatible_method'
    assert offset['status']=='incompatible_method'


def test_result_envelope_names_continuous_midi_and_section_midi():
    verification=mo.verify_manual_anchors(lambda t:t,{"quality":{"anchors":[]}},'x')
    continuous=mo.build_result_envelope(alignment_mode='continuous-anchor-dtw',decision={'timing_output':'trusted-midi-baseline'},
        prepared_midi_mode='auto',manual_anchor_verification=verification,
        transcription_report={'summary':{'selected_monotonic_candidates':3}})
    assert continuous['method_id']=='continuous-midi-landmark'
    section=mo.build_result_envelope(alignment_mode='continuous-anchor-dtw',decision={'timing_output':'midi-section-fit-permissive'},
        prepared_midi_mode='auto',manual_anchor_verification=verification,
        midi_section_fit_report={'fusion_report':{'regions':[{}]}})
    assert section['method_id']=='midi-section-fit-permissive'
    assert section['midi_method_comparison']['timing_map_averaged'] is False
