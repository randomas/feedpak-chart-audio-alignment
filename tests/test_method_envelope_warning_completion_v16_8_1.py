import method_orchestration as mo


def verification():
    return {"manual_anchor_count":0,"manual_anchor_status":"not_supplied","maximum_manual_anchor_error_ms":0.0,
        "manual_anchor_conflicts":[],"verification_tolerance_ms":1.0,"method_id":"x"}


def test_clean_result_can_be_complete():
    out=mo.build_result_envelope(alignment_mode='continuous-anchor-dtw',decision={'timing_output':'continuous-anchor-dtw'},
        prepared_midi_mode='off',manual_anchor_verification=verification(),
        strict_query_report={"version":"q","queries":[{"query_type":"opening_root_uncertain","answer":{"status":"strict_audio_support_available"}}]})
    assert out['status']=='complete'
    assert out['opening_root_status']=='supported'
    assert out['orchestration_version']=='v16.8-lite.2-reporting-corrections-v1'


def test_occurrence_ambiguity_creates_warning_review_and_manual_recommendation():
    occurrence={"version":"occ","qualifications":[{"window_id":"w1","role":"guitar","chart_start":90.,"chart_end":100.,
        "occurrence_identity_support":"phrase_supported_occurrence_ambiguous"}]}
    out=mo.build_result_envelope(alignment_mode='continuous-anchor-dtw',decision={'timing_output':'continuous-anchor-dtw'},
        prepared_midi_mode='auto',manual_anchor_verification=verification(),occurrence_report=occurrence)
    assert out['status']=='complete_with_warnings'
    assert out['occurrence_warnings']
    assert out['listening_review_points'][0]['chart_time']==90.
    assert out['manual_anchor_recommended'] is True


def test_blocked_materialization_is_partial():
    midi={"version":"midi","application_report":{"status":"materialization_blocked"}}
    out=mo.build_result_envelope(alignment_mode='continuous-anchor-structural',decision={'timing_output':'continuous-anchor-structural'},
        prepared_midi_mode='off',manual_anchor_verification=verification(),midi_section_fit_report=midi)
    assert out['status']=='partial'
    assert out['materialization_status']=='partial'


def test_comparative_conflict_is_visible_but_does_not_recommend_anchor_alone():
    comp={"version":"comp","pair_evaluations":[{"window_id":"w","role":"drums","pair_classification":"comparative_sources_conflict"}]}
    out=mo.build_result_envelope(alignment_mode='continuous-anchor-dtw',decision={'timing_output':'continuous-anchor-dtw'},
        prepared_midi_mode='off',manual_anchor_verification=verification(),comparative_report=comp)
    assert any(x['classification']=='comparative_sources_conflict' for x in out['occurrence_warnings'])
    assert out['manual_anchor_recommended'] is False


def test_component_versions_are_reported():
    out=mo.build_result_envelope(alignment_mode='continuous-anchor-dtw',decision={'timing_output':'continuous-anchor-dtw'},
        prepared_midi_mode='auto',manual_anchor_verification=verification(),transcription_report={'version':'ti'},
        midi_section_fit_report={'version':'msf'},occurrence_report={'version':'occ'})
    versions=out['midi_method_comparison']['component_versions']
    assert versions['transcription_investigator']=='ti'
    assert versions['midi_section_fit']=='msf'
    assert versions['occurrence_resolution']=='occ'
