import strict_audio_queries as q
import query_result_evaluation as qre


def qualification():
    return {'qualifications':[{'evidence_id':'e1','source':'guitar','physical_source_id':'stem',
        'strict_eligible':True,'confidence':.9,'residual_seconds':.05,
        'clocks':{'chart_time':10.0,'padded_audio_time':11.0,'mapped_audio_time':10.0}}]}


def test_unbounded_query_never_selects_global_evidence():
    queries=[{'query_id':'q1','query_type':'materialization_gap_boundary','chart_start':None,'chart_end':None,
        'bounded_query_status':'unbounded_origin_missing_chart_location'}]
    row=q.answer_queries(qualification(),queries)[0]
    assert row['answer']['status']=='insufficient_query_localization'
    assert row['answer']['selected_support']==[]
    assert row['answer']['alternative_count']==0


def test_q2_classifies_localization_failure_separately():
    strict={'queries':[{'query_id':'q1','query_type':'materialization_gap_boundary','chart_start':None,'chart_end':None,
        'bounded_query_status':'unbounded_origin_missing_chart_location','answer':{'status':'insufficient_query_localization',
        'selected_support':[],'independent_physical_source_count':0,'alternative_count':0}}]}
    row=qre.build_report(strict_query_report=strict,midi_section_fit_report={})['evaluations'][0]
    assert row['classification']=='insufficient_query_localization'
    assert row['bridge_geometry']['status']=='not_evaluated'


def test_bounded_query_still_selects_local_evidence():
    queries=[{'query_id':'q1','query_type':'materialization_gap_boundary','chart_start':9.5,'chart_end':10.5,
        'bounded_query_status':'bounded'}]
    row=q.answer_queries(qualification(),queries)[0]
    assert row['answer']['status']=='strict_audio_support_available'
    assert [x['evidence_id'] for x in row['answer']['selected_support']]==['e1']


def test_shared_evidence_usage_is_explicit():
    queries=[
      {'query_id':'q1','query_type':'persistent_phase_transition','chart_start':9,'chart_end':11,'bounded_query_status':'bounded'},
      {'query_id':'q2','query_type':'rejected_region_boundary','chart_start':9.5,'chart_end':10.5,'bounded_query_status':'bounded'}]
    rows=q.answer_queries(qualification(),queries)
    assert rows[0]['shared_evidence_usage']['e1']==['q1','q2']
    assert rows[1]['shared_evidence_query_count']==2


def test_generated_missing_point_is_unbounded_and_localization_insufficient():
    midi={'fusion_report':{'unsupported_points':[{'reason':'no_role_support'}]}}
    queries=q.generate_queries(midi_section_fit_report=midi)
    assert queries[0]['bounded_query_status']=='unbounded_origin_missing_chart_location'
    answered=q.answer_queries(qualification(),queries)[0]
    assert answered['answer']['status']=='insufficient_query_localization'


def test_safety_contract_remains_diagnostic_only():
    report=q.build_report(qualification_report=qualification(),midi_section_fit_report={'fusion_report':{'unsupported_points':[{'reason':'no_role_support'}]}})
    assert report['summary']['insufficient_localization_count']==1
    assert report['summary']['unbounded_query_count']==1
    assert report['anchors_created'] is False and report['timing_changes_applied'] is False and report['midi_authority_changed'] is False
