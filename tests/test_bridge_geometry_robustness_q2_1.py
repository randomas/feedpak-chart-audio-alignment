import query_result_evaluation as q
import strict_audio_queries as saq


def midi(points):
    return {'fusion_report':{'regions':[{'region_id':'r','controls':[{'score_time':c,'audio_time':a,'roles':['drums']} for c,a in points]}]}}


def query(chart,audio,qtype='materialization_gap_boundary'):
    return {'query_id':'q','query_type':qtype,'chart_start':chart-.5,'chart_end':chart+.5,'reason':'x',
      'minimum_independent_physical_sources':1,'answer':{'status':'strict_audio_support_available','independent_physical_source_count':1,
      'alternative_count':1,'selected_support':[{'evidence_id':'e','source':'guitar','physical_source_id':'stem',
      'confidence':.9,'residual_seconds':.05,'clocks':{'chart_time':chart,'padded_audio_time':audio,'mapped_audio_time':chart}}]}}


def evaluate(points,chart,audio,qtype='materialization_gap_boundary'):
    return q.build_report(strict_query_report={'queries':[query(chart,audio,qtype)]},midi_section_fit_report=midi(points))['evaluations'][0]


def test_coincident_agreement_is_direct_boundary_corroboration():
    row=evaluate([(5,6),(10,11)],5.1,6.05)
    assert row['bridge_geometry']['status']=='coincident_with_midi_control'
    assert row['classification']=='direct_boundary_corroboration'
    assert not row['future_audio_only_anchor_candidate']


def test_coincident_disagreement_is_conflict_not_huge_stretch():
    row=evaluate([(5,6),(10,11)],5.01,9)
    assert row['bridge_geometry']['status']=='coincident_with_midi_control'
    assert row['classification']=='audio_midi_conflict'
    assert row['bridge_geometry']['edges']==[]


def test_duplicate_controls_are_collapsed_before_neighbor_selection():
    controls=q._midi_controls(midi([(0,1),(5,6),(5.05,6.05),(10,11)]))
    assert len(controls)==3
    assert any(x['collapsed_control_count']==2 for x in controls)


def test_safe_one_sided_opening_bridge_is_future_candidate():
    row=evaluate([(10,11),(20,21)],2,3,'opening_root_uncertain')
    assert row['bridge_geometry']['status']=='safe_one_sided_opening_bridge'
    assert row['future_audio_only_anchor_candidate']


def test_safe_one_sided_terminal_bridge_is_future_candidate():
    row=evaluate([(0,1),(10,11)],20,21,'terminal_mismatch')
    assert row['bridge_geometry']['status']=='safe_one_sided_terminal_bridge'
    assert row['future_audio_only_anchor_candidate']


def test_generic_one_sided_gap_is_insufficient_geometry():
    row=evaluate([(10,11)],2,3)
    assert row['bridge_geometry']['status']=='insufficient_geometric_separation'
    assert row['classification']=='insufficient_geometric_separation'


def test_point_uncertainties_are_bounded_plus_minus_half_second():
    report={'fusion_report':{'unsupported_points':[{'chart_time':20,'reason':'no_role_support'}]}}
    rows=saq.generate_queries(midi_section_fit_report=report)
    assert rows[0]['chart_start']==19.5 and rows[0]['chart_end']==20.5
    assert rows[0]['bounded_query_status']=='bounded'


def test_missing_chart_location_is_explicitly_unbounded_not_global_support():
    report={'fusion_report':{'unsupported_points':[{'reason':'no_role_support'}]}}
    rows=saq.generate_queries(midi_section_fit_report=report)
    assert rows[0]['bounded_query_status']=='unbounded_origin_missing_chart_location'
