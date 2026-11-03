import section_evidence_quality as seq


def base_report(rooted=True,with_region=True,application=True):
    inventory=[{'start':i*10.0,'end':(i+1)*10.0} for i in range(6)]
    fusion={'regions':[], 'role_conflicts':[],
      'production_phase_diagnostics':{
        'opening':{'first_fused_control_chart_time':2.0,'validated_acquisition_root':rooted,
                   'status':'rooted' if rooted else 'unrooted_opening'},
        'terminal':{'chart_terminal_time':60.0},'regions':[],'review_points':[]},
      'structural_repeat_diagnostics':{'playback_structure_inventory':inventory,
        'phrase_repeat_runs':[],'phrase_run_correspondences':[]}}
    if with_region:
        fusion['regions']=[{'region_id':'r1','chart_start':0.0,'chart_end':50.0,
          'control_count':6,'strong_control_count':4,'controls':[1]*6}]
    report={'fusion_report':fusion}
    if application:
        report['application_report']={'applied_regions':[{'region_id':'a1','source_region_id':'r1','chart_start':0.0,'chart_end':50.0}],
                                      'transitions':[]}
    return report


def test_section_summary_is_diagnostic_only_and_covers_timeline():
    result=seq.build(base_report())
    assert result['diagnostic_only'] is True
    assert result['selection_behavior_changed'] is False
    assert result['timing_changes_applied'] is False
    assert result['sections'][0]['section_start']==0.0
    assert result['sections'][-1]['section_end']==60.0
    assert all(x['summary'] for x in result['sections'])


def test_rooted_applied_section_rates_better_than_unrooted_unsupported():
    good=seq.build(base_report())
    weak=seq.build(base_report(rooted=False,with_region=False,application=True))
    assert good['overall_confidence']>weak['overall_confidence']
    assert any(x['quality'] in {'unsupported','structurally_uncertain'} for x in weak['sections'] if x['semantic_type']!='pre_roll')
    assert weak['primary_concern']


def test_phase_review_creates_causal_warning_and_review_point():
    report=base_report()
    report['fusion_report']['production_phase_diagnostics']['review_points']=[
      {'kind':'gradual_region_drift','severity_score':5,'chart_start':20.0,'chart_end':40.0}]
    result=seq.build(report)
    affected=[x for x in result['sections'] if x['section_start']<=20<x['section_end'] or 20<=x['section_start']<40]
    assert any('cumulative phase movement' in ' '.join(x['risk_factors']) for x in affected)
    assert 20.0 in result['suggested_review_points']


def test_phrase_ambiguity_is_descriptive_not_pruning():
    report=base_report()
    report['fusion_report']['structural_repeat_diagnostics']['phrase_repeat_runs']=[
      {'chart_start':10.0,'chart_end':30.0}]
    report['fusion_report']['structural_repeat_diagnostics']['phrase_run_correspondences']=[
      {'chart_span':[10.0,30.0],'classification':'multiple_possible_run_instances'}]
    result=seq.build(report)
    section=next(x for x in result['sections'] if x['section_start']<=10<x['section_end'])
    assert 'phrase has multiple plausible MIDI instances' in section['risk_factors']
    assert section['production_use']=='applied'
    assert result['timing_changes_applied'] is False


def test_selected_strong_path_softens_rejected_alternative_role_conflicts():
    report=base_report()
    report['fusion_report']['role_conflicts']=[{'chart_time':20.0},{'chart_time':30.0}]
    result=seq.build(report)
    applied=[x for x in result['sections'] if x['production_use']=='applied']
    assert applied
    assert all(x['quality']!='conflicting' for x in applied)
    assert any('alternative role hypotheses were rejected by fusion' in x['risk_factors'] for x in applied)


def test_unapplied_role_conflict_remains_hard_conflict():
    report=base_report(with_region=True,application=False)
    report['fusion_report']['role_conflicts']=[{'chart_time':20.0}]
    result=seq.build(report)
    assert any(x['quality']=='conflicting' for x in result['sections'])


def test_pre_roll_is_excluded_from_musical_quality_distribution():
    report=base_report()
    result=seq.build(report)
    assert result['sections'][0]['semantic_type']=='pre_roll'
    assert 'pre_roll' not in result['quality_distribution_percent']


def test_review_points_are_deduplicated():
    report=base_report()
    report['fusion_report']['production_phase_diagnostics']['review_points']=[
      {'kind':'persistent_phase_change','severity_score':3,'chart_time':20.0},
      {'kind':'gradual_region_drift','severity_score':5,'chart_start':20.0,'chart_end':50.0}]
    result=seq.build(report)
    assert result['suggested_review_points'].count(20.0)==1
