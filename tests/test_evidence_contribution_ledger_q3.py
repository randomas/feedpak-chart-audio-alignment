import evidence_contribution_ledger as q


def audio(t,eid='a1',source='guitar'):
    return {'evidence_id':eid,'evidence_type':'retained_local_audio_hypothesis','source':source,'confidence':.9,
        'clocks':{'chart_time':t,'mapped_audio_time':t+1,'padded_audio_time':t+1},'limitations':[]}


def midi(t,applied=True):
    report={'fusion_report':{'regions':[{'region_id':'r1','controls':[{'score_time':t,'audio_time':t+1,'roles':['guitar'],'classification':'strong'}]}]},
      'application_report':{'status':'available','global_monotonicity_validated':True,'applied_regions':[]}}
    if applied:report['application_report']['applied_regions']=[{'region_id':'r1'}]
    return report


def test_cross_modal_event_reports_both_modalities_and_midi_materialization():
    result=q.build_ledger(audio_evidence_bundle={'evidence':[audio(100)]},midi_section_fit_report=midi(100))
    event=result['events'][0]
    assert event['status'] in {'cross_modal_structural_corroboration','exact_cross_modal_onset'}
    assert event['midi_contributed_to_region'] and event['midi_survived_materialization']
    assert event['production_effect']=='midi_materialized_to_production'


def test_audio_only_and_midi_only_remain_visible():
    result=q.build_ledger(audio_evidence_bundle={'evidence':[audio(10)]},midi_section_fit_report=midi(50))
    assert {'audio_only_coverage','midi_only_coverage'} <= {x['status'] for x in result['events']}


def test_local_midi_contribution_survives_global_blocking_explanation():
    report=midi(143,applied=False);report['application_report']['status']='materialization_blocked';report['application_report']['global_monotonicity_validated']=False
    result=q.build_ledger(audio_evidence_bundle={'evidence':[audio(143)]},midi_section_fit_report=report)
    event=result['events'][0]
    assert event['status'] in {'cross_modal_structural_corroboration','exact_cross_modal_onset'}
    assert event['midi_contributed_to_region']
    assert not event['midi_survived_materialization']
    assert event['materialization_blocked_elsewhere']
    assert event['production_effect']=='midi_region_contributor_materialization_blocked_elsewhere'


def test_audio_production_control_is_reported_as_effect_not_erased_by_q2():
    decision={'continuous_anchor_linear':{'timing_changes_applied':True,'controls':[{'chart_time':20,'audio_time':21,'source':'guitar'}]}}
    result=q.build_ledger(audio_evidence_bundle={'evidence':[audio(20)]},alignment_decision=decision,
        query_result_evaluation={'summary':{'future_audio_only_anchor_candidate_count':0}})
    event=result['events'][0]
    assert event['audio_contributed_to_production_warp']
    assert event['production_effect']=='audio_structural_warp_contributor'


def test_safety_contract_and_runtime_wiring():
    result=q.build_ledger(audio_evidence_bundle={'evidence':[]})
    assert result['anchors_created'] is False and result['timing_changes_applied'] is False and result['midi_authority_changed'] is False
    process=open('process_gp_alignment.py',encoding='utf-8').read();builder=open('build_feedpak.py',encoding='utf-8').read();collector=open('collect_run_diagnostics.py',encoding='utf-8').read()
    assert 'ecl.build_ledger' in process and 'evidence_contribution_ledger_summary' in process
    assert '.evidence_contribution_ledger.json' in builder
    assert 'skeleton_contribution_ledger' in collector
