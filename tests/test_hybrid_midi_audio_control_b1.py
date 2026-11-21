import hybrid_midi_audio_control as h


def report(applied=True):
 return {'fusion_report':{'regions':[{'region_id':'r1','chart_start':10,'chart_end':20,'controls':[{'score_time':12,'audio_time':13},{'score_time':18,'audio_time':19}]}]},
  'application_report':{'applied_regions':[{'chart_start':10,'chart_end':20}] if applied else []}}


def event(status='exact_cross_modal_onset', audio_time=13.2, confidence=.95, limitations=None, chart=12):
 return {'event_id':'event-1','chart_time':chart,'status':status,
  'audio_clocks':[{'event_id':'a1','chart_time':chart,'mapped_audio_time':audio_time,'confidence':confidence,'limitations':limitations or [],'source':'guitar'}],
  'midi_clocks':[{'event_id':'m1','chart_time':chart,'mapped_audio_time':13.0}]}


def test_proposes_bounded_local_correction_but_applies_nothing():
 result=h.build_proposal(evidence_contribution_ledger={'events':[event()]},midi_section_fit_report=report())
 p=result['proposals'][0]
 assert p['classification']=='audio_corrected_midi_geometry'
 assert round(p['proposed_control']['delta_seconds'],3)==.2
 assert p['proposed_control']['influence_chart_start']==10
 assert p['proposed_control']['influence_chart_end']==14
 assert not p['downstream_warp_authorized']
 assert result['timing_changes_applied'] is False


def test_small_delta_confirms_midi_without_correction():
 result=h.build_proposal(evidence_contribution_ledger={'events':[event(audio_time=13.03)]},midi_section_fit_report=report())
 p=result['proposals'][0]
 assert p['classification']=='audio_confirmed_midi_geometry'
 assert p['proposed_control'] is None


def test_low_quality_audio_does_not_correct():
 result=h.build_proposal(evidence_contribution_ledger={'events':[event(confidence=.5)]},midi_section_fit_report=report())
 assert result['proposals'][0]['classification']=='audio_confirmed_midi_geometry'
 assert result['summary']['bounded_local_control_count']==0


def test_audio_only_outside_midi_is_unresolved_and_neutral():
 e=event(status='audio_only_coverage',chart=40); e['midi_clocks']=[]
 result=h.build_proposal(evidence_contribution_ledger={'events':[e]},midi_section_fit_report=report())
 unresolved=[p for p in result['proposals'] if p['classification']=='unresolved_audio_only_evidence']
 assert len(unresolved)==1 and unresolved[0]['decision']=='no_intervention'
